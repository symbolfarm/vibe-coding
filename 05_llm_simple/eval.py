import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from config import Config
from model import create_model
from data import load_and_preprocess_data

class ModelEvaluator:
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load data (we need tokenizer)
        print("Loading data...")
        _, self.val_loader, self.tokenizer = load_and_preprocess_data(config)
        
        # Create and load model
        print("Loading model...")
        self.model = create_model(config.model).to(self.device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.step = checkpoint.get('step', 0)
            print(f"Loaded checkpoint from step {self.step}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model.eval()
    
    @torch.no_grad()
    def calculate_perplexity(self, num_batches: int = None) -> float:
        """Calculate perplexity on validation set"""
        total_loss = 0
        total_tokens = 0
        num_processed = 0
        
        for batch in self.val_loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            logits, loss = self.model(x, y)
            
            # Count non-padding tokens
            mask = (y != 0).float()
            tokens_in_batch = mask.sum().item()
            
            total_loss += loss.item() * tokens_in_batch
            total_tokens += tokens_in_batch
            num_processed += 1
            
            if num_batches and num_processed >= num_batches:
                break
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    @torch.no_grad()
    def generate_samples(self, prompts: List[str], max_tokens: int = 100, 
                        temperature: float = 0.8, top_k: int = 50) -> List[str]:
        """Generate text samples from prompts"""
        samples = []
        
        for prompt in prompts:
            if prompt:
                tokens = self.tokenizer.encode(prompt)
            else:
                tokens = [0]  # Start with pad token
            
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )
            
            generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
            samples.append(generated_text)
        
        return samples
    
    @torch.no_grad()
    def analyze_attention(self, text: str, layer_idx: int = -1) -> Dict:
        """Analyze attention patterns for given text"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.config.model.max_seq_len:
            tokens = tokens[:self.config.model.max_seq_len]
        
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Forward pass with attention extraction
        x = self.model.token_embedding(input_ids)
        x = self.model.pos_encoding(x)
        
        attentions = []
        for i, block in enumerate(self.model.blocks):
            # Get attention weights from the block
            ln_x = block.ln1(x)
            qkv = block.attn.qkv(ln_x)
            q, k, v = qkv.split(self.config.model.d_model, dim=2)
            
            B, T, C = q.shape
            q = q.view(B, T, self.config.model.n_heads, -1).transpose(1, 2)
            k = k.view(B, T, self.config.model.n_heads, -1).transpose(1, 2)
            
            # Attention weights
            att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(q.size(-1)))
            att = att.masked_fill(block.attn.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            
            attentions.append(att.cpu().numpy())
            
            # Continue forward pass
            x = x + block.attn(ln_x)
            x = x + block.mlp(block.ln2(x))
        
        return {
            'tokens': [self.tokenizer.decode([t]) for t in tokens],
            'attentions': attentions,
            'layer_idx': layer_idx if layer_idx >= 0 else len(attentions) - 1
        }
    
    def evaluate_generation_quality(self, num_samples: int = 10) -> Dict:
        """Evaluate generation quality with various metrics"""
        prompts = [
            "Once upon a time",
            "The little girl",
            "In a forest",
            "The magic",
            "Every day",
            "",  # Unconditional generation
        ] * (num_samples // 6 + 1)
        prompts = prompts[:num_samples]
        
        samples = self.generate_samples(prompts, max_tokens=100, temperature=0.8)
        
        # Calculate basic statistics
        lengths = [len(sample.split()) for sample in samples]
        unique_words = set()
        total_words = 0
        
        for sample in samples:
            words = sample.split()
            unique_words.update(words)
            total_words += len(words)
        
        # Repetition analysis
        repetition_scores = []
        for sample in samples:
            words = sample.split()
            if len(words) > 1:
                unique_ratio = len(set(words)) / len(words)
                repetition_scores.append(1 - unique_ratio)
        
        return {
            'num_samples': len(samples),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'vocabulary_size': len(unique_words),
            'avg_repetition': np.mean(repetition_scores) if repetition_scores else 0,
            'samples': samples[:5]  # First 5 samples for inspection
        }
    
    def plot_training_curves(self, log_dir: str, save_path: str = None):
        """Plot training and validation loss curves"""
        metrics_file = os.path.join(log_dir, 'metrics.jsonl')
        
        if not os.path.exists(metrics_file):
            print(f"Metrics file not found: {metrics_file}")
            return
        
        # Load metrics
        train_steps, train_losses = [], []
        val_steps, val_losses = [], []
        
        with open(metrics_file, 'r') as f:
            for line in f:
                metrics = json.loads(line.strip())
                train_steps.append(metrics['step'])
                train_losses.append(metrics['train_loss'])
                
                if metrics['val_loss'] is not None:
                    val_steps.append(metrics['step'])
                    val_losses.append(metrics['val_loss'])
        
        # Create plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_steps, train_losses, label='Training Loss', alpha=0.7)
        if val_losses:
            plt.plot(val_steps, val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if len(train_losses) > 100:
            # Smooth the training loss for better visualization
            window = len(train_losses) // 50
            smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            smoothed_steps = train_steps[window-1:]
            plt.plot(smoothed_steps, smoothed, label=f'Training Loss (smoothed, window={window})')
        else:
            plt.plot(train_steps, train_losses, label='Training Loss')
            
        if val_losses:
            plt.plot(val_steps, val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Progress (Smoothed)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        plt.show()
    
    def run_full_evaluation(self) -> Dict:
        """Run comprehensive evaluation"""
        print("Running comprehensive evaluation...")
        
        results = {}
        
        # Perplexity
        print("Calculating perplexity...")
        results['perplexity'] = self.calculate_perplexity(num_batches=100)
        
        # Generation quality
        print("Evaluating generation quality...")
        results['generation'] = self.evaluate_generation_quality(num_samples=20)
        
        # Model info
        results['model_info'] = {
            'parameters': self.model.count_parameters(),
            'training_step': getattr(self, 'step', 0),
            'vocab_size': self.config.model.vocab_size,
            'tokenizer_type': self.config.training.tokenizer_type
        }
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained LLM')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate training curve plots')
    parser.add_argument('--generate', type=str, nargs='*',
                       help='Generate text from given prompts')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Create evaluator
    evaluator = ModelEvaluator(config, args.checkpoint)
    
    if args.generate:
        # Generate samples
        print("Generating samples...")
        samples = evaluator.generate_samples(args.generate, max_tokens=150)
        for i, (prompt, sample) in enumerate(zip(args.generate, samples)):
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Generated: {sample}")
    else:
        # Full evaluation
        results = evaluator.run_full_evaluation()
        
        # Print results
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Model Parameters: {results['model_info']['parameters']:,}")
        print(f"Training Step: {results['model_info']['training_step']:,}")
        print(f"Vocabulary Size: {results['model_info']['vocab_size']:,}")
        print(f"Tokenizer Type: {results['model_info']['tokenizer_type']}")
        print(f"\nPerplexity: {results['perplexity']:.2f}")
        print(f"\nGeneration Quality:")
        print(f"  Average Length: {results['generation']['avg_length']:.1f} words")
        print(f"  Vocabulary Size: {results['generation']['vocabulary_size']:,}")
        print(f"  Average Repetition: {results['generation']['avg_repetition']:.3f}")
        
        print(f"\nSample Generations:")
        for i, sample in enumerate(results['generation']['samples']):
            print(f"  {i+1}: {sample[:100]}...")
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    if args.plot:
        evaluator.plot_training_curves(
            config.training.log_dir,
            save_path='training_curves.png'
        )

if __name__ == "__main__":
    main()