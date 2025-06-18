import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import time
import json
from datetime import datetime
from typing import Optional

from config import Config
from model import create_model
from data import load_and_preprocess_data

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # Create directories
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(config.training.log_dir, exist_ok=True)
        
        # Load data
        print("Loading data...")
        self.train_loader, self.val_loader, self.tokenizer = load_and_preprocess_data(config)
        
        # Create model
        print("Creating model...")
        self.model = create_model(config.model).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(config.training.beta1, config.training.beta2)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.training.max_steps,
            eta_min=config.training.learning_rate * 0.1
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Training state
        self.step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def get_lr(self):
        """Get current learning rate with warmup"""
        if self.step < self.config.training.warmup_steps:
            return self.config.training.learning_rate * self.step / self.config.training.warmup_steps
        return self.scheduler.get_last_lr()[0]
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.scaler:
            with autocast():
                logits, loss = self.model(x, y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, loss = self.model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            self.optimizer.step()
        
        # Update learning rate
        if self.step >= self.config.training.warmup_steps:
            self.scheduler.step()
        else:
            # Manual warmup
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            if self.scaler:
                with autocast():
                    logits, loss = self.model(x, y)
            else:
                logits, loss = self.model(x, y)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Limit validation batches for speed
            if num_batches >= 50:
                break
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Regular checkpoint
        checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir, 
            f'checkpoint_{self.step}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Best model checkpoint
        if is_best:
            best_path = os.path.join(self.config.training.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        # Latest checkpoint
        latest_path = os.path.join(self.config.training.checkpoint_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path} (step {self.step})")
    
    def log_metrics(self, train_loss: float, val_loss: Optional[float] = None):
        """Log training metrics"""
        self.train_losses.append((self.step, train_loss))
        
        if val_loss is not None:
            self.val_losses.append((self.step, val_loss))
        
        # Save metrics to file
        metrics = {
            'step': self.step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': self.get_lr(),
            'timestamp': datetime.now().isoformat()
        }
        
        log_file = os.path.join(self.config.training.log_dir, 'metrics.jsonl')
        with open(log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def generate_sample(self, prompt: str = "", max_tokens: int = 100):
        """Generate a sample from the model"""
        self.model.eval()
        
        if prompt:
            tokens = self.tokenizer.encode(prompt)
        else:
            tokens = [0]  # Start with pad token
        
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                input_ids, 
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50
            )
        
        generated_tokens = generated[0].cpu().tolist()
        return self.tokenizer.decode(generated_tokens)
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop"""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"Starting training from step {self.step}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        train_iter = iter(self.train_loader)
        
        while self.step < self.config.training.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Training step
            train_loss = self.train_step(batch)
            self.step += 1
            
            # Validation
            if self.step % self.config.training.eval_interval == 0:
                val_loss = self.validate()
                
                # Log metrics
                self.log_metrics(train_loss, val_loss)
                
                # Print progress
                elapsed = time.time() - start_time
                tokens_per_sec = (self.step * self.config.training.batch_size * self.config.model.max_seq_len) / elapsed
                
                print(f"Step {self.step:6d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"LR: {self.get_lr():.2e} | "
                      f"Tokens/sec: {tokens_per_sec:.0f}")
                
                # Generate sample
                if self.step % (self.config.training.eval_interval * 2) == 0:
                    sample = self.generate_sample("Once upon a time", max_tokens=50)
                    print(f"Sample: {sample[:200]}...")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
            
            else:
                # Log only training loss
                self.log_metrics(train_loss)
            
            # Save checkpoint
            if self.step % self.config.training.save_interval == 0:
                self.save_checkpoint()
        
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

def main():
    config = Config()
    trainer = Trainer(config)
    
    # Check for existing checkpoint
    latest_checkpoint = os.path.join(config.training.checkpoint_dir, 'latest.pt')
    resume_from = latest_checkpoint if os.path.exists(latest_checkpoint) else None
    
    trainer.train(resume_from=resume_from)

if __name__ == "__main__":
    main()