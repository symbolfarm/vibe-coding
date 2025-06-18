# Basic LLM Training with PyTorch and TinyStories

A simple implementation of a transformer-based language model trained on the TinyStories dataset. Designed to fit on a single RTX 4090 GPU with ~10M parameters.

## Features

- **Transformer Architecture**: Decoder-only transformer with multi-head attention
- **Dual Tokenization**: Support for both character-level and GPT-2 BPE tokenization  
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Checkpointing**: Automatic model saving and resuming
- **Evaluation Tools**: Comprehensive model evaluation including perplexity and generation quality
- **RTX 4090 Optimized**: Configured for single GPU training

## Model Architecture

- **Parameters**: ~10M (configurable)
- **Layers**: 8 transformer blocks
- **Attention Heads**: 8
- **Model Dimension**: 256
- **Feed Forward**: 1024
- **Max Sequence Length**: 512 tokens
- **Dropout**: 0.1

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Dataset

The TinyStories dataset will be automatically downloaded on first run. The training script uses the first 10,000 stories for faster experimentation.

## Usage

### Training

Start training with default settings:

```bash
python train.py
```

The training will:
- Download TinyStories dataset automatically
- Create checkpoints in `./checkpoints/`
- Log metrics to `./logs/`
- Resume from the latest checkpoint if available

### Configuration

Modify hyperparameters in `config.py`:

```python
# Model size
config.model.n_layers = 8        # Number of transformer layers
config.model.d_model = 256       # Model dimension
config.model.n_heads = 8         # Attention heads

# Training
config.training.batch_size = 32  # Batch size
config.training.learning_rate = 3e-4
config.training.max_steps = 100000

# Tokenizer
config.training.tokenizer_type = "gpt2"  # or "char"
```

### Tokenizer Comparison

**GPT-2 BPE Tokenizer** (default):
- Vocabulary: ~50,000 tokens
- Better compression, fewer tokens per text
- More standard for modern LLMs

**Character-level Tokenizer**:
- Vocabulary: ~100 characters  
- Simpler, no out-of-vocabulary issues
- Longer sequences for same text

Switch tokenizers by changing `config.training.tokenizer_type` in `config.py`.

### Evaluation

Evaluate a trained model:

```bash
# Full evaluation with metrics
python eval.py --checkpoint checkpoints/best_model.pt

# Generate text from prompts
python eval.py --checkpoint checkpoints/best_model.pt --generate "Once upon a time" "The little girl"

# Plot training curves
python eval.py --checkpoint checkpoints/best_model.pt --plot
```

### Monitoring Training

Training metrics are logged to `logs/metrics.jsonl`. Each line contains:

```json
{
  "step": 1000,
  "train_loss": 2.45,
  "val_loss": 2.38,
  "lr": 0.0003,
  "timestamp": "2024-01-01T12:00:00"
}
```

## File Structure

```
05/
├── config.py          # Model and training configuration
├── model.py           # Transformer model implementation  
├── data.py            # Dataset loading and tokenization
├── train.py           # Training loop with checkpointing
├── eval.py            # Model evaluation and generation
├── requirements.txt   # Python dependencies
└── README.md          # This file

# Generated during training:
├── data/              # Downloaded dataset
├── checkpoints/       # Model checkpoints
└── logs/              # Training metrics
```

## Memory Usage

Configured for RTX 4090 (24GB VRAM):
- **Model**: ~40MB (10M parameters)
- **Training batch**: ~2GB (batch_size=32, seq_len=512)
- **Optimizer states**: ~80MB
- **Total**: ~3-4GB VRAM usage

## Scaling the Model

To adjust model size, modify `config.py`:

```python
# Smaller model (~5M parameters)
config.model.n_layers = 6
config.model.d_model = 192
config.model.d_ff = 768

# Larger model (~25M parameters)  
config.model.n_layers = 12
config.model.d_model = 384
config.model.d_ff = 1536
```

## Training Tips

1. **Start Small**: Begin with the default 10M parameter model
2. **Monitor Validation**: Watch for overfitting (val_loss > train_loss)
3. **Checkpointing**: Training automatically resumes from latest checkpoint
4. **Generation**: Use `temperature=0.8` and `top_k=50` for balanced generation
5. **Tokenizer Choice**: GPT-2 BPE generally works better for English text

## Expected Results

With default settings after ~50k steps:
- **Training Loss**: ~1.5-2.0
- **Validation Loss**: ~2.0-2.5  
- **Perplexity**: ~7-12
- **Generation**: Coherent short stories with some repetition

## Troubleshooting

**CUDA Out of Memory**: Reduce `batch_size` in `config.py`

**Slow Training**: Ensure CUDA is available and `use_amp=True`

**Poor Generation**: Try different temperature/top_k values or train longer

**Download Issues**: Check internet connection, dataset downloads to `data/`

## License

MIT License - see parent directory for details.