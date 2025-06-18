import torch
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ModelConfig:
    vocab_size: int = 50257  # GPT-2 vocab size, will be overridden for char-level
    max_seq_len: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.1
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100000
    eval_interval: int = 1000
    save_interval: int = 5000
    
    # Mixed precision training
    use_amp: bool = True
    
    # Dataset
    tokenizer_type: Literal["char", "gpt2"] = "gpt2"
    train_split: float = 0.9
    
    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42