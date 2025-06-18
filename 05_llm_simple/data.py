import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import requests
from typing import List, Tuple, Optional
import json
import pickle
from transformers import GPT2TokenizerFast
from config import Config

class CharTokenizer:
    def __init__(self, texts: List[str]):
        # Build character vocabulary
        chars = set()
        for text in texts:
            chars.update(text)
        
        self.chars = sorted(list(chars))
        self.vocab_size = len(self.chars)
        
        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode(self, indices: List[int]) -> str:
        return ''.join([self.idx_to_char[i] for i in indices if i in self.idx_to_char])
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'chars': self.chars,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls.__new__(cls)
        tokenizer.chars = data['chars']
        tokenizer.char_to_idx = data['char_to_idx']
        tokenizer.idx_to_char = data['idx_to_char']
        tokenizer.vocab_size = data['vocab_size']
        return tokenizer

class GPT2Tokenizer:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.vocab_size = len(self.tokenizer)
        
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def decode(self, indices: List[int]) -> str:
        return self.tokenizer.decode(indices)

class TinyStoriesDataset(Dataset):
    def __init__(self, data: List[List[int]], max_seq_len: int):
        self.data = data
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        
        # Truncate if too long
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        
        # Pad if too short
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
        
        # Input and target (shifted by 1)
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y

def download_tinystories(data_dir: str) -> str:
    """Download TinyStories dataset"""
    os.makedirs(data_dir, exist_ok=True)
    
    # Use the smaller version for faster download
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    filepath = os.path.join(data_dir, "TinyStories-train.txt")
    
    if not os.path.exists(filepath):
        print("Downloading TinyStories dataset...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download completed!")
    
    return filepath

def load_and_preprocess_data(config: Config) -> Tuple[DataLoader, DataLoader, object]:
    """Load and preprocess TinyStories data"""
    data_dir = config.training.data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    # Download dataset
    filepath = download_tinystories(data_dir)
    
    # Load stories
    print("Loading stories...")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into individual stories (assuming stories are separated by double newlines)
    stories = [story.strip() for story in text.split('\n\n') if story.strip()]
    
    # Limit dataset size for faster training
    stories = stories[:10000]  # Use first 10k stories
    
    print(f"Loaded {len(stories)} stories")
    
    # Create tokenizer
    tokenizer_path = os.path.join(data_dir, f'tokenizer_{config.training.tokenizer_type}.pkl')
    
    if config.training.tokenizer_type == "char":
        if os.path.exists(tokenizer_path):
            tokenizer = CharTokenizer.load(tokenizer_path)
        else:
            print("Building character tokenizer...")
            tokenizer = CharTokenizer(stories)
            tokenizer.save(tokenizer_path)
    else:  # gpt2
        tokenizer = GPT2Tokenizer()
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Update model config with actual vocab size
    config.model.vocab_size = tokenizer.vocab_size
    
    # Tokenize stories
    print("Tokenizing stories...")
    tokenized_stories = []
    for story in stories:
        tokens = tokenizer.encode(story)
        if len(tokens) > 10:  # Filter out very short stories
            tokenized_stories.append(tokens)
    
    print(f"Tokenized {len(tokenized_stories)} stories")
    
    # Split train/validation
    split_idx = int(len(tokenized_stories) * config.training.train_split)
    train_stories = tokenized_stories[:split_idx]
    val_stories = tokenized_stories[split_idx:]
    
    # Create datasets
    train_dataset = TinyStoriesDataset(train_stories, config.model.max_seq_len)
    val_dataset = TinyStoriesDataset(val_stories, config.model.max_seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, tokenizer

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    xs, ys = zip(*batch)
    
    # Pad sequences to same length
    max_len = max(len(x) for x in xs)
    
    padded_xs = []
    padded_ys = []
    
    for x, y in zip(xs, ys):
        pad_len = max_len - len(x)
        padded_x = F.pad(x, (0, pad_len), value=0)
        padded_y = F.pad(y, (0, pad_len), value=0)
        padded_xs.append(padded_x)
        padded_ys.append(padded_y)
    
    return torch.stack(padded_xs), torch.stack(padded_ys)