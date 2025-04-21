# ------------------------------------------------------------
# Transformer-based Language Model for Word Prediction
# ------------------------------------------------------------
# This script trains a small Transformer model to predict
# the central word given surrounding context words.
#
# User can select:
# - Single-head (self-attention) like standard Transformer
# - Multi-head attention (advanced)
#
# ------------------------------------------------------------
# Author: Designed for UPC SLPDL course and after developed by İsmail Eren Küçükali
# 
# For educational purposes, works best with large datasets.
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

TRAIN_FILE = "en.wiki.train.tokens"  # Training data file (tokens separated by spaces)
VALID_FILE = "en.wiki.valid.tokens"  # Validation data file
TEST_FILE = "en.wiki.test.tokens"    # Test data file (optional)

EMBEDDING_DIM = 256          # Dimension of word embeddings
BATCH_SIZE = 32              # Batch size for training
EPOCHS = 30                  # Number of training epochs
CONTEXT_SIZE = 5             # How many words before and after the central word
LEARNING_RATE = 0.001        # Initial learning rate

USE_MULTIHEAD = False        # Set True for Multi-Head Attention, False for Single-Head
NUM_HEADS = 4                # Number of attention heads (only used if USE_MULTIHEAD = True)
NUM_LAYERS = 1               # Number of stacked Transformer layers

# HELPER FUNCTIONS
def load_tokens(file_path):
    """Loads tokens from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens = f.read().split()
    return tokens

def build_vocab(tokens, min_count=1):
    """Builds vocabulary from tokens."""
    counter = Counter(tokens)
    vocab = {word: idx for idx, (word, count) in enumerate(counter.items()) if count >= min_count}
    idx_to_word = {idx: word for word, idx in vocab.items()}
    return vocab, idx_to_word

# DATASET CLASS
class LanguageModelDataset(Dataset):
    """Custom Dataset for Word Prediction Task."""
    def __init__(self, tokens, vocab, context_size):
        self.data = []
        self.vocab = vocab
        self.context_size = context_size
        for i in range(context_size, len(tokens) - context_size):
            context = [tokens[i + j] for j in range(-context_size, context_size + 1) if j != 0]
            target = tokens[i]
            if all(word in vocab for word in context + [target]):
                context_indices = [vocab[w] for w in context]
                target_index = vocab[target]
                self.data.append((context_indices, target_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

# TRANSFORMER BLOCK
class TransformerBlock(nn.Module):
    """Single Transformer Block with Self-Attention or Multi-Head Attention."""
    def __init__(self, embedding_dim, use_multihead, num_heads):
        super(TransformerBlock, self).__init__()
        if use_multihead:
            self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        else:
            self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)
        return x

# FULL LANGUAGE MODEL
class TransformerLanguageModel(nn.Module):
    """Complete Language Model using stacked Transformer Blocks."""
    def __init__(self, vocab_size, embedding_dim, context_size, use_multihead, num_heads, num_layers):
        super(TransformerLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embedding_dim, use_multihead, num_heads) for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        x = self.embedding(context_idxs)  # (batch_size, context_len, embedding_dim)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, context_len)
        pooled = self.pool(x).squeeze(-1)  # (batch_size, embedding_dim)
        output = self.linear(pooled)  # (batch_size, vocab_size)
        return output

# TRAINING FUNCTION
def train_model(model, train_loader, valid_loader, optimizer, criterion, device):
    """Training Loop."""
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for contexts, targets in train_loader:
            contexts, targets = contexts.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(contexts)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {total_loss/len(train_loader):.4f}")
        validate_model(model, valid_loader, criterion, device)

# VALIDATION FUNCTION
def validate_model(model, valid_loader, criterion, device):
    """Validation Loop."""
    model.eval()
    if len(valid_loader) == 0:
        print("Warning: Validation set is empty. Skipping validation.")
        return
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for contexts, targets in valid_loader:
            contexts, targets = contexts.to(device), targets.to(device)
            output = model(contexts)
            loss = criterion(output, targets)
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    perplexity = np.exp(total_loss/len(valid_loader))
    accuracy = correct/total if total > 0 else 0
    print(f"Validation Loss: {total_loss/len(valid_loader):.4f}, Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.4f}")

# MAIN EXECUTION
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tokens = load_tokens(TRAIN_FILE)
    valid_tokens = load_tokens(VALID_FILE)
    test_tokens = load_tokens(TEST_FILE)

    full_tokens = train_tokens + valid_tokens
    vocab, idx_to_word = build_vocab(full_tokens)

    train_dataset = LanguageModelDataset(train_tokens, vocab, CONTEXT_SIZE)
    valid_dataset = LanguageModelDataset(valid_tokens, vocab, CONTEXT_SIZE)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    model = TransformerLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        context_size=CONTEXT_SIZE,
        use_multihead=USE_MULTIHEAD,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, valid_loader, optimizer, criterion, device)
