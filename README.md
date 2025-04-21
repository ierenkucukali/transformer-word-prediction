# Transformer-based Word Prediction Model

This repository contains a simple but flexible Transformer-based language model designed to predict a central word given its surrounding context.

---

## 🎯 Project Aim

The goal is to build a small Transformer model that:
- Learns word embeddings
- Predicts the central word from previous and next words
- Supports both Self-Attention and Multi-Head Attention mechanisms

It is **inspired by CBOW (Continuous Bag of Words)**, but **powered by Transformer architectures**.

---

## 🛠️ Features

- Supports **Single-Head Self-Attention** (classic Transformer)
- Supports **Multi-Head Attention** (advanced)
- Easy to configure model size, batch size, context size, etc.
- Fully written in PyTorch
- Clear structure with comments for easy modifications

---

## 🧠 Self-Attention vs Multi-Head Attention

| Aspect | Self-Attention | Multi-Head Attention |
|:------|:---------------|:---------------------|
| Heads | Single attention head | Multiple attention heads |
| Information learned | Single type of relationship | Multiple parallel relationships |
| Expressiveness | Basic | Richer, captures more patterns |
| Computation | Simpler | Slightly heavier |

✅ You can easily switch between them by changing `USE_MULTIHEAD = True` or `False` in the code.

---

## 📚 Dataset Requirements

To train this model properly, **you need a big dataset**:

- **At least** 50,000 tokens for training
- **At least** 10,000 tokens for validation
- Tokens should be **space-separated words**, like:

```text
the cat sat on the mat
the dog barked at the stranger
the sun rose behind the mountains
