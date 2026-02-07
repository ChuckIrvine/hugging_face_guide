"""
sentiment_basic.py
Demonstrates the simplest possible use of the Hugging Face Pipeline API
to perform sentiment analysis on a single sentence.
"""

import torch
from transformers import pipeline

# ---------------------------------------------------------------
# Device selection: prefer Apple MPS, then CUDA, then fall back to CPU
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# ---------------------------------------------------------------
# Create a sentiment-analysis pipeline.
# The first call downloads and caches the default model (~260 MB).
# ---------------------------------------------------------------
classifier = pipeline("sentiment-analysis", device=device)

# ---------------------------------------------------------------
# Run inference on a single input string
# ---------------------------------------------------------------
result = classifier("I absolutely love learning about transformers!")
print(result)