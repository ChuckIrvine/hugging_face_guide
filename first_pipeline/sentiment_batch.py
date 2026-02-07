"""
sentiment_batch.py
Demonstrates batch inference with the Pipeline API, processing
multiple sentences in a single call for efficiency.
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
# Instantiate the sentiment-analysis pipeline
# ---------------------------------------------------------------
classifier = pipeline("sentiment-analysis", device=device)

# ---------------------------------------------------------------
# Define a batch of sentences with mixed sentiment
# ---------------------------------------------------------------
sentences = [
    "The new update is fantastic—everything runs so smoothly now.",
    "I'm really disappointed with the customer service I received.",
    "The weather today is okay, nothing special.",
    "This is the worst product I have ever purchased.",
    "Hugging Face makes NLP accessible to everyone!",
]

# ---------------------------------------------------------------
# Run batch inference and display results
# ---------------------------------------------------------------
results = classifier(sentences)

for sentence, result in zip(sentences, results):
    label = result["label"]
    score = result["score"]
    print(f"[{label} {score:.4f}]  {sentence}")