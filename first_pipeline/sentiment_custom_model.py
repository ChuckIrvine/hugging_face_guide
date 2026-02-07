"""
sentiment_custom_model.py
Shows how to specify a particular model from the Hugging Face Hub
instead of using the default checkpoint.
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
# Use a multilingual sentiment model trained on Twitter data.
# This model outputs labels from 1 star to 5 stars.
# ---------------------------------------------------------------
classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=device,
)

# ---------------------------------------------------------------
# Analyze text—this model supports multiple languages
# ---------------------------------------------------------------
texts = [
    "I love this product!",
    "Ce produit est terrible.",   # French: "This product is terrible."
    "Das Essen war mittelmäßig.", # German: "The food was mediocre."
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(f"[{result['label']}  {result['score']:.4f}]  {text}")