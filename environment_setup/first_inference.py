"""
first_inference.py
Load a pre-trained sentiment-analysis model via the pipeline API
and run inference on sample text.
"""

import torch
from transformers import pipeline

# ── Device detection ────────────────────────────────────────────────
# Check for Apple Silicon GPU first, then NVIDIA CUDA, then CPU.
if torch.backends.mps.is_available():
    device = 0  # pipeline accepts 0 for the first available GPU
    device_name = "Apple MPS"
    # Note: some pipelines may not fully support MPS yet.
    # If you encounter errors, set device = -1 to fall back to CPU.
elif torch.cuda.is_available():
    device = 0
    device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
else:
    device = -1  # -1 tells the pipeline to use CPU
    device_name = "CPU"

print(f"Running inference on: {device_name}\n")

# ── Create the sentiment-analysis pipeline ──────────────────────────
# The first call downloads the model and tokenizer. Subsequent calls
# load from the local cache (~/.cache/huggingface/).
classifier = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
)

# ── Run predictions on sample inputs ───────────────────────────────
samples = [
    "Hugging Face makes NLP incredibly accessible and fun!",
    "I spent three hours debugging a dependency conflict.",
]

results = classifier(samples)

# ── Display results ─────────────────────────────────────────────────
for text, result in zip(samples, results):
    label = result["label"]
    score = result["score"]
    print(f"Text  : {text}")
    print(f"Label : {label}  (confidence: {score:.4f})")
    print("-" * 60)