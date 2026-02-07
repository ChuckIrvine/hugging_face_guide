"""
load_from_hub.py
Demonstrates loading a dataset directly from the Hugging Face Hub,
inspecting its structure, and performing basic exploration.
"""

import torch
from datasets import load_dataset

# ──────────────────────────────────────────────
# Check for Apple GPU (MPS) availability
# ──────────────────────────────────────────────
if torch.backends.mps.is_available():
    print("Apple MPS GPU detected — available for downstream tasks.")
else:
    print("No Apple MPS GPU detected — CPU will be used.")

# ──────────────────────────────────────────────
# Load the IMDB dataset from the Hugging Face Hub
# This downloads and caches the data locally on first run.
# ──────────────────────────────────────────────
dataset = load_dataset("imdb")

# ──────────────────────────────────────────────
# Inspect the DatasetDict: list splits and their sizes
# ──────────────────────────────────────────────
print("\n=== Dataset Overview ===")
print(f"Type: {type(dataset)}")
print(f"Splits: {list(dataset.keys())}")
for split_name, split_data in dataset.items():
    print(f"  {split_name}: {len(split_data)} rows")

# ──────────────────────────────────────────────
# Examine the feature schema (column names, types)
# ──────────────────────────────────────────────
print("\n=== Features ===")
print(dataset["train"].features)

# ──────────────────────────────────────────────
# Preview the first record in the training split
# ──────────────────────────────────────────────
print("\n=== First Training Example ===")
example = dataset["train"][0]
print(f"Label: {example['label']}")
print(f"Text (first 200 chars): {example['text'][:200]}...")