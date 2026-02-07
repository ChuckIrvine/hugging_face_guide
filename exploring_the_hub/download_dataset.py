"""
download_dataset.py
Downloads a dataset from the Hugging Face Hub and inspects
its structure, splits, and sample rows.
"""

import torch
from datasets import load_dataset

# -------------------------------------------------------
# Check for Apple Silicon GPU (MPS) availability
# -------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple MPS GPU detected — available for downstream tasks.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU detected.")
else:
    device = torch.device("cpu")
    print("No GPU detected — using CPU.")

print()

# -------------------------------------------------------
# Download the rotten_tomatoes dataset from the Hub
# -------------------------------------------------------
dataset = load_dataset("rotten_tomatoes")

# -------------------------------------------------------
# Inspect the dataset object: splits, features, size
# -------------------------------------------------------
print("=== Dataset Structure ===")
print(dataset)
print()

print("=== Features ===")
print(dataset["train"].features)
print()

print(f"Training samples   : {len(dataset['train']):,}")
print(f"Validation samples : {len(dataset['validation']):,}")
print(f"Test samples       : {len(dataset['test']):,}")
print()

# -------------------------------------------------------
# Display a few sample rows from the training split
# -------------------------------------------------------
print("=== Sample Training Rows ===")
for i in range(3):
    row = dataset["train"][i]
    label_name = "positive" if row["label"] == 1 else "negative"
    print(f"  [{label_name}] {row['text'][:100]}...")
    print()