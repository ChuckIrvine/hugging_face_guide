"""
load_from_local.py
Demonstrates loading datasets from local CSV and JSONL files.
"""

import json
import torch
from datasets import load_dataset

# ──────────────────────────────────────────────
# Check for Apple GPU (MPS) availability
# ──────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# ──────────────────────────────────────────────
# Create a sample CSV file for demonstration
# ──────────────────────────────────────────────
csv_path = "sample_reviews.csv"
with open(csv_path, "w") as f:
    f.write("text,label\n")
    f.write('"Great film, loved every minute of it.",positive\n')
    f.write('"Terrible acting and a boring plot.",negative\n')
    f.write('"A masterpiece of modern cinema.",positive\n')

# ──────────────────────────────────────────────
# Load the CSV file as a Dataset
# ──────────────────────────────────────────────
csv_dataset = load_dataset("csv", data_files=csv_path, split="train")
print("\n=== CSV Dataset ===")
print(csv_dataset)
print(csv_dataset.to_pandas())

# ──────────────────────────────────────────────
# Create a sample JSON Lines file for demonstration
# ──────────────────────────────────────────────
jsonl_path = "sample_reviews.jsonl"
records = [
    {"text": "Absolutely wonderful storytelling.", "rating": 5},
    {"text": "Not worth the ticket price.", "rating": 1},
]
with open(jsonl_path, "w") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")

# ──────────────────────────────────────────────
# Load the JSONL file as a Dataset
# ──────────────────────────────────────────────
jsonl_dataset = load_dataset("json", data_files=jsonl_path, split="train")
print("\n=== JSONL Dataset ===")
print(jsonl_dataset)
print(jsonl_dataset.to_pandas())