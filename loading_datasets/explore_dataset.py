"""
explore_dataset.py
Demonstrates slicing, filtering, shuffling, and converting datasets.
"""

import torch
from datasets import load_dataset

# ──────────────────────────────────────────────
# Check for Apple GPU (MPS) availability
# ──────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# ──────────────────────────────────────────────
# Load only the training split of IMDB (faster than loading all splits)
# ──────────────────────────────────────────────
train_ds = load_dataset("imdb", split="train")
print(f"Training set size: {len(train_ds)}")

# ──────────────────────────────────────────────
# Slicing: access rows by index or range
# Indexing returns a dict; slicing returns a dict of lists.
# ──────────────────────────────────────────────
print("\n=== Slicing ===")
single = train_ds[0]
print(f"Row 0 label: {single['label']}")

batch = train_ds[:3]
print(f"First 3 labels: {batch['label']}")

# ──────────────────────────────────────────────
# Filtering: keep only positive reviews (label == 1)
# The filter function applies a callable row-by-row.
# ──────────────────────────────────────────────
print("\n=== Filtering ===")
positive_ds = train_ds.filter(lambda row: row["label"] == 1)
print(f"Positive reviews: {len(positive_ds)}")

# ──────────────────────────────────────────────
# Shuffling and selecting a small subset
# Useful for quick prototyping and sanity checks.
# ──────────────────────────────────────────────
print("\n=== Shuffle + Select ===")
small_ds = train_ds.shuffle(seed=42).select(range(5))
for i, row in enumerate(small_ds):
    snippet = row["text"][:80].replace("\n", " ")
    print(f"  [{i}] label={row['label']}  {snippet}...")

# ──────────────────────────────────────────────
# Convert to pandas DataFrame for ad-hoc analysis
# ──────────────────────────────────────────────
print("\n=== Pandas Conversion ===")
df = train_ds.to_pandas()
print(f"DataFrame shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# ──────────────────────────────────────────────
# Train/test split from a single Dataset
# Handy when your data has no predefined splits.
# ──────────────────────────────────────────────
print("\n=== Train/Test Split ===")
split_result = train_ds.train_test_split(test_size=0.1, seed=42)
print(f"New train size: {len(split_result['train'])}")
print(f"New test size:  {len(split_result['test'])}")