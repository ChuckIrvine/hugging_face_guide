"""
Advanced Multi-Step Preprocessing Pipeline
============================================
Chains multiple map/filter operations and demonstrates
disk caching for reproducible, efficient workflows.
"""

import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

# ---------------------------------------------------------------
# Device detection – Apple MPS / CUDA / CPU
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple Silicon GPU (MPS) detected.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ CUDA GPU detected.")
else:
    device = torch.device("cpu")
    print("ℹ️  Using CPU.")

SAVE_PATH = os.path.expanduser(
    "~/hugging_face_guide/data_preprocessing/processed_dataset"
)

# ---------------------------------------------------------------
# Step 1: Load a small subset for fast demonstration
# ---------------------------------------------------------------
print("\n📥 Loading 5,000 examples from IMDB...")
dataset = load_dataset("imdb", split="train[:5000]")
print(f"   Loaded {len(dataset)} examples.")

# ---------------------------------------------------------------
# Step 2: Lowercase normalization
# ---------------------------------------------------------------
print("\n🔡 Lowercasing text...")
dataset = dataset.map(
    lambda ex: {"text": ex["text"].lower()},
    desc="Lowercasing",
)

# ---------------------------------------------------------------
# Step 3: Add a length_bin feature (short / medium / long)
# ---------------------------------------------------------------
print("\n📏 Computing length bins...")

def add_length_bin(example):
    """Classify review length into short, medium, or long."""
    char_len = len(example["text"])
    if char_len < 500:
        bin_label = "short"
    elif char_len < 2000:
        bin_label = "medium"
    else:
        bin_label = "long"
    return {"length_bin": bin_label}

dataset = dataset.map(add_length_bin, desc="Length binning")
print(f"   Columns: {dataset.column_names}")

# ---------------------------------------------------------------
# Step 4: Filter to keep only medium and long reviews
# ---------------------------------------------------------------
print("\n🔍 Filtering to medium and long reviews...")
dataset = dataset.filter(
    lambda ex: ex["length_bin"] in ("medium", "long"),
    desc="Filtering by length",
)
print(f"   Remaining: {len(dataset)} examples")

# ---------------------------------------------------------------
# Step 5: Batched tokenization
# ---------------------------------------------------------------
print("\n🔤 Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

dataset = dataset.map(
    lambda batch: tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    ),
    batched=True,
    batch_size=512,
    num_proc=2,
    desc="Tokenizing",
)

# ---------------------------------------------------------------
# Step 6: Clean up columns and set format
# ---------------------------------------------------------------
dataset = dataset.remove_columns(["text", "length_bin"])
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ---------------------------------------------------------------
# Step 7: Save to disk and reload to verify caching
# ---------------------------------------------------------------
print(f"\n💾 Saving processed dataset to {SAVE_PATH}...")
dataset.save_to_disk(SAVE_PATH)

print("📂 Reloading from disk...")
reloaded = load_from_disk(SAVE_PATH)
reloaded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

print(f"\n✅ Reloaded dataset: {len(reloaded)} examples")
print(f"   Sample input_ids shape: {reloaded[0]['input_ids'].shape}")
print(f"   Device target: {device}")