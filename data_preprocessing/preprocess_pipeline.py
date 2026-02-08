"""
Data Preprocessing Pipeline
============================
Demonstrates filter, map (batched), column removal, and format
setting to produce model-ready PyTorch tensors from raw text.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------
# Device detection – prefer Apple MPS, then CUDA, then CPU
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

# ---------------------------------------------------------------
# Step 1: Load raw dataset
# ---------------------------------------------------------------
print("\n📥 Loading IMDB dataset (train split)...")
dataset = load_dataset("imdb", split="train")
print(f"   Raw size: {len(dataset)} examples")
print(f"   Columns : {dataset.column_names}")
print(f"   Sample  : {dataset[0]['text'][:120]}...")

# ---------------------------------------------------------------
# Step 2: Filter – remove very short reviews (< 30 characters)
# ---------------------------------------------------------------
print("\n🔍 Filtering short reviews...")
filtered_dataset = dataset.filter(
    lambda example: len(example["text"]) >= 30,
    desc="Filtering short reviews",
)
print(f"   Size after filter: {len(filtered_dataset)} examples")

# ---------------------------------------------------------------
# Step 3: Tokenize with batched map for speed
# ---------------------------------------------------------------
print("\n🔤 Tokenizing with DistilBERT tokenizer (batched)...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_batch(batch):
    """
    Tokenize a batch of text examples. Padding and truncation
    ensure uniform sequence lengths suitable for model input.
    """
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

tokenized_dataset = filtered_dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=1000,
    num_proc=4,
    desc="Tokenizing",
)
print(f"   Columns after map: {tokenized_dataset.column_names}")

# ---------------------------------------------------------------
# Step 4: Remove raw text column and set PyTorch format
# ---------------------------------------------------------------
print("\n🧹 Removing 'text' column and setting PyTorch format...")
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
print(f"   Final columns: {tokenized_dataset.column_names}")
print(f"   Format       : {tokenized_dataset.format}")

# ---------------------------------------------------------------
# Step 5: Inspect a sample
# ---------------------------------------------------------------
sample = tokenized_dataset[0]
print("\n📊 Sample output:")
print(f"   input_ids shape    : {sample['input_ids'].shape}")
print(f"   attention_mask shape: {sample['attention_mask'].shape}")
print(f"   label              : {sample['label']}")
print(f"   dtype              : {sample['input_ids'].dtype}")
print(f"\n✅ Pipeline complete. {len(tokenized_dataset)} examples ready for training on {device}.")