import json
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------
# Device selection: prefer Apple MPS GPU, then CUDA, then CPU
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ---------------------------------------------------------------
# Download model and tokenizer (cached after first run)
# ---------------------------------------------------------------
model_name = "distilbert-base-uncased"
print(f"\nDownloading / loading '{model_name}'...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# ---------------------------------------------------------------
# Locate the snapshot directory inside the HF cache
# ---------------------------------------------------------------
from huggingface_hub import snapshot_download

cache_dir = snapshot_download(model_name, local_files_only=True)
print(f"\nLocal cache directory:\n  {cache_dir}\n")

# ---------------------------------------------------------------
# Walk the cached directory and describe each file
# ---------------------------------------------------------------
print("=" * 65)
print(f"{'File':<35} {'Size':>10}")
print("=" * 65)

for file_path in sorted(Path(cache_dir).rglob("*")):
    if file_path.is_dir():
        continue
    rel = file_path.relative_to(cache_dir)
    size_kb = file_path.stat().st_size / 1024
    print(f"{str(rel):<35} {size_kb:>8.1f} KB")

    # Pretty-print JSON files (first 20 lines)
    if file_path.suffix == ".json":
        with open(file_path, "r") as f:
            data = json.load(f)
        lines = json.dumps(data, indent=2).split("\n")
        for line in lines[:20]:
            print(f"    {line}")
        if len(lines) > 20:
            print(f"    ... ({len(lines) - 20} more lines)")
        print()

print("=" * 65)