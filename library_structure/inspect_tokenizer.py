import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# ---------------------------------------------------------------
# Apple MPS check
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    print("Apple MPS GPU detected.\n")

# ---------------------------------------------------------------
# Load tokenizer and display vocabulary stats
# ---------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print(f"Tokenizer class  : {tokenizer.__class__.__name__}")
print(f"Vocabulary size   : {tokenizer.vocab_size}")
print(f"Is fast tokenizer : {tokenizer.is_fast}\n")

# ---------------------------------------------------------------
# List special tokens and their integer IDs
# ---------------------------------------------------------------
print("Special tokens:")
for name in ["bos_token", "eos_token", "unk_token", "sep_token",
             "pad_token", "cls_token", "mask_token"]:
    token = getattr(tokenizer, name, None)
    if token:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {name:<15} -> '{token}' (id: {token_id})")

# ---------------------------------------------------------------
# Tokenize a sample sentence and inspect output
# ---------------------------------------------------------------
text = "Hugging Face makes NLP accessible."
encoded = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
print(f"\nInput text : {text}")
print(f"Token IDs  : {encoded['input_ids'][0].tolist()}")
print(f"Tokens     : {tokens}")

# ---------------------------------------------------------------
# Read the raw special_tokens_map.json from cache
# ---------------------------------------------------------------
cache_dir = snapshot_download("distilbert-base-uncased", local_files_only=True)
stm_path = Path(cache_dir) / "special_tokens_map.json"
if stm_path.exists():
    with open(stm_path) as f:
        stm = json.load(f)
    print("\nspecial_tokens_map.json contents:")
    print(json.dumps(stm, indent=2))