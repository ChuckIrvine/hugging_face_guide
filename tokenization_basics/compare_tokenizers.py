"""
compare_tokenizers.py

Compare tokenization outputs across BERT, GPT-2, and T5 architectures.
Demonstrates how different subword algorithms fragment identical text.
"""

import torch
from transformers import AutoTokenizer

# ---------------------------------------------------------------
# Device detection: check for Apple Silicon GPU (MPS), CUDA, or CPU
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("Device: CPU")

# ---------------------------------------------------------------
# Load tokenizers for three major architectures
# ---------------------------------------------------------------
tokenizers = {
    "BERT (WordPiece)": AutoTokenizer.from_pretrained("bert-base-uncased"),
    "GPT-2 (BPE)": AutoTokenizer.from_pretrained("gpt2"),
    "T5 (SentencePiece)": AutoTokenizer.from_pretrained("t5-small"),
}

# ---------------------------------------------------------------
# Define test sentences that expose tokenization differences
# ---------------------------------------------------------------
test_sentences = [
    "Tokenization is fundamental to NLP.",
    "The unhappiness of the llama was unforgettable.",
    "Hugging Face transformers simplify deep learning workflows.",
    "COVID-19 spread rapidly in 2020.",
]

# ---------------------------------------------------------------
# Print vocabulary sizes for each tokenizer
# ---------------------------------------------------------------
print("\n" + "=" * 65)
print("VOCABULARY SIZES")
print("=" * 65)
for name, tokenizer in tokenizers.items():
    print(f"  {name:25s} -> {tokenizer.vocab_size:,} tokens")

# ---------------------------------------------------------------
# Compare tokenizer outputs for each test sentence
# ---------------------------------------------------------------
for sentence in test_sentences:
    print("\n" + "=" * 65)
    print(f'Input: "{sentence}"')
    print("-" * 65)

    for name, tokenizer in tokenizers.items():
        # Encode the sentence into token IDs
        encoded = tokenizer(sentence, add_special_tokens=False)
        input_ids = encoded["input_ids"]

        # Decode individual IDs back to token strings for inspection
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        print(f"\n  {name}")
        print(f"    Tokens ({len(tokens):2d}): {tokens}")
        print(f"    IDs:          {input_ids}")

print("\n" + "=" * 65)
print("Comparison complete.")