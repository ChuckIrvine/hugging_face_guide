"""
visualize_tokens.py

Generate an HTML visualization comparing token boundaries
across BERT, GPT-2, and T5 tokenizers side by side.
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path

# ---------------------------------------------------------------
# Device detection: check for Apple Silicon GPU (MPS), CUDA, or CPU
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    print("Device: Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
else:
    print("Device: CPU")

# ---------------------------------------------------------------
# Load tokenizers
# ---------------------------------------------------------------
tokenizer_configs = [
    ("BERT (WordPiece)", "bert-base-uncased"),
    ("GPT-2 (BPE)", "gpt2"),
    ("T5 (SentencePiece)", "t5-small"),
]
tokenizers = [(name, AutoTokenizer.from_pretrained(ckpt)) for name, ckpt in tokenizer_configs]

# ---------------------------------------------------------------
# Sentences to visualize
# ---------------------------------------------------------------
sentences = [
    "Tokenization is fundamental to NLP.",
    "The unhappiness of the llama was unforgettable.",
    "COVID-19 spread rapidly in 2020.",
]

# ---------------------------------------------------------------
# Alternating color palette for token spans
# ---------------------------------------------------------------
COLORS = ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#F8BBD0", "#D1C4E9", "#B2EBF2"]

def tokens_to_html(tokenizer, sentence):
    """Convert a sentence into colored HTML spans, one per token."""
    encoded = tokenizer(sentence, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    spans = []
    for i, token in enumerate(tokens):
        # Clean up subword prefixes for display
        display = token.replace("##", "·").replace("▁", "⎵").replace("Ġ", "⎵")
        color = COLORS[i % len(COLORS)]
        spans.append(
            f'<span style="background:{color};padding:2px 4px;'
            f'border-radius:3px;margin:1px;display:inline-block;'
            f'font-family:monospace;">{display}</span>'
        )
    return " ".join(spans)

# ---------------------------------------------------------------
# Build the HTML document
# ---------------------------------------------------------------
html_parts = [
    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
    "<title>Tokenizer Comparison</title>",
    "<style>body{font-family:sans-serif;max-width:900px;margin:40px auto;line-height:1.8}",
    "h2{color:#333}h3{color:#555;margin-bottom:4px}",
    ".sentence{background:#f5f5f5;padding:12px;border-radius:6px;margin-bottom:20px}",
    "</style></head><body>",
    "<h1>Tokenizer Output Comparison</h1>",
]

for sentence in sentences:
    html_parts.append(f'<div class="sentence"><h2>"{sentence}"</h2>')
    for name, tokenizer in tokenizers:
        html_parts.append(f"<h3>{name}</h3><p>{tokens_to_html(tokenizer, sentence)}</p>")
    html_parts.append("</div>")

html_parts.append("</body></html>")

# ---------------------------------------------------------------
# Write HTML file to disk
# ---------------------------------------------------------------
output_path = Path(__file__).parent / "token_comparison.html"
output_path.write_text("\n".join(html_parts), encoding="utf-8")
print(f"\nVisualization saved to: {output_path}")
print("Open this file in a browser to see color-coded token boundaries.")
