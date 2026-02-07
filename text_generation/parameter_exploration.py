"""
parameter_exploration.py
Explores how temperature, top-k, and top-p affect the
creativity and coherence of GPT-2 text generation.
"""

import torch
from transformers import pipeline, set_seed

# -----------------------------------------------------------
# Device detection: prefer Apple MPS GPU, fall back to CPU
# -----------------------------------------------------------
if torch.backends.mps.is_available():
    device = 0
    print("Using Apple MPS GPU")
else:
    device = -1
    print("Using CPU")

# -----------------------------------------------------------
# Load the GPT-2 pipeline
# -----------------------------------------------------------
generator = pipeline("text-generation", model="gpt2", device=device)

# -----------------------------------------------------------
# Define the prompt and generation configurations
# Each config is a dict of keyword arguments passed to the
# generator, plus a human-readable label for display.
# -----------------------------------------------------------
prompt = "The invention of the printing press"

configs = [
    {
        "label": "Low Temperature (0.3) — conservative, repetitive",
        "params": {
            "max_new_tokens": 80,
            "temperature": 0.3,
            "do_sample": True,
        },
    },
    {
        "label": "High Temperature (1.5) — creative, possibly incoherent",
        "params": {
            "max_new_tokens": 80,
            "temperature": 1.5,
            "do_sample": True,
        },
    },
    {
        "label": "Top-k=50, Temperature=0.7 — balanced diversity",
        "params": {
            "max_new_tokens": 80,
            "temperature": 0.7,
            "top_k": 50,
            "do_sample": True,
        },
    },
    {
        "label": "Top-p=0.92, Temperature=0.8 — nucleus sampling",
        "params": {
            "max_new_tokens": 80,
            "temperature": 0.8,
            "top_p": 0.92,
            "do_sample": True,
        },
    },
]

# -----------------------------------------------------------
# Generate and display text for each configuration
# set_seed ensures reproducibility within each run
# -----------------------------------------------------------
for i, cfg in enumerate(configs, 1):
    set_seed(42)
    result = generator(prompt, num_return_sequences=1, **cfg["params"])
    print(f"\n{'='*60}")
    print(f"Config {i}: {cfg['label']}")
    print(f"{'='*60}")
    print(result[0]["generated_text"])
    