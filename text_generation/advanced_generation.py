"""
advanced_generation.py
Demonstrates beam search and repetition penalty alongside
nucleus sampling for comparison.
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
# Load pipeline
# -----------------------------------------------------------
generator = pipeline("text-generation", model="gpt2", device=device)

prompt = "Artificial intelligence will transform healthcare by"
set_seed(42)

# -----------------------------------------------------------
# Strategy 1: Beam search with repetition penalty
# num_beams controls how many hypotheses are maintained;
# repetition_penalty > 1.0 penalizes repeated tokens.
# -----------------------------------------------------------
beam_result = generator(
    prompt,
    max_new_tokens=100,
    num_beams=5,
    no_repeat_ngram_size=2,
    repetition_penalty=1.3,
    early_stopping=True,
)

print("\n--- Beam Search (5 beams, repetition_penalty=1.3) ---")
print(beam_result[0]["generated_text"])

# -----------------------------------------------------------
# Strategy 2: Nucleus sampling (top-p) for comparison
# -----------------------------------------------------------
set_seed(42)
nucleus_result = generator(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    top_p=0.92,
    temperature=0.8,
    repetition_penalty=1.2,
)

print("\n--- Nucleus Sampling (top_p=0.92, temp=0.8) ---")
print(nucleus_result[0]["generated_text"])