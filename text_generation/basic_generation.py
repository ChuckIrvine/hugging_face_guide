"""
basic_generation.py
Demonstrates simple text generation with GPT-2 using the
Hugging Face pipeline abstraction and default greedy decoding.
"""

import torch
from transformers import pipeline

# -----------------------------------------------------------
# Device detection: prefer Apple MPS GPU, fall back to CPU
# -----------------------------------------------------------
if torch.backends.mps.is_available():
    device = 0  # pipeline uses device index; 0 maps to mps
    print("Using Apple MPS GPU")
else:
    device = -1  # CPU
    print("Using CPU")

# -----------------------------------------------------------
# Load GPT-2 text-generation pipeline
# The model and tokenizer are downloaded automatically on
# the first run and cached in ~/.cache/huggingface/
# -----------------------------------------------------------
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=device,
)

# -----------------------------------------------------------
# Generate text with a simple prompt (greedy / default)
# max_new_tokens limits the number of tokens appended
# -----------------------------------------------------------
prompt = "In a distant galaxy, a lone starship"
results = generator(
    prompt,
    max_new_tokens=60,
    num_return_sequences=1,
)

# -----------------------------------------------------------
# Display the generated continuation
# -----------------------------------------------------------
print("\n--- Generated Text (Greedy) ---")
print(results[0]["generated_text"])