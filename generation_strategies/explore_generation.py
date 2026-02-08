"""
explore_generation.py
Demonstrates how different generation strategies affect text output
from a causal language model (GPT-2).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------
# Device detection: prefer Apple MPS GPU, then CUDA, then CPU
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
# Load model and tokenizer
# ---------------------------------------------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# GPT-2 has no pad token by default; set it to eos_token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# ---------------------------------------------------------------
# Shared prompt for all experiments
# ---------------------------------------------------------------
prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

MAX_NEW_TOKENS = 60

def generate_and_print(label, **gen_kwargs):
    """Helper to generate text with given parameters and print results."""
    print(f"\n{'='*70}")
    print(f"Strategy: {label}")
    print(f"Parameters: {gen_kwargs}")
    print(f"{'='*70}")
    output = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        **gen_kwargs,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)
    print()

# ---------------------------------------------------------------
# 1. Greedy decoding — always pick the highest-probability token
# ---------------------------------------------------------------
generate_and_print(
    "Greedy Decoding",
    do_sample=False,
)

# ---------------------------------------------------------------
# 2. Low temperature (0.3) — sharper distribution, more focused
# ---------------------------------------------------------------
generate_and_print(
    "Low Temperature (T=0.3)",
    do_sample=True,
    temperature=0.3,
)

# ---------------------------------------------------------------
# 3. High temperature (1.5) — flatter distribution, more random
# ---------------------------------------------------------------
generate_and_print(
    "High Temperature (T=1.5)",
    do_sample=True,
    temperature=1.5,
)

# ---------------------------------------------------------------
# 4. Top-k sampling — restrict to k most probable tokens
# ---------------------------------------------------------------
generate_and_print(
    "Top-k Sampling (k=10)",
    do_sample=True,
    top_k=10,
    temperature=1.0,
)

# ---------------------------------------------------------------
# 5. Top-p (nucleus) sampling — dynamic candidate set
# ---------------------------------------------------------------
generate_and_print(
    "Top-p Nucleus Sampling (p=0.9)",
    do_sample=True,
    top_p=0.9,
    temperature=1.0,
)

# ---------------------------------------------------------------
# 6. Beam search — explore multiple paths simultaneously
# ---------------------------------------------------------------
generate_and_print(
    "Beam Search (num_beams=5)",
    do_sample=False,
    num_beams=5,
    early_stopping=True,
    no_repeat_ngram_size=2,
)

# ---------------------------------------------------------------
# 7. Combined strategy — temperature + top-k + top-p together
# ---------------------------------------------------------------
generate_and_print(
    "Combined: T=0.7, top_k=50, top_p=0.92",
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.92,
)

print("\nAll generation strategies complete.")