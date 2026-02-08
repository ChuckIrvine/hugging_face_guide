"""
inspect_lora.py
Demonstrates LoRA configuration and parameter reduction on GPT-2.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# -------------------------------------------------------
# Device detection: prefer Apple MPS GPU, fallback to CPU
# -------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# -------------------------------------------------------
# Load the base model and tokenizer
# -------------------------------------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# -------------------------------------------------------
# Count parameters before LoRA
# -------------------------------------------------------
total_params_before = sum(p.numel() for p in model.parameters())
print(f"\nBase model parameters: {total_params_before:,}")

# -------------------------------------------------------
# Define the LoRA configuration
# - r: rank of the decomposition
# - lora_alpha: scaling factor (alpha/r scales the update)
# - target_modules: which linear layers receive adapters
# - lora_dropout: dropout applied to LoRA layers
# -------------------------------------------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],
)

# -------------------------------------------------------
# Wrap the model with LoRA adapters
# -------------------------------------------------------
peft_model = get_peft_model(model, lora_config)

# -------------------------------------------------------
# Print parameter comparison
# -------------------------------------------------------
peft_model.print_trainable_parameters()