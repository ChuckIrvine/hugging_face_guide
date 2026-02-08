"""
merge_adapter.py
Loads a trained LoRA adapter and merges it into the base model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -------------------------------------------------------
# Device detection
# -------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# -------------------------------------------------------
# Load base model and tokenizer
# -------------------------------------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# -------------------------------------------------------
# Load the LoRA adapter on top of the base model
# -------------------------------------------------------
peft_model = PeftModel.from_pretrained(base_model, "./lora_adapter")
peft_model.to(device)

# -------------------------------------------------------
# Generate text with the adapter loaded (not merged)
# -------------------------------------------------------
prompt = "This movie was absolutely"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = peft_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
    )
print(f"\nAdapter output:\n{tokenizer.decode(output[0], skip_special_tokens=True)}")

# -------------------------------------------------------
# Merge LoRA weights into the base model permanently
# This eliminates the PEFT overhead during inference
# -------------------------------------------------------
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
print("\nMerged model saved to ./merged_model")