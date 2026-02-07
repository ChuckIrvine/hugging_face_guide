"""
compare_automodels.py
Load bert-base-uncased with multiple AutoModel classes and compare
their architectures and output tensor shapes.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
)

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
# Load the shared tokenizer (same for all model variants)
# ---------------------------------------------------------------
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

sample_text = "Hugging Face makes NLP accessible to everyone."
inputs = tokenizer(sample_text, return_tensors="pt").to(device)

print(f"\nTokenized input shape: {inputs['input_ids'].shape}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}\n")

# ---------------------------------------------------------------
# Define the AutoModel variants to compare
# ---------------------------------------------------------------
model_classes = {
    "AutoModel (base)": AutoModel,
    "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
    "AutoModelForTokenClassification": AutoModelForTokenClassification,
    "AutoModelForMaskedLM": AutoModelForMaskedLM,
}

# ---------------------------------------------------------------
# Load each variant, inspect architecture, and run inference
# ---------------------------------------------------------------
for label, model_class in model_classes.items():
    print("=" * 65)
    print(f"  {label}")
    print("=" * 65)

    model = model_class.from_pretrained(model_name).to(device)
    model.eval()

    # Print top-level module names to reveal the task head
    top_modules = [name for name, _ in model.named_children()]
    print(f"  Top-level modules: {top_modules}")

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters:  {total_params:,}")

    # Run forward pass (no gradient computation needed)
    with torch.no_grad():
        outputs = model(**inputs)

    # The first element of the output is the primary tensor
    if hasattr(outputs, "last_hidden_state"):
        primary = outputs.last_hidden_state
        desc = "last_hidden_state"
    elif hasattr(outputs, "logits"):
        primary = outputs.logits
        desc = "logits"
    else:
        primary = outputs[0]
        desc = "outputs[0]"

    print(f"  Output attribute:  {desc}")
    print(f"  Output shape:      {primary.shape}\n")

    # Clean up to save memory
    del model