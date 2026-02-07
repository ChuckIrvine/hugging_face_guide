"""
Named Entity Recognition with direct tokenizer/model access.
Demonstrates lower-level control over tokenization and inference.
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ---------------------------------------------------------------
# Device detection: prefer Apple MPS GPU, fall back to CUDA or CPU
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------
# Load the pretrained tokenizer and model
# ---------------------------------------------------------------
model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

# ---------------------------------------------------------------
# Retrieve the label mapping from the model config.
# id2label maps integer indices to BIO-tagged entity labels.
# ---------------------------------------------------------------
id2label = model.config.id2label
print(f"Label set: {id2label}")

# ---------------------------------------------------------------
# Tokenize input and run inference
# ---------------------------------------------------------------
text = "Marie Curie conducted research at the University of Paris and won the Nobel Prize."

inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
offset_mapping = inputs.pop("offset_mapping")[0]  # not a model input
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

# ---------------------------------------------------------------
# Convert logits to predicted label indices via argmax,
# then map indices to human-readable label strings.
# ---------------------------------------------------------------
predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

# ---------------------------------------------------------------
# Display token-level predictions, skipping special tokens
# ---------------------------------------------------------------
print(f"\nInput: {text}")
print(f"{'Token':>20s}  {'Label':>10s}  {'Offset'}")
print("-" * 55)
for token, pred, offset in zip(tokens, predictions, offset_mapping.tolist()):
    if token in ("[CLS]", "[SEP]", "[PAD]"):
        continue
    label = id2label[pred]
    print(f"{token:>20s}  {label:>10s}  {offset}")
    