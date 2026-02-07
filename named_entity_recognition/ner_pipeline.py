"""
Named Entity Recognition using the Hugging Face pipeline API.
Demonstrates high-level NER with automatic sub-word aggregation.
"""

import torch
from transformers import pipeline

# ---------------------------------------------------------------
# Device detection: prefer Apple MPS GPU, fall back to CUDA or CPU
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# ---------------------------------------------------------------
# Create the NER pipeline with a well-known pretrained model.
# "dslim/bert-base-NER" is a BERT model fine-tuned on CoNLL-2003.
# aggregation_strategy="simple" merges sub-word tokens into spans.
# ---------------------------------------------------------------
ner_pipeline = pipeline(
    task="ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
    device=device,
)

# ---------------------------------------------------------------
# Sample texts to analyze
# ---------------------------------------------------------------
texts = [
    "Elon Musk announced that SpaceX will launch a mission to Mars from Cape Canaveral in December.",
    "The United Nations held a summit in Geneva, where Angela Merkel discussed NATO expansion.",
]

# ---------------------------------------------------------------
# Run NER on each text and display results
# ---------------------------------------------------------------
for text in texts:
    print(f"\nInput: {text}")
    print("-" * 80)
    entities = ner_pipeline(text)
    for ent in entities:
        print(
            f"  Entity: {ent['word']:25s}  "
            f"Label: {ent['entity_group']:5s}  "
            f"Score: {ent['score']:.4f}  "
            f"Span: [{ent['start']}:{ent['end']}]"
        )
    if not entities:
        print("  No entities found.")