"""
Semantic Similarity Tool
========================
Extracts sentence embeddings from a pretrained BERT model
and computes cosine similarity between text pairs.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

# ---------------------------------------------------------------
# Device detection: prefer Apple MPS GPU, then CUDA, then CPU
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (Metal Performance Shaders) GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ---------------------------------------------------------------
# Load pretrained BERT tokenizer and model
# ---------------------------------------------------------------
model_name = "bert-base-uncased"
print(f"\nLoading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)
model.eval()  # Set to evaluation mode (disables dropout)

def get_embedding(text: str) -> torch.Tensor:
    """
    Compute a sentence-level embedding via mean pooling
    over the last hidden state of BERT.

    Steps:
      1. Tokenize the input text with padding and truncation.
      2. Move input tensors to the selected device.
      3. Run the model forward pass (no gradients needed).
      4. Extract the last_hidden_state from the model output.
      5. Apply the attention mask so padding tokens contribute
         zero weight to the mean.
      6. Return a 1-D embedding tensor on CPU.
    """
    # ---------------------------------------------------------------
    # Tokenize and move inputs to device
    # ---------------------------------------------------------------
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # ---------------------------------------------------------------
    # Forward pass with no gradient computation
    # ---------------------------------------------------------------
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # ---------------------------------------------------------------
    # Mean pooling: mask-aware averaging of token embeddings
    # ---------------------------------------------------------------
    # last_hidden_state shape: (batch_size, seq_len, hidden_dim)
    last_hidden_state = outputs.last_hidden_state

    # Expand attention_mask to match hidden state dimensions
    # (batch_size, seq_len) -> (batch_size, seq_len, hidden_dim)
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    # Zero out padding positions, then sum and divide by token count
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    sentence_embedding = sum_embeddings / sum_mask

    # Return as a 1-D CPU tensor
    return sentence_embedding.squeeze(0).cpu()

# ---------------------------------------------------------------
# Define sentence pairs to compare
# ---------------------------------------------------------------
sentence_pairs = [
    ("The cat sat on the mat.", "A kitten was resting on the rug."),
    ("The cat sat on the mat.", "Stock prices fell sharply today."),
    ("I love programming in Python.", "Python is my favorite coding language."),
    ("I love programming in Python.", "The weather is beautiful outside."),
    ("The doctor performed the surgery.", "The surgeon operated on the patient."),
]

# ---------------------------------------------------------------
# Compute and display cosine similarity for each pair
# ---------------------------------------------------------------
print("\n" + "=" * 65)
print("SEMANTIC SIMILARITY RESULTS")
print("=" * 65)

results = []
for sent_a, sent_b in sentence_pairs:
    emb_a = get_embedding(sent_a)
    emb_b = get_embedding(sent_b)

    # scipy's cosine() returns cosine *distance*, so similarity = 1 - distance
    similarity = 1.0 - cosine(emb_a.numpy(), emb_b.numpy())
    results.append((similarity, sent_a, sent_b))

# Sort by similarity (highest first)
results.sort(key=lambda x: x[0], reverse=True)

for rank, (sim, sent_a, sent_b) in enumerate(results, 1):
    print(f"\n  Rank {rank}  |  Similarity: {sim:.4f}")
    print(f"    A: \"{sent_a}\"")
    print(f"    B: \"{sent_b}\"")

print("\n" + "=" * 65)

# ---------------------------------------------------------------
# Show embedding shape for reference
# ---------------------------------------------------------------
sample_emb = get_embedding("Hello world")
print(f"\nEmbedding dimensionality: {sample_emb.shape[0]}")
print(f"Embedding dtype: {sample_emb.dtype}")