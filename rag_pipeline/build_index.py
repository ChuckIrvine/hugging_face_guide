import json
import os
import pickle

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

"""
Build a FAISS vector index from the knowledge base documents.
This index enables fast approximate nearest-neighbor retrieval
at query time.
"""

## -------------------------------------------
## Detect device: prefer Apple MPS, then CUDA, fallback to CPU
## -------------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

## -------------------------------------------
## Load documents from the knowledge base
## -------------------------------------------
base_dir = os.path.dirname(__file__)
with open(os.path.join(base_dir, "knowledge_base.json"), "r") as f:
    documents = json.load(f)

texts = [doc["text"] for doc in documents]

## -------------------------------------------
## Compute dense embeddings using sentence-transformers
## -------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

## Normalize embeddings for cosine similarity via inner product
faiss.normalize_L2(embeddings)

## -------------------------------------------
## Build and save the FAISS index
## -------------------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product on L2-normalized vectors = cosine sim
index.add(embeddings)

faiss.write_index(index, os.path.join(base_dir, "docs.index"))

## Save document metadata mapping (position -> document)
with open(os.path.join(base_dir, "doc_map.pkl"), "wb") as f:
    pickle.dump(documents, f)

print(f"Indexed {index.ntotal} documents with dimension {dimension}")