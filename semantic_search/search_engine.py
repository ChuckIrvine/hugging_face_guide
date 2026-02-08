"""
Semantic Search Engine
======================
A complete document search system using sentence-transformers and FAISS.
Retrieves documents based on meaning rather than keyword overlap.
"""

import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# Device Detection
# Automatically selects the best available accelerator.
# Checks for Apple MPS (Metal Performance Shaders) on macOS,
# then CUDA for NVIDIA GPUs, and falls back to CPU.
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# ---------------------------------------------------------------
# Load the Sentence Transformer Model
# 'all-MiniLM-L6-v2' is a compact, high-quality model that
# produces 384-dimensional embeddings. It balances speed and
# accuracy well for general-purpose semantic search.
# ---------------------------------------------------------------
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device=device)
print(f"Loaded model: {model_name}")

# ---------------------------------------------------------------
# Document Corpus
# A diverse set of documents covering technology, science,
# cooking, sports, and finance. Note that semantically related
# documents use different vocabulary — this is what semantic
# search is designed to handle.
# ---------------------------------------------------------------
documents = [
    "The electric vehicle market is expanding rapidly as battery technology improves and charging infrastructure grows.",
    "Photosynthesis converts sunlight into chemical energy, allowing plants to produce glucose from carbon dioxide and water.",
    "To make a perfect risotto, slowly add warm broth while stirring continuously to release the starch from the rice.",
    "Neural networks loosely mimic the structure of biological brains, using layers of interconnected nodes to learn patterns.",
    "The stock market experienced significant volatility this quarter due to rising interest rates and geopolitical tensions.",
    "Regular cardiovascular exercise strengthens the heart muscle and improves blood circulation throughout the body.",
    "Quantum computers use qubits that can exist in superposition, enabling them to solve certain problems exponentially faster.",
    "The Mediterranean diet emphasizes whole grains, olive oil, fish, and fresh vegetables for improved longevity.",
    "Basketball players must develop both offensive skills like shooting and defensive abilities such as blocking and stealing.",
    "Machine learning algorithms identify patterns in data to make predictions without being explicitly programmed for each task.",
    "Renewable energy sources like solar and wind power are becoming cost-competitive with fossil fuels worldwide.",
    "The human genome contains approximately 20,000 protein-coding genes spread across 23 pairs of chromosomes.",
    "Effective portfolio diversification reduces investment risk by spreading capital across uncorrelated asset classes.",
    "Deep learning has revolutionized computer vision, enabling machines to recognize objects in images with superhuman accuracy.",
    "Fermentation is a metabolic process where microorganisms convert sugars into alcohol, gases, or organic acids.",
    "The Tour de France is one of the most grueling endurance sporting events, covering over 3,500 kilometers in three weeks.",
    "Autonomous vehicles rely on a combination of LIDAR, cameras, and radar to perceive their environment.",
    "Inflation erodes purchasing power, meaning the same amount of money buys fewer goods and services over time.",
    "CRISPR-Cas9 gene editing technology allows scientists to precisely modify DNA sequences in living organisms.",
    "Soccer, known as football outside North America, is the most widely played sport on the planet.",
]

# ---------------------------------------------------------------
# Encode Documents into Dense Vectors
# The model converts each document string into a fixed-size
# embedding vector. We normalize embeddings so that cosine
# similarity equals inner product (dot product), which FAISS
# can compute very efficiently.
# ---------------------------------------------------------------
print(f"\nEncoding {len(documents)} documents...")
doc_embeddings = model.encode(
    documents,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,  # L2-normalize for cosine similarity
)

print(f"Embedding shape: {doc_embeddings.shape}")
# Expected: (20, 384) — 20 documents, 384 dimensions each

# ---------------------------------------------------------------
# Build the FAISS Index
# We use IndexFlatIP (Inner Product) since our embeddings are
# L2-normalized, making inner product equivalent to cosine
# similarity. For larger corpora (millions of docs), you would
# use approximate indices like IndexIVFFlat or IndexHNSW.
# ---------------------------------------------------------------
embedding_dim = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(doc_embeddings.astype(np.float32))
print(f"FAISS index built with {index.ntotal} vectors (dim={embedding_dim})")

# ---------------------------------------------------------------
# Search Function
# Encodes a query using the same model, then retrieves the top-K
# nearest neighbors from the FAISS index. Returns documents
# ranked by cosine similarity score.
# ---------------------------------------------------------------
def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Perform semantic search over the document corpus.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.

    Returns:
        List of dicts with 'document', 'score', and 'index' keys.
    """
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores, indices = index.search(query_embedding.astype(np.float32), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "index": int(idx),
            "score": float(score),
            "document": documents[idx],
        })
    return results

# ---------------------------------------------------------------
# Run Example Queries
# These queries deliberately use different vocabulary than the
# documents to demonstrate that semantic search captures meaning.
# "automobile" should match "electric vehicle" / "autonomous vehicles",
# "healthy eating" should match diet-related documents, etc.
# ---------------------------------------------------------------
queries = [
    "How do self-driving cars work?",
    "healthy eating habits",
    "artificial intelligence and deep learning",
    "ways to grow your savings",
    "popular team sports around the world",
]

TOP_K = 3

print("\n" + "=" * 70)
print("SEMANTIC SEARCH RESULTS")
print("=" * 70)

for query in queries:
    print(f"\n🔍 Query: \"{query}\"")
    print("-" * 50)
    results = search(query, top_k=TOP_K)
    for rank, result in enumerate(results, 1):
        print(f"  [{rank}] (score: {result['score']:.4f}) {result['document']}")

# ---------------------------------------------------------------
# Interactive Search Mode
# Allows the user to type custom queries and see results in
# real time. Type 'quit' or 'exit' to stop.
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("INTERACTIVE MODE — Type a query and press Enter (type 'quit' to exit)")
print("=" * 70)

while True:
    try:
        user_query = input("\n🔍 Your query: ").strip()
        if user_query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_query:
            continue
        results = search(user_query, top_k=TOP_K)
        for rank, result in enumerate(results, 1):
            print(f"  [{rank}] (score: {result['score']:.4f}) {result['document']}")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break