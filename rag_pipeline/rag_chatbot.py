import os

# Disable parallelism to prevent segfaults from FAISS/tokenizers/MPS conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

"""
Interactive RAG chatbot that retrieves relevant passages
from the knowledge base and generates grounded answers.
"""

## -------------------------------------------
## Detect device: prefer Apple MPS, then CUDA, fallback to CPU
## -------------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
    torch_device = torch.device("mps")
elif torch.cuda.is_available():
    device = "cuda"
    torch_device = torch.device("cuda")
else:
    device = "cpu"
    torch_device = torch.device("cpu")
print(f"Using device: {device}")

## -------------------------------------------
## Load the FAISS index and document metadata
## -------------------------------------------
base_dir = os.path.dirname(__file__)
index = faiss.read_index(os.path.join(base_dir, "docs.index"))

with open(os.path.join(base_dir, "doc_map.pkl"), "rb") as f:
    documents = pickle.load(f)

## -------------------------------------------
## Load the embedding model (same one used at index time)
## -------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

## -------------------------------------------
## Load the generative language model and tokenizer
## -------------------------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

gen_model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=gen_model,
    tokenizer=tokenizer,
)

def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """
    Encode the query and return the top-k most similar documents
    from the FAISS index.
    """
    ## Embed and normalize the query vector
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    ## Search the index
    scores, indices = index.search(query_vec, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        doc = documents[idx]
        results.append({
            "rank": rank + 1,
            "score": float(score),
            "title": doc["title"],
            "text": doc["text"],
        })
    return results

def build_prompt(query: str, contexts: list[dict]) -> str:
    """
    Construct a prompt that presents retrieved passages as context
    and instructs the model to answer based on that context.
    """
    context_block = "\n\n".join(
        f"[Document: {ctx['title']}]\n{ctx['text']}" for ctx in contexts
    )

    prompt = (
        f"Use the following documents to answer the question. "
        f"If the answer is not in the documents, say 'I don't know.'\n\n"
        f"{context_block}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    return prompt

def generate_answer(prompt: str, max_new_tokens: int = 150) -> str:
    """
    Generate an answer from the language model given the RAG prompt.
    """
    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    ## Extract only the generated portion after the prompt
    full_text = outputs[0]["generated_text"]
    answer = full_text[len(prompt):].strip()
    return answer

## -------------------------------------------
## Interactive chat loop
## -------------------------------------------
def main():
    print("\n=== RAG Chatbot ===")
    print("Ask questions about the knowledge base. Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        ## Step 1: Retrieve relevant passages
        retrieved = retrieve(query, top_k=2)
        print(f"\n  [Retrieved {len(retrieved)} passages]")
        for r in retrieved:
            print(f"    #{r['rank']} (score={r['score']:.3f}) {r['title']}")

        ## Step 2: Build the augmented prompt
        prompt = build_prompt(query, retrieved)

        ## Step 3: Generate the answer
        answer = generate_answer(prompt)
        print(f"\nBot: {answer}\n")

if __name__ == "__main__":
    main()