"""
multi_doc_qa.py
A multi-document Q&A tool that searches across several passages
and returns the best answer with provenance information.
"""

import torch
from transformers import pipeline

# -------------------------------------------------------
# Device detection: prefer Apple MPS GPU, then CUDA, else CPU
# -------------------------------------------------------
if torch.backends.mps.is_available():
    device = 0
elif torch.cuda.is_available():
    device = 0
else:
    device = -1

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/minilm-uncased-squad2",
    device=device,
)

# -------------------------------------------------------
# Define a small corpus of documents with identifiers
# -------------------------------------------------------
documents = {
    "doc_transformers": (
        "The Transformer architecture uses self-attention to process input sequences "
        "in parallel. It was introduced by Vaswani et al. in 2017 and forms the basis "
        "of models like BERT, GPT-2, and T5."
    ),
    "doc_bert": (
        "BERT (Bidirectional Encoder Representations from Transformers) was released "
        "by Google in October 2018. It is pre-trained using masked language modeling "
        "and next sentence prediction on a large corpus of English text."
    ),
    "doc_gpt": (
        "GPT-2 was released by OpenAI in February 2019. It is a unidirectional "
        "language model with 1.5 billion parameters, trained on WebText, a dataset "
        "of 8 million web pages."
    ),
}

def ask(question: str, docs: dict, threshold: float = 0.05) -> dict:
    """
    Search all documents for the best answer to a question.
    Returns the top answer, its source document, and score.
    Falls back to 'unanswerable' if below threshold.
    """
    # --------------------------------------------------
    # Iterate over each document, collecting QA results
    # --------------------------------------------------
    candidates = []
    for doc_id, text in docs.items():
        result = qa_pipeline(question=question, context=text)
        result["doc_id"] = doc_id
        candidates.append(result)

    # --------------------------------------------------
    # Rank candidates by score and apply threshold
    # --------------------------------------------------
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    if best["score"] < threshold:
        return {"answer": "Unanswerable", "score": best["score"], "doc_id": None}

    return best

# -------------------------------------------------------
# Test with several questions, including one that is unanswerable
# -------------------------------------------------------
test_questions = [
    "When was BERT released?",
    "How many parameters does GPT-2 have?",
    "Who introduced the Transformer?",
    "What is the capital of France?",  # unanswerable from these docs
]

for q in test_questions:
    result = ask(q, documents)
    print(f"Q: {q}")
    print(f"A: {result.get('answer', 'N/A')}")
    print(f"   Score: {result['score']:.4f} | Source: {result.get('doc_id', 'N/A')}")
    print()