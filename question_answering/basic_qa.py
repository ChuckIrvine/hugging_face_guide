"""
basic_qa.py
Demonstrates extractive question answering using the Hugging Face pipeline API.
"""

import torch
from transformers import pipeline

# -------------------------------------------------------
# Device detection: prefer Apple MPS GPU, then CUDA, else CPU
# -------------------------------------------------------
if torch.backends.mps.is_available():
    device = 0  # pipeline uses device index; MPS is default on Apple Silicon
    print("Using Apple MPS GPU")
elif torch.cuda.is_available():
    device = 0
    print("Using CUDA GPU")
else:
    device = -1
    print("Using CPU")

# -------------------------------------------------------
# Initialize the question-answering pipeline with a SQuAD2-trained model.
# SQuAD2 includes unanswerable questions, making the model more robust.
# -------------------------------------------------------
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/minilm-uncased-squad2",
    device=device,
)

# -------------------------------------------------------
# Define a context document and a set of questions
# -------------------------------------------------------
context = """
The Transformer architecture was introduced in the paper "Attention Is All You Need"
by Vaswani et al. in 2017. It replaced recurrent layers with self-attention mechanisms,
enabling significantly more parallelization during training. The original Transformer
was designed for machine translation and achieved state-of-the-art results on the
WMT 2014 English-to-German and English-to-French translation tasks. BERT, GPT, and
T5 are all architectures built upon the Transformer foundation. The model uses
multi-head attention to attend to different representation subspaces at different positions.
"""

questions = [
    "When was the Transformer architecture introduced?",
    "What did the Transformer replace?",
    "What tasks did the original Transformer achieve state-of-the-art results on?",
]

# -------------------------------------------------------
# Run each question against the context and display results
# -------------------------------------------------------
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Span:  [{result['start']}:{result['end']}]")
    print()