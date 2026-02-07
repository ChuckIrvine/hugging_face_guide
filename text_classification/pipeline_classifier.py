"""
Text Classification with Hugging Face Pipeline API
Classifies news article snippets into topic categories
using zero-shot classification.
"""

import torch
from transformers import pipeline

# -----------------------------------------------
# Device Detection
# Check for Apple Silicon GPU (MPS), CUDA, or CPU
# -----------------------------------------------
if torch.backends.mps.is_available():
    device = 0  # pipeline uses device index; MPS maps to 0
    print("Using Apple MPS (Metal Performance Shaders) GPU")
elif torch.cuda.is_available():
    device = 0
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = -1
    print("Using CPU")

# -----------------------------------------------
# Initialize the Zero-Shot Classification Pipeline
# facebook/bart-large-mnli can classify text into
# arbitrary labels without task-specific fine-tuning
# -----------------------------------------------
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

# -----------------------------------------------
# Define candidate topic labels for news articles
# -----------------------------------------------
candidate_labels = ["Business", "Technology", "Sports", "Politics", "Science", "Entertainment"]

# -----------------------------------------------
# Sample news article snippets to classify
# -----------------------------------------------
articles = [
    "The Federal Reserve raised interest rates by 25 basis points, citing persistent inflation pressures across the economy.",
    "SpaceX successfully launched its Starship rocket on a test flight, reaching orbital velocity before a controlled splashdown.",
    "The defending champions secured a dramatic last-minute victory with a 95-yard touchdown drive in the fourth quarter.",
    "Researchers at MIT have developed a new quantum computing chip that operates at room temperature.",
    "The latest blockbuster sequel grossed over $200 million in its opening weekend, breaking box office records.",
    "Congress passed a bipartisan infrastructure bill allocating $1.2 trillion for roads, bridges, and broadband."
]

# -----------------------------------------------
# Classify each article and display results
# -----------------------------------------------
print("\n" + "=" * 60)
print("NEWS ARTICLE TOPIC CLASSIFICATION")
print("=" * 60)

for i, article in enumerate(articles, 1):
    result = classifier(article, candidate_labels)

    # result contains 'labels' sorted by score and 'scores'
    top_label = result["labels"][0]
    top_score = result["scores"][0]

    print(f"\nArticle {i}: {article[:80]}...")
    print(f"  Predicted Topic: {top_label} (confidence: {top_score:.4f})")

    # Show runner-up for context
    runner_up_label = result["labels"][1]
    runner_up_score = result["scores"][1]
    print(f"  Runner-up:       {runner_up_label} (confidence: {runner_up_score:.4f})")