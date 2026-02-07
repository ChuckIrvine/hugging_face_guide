"""
Manual Text Classification
Demonstrates direct use of tokenizer and model
for fine-grained control over the classification process.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------------------------
# Device Detection
# Check for Apple Silicon GPU (MPS), CUDA, or CPU
# -----------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# -----------------------------------------------
# Load pre-trained model and tokenizer
# This model is fine-tuned on AG News dataset
# Classes: World (0), Sports (1), Business (2), Sci/Tech (3)
# -----------------------------------------------
model_name = "textattack/bert-base-uncased-ag-news"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# -----------------------------------------------
# Define AG News label mapping
# (model config has generic LABEL_0, etc.)
# -----------------------------------------------
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
print(f"Label mapping: {id2label}")

# -----------------------------------------------
# Sample articles for classification
# -----------------------------------------------
articles = [
    "Oil prices surged to a three-month high as OPEC announced further production cuts.",
    "The women's national soccer team advanced to the World Cup semifinals with a 3-1 victory.",
    "A magnitude 7.2 earthquake struck off the coast of Japan, triggering tsunami warnings across the Pacific.",
    "Nvidia unveiled its next-generation GPU architecture, promising a 4x improvement in AI training performance."
]

# -----------------------------------------------
# Tokenize, run inference, and interpret outputs
# -----------------------------------------------
print("\n" + "=" * 60)
print("MANUAL CLASSIFICATION WITH AG NEWS MODEL")
print("=" * 60)

for i, article in enumerate(articles, 1):
    # Tokenize the input text
    # return_tensors="pt" gives us PyTorch tensors
    # truncation ensures long texts don't exceed model limits
    inputs = tokenizer(
        article,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # Move input tensors to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Run inference without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.logits contains raw unnormalized scores
    # Apply softmax to convert to probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted class
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    predicted_label = id2label[predicted_class_id]
    confidence = probabilities[0][predicted_class_id].item()

    print(f"\nArticle {i}: {article[:75]}...")
    print(f"  Predicted: {predicted_label} (confidence: {confidence:.4f})")
    print("  Full distribution:")
    for class_id, prob in enumerate(probabilities[0]):
        label = id2label[class_id]
        bar = "█" * int(prob.item() * 30)
        print(f"    {label:10s} {prob.item():.4f} {bar}")