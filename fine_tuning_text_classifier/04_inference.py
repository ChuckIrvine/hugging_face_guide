from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

"""
=== LOAD THE FINE-TUNED MODEL ===
The Trainer saved the best model in ./results. We load both the
model and tokenizer from there. If the Trainer saved the best
checkpoint to a subdirectory, adjust the path accordingly.
We look for the model in ./results since load_best_model_at_end
ensures the best checkpoint is saved to output_dir.
"""
model_path = "./results/checkpoint-375"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

"""
=== INFERENCE ON CUSTOM REVIEWS ===
Tokenize new text, pass through the model, and convert logits to
predictions. Logits are raw scores; we apply softmax to get
probabilities and argmax to get the predicted class.
"""
reviews = [
    "This movie was absolutely fantastic! The acting was superb and the plot kept me on the edge of my seat.",
    "Terrible film. The dialogue was wooden, the pacing was awful, and I nearly fell asleep halfway through.",
    "It was okay. Some good moments but overall pretty forgettable.",
]

label_map = {0: "NEGATIVE", 1: "POSITIVE"}

for review in reviews:
    inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    """
    Extract logits and compute softmax probabilities.
    """
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()

    print(f"Review: {review[:80]}...")
    print(f"  Prediction: {label_map[predicted_class]} (confidence: {confidence:.2%})")
    print(f"  Raw logits: negative={logits[0][0]:.3f}, positive={logits[0][1]:.3f}")
    print()