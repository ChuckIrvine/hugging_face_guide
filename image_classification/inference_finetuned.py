from transformers import pipeline
from datasets import load_dataset

"""
Load a few test images from the beans dataset.
"""
test_dataset = load_dataset("beans", split="test")
labels = test_dataset.features["labels"].names

"""
Load the fine-tuned model as a classification pipeline.
"""
classifier = pipeline(
    "image-classification",
    model="./vit-beans-finetuned",
)

"""
Run inference on the first 5 test images and compare
predictions against ground truth.
"""
print(f"{'Index':<8}{'Predicted':<25}{'Actual':<25}{'Correct'}")
print("-" * 70)

for i in range(5):
    image = test_dataset[i]["image"]
    true_label = labels[test_dataset[i]["labels"]]
    prediction = classifier(image, top_k=1)[0]
    pred_label = prediction["label"]
    match = "✓" if pred_label == true_label else "✗"
    print(f"{i:<8}{pred_label:<25}{true_label:<25}{match}")