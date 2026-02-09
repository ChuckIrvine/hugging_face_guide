from datasets import load_dataset

"""
Load the IMDB dataset from the Hugging Face Hub.
The dataset has 'train' and 'test' splits, each with 25,000 examples.
"""
dataset = load_dataset("imdb")
print(dataset)
print(f"\nSample review (first 300 chars): {dataset['train'][0]['text'][:300]}...")
print(f"Label: {dataset['train'][0]['label']}  (0=negative, 1=positive)")

"""
Create smaller subsets for faster training during this lesson.
We use 2,000 training examples and 500 test examples.
Shuffle first to ensure a representative sample.
"""
small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

print(f"\nTraining subset size: {len(small_train)}")
print(f"Test subset size: {len(small_test)}")

"""
Check label distribution to confirm the subset is balanced.
"""
from collections import Counter
train_labels = Counter(small_train["label"])
print(f"Train label distribution: {dict(train_labels)}")
