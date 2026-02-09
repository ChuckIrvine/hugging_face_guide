from datasets import load_dataset
from transformers import AutoTokenizer

"""
Load dataset and create subsets (same as step 1).
"""
dataset = load_dataset("imdb")
small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

"""
Load the tokenizer for distilbert-base-uncased.
This tokenizer uses WordPiece and lowercases input text.
"""
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

"""
Define a tokenization function that truncates to max length 512.
We set padding=False here because the Trainer will handle dynamic
padding via a data collator, which is more memory-efficient.
"""
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

"""
Apply tokenization to both splits using batched map for speed.
"""
tokenized_train = small_train.map(tokenize_function, batched=True)
tokenized_test = small_test.map(tokenize_function, batched=True)

"""
Inspect the tokenized output to verify fields are present.
"""
print("Columns after tokenization:", tokenized_train.column_names)
print(f"Sample input_ids length: {len(tokenized_train[0]['input_ids'])}")
print(f"First 20 tokens: {tokenized_train[0]['input_ids'][:20]}")

"""
Decode tokens back to text to verify correctness.
"""
decoded = tokenizer.decode(tokenized_train[0]["input_ids"][:20])
print(f"Decoded: {decoded}")