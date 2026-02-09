from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
import numpy as np

"""
=== DATA PREPARATION ===
Load the IMDB dataset and create small subsets for manageable training.
"""
dataset = load_dataset("imdb")
small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

"""
=== TOKENIZATION ===
Tokenize both splits using the DistilBERT tokenizer.
Dynamic padding will be applied by the data collator during training.
"""
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_train = small_train.map(tokenize_function, batched=True)
tokenized_test = small_test.map(tokenize_function, batched=True)

"""
=== DATA COLLATOR ===
DataCollatorWithPadding pads each batch to the length of its longest
sequence, rather than padding all sequences to 512 tokens upfront.
This significantly reduces computation and memory usage.
"""
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

"""
=== MODEL ===
Load DistilBERT with a sequence classification head (2 labels).
The classification head is randomly initialized; the base model
weights come from pre-training. A warning about uninitialized
weights for the classifier head is expected and normal.
"""
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

"""
=== METRICS ===
Define a compute_metrics function that the Trainer calls after each
evaluation. We report both accuracy and F1 score.
"""
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    return acc

"""
=== TRAINING ARGUMENTS ===
Configure the training run:
- output_dir: where checkpoints and logs are saved
- eval_strategy: evaluate at the end of each epoch
- learning_rate: 2e-5 is a standard fine-tuning rate for BERT variants
- per_device_train_batch_size: 16 balances speed and memory
- num_train_epochs: 3 epochs is typical for fine-tuning
- weight_decay: small regularization to prevent overfitting
- save_strategy: save at each epoch to enable checkpoint recovery
- load_best_model_at_end: automatically load the best checkpoint
- metric_for_best_model: use accuracy to determine the best checkpoint
"""
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    seed=42,
)

"""
=== TRAINER ===
Assemble the Trainer with all components and start training.
The Trainer handles the training loop, gradient accumulation,
optimizer (AdamW by default), learning rate scheduling, and evaluation.
"""
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
train_result = trainer.train()

"""
=== TRAINING RESULTS ===
Print training metrics: total training time and final training loss.
"""
print("\n=== Training Results ===")
print(f"Training loss: {train_result.training_loss:.4f}")
print(f"Training runtime: {train_result.metrics['train_runtime']:.1f} seconds")

"""
=== EVALUATION ===
Run a final evaluation pass on the test set and print metrics.
"""
print("\n=== Evaluation Results ===")
eval_results = trainer.evaluate()
for key, value in sorted(eval_results.items()):
    print(f"  {key}: {value}")