import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)

"""
Load the beans dataset. It contains train, validation, and test splits
with three classes of bean leaf conditions.
"""
dataset = load_dataset("beans")
labels = dataset["train"].features["labels"].names
num_labels = len(labels)
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

print(f"Classes ({num_labels}): {labels}")
print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

"""
Load the ViT image processor to handle resizing and normalization.
This must match the pre-trained model's expected input format.
"""
model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)

def preprocess(batch):
    """
    Apply the ViT processor to each image in the batch.
    The processor resizes to 224x224 and normalizes pixel values.
    """
    inputs = processor(images=batch["image"], return_tensors="pt")
    inputs["labels"] = batch["labels"]
    return inputs

"""
Apply preprocessing. We set the format to PyTorch tensors
and use batched processing for efficiency.
"""
dataset = dataset.with_transform(preprocess)

"""
Load the pre-trained ViT model with a classification head
configured for our number of classes.
"""
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

"""
Define the evaluation metric — simple accuracy.
"""
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

"""
Configure training arguments. We use a small number of epochs
and a modest batch size suitable for demonstration.
"""
training_args = TrainingArguments(
    output_dir="./vit-beans-output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=False,
    report_to="none",
)

"""
Initialize the Trainer with model, data, and evaluation function,
then start fine-tuning.
"""
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

"""
Evaluate on the held-out test set and print results.
"""
test_results = trainer.evaluate(dataset["test"])
print(f"\nTest Accuracy: {test_results['eval_accuracy']:.4f}")

"""
Save the fine-tuned model and processor for later inference.
"""
trainer.save_model("./vit-beans-finetuned")
processor.save_pretrained("./vit-beans-finetuned")
print("Model saved to ./vit-beans-finetuned")