"""
Custom Training Loop for Sentiment Classification
===================================================
Fine-tunes DistilBERT on SST-2 using a pure PyTorch training loop,
demonstrating device detection, gradient accumulation, and evaluation.
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

# ---------------------------------------------------------------
# 1. Device detection — prefer CUDA, then Apple MPS, then CPU
# ---------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ---------------------------------------------------------------
# 2. Hyperparameters and configuration
# ---------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
GRADIENT_ACCUMULATION_STEPS = 1  # effective batch size = 32
MAX_LENGTH = 128

# ---------------------------------------------------------------
# 3. Load and tokenize the SST-2 dataset
# ---------------------------------------------------------------
dataset = load_dataset("glue", "sst2")

# Use a subset for faster training (set to None for full dataset)
TRAIN_SUBSET = 10000
if TRAIN_SUBSET:
    dataset["train"] = dataset["train"].select(range(TRAIN_SUBSET))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    """Tokenize sentences with truncation and a fixed max length."""
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized = dataset.map(tokenize_fn, batched=True)

# Keep only the columns the model expects
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ---------------------------------------------------------------
# 4. Create DataLoaders with dynamic padding
# ---------------------------------------------------------------
collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    tokenized["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
)

eval_loader = DataLoader(
    tokenized["validation"],
    batch_size=BATCH_SIZE * 2,  # larger batch for eval (no gradients)
    shuffle=False,
    collate_fn=collator,
)

# ---------------------------------------------------------------
# 5. Instantiate model, optimizer, and learning-rate scheduler
# ---------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# ---------------------------------------------------------------
# 6. Training loop with gradient accumulation
# ---------------------------------------------------------------
print(f"\nTraining for {NUM_EPOCHS} epochs on {device}")
print(f"  Batches per epoch : {len(train_loader)}")
print(f"  Accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch   : {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Total optim steps : {num_training_steps}\n")

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    start = time.time()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for step, batch in enumerate(progress_bar):
        # Move every tensor in the batch to the target device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass — the model returns a loss when labels are provided
        outputs = model(**batch)
        loss = outputs.loss

        # Scale loss by accumulation steps so gradients average correctly
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        
        # Update progress bar less frequently to avoid display issues
        if step % 100 == 0:
            progress_bar.set_postfix(loss=f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}", refresh=False)

        # Only step the optimizer every GRADIENT_ACCUMULATION_STEPS batches
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(
            train_loader
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    avg_loss = epoch_loss / len(train_loader)
    elapsed = time.time() - start

    # -----------------------------------------------------------
    # 7. Evaluation after each epoch
    # -----------------------------------------------------------
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}  |  "
        f"Loss: {avg_loss:.4f}  |  "
        f"Val Accuracy: {accuracy:.4f}  |  "
        f"Time: {elapsed:.1f}s"
    )

# ---------------------------------------------------------------
# 8. Save the fine-tuned model and tokenizer
# ---------------------------------------------------------------
output_dir = "fine_tuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\nModel saved to {output_dir}/")