"""
train_lora.py
Fine-tunes GPT-2 with LoRA on a subset of the IMDB dataset.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# -------------------------------------------------------
# Device detection: prefer Apple MPS GPU, fallback to CPU
# -------------------------------------------------------
if torch.backends.mps.is_available():
    device_type = "mps"
    print("Using Apple MPS GPU")
elif torch.cuda.is_available():
    device_type = "cuda"
    print("Using CUDA GPU")
else:
    device_type = "cpu"
    print("Using CPU")

# -------------------------------------------------------
# Load tokenizer and set pad token
# -------------------------------------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------------------------------
# Load and prepare a small subset of IMDB for speed
# -------------------------------------------------------
dataset = load_dataset("imdb", split="train[:500]")

def tokenize_fn(examples):
    """Tokenize text and truncate to 128 tokens for efficiency."""
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
tokenized_dataset.set_format("torch")

# -------------------------------------------------------
# Split into train and eval
# -------------------------------------------------------
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# -------------------------------------------------------
# Load model and apply LoRA
# -------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# -------------------------------------------------------
# Data collator for causal language modeling
# -------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# -------------------------------------------------------
# Training arguments — configured for consumer hardware
# Note: no deprecated arguments; MPS is auto-detected
# -------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=50,
    report_to="none",
    load_best_model_at_end=True,
)

# -------------------------------------------------------
# Initialize Trainer and start training
# -------------------------------------------------------
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

print("\nStarting LoRA fine-tuning...")
trainer.train()

# -------------------------------------------------------
# Save only the LoRA adapter weights (not the full model)
# -------------------------------------------------------
peft_model.save_pretrained("./lora_adapter")
print("\nLoRA adapter saved to ./lora_adapter")
print(f"Adapter size on disk: check ./lora_adapter directory")