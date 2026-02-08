"""
LoRA Fine-Tuning with PEFT
===========================
Demonstrates Low-Rank Adaptation on distilgpt2 using the IMDB dataset.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------
# 1. Device Detection — Check for Apple MPS GPU, CUDA, or CPU
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU (training will be slow)")

# ---------------------------------------------------------------
# 2. Load Base Model and Tokenizer
# ---------------------------------------------------------------
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Print base model parameter count before LoRA
total_base_params = sum(p.numel() for p in model.parameters())
print(f"\nBase model parameters: {total_base_params:,}")

# ---------------------------------------------------------------
# 3. Configure LoRA Adapter
#    - r: rank of decomposition (lower = fewer params, higher = more capacity)
#    - lora_alpha: scaling factor (commonly set to 2*r)
#    - target_modules: which linear layers receive LoRA adapters
#    - lora_dropout: regularization on the adapter layers
# ---------------------------------------------------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],  # GPT-2 attention layer names
    bias="none",
)

# ---------------------------------------------------------------
# 4. Apply LoRA and Print Trainable Parameter Summary
# ---------------------------------------------------------------
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params ~0.6M out of ~82M total (~0.7%)

model.to(device)

# ---------------------------------------------------------------
# 5. Load and Tokenize a Subset of IMDB
#    We use only 1000 training samples to keep this demo fast.
# ---------------------------------------------------------------
dataset = load_dataset("imdb", split="train[:1000]")

def tokenize_fn(examples):
    """Tokenize text with truncation for causal LM training."""
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
tokenized_dataset.set_format("torch")

# ---------------------------------------------------------------
# 6. Configure Training Arguments
#    Small batch size and few epochs for a quick demonstration.
# ---------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),        # fp16 only on CUDA
    logging_steps=25,
    save_strategy="epoch",
    report_to="none",                      # disable wandb/tensorboard
    use_mps_device=(device.type == "mps"), # enable MPS if available
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM, not masked LM
)

# ---------------------------------------------------------------
# 7. Train the LoRA Adapter
# ---------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("\nStarting LoRA fine-tuning...")
trainer.train()

# ---------------------------------------------------------------
# 8. Save Only the Adapter Weights
#    The saved adapter is typically < 5 MB vs. the full model at ~300+ MB.
# ---------------------------------------------------------------
adapter_path = "./lora_adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"\nAdapter saved to {adapter_path}/")

# Show saved file sizes
import os
for f in os.listdir(adapter_path):
    size_kb = os.path.getsize(os.path.join(adapter_path, f)) / 1024
    print(f"  {f}: {size_kb:.1f} KB")