import json
import torch
from transformers import AutoConfig

# ---------------------------------------------------------------
# Device check for Apple MPS GPU
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    print("Apple MPS GPU detected.\n")

# ---------------------------------------------------------------
# Load the model configuration
# ---------------------------------------------------------------
config = AutoConfig.from_pretrained("distilbert-base-uncased")
print("Model type        :", config.model_type)
print("Hidden size        :", config.hidden_size)
print("Num hidden layers  :", config.num_hidden_layers)
print("Num attention heads:", config.n_heads)
print("Vocab size         :", config.vocab_size)
print("Max position embeds:", config.max_position_embeddings)

# ---------------------------------------------------------------
# Examine the full config as a dictionary
# ---------------------------------------------------------------
config_dict = config.to_dict()
print(f"\nTotal config keys: {len(config_dict)}")
print("Keys:", sorted(config_dict.keys()))

# ---------------------------------------------------------------
# Modify a config value (e.g., reduce layers for experimentation)
# ---------------------------------------------------------------
custom_config = AutoConfig.from_pretrained(
    "distilbert-base-uncased",
    num_hidden_layers=2
)
print(f"\nCustom config layers: {custom_config.num_hidden_layers}")