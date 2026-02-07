"""
browse_hub.py
Demonstrates programmatic browsing of the Hugging Face Hub
for both models and datasets.
"""

from huggingface_hub import HfApi

# -------------------------------------------------------
# Initialize the Hub API client
# -------------------------------------------------------
api = HfApi()

# -------------------------------------------------------
# Search for text-classification models, sorted by downloads
# -------------------------------------------------------
print("=== Top 5 Text Classification Models ===\n")
models = api.list_models(
    pipeline_tag="text-classification",
    sort="downloads",
    limit=5,
)

for model in models:
    print(f"  Model ID   : {model.modelId}")
    print(f"  Downloads  : {model.downloads:,}")
    print(f"  Likes      : {model.likes}")
    print(f"  Pipeline   : {model.pipeline_tag}")
    print()

# -------------------------------------------------------
# Search for datasets tagged with 'text-classification'
# -------------------------------------------------------
print("=== Top 5 Text Classification Datasets ===\n")
datasets = api.list_datasets(
    task_categories="text-classification",
    sort="downloads",
    limit=5,
)

for ds in datasets:
    print(f"  Dataset ID : {ds.id}")
    print(f"  Downloads  : {ds.downloads:,}")
    print(f"  Likes      : {ds.likes}")
    print()