from transformers import pipeline
from PIL import Image
import numpy as np

"""
Create a synthetic test image programmatically.
We generate a 224x224 image with a gradient pattern
so we have something to classify without downloading.
"""
array = np.zeros((224, 224, 3), dtype=np.uint8)
array[:, :, 0] = 200  # red channel dominant
array[:, :, 1] = np.linspace(0, 150, 224, dtype=np.uint8)  # green gradient
array[:, :, 2] = 50
image = Image.fromarray(array)

"""
Load the pre-trained ViT image classification pipeline
and run inference on the synthetic image.
"""
classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224"
)

results = classifier(image, top_k=5)

"""
Display the top predictions with labels and scores.
"""
print("Top-5 predictions:")
for r in results:
    print(f"  {r['label']:40s} score: {r['score']:.4f}")