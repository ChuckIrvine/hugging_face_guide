"""
basic_summarizer.py
Demonstrates text summarization using Hugging Face models directly.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------------------------------------
# Device detection: prefer Apple MPS GPU, then CUDA, else CPU
# -----------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# -----------------------------------------------------------
# Initialize the model and tokenizer
# (sshleifer/distilbart-cnn-12-6 — a distilled BART variant)
# -----------------------------------------------------------
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# -----------------------------------------------------------
# Sample article text to summarize
# -----------------------------------------------------------
article = """
The James Webb Space Telescope (JWST) has fundamentally transformed our understanding
of the early universe. Launched on December 25, 2021, the telescope orbits the Sun at
the second Lagrange point, approximately 1.5 million kilometers from Earth. Its 6.5-meter
primary mirror, composed of 18 gold-plated beryllium segments, collects infrared light
from the most distant galaxies ever observed. In its first year of operation, JWST
discovered galaxies that formed just 300 million years after the Big Bang, far earlier
than previous models predicted. The telescope has also provided unprecedented views of
exoplanet atmospheres, detecting water vapor, carbon dioxide, and methane in the
atmospheres of planets orbiting distant stars. Scientists believe these findings will
reshape theories of galaxy formation, stellar evolution, and the potential for life
beyond our solar system. The mission, a collaboration between NASA, ESA, and CSA, is
expected to operate for at least 20 years thanks to a precise launch that conserved fuel.
"""

# -----------------------------------------------------------
# Run summarization and print result
# -----------------------------------------------------------
inputs = tokenizer(article, return_tensors="pt", truncation=True, max_length=1024).to(device)

with torch.no_grad():
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=80,
        min_length=30,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\n=== Summary ===")
print(summary)