"""
01_basic_translate.py
Demonstrates basic English-to-French translation using MarianMT.
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate(texts, tokenizer, model, device):
    """
    Translate a list of source-language strings to the target language.
    Tokenizes input, runs generation on the appropriate device,
    and decodes the result.
    """
    # Tokenize source texts with padding and truncation
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate translations using beam search (default num_beams=4 for Marian)
    with torch.no_grad():
        translated_ids = model.generate(**inputs, max_length=512)

    # Decode generated token IDs back to strings
    translations = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
    return translations

def main():
    model_name = "Helsinki-NLP/opus-mt-en-fr"

    # --- Device detection (manual since we are not using Trainer) ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load model and tokenizer with error handling ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        sys.exit(1)

    # --- Sample sentences to translate ---
    english_sentences = [
        "Machine learning is transforming the world.",
        "The weather is beautiful today.",
        "Open-source tools empower developers everywhere.",
    ]

    print("\n=== English → French Translation ===\n")
    results = translate(english_sentences, tokenizer, model, device)

    for src, tgt in zip(english_sentences, results):
        print(f"  EN: {src}")
        print(f"  FR: {tgt}")
        print()

if __name__ == "__main__":
    main()
    