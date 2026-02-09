"""
02_multi_translate.py
Multi-language translator with lazy model loading and multi-hop support.
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class MultiTranslator:
    """
    Translates between arbitrary language pairs using Helsinki-NLP MarianMT models.
    Models are loaded on first request and cached for subsequent calls.
    """

    def __init__(self, device=None):
        # --- Automatic device detection ---
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        print(f"MultiTranslator using device: {self.device}")

        # Cache: maps "src-tgt" to (tokenizer, model) tuples
        self._cache = {}

    def _load_pair(self, src: str, tgt: str):
        """
        Load and cache the tokenizer and model for a given language pair.
        Raises a clear error if the model does not exist on the Hub.
        """
        key = f"{src}-{tgt}"
        if key not in self._cache:
            model_name = f"Helsinki-NLP/opus-mt-{key}"
            print(f"  Loading model: {model_name} ...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
                self._cache[key] = (tokenizer, model)
            except Exception as e:
                raise RuntimeError(
                    f"Could not load model '{model_name}'. "
                    f"Verify the language pair '{key}' exists at https://huggingface.co/Helsinki-NLP. "
                    f"Original error: {e}"
                )
        return self._cache[key]

    def translate(self, texts, src: str, tgt: str):
        """Translate a list of texts from src language to tgt language."""
        tokenizer, model = self._load_pair(src, tgt)

        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(**inputs, max_length=512)

        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    def translate_multihop(self, texts, hops):
        """
        Translate through a chain of language pairs.
        hops is a list of (src, tgt) tuples, e.g. [("fr", "en"), ("en", "de")]
        """
        current_texts = texts
        for src, tgt in hops:
            print(f"  Hop: {src} → {tgt}")
            current_texts = self.translate(current_texts, src, tgt)
        return current_texts

def main():
    translator = MultiTranslator()

    # --- 1. English to German ---
    print("\n=== English → German ===\n")
    en_texts = [
        "Artificial intelligence will reshape every industry.",
        "I would like a cup of coffee, please.",
    ]
    de_results = translator.translate(en_texts, src="en", tgt="de")
    for src, tgt in zip(en_texts, de_results):
        print(f"  EN: {src}")
        print(f"  DE: {tgt}\n")

    # --- 2. English to Spanish ---
    print("=== English → Spanish ===\n")
    es_results = translator.translate(en_texts, src="en", tgt="es")
    for src, tgt in zip(en_texts, es_results):
        print(f"  EN: {src}")
        print(f"  ES: {tgt}\n")

    # --- 3. Multi-hop: French → English → German ---
    print("=== Multi-hop: French → English → German ===\n")
    fr_texts = [
        "La traduction automatique est un domaine fascinant.",
    ]
    de_via_en = translator.translate_multihop(fr_texts, hops=[("fr", "en"), ("en", "de")])
    for src, tgt in zip(fr_texts, de_via_en):
        print(f"  FR (original): {src}")
        print(f"  DE (via EN):   {tgt}\n")

    # --- 4. Show cached models ---
    print(f"Cached model pairs: {list(translator._cache.keys())}")

if __name__ == "__main__":
    main()