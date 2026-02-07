"""
Reusable NER utility class built on top of the Hugging Face pipeline.
Produces structured JSON-ready output for integration into applications.
"""

import json
import torch
from transformers import pipeline

class NERExtractor:
    """
    Wraps a Hugging Face NER pipeline into a reusable extractor
    with structured output.
    """

    def __init__(self, model_name="dslim/bert-base-NER"):
        # -----------------------------------------------------------
        # Detect the best available device (Apple MPS, CUDA, or CPU)
        # -----------------------------------------------------------
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # -----------------------------------------------------------
        # Initialize the pipeline once for reuse across calls
        # -----------------------------------------------------------
        self.pipe = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=self.device,
        )

    def extract_entities(self, text):
        """
        Extract named entities from a single text string.
        Returns a list of dicts with entity, label, score, start, end.
        """
        raw = self.pipe(text)
        return [
            {
                "entity": ent["word"],
                "label": ent["entity_group"],
                "score": round(float(ent["score"]), 4),
                "start": ent["start"],
                "end": ent["end"],
            }
            for ent in raw
        ]

    def extract_batch(self, texts):
        """
        Process multiple texts and return a list of entity lists.
        """
        return [self.extract_entities(t) for t in texts]

# ---------------------------------------------------------------
# Demonstration: process several documents and print JSON output
# ---------------------------------------------------------------
if __name__ == "__main__":
    extractor = NERExtractor()

    documents = [
        "Tim Cook unveiled the new iPhone at Apple Park in Cupertino, California.",
        "The European Central Bank, based in Frankfurt, raised interest rates.",
        "Serena Williams competed at Wimbledon and later visited London.",
    ]

    for i, doc in enumerate(documents):
        entities = extractor.extract_entities(doc)
        print(f"\nDocument {i + 1}: {doc}")
        print(json.dumps(entities, indent=2))