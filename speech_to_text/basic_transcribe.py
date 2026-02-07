"""
basic_transcribe.py
Demonstrates basic audio transcription using OpenAI Whisper via
the Hugging Face Transformers pipeline API.
"""

import torch
from transformers import pipeline
from datasets import load_dataset

# ---------------------------------------------------------------
# Device selection: prefer Apple MPS GPU, then CUDA, then CPU
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# ---------------------------------------------------------------
# Load the Whisper automatic-speech-recognition pipeline
# whisper-small offers a good speed/accuracy tradeoff (~461M params)
# ---------------------------------------------------------------
transcriber = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-small",
    device=device,
)

# ---------------------------------------------------------------
# Load a sample audio file from the LibriSpeech dummy dataset
# This provides a short English speech clip at 16 kHz
# ---------------------------------------------------------------
dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy",
    "clean",
    split="validation",
    trust_remote_code=True,
)
sample = dataset[0]["audio"]
print(f"Sample rate: {sample['sampling_rate']} Hz")
print(f"Audio length: {len(sample['array']) / sample['sampling_rate']:.2f} seconds")

# ---------------------------------------------------------------
# Run transcription and display the result
# ---------------------------------------------------------------
result = transcriber(sample)
print(f"\nTranscription:\n{result['text']}")