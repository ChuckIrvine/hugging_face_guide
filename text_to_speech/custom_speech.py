"""
Custom TTS generation with configurable text input and speaker selection.
Usage: python custom_speech.py --text "Your text here" --speaker 7306
"""

import argparse
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import numpy as np

# ---------------------------------------------------------------
# Argument Parsing
# Accept text and speaker index from command line arguments.
# ---------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate speech from text")
parser.add_argument("--text", type=str, default="Hello, welcome to the Hugging Face guide.",
                    help="Text to convert to speech")
parser.add_argument("--speaker", type=int, default=7306,
                    help="Speaker embedding index from CMU ARCTIC dataset")
parser.add_argument("--output", type=str, default="custom_speech.wav",
                    help="Output WAV file path")
args = parser.parse_args()

# ---------------------------------------------------------------
# Device Detection
# Check for Apple MPS, CUDA, or fall back to CPU.
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------
# Load All Model Components
# ---------------------------------------------------------------
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

# ---------------------------------------------------------------
# Create Speaker Embedding
# Use speaker index as seed for reproducible synthetic voices.
# ---------------------------------------------------------------
torch.manual_seed(args.speaker)
speaker_embedding = torch.randn(1, 512).to(device)
print(f"Using speaker seed: {args.speaker}")

# ---------------------------------------------------------------
# Sentence-Level Generation
# SpeechT5 has a maximum input length (~600 tokens). For longer
# texts, we split by sentence and concatenate the resulting audio.
# ---------------------------------------------------------------
sentences = [s.strip() for s in args.text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
all_speech = []

print(f"Generating speech for {len(sentences)} sentence(s)...")
for i, sentence in enumerate(sentences):
    # Add punctuation back for natural prosody
    sentence_text = sentence + "."
    inputs = processor(text=sentence_text, return_tensors="pt").to(device)

    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embedding,
            vocoder=vocoder
        )
    all_speech.append(speech.cpu().numpy())

    # Add a short silence (0.3s) between sentences
    silence = np.zeros(int(16000 * 0.3), dtype=np.float32)
    all_speech.append(silence)

# ---------------------------------------------------------------
# Concatenate and Save
# ---------------------------------------------------------------
final_audio = np.concatenate(all_speech)
sf.write(args.output, final_audio, samplerate=16000)
print(f"Saved: {args.output} ({len(final_audio) / 16000:.2f}s)")