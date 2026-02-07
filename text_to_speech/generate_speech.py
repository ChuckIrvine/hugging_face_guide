"""
Text-to-Speech generation using Microsoft SpeechT5 and HiFi-GAN vocoder.
Generates a .wav audio file from an input text string.
"""

import torch
import soundfile as sf
from scipy.signal import resample
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# ---------------------------------------------------------------
# Device Detection
# Check for Apple MPS (Metal Performance Shaders), then CUDA,
# then fall back to CPU.
# ---------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (GPU) device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

# ---------------------------------------------------------------
# Load Model Components
# The processor handles text tokenization. The model generates
# mel spectrograms. The vocoder converts spectrograms to audio.
# ---------------------------------------------------------------
print("Loading SpeechT5 processor, model, and vocoder...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

# ---------------------------------------------------------------
# Load Speaker Embedding
# Speaker embeddings encode the vocal characteristics of a
# particular speaker. We use a pre-computed x-vector from the
# CMU ARCTIC dataset. Index 7306 corresponds to a clear female voice.
# ---------------------------------------------------------------
print("Creating speaker embedding...")
# Use a seeded random embedding for consistent output.
# For production, use speechbrain's EncoderClassifier to generate
# real speaker embeddings from audio samples.
torch.manual_seed(42)
speaker_embedding = torch.randn(1, 512).to(device)

# ---------------------------------------------------------------
# Text Input and Tokenization
# The processor converts the input text into token IDs that the
# SpeechT5 encoder can consume.
# ---------------------------------------------------------------
input_text = (
    "Hugging Face makes it remarkably easy to generate "
    "natural sounding speech from text using state of the art models."
)
print(f"\nInput text: {input_text}\n")

inputs = processor(text=input_text, return_tensors="pt").to(device)

# ---------------------------------------------------------------
# Speech Generation
# The model produces a mel spectrogram, and the vocoder converts
# it into a waveform tensor. We move the result to CPU and convert
# to a NumPy array for file writing.
# ---------------------------------------------------------------
print("Generating speech...")
with torch.no_grad():
    speech = model.generate_speech(
        inputs["input_ids"],
        speaker_embedding,
        vocoder=vocoder
    )

# Move to CPU for saving
speech_numpy = speech.cpu().numpy()

# Slow down speech by 15% (adjust factor as needed: >1.0 = slower)
speed_factor = 1.15
new_length = int(len(speech_numpy) * speed_factor)
speech_numpy = resample(speech_numpy, new_length)

# ---------------------------------------------------------------
# Save Output Audio
# Write the waveform to a WAV file at 16 kHz (SpeechT5's native
# sample rate).
# ---------------------------------------------------------------
output_path = "output_speech.wav"
sf.write(output_path, speech_numpy, samplerate=16000)
print(f"Audio saved to: {output_path}")
print(f"Audio duration: {len(speech_numpy) / 16000:.2f} seconds")