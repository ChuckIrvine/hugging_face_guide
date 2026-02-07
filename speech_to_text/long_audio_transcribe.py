"""
long_audio_transcribe.py
Transcribes long audio files using chunked inference with timestamps.
"""

import sys
import torch
import librosa
from transformers import pipeline

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

if len(sys.argv) < 2:
    print("Usage: python long_audio_transcribe.py <audio_file>")
    sys.exit(1)

audio_path = sys.argv[1]

# ---------------------------------------------------------------
# Load and resample audio to 16 kHz mono
# ---------------------------------------------------------------
audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
duration = len(audio_array) / sr
print(f"Loaded: {audio_path} ({duration:.1f} seconds)")

# ---------------------------------------------------------------
# Build pipeline with chunking enabled
# chunk_length_s: each chunk is 30 seconds of audio
# stride_length_s: 5-second overlap on each side prevents
#   word loss at chunk boundaries
# ---------------------------------------------------------------
transcriber = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-small",
    device=device,
    chunk_length_s=30,
    stride_length_s=(5, 5),
)

# ---------------------------------------------------------------
# Transcribe with timestamp output
# return_timestamps=True provides start/end times per chunk
# ---------------------------------------------------------------
result = transcriber(
    {"array": audio_array, "sampling_rate": 16000},
    return_timestamps=True,
)

# ---------------------------------------------------------------
# Display timestamped segments
# ---------------------------------------------------------------
print("\nTimestamped Transcription:")
print("-" * 60)
for chunk in result["chunks"]:
    start, end = chunk["timestamp"]
    text = chunk["text"].strip()
    start_str = f"{start:.1f}" if start is not None else "?"
    end_str = f"{end:.1f}" if end is not None else "?"
    print(f"[{start_str}s - {end_str}s] {text}")

print(f"\nFull text:\n{result['text']}")