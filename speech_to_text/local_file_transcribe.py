"""
local_file_transcribe.py
Transcribes a local audio file with explicit language control.
Supports WAV, MP3, FLAC, OGG, and other formats via librosa.
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

# ---------------------------------------------------------------
# Accept an audio file path from the command line
# ---------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python local_file_transcribe.py <audio_file> [language_code]")
    print("Example: python local_file_transcribe.py interview.mp3 en")
    sys.exit(1)

audio_path = sys.argv[1]
language = sys.argv[2] if len(sys.argv) > 2 else None

# ---------------------------------------------------------------
# Load audio and resample to 16 kHz (Whisper's expected rate)
# librosa returns a float32 NumPy array normalized to [-1, 1]
# ---------------------------------------------------------------
audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
print(f"Loaded: {audio_path}")
print(f"Duration: {len(audio_array) / sr:.2f} seconds")

# ---------------------------------------------------------------
# Build the transcription pipeline
# ---------------------------------------------------------------
transcriber = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-small",
    device=device,
)

# ---------------------------------------------------------------
# Configure generation parameters for language and task
# Setting task="transcribe" keeps the original language;
# task="translate" translates any language into English
# ---------------------------------------------------------------
generate_kwargs = {"task": "transcribe"}
if language:
    generate_kwargs["language"] = language
    print(f"Forcing language: {language}")

result = transcriber(
    {"array": audio_array, "sampling_rate": 16000},
    generate_kwargs=generate_kwargs,
)

print(f"\nTranscription:\n{result['text']}")