"""
check_environment.py
Verify that all dependencies are installed and detect the best available
compute device (Apple MPS, NVIDIA CUDA, or CPU).
"""

import torch
import transformers

# ── Print library versions ──────────────────────────────────────────
print(f"Transformers version : {transformers.__version__}")
print(f"PyTorch version      : {torch.__version__}")

# ── Detect the best available compute device ────────────────────────
# Apple Silicon Macs expose GPU acceleration through the MPS backend.
# NVIDIA GPUs are accessed via CUDA. If neither is available, fall
# back to CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device               : Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device               : CUDA GPU ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("Device               : CPU")

# ── Quick tensor sanity check ───────────────────────────────────────
# Create a small tensor on the detected device to confirm it works.
t = torch.tensor([1.0, 2.0, 3.0], device=device)
print(f"Tensor on {device}   : {t}")
print("\nEnvironment is ready.")