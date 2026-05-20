#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Generated for Zyphra/ZAYA1-8B
Scheme: W4A16 (RTN mode - iters=0)
Format: auto_round
Device: cuda (single GPU)
"""

import os
import sys
import time

HF_TOKEN = os.environ.get("HF_TOKEN", None)

print(f"Python: {sys.executable}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

from auto_round import AutoRound

# Configuration
model_name_or_path = "Zyphra/ZAYA1-8B"
output_dir = "/root/.openclaw/workspace/quantized/Zyphra_ZAYA1-8B-W4A16"
scheme = "W4A16"
iters = 0          # RTN mode (no training iterations)
nsamples = 128
format_str = "auto_round"

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Mode: RTN (iters={iters})")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Output: {output_dir}")

start_time = time.time()

# Create AutoRound instance
# Use disable_opt_rtn=True to skip imatrix calibration which causes tuple-return issue with Zaya architecture
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    device="cuda",
    disable_opt_rtn=True,  # Force pure RTN without imatrix calibration
)

# Quantize and save
print("Starting quantization (RTN, pure mode)...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

end_time = time.time()
duration = end_time - start_time

print(f"Quantization complete in {duration:.2f}s")
print(f"Output: {output_dir}")

# Print output files
for root, dirs, files in os.walk(output_dir):
    for f in files:
        fp = os.path.join(root, f)
        size_mb = os.path.getsize(fp) / 1024 / 1024
        print(f"  {fp} ({size_mb:.2f} MB)")