#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Generated for Qwen/Qwen3-14B W4A16 RTN quantization

Model: Qwen/Qwen3-14B
Scheme: W4A16 (INT4 weight, FP16 activation)
Method: RTN (Round-To-Nearest, iters=0)
Format: auto_round
Device: cuda (single GPU)
"""

import os
import sys

# Ensure output directories exist
output_dir = "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-14B-W4A16"
model_output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3-14B-W4A16"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)

from auto_round import AutoRound

# Configuration
model_name_or_path = "Qwen/Qwen3-14B"
scheme = "W4A16"
iters = 0  # RTN mode - no training iterations
nsamples = 128
format_str = "auto_round"
num_gpus = 1

# Device selection: single GPU uses device="cuda"
autoround_kwargs = {"device": "cuda"}

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Method: RTN (iters={iters})")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device args: {autoround_kwargs}")

# Create AutoRound instance for RTN quantization
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    **autoround_kwargs,
)

# Quantize and save to model_output_dir
print("Starting quantization (RTN mode)...")
ar.quantize_and_save(output_dir=model_output_dir, format=format_str)

print(f"Quantization complete!")
print(f"Output saved to: {model_output_dir}")

# List output files for verification
import glob
output_files = glob.glob(os.path.join(model_output_dir, "*"))
print(f"\nOutput files ({len(output_files)}):")
for f in output_files:
    size = os.path.getsize(f)
    print(f"  {os.path.basename(f)}: {size / 1024 / 1024:.2f} MB")