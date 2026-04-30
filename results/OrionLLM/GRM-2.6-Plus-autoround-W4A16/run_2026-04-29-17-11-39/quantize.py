#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Generated for: OrionLLM/GRM-2.6-Plus
Scheme: W4A16 / RTN
Format: auto_round
Device: cuda (1 GPU)
"""

import os
import sys
import time

# Configuration
model_name_or_path = "OrionLLM/GRM-2.6-Plus"
output_dir = "/root/.openclaw/workspace/quantized/OrionLLM_GRM-2.6-Plus-W4A16"
scheme = "W4A16"
iters = 0  # RTN mode - no training iterations
nsamples = 128
format_str = "auto_round"
num_gpus = 1

# CUDA device selection: single GPU -> device="cuda"
autoround_kwargs = {"device": "cuda"}

start_time = time.time()

print(f"=" * 60)
print(f"Auto-Round Quantization - GRM-2.6-Plus (W4A16 / RTN)")
print(f"=" * 60)
print(f"Model: {model_name_or_path}")
print(f"Output: {output_dir}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device kwargs: {autoround_kwargs}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Import after config
from auto_round import AutoRound

print("Loading AutoRound...")
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    trust_remote_code=True,
    **autoround_kwargs,
)

print("Starting quantization and export...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

end_time = time.time()
duration = end_time - start_time

print()
print(f"=" * 60)
print(f"Quantization complete!")
print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
print(f"Output directory: {output_dir}")
print(f"=" * 60)