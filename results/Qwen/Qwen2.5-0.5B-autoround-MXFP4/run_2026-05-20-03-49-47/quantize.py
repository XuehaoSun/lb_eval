#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: Qwen/Qwen2.5-0.5B
Scheme: MXFP4 / RTN (iters=0)
Format: auto_round
Device: cuda (single GPU)
"""

import os
import sys
import time

# Ensure CUDA device is visible
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from auto_round import AutoRound

# Configuration
model_name_or_path = "Qwen/Qwen2.5-0.5B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen2.5-0.5B-MXFP4"
scheme = "MXFP4"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"

num_gpus = 1
autoround_device_kwargs = {"device": "cuda"} if num_gpus <= 1 else {"device_map": "auto"}

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device args: {autoround_device_kwargs}")
print(f"Output dir: {output_dir}")
sys.stdout.flush()

start_time = time.time()

# Create AutoRound instance
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    **autoround_device_kwargs,
)

print("Starting quantization and export...", flush=True)

# Quantize and save
ar.quantize_and_save(output_dir=output_dir, format=format_str)

end_time = time.time()
duration = end_time - start_time
print(f"Quantization complete in {duration:.2f} seconds", flush=True)
print(f"Output: {output_dir}", flush=True)