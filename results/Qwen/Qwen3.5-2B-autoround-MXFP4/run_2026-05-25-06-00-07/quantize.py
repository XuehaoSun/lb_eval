#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: Qwen/Qwen3.5-2B
Scheme: MXFP4
Method: RTN (iters=0)
Format: auto_round
Device: cuda
"""

import time
import sys

start_time = time.time()

# Configuration
model_name_or_path = "Qwen/Qwen3.5-2B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-2B-MXFP4"
scheme = "MXFP4"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device: cuda")

from auto_round import AutoRound

autoround_device_kwargs = {"device": "cuda"}

ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    **autoround_device_kwargs,
)

print("Starting quantization...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

end_time = time.time()
duration = end_time - start_time
print(f"Quantization complete! Duration: {duration:.2f}s")
print(f"Output: {output_dir}")