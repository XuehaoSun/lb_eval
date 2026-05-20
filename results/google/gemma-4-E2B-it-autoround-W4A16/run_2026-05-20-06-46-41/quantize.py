#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Generated for google/gemma-4-E2B-it

Model: google/gemma-4-E2B-it
Scheme: W4A16
Method: RTN (iters=0)
Format: auto_round
Device: cuda (single GPU)
"""

import os
import sys

# Configuration
model_name_or_path = "google/gemma-4-E2B-it"
output_dir = "/root/.openclaw/workspace/quantized/google_gemma-4-E2B-it-W4A16"
scheme = "W4A16"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

# Use device_map=0 for single GPU
autoround_kwargs = {"device_map": 0, "low_gpu_mem_usage": True}

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device args: {autoround_kwargs}")

from auto_round import AutoRound

ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    seqlen=512,  # Reduce to avoid RoPE mismatch with Gemma4 p-RoPE
    disable_opt_rtn=True,
    **autoround_kwargs,
)

print("Starting quantization and export...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

print(f"Quantization complete! Output: {output_dir}")