#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: Qwen/Qwen3-0.6B
Scheme: W4A16 / RTN (iters=0)
Format: auto_round
Device: cuda (single GPU)
"""

import os

os.environ["HF_TOKEN"] = ""

from auto_round import AutoRound

# Configuration
model_name_or_path = "Qwen/Qwen3-0.6B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16"
scheme = "W4A16"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device: cuda (num_gpus={num_gpus})")

# Create AutoRound instance
# Single GPU: use device="cuda"
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    device="cuda",
)

# Quantize and save
print("Starting quantization...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

print(f"Quantization complete! Output: {output_dir}")