#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: Qwen/Qwen3-32B
Scheme: MXFP4 (RTN mode - iters=0)
Format: auto_round
"""

from auto_round import AutoRound
import os

# Configuration
model_name_or_path = "/root/.openclaw/workspace/quantized/Qwen_Qwen3-32B-MXFP4"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3-32B-MXFP4"
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
print(f"Device: cuda (single GPU)")

# Create AutoRound instance
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