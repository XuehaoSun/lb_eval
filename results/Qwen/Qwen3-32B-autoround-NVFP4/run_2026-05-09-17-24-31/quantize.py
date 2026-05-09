#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Generated for NVFP4/RTN quantization of Qwen/Qwen3-32B

Model: Qwen/Qwen3-32B
Scheme: NVFP4
Method: RTN (iters=0)
Format: auto_round
Device: cuda (single GPU)
"""

import os
import sys

# Force CUDA device assignment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from auto_round import AutoRound

# Configuration
model_name_or_path = "Qwen/Qwen3-32B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3-32B-NVFP4"
runtime_output_dir = "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-32B-NVFP4"
scheme = "NVFP4"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device: cuda")

# Create AutoRound instance for single GPU CUDA
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