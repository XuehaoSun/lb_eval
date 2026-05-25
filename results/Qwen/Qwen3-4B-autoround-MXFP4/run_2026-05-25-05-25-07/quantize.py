#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Generated for Qwen/Qwen3-4B MXFP4 quantization

Model: Qwen/Qwen3-4B
Scheme: MXFP4
Method: RTN (iters=0)
Format: auto_round
Device: cuda
"""

import sys
import os

# Ensure output directories exist
OUTPUT_DIR = "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-4B-MXFP4"
MODEL_EXPORT_DIR = "/root/.openclaw/workspace/quantized/Qwen_Qwen3-4B-MXFP4"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)

from auto_round import AutoRound

# Configuration
model_name_or_path = "Qwen/Qwen3-4B"
export_dir = MODEL_EXPORT_DIR
scheme = "MXFP4"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

# CUDA device selection: single GPU uses device="cuda"
autoround_kwargs = {"device": "cuda"} if num_gpus <= 1 else {"device_map": "auto"}

print(f"=" * 60)
print(f"Auto-Round Quantization")
print(f"=" * 60)
print(f"Model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device: {autoround_kwargs}")
print(f"Output: {export_dir}")
print(f"=" * 60)

# Create AutoRound instance
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    **autoround_kwargs,
)

# Quantize and save
print("Starting quantization...")
ar.quantize_and_save(output_dir=export_dir, format=format_str)

print(f"Quantization complete! Output: {export_dir}")