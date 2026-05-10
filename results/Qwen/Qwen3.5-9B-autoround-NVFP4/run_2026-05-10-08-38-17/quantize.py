#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Generated for Qwen/Qwen3.5-9B

Model: Qwen/Qwen3.5-9B
Scheme: NVFP4
Method: RTN (iters=0)
Export Format: auto_round
Output: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-9B-NVFP4
"""

import os

# Environment setup
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from auto_round import AutoRound

# Configuration
model_name_or_path = "Qwen/Qwen3.5-9B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-9B-NVFP4"

# NVFP4 with RTN (iters=0)
scheme = "NVFP4"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

# CUDA device selection: single GPU uses device="cuda"
autoround_device_kwargs = {"device": "cuda"} if num_gpus <= 1 else {"device_map": "auto"}

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Method: RTN (iters={iters})")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device args: {autoround_device_kwargs}")

# Create AutoRound instance
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    **autoround_device_kwargs,
)

# Quantize and save
print("Starting quantization...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

print(f"Quantization complete! Output: {output_dir}")