#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: Qwen/Qwen3.5-0.8B
Scheme: NVFP4 (RTN mode)
Format: auto_round
"""

import os
import sys

# Configuration
model_name_or_path = "Qwen/Qwen3.5-0.8B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-NVFP4"
scheme = "NVFP4"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

# CUDA device selection
if num_gpus <= 1:
    autoround_device_kwargs = {"device": "cuda"}
else:
    autoround_device_kwargs = {"device_map": "auto"}

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters}")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device kwargs: {autoround_device_kwargs}")

from auto_round import AutoRound

ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    **autoround_device_kwargs,
)

print("Starting quantization...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

print(f"Quantization complete! Output: {output_dir}")