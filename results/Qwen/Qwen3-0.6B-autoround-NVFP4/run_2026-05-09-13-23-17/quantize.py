#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: Qwen/Qwen3-0.6B
Scheme: NVFP4 (RTN mode)
Format: auto_round
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from auto_round import AutoRound

# Configuration
model_name_or_path = "Qwen/Qwen3-0.6B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-NVFP4"
scheme = "NVFP4"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")

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