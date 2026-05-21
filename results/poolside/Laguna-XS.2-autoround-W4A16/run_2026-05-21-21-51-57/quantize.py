#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: poolside/Laguna-XS.2
Scheme: W4A16 (RTN mode - iters=0)
Format: auto_round
"""

import os
import sys

# Ensure output directories exist
os.makedirs("/root/.openclaw/workspace/quantized/poolside_Laguna-XS.2-W4A16", exist_ok=True)
os.makedirs("/root/.openclaw/workspace/quantized/runs/poolside_Laguna-XS.2-W4A16/logs", exist_ok=True)

from auto_round import AutoRound

# Configuration
model_name_or_path = "poolside/Laguna-XS.2"
output_dir = "/root/.openclaw/workspace/quantized/poolside_Laguna-XS.2-W4A16"
scheme = "W4A16"
iters = 0        # RTN mode (no training, no calibration samples needed)
nsamples = 128   # still needed for AutoRound init, but will be ignored in RTN
format_str = "auto_round"
num_gpus = 1

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device: cuda")

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
print("Starting quantization (RTN mode)...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

print(f"Quantization complete! Output: {output_dir}")