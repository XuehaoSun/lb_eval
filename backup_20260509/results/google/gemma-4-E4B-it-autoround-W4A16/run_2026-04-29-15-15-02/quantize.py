#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: google/gemma-4-E4B-it
Scheme: W4A16
Method: RTN (iters=0)
Format: auto_round
Device: cuda (single GPU)
"""

import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "0"

from auto_round import AutoRound

# Configuration
model_name_or_path = "google/gemma-4-E4B-it"
output_dir = "/root/.openclaw/workspace/quantized/google_gemma-4-E4B-it-W4A16"
scheme = "W4A16"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

print(f"Model: {model_name_or_path}")
print(f"Output: {output_dir}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device: cuda (num_gpus={num_gpus})")

# Single GPU: use device_map="auto" for better compatibility
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    device_map="auto",
    seqlen=512,
    disable_opt_rtn=True,
    enable_torch_compile=False,
)

print("Starting quantization...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)
print(f"Quantization complete! Output: {output_dir}")