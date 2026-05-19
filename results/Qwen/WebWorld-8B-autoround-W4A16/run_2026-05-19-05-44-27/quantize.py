#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: Qwen/WebWorld-8B
Scheme: W4A16 (INT4 weight, FP16 activation)
Method: RTN (iters=0)
Format: auto_round
Device: cuda (single GPU)
"""

from auto_round import AutoRound

# Configuration
model_name_or_path = "Qwen/WebWorld-8B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16"
scheme = "W4A16"
iters = 0  # RTN mode - no calibration training
nsamples = 128
format_str = "auto_round"
num_gpus = 1

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Method: RTN (iters={iters})")
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