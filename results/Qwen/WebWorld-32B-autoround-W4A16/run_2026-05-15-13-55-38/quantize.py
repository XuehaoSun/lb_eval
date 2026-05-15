#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: Qwen/WebWorld-32B
Scheme: W4A16
Method: RTN (iters=0)
Format: auto_round
"""

from auto_round import AutoRound

# Configuration
model_name_or_path = "Qwen/WebWorld-32B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_WebWorld-32B-W4A16"
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
print(f"Num GPUs: {num_gpus}")

# Single GPU: use device="cuda"
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    device="cuda",
)

print("Starting quantization...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

print(f"Quantization complete! Output: {output_dir}")