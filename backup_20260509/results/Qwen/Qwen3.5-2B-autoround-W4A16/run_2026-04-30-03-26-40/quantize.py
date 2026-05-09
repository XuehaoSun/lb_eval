#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Generated for Qwen/Qwen3.5-2B

Model: Qwen/Qwen3.5-2B
Scheme: W4A16
Method: RTN (iters=0)
Format: auto_round
Device: cuda (single GPU)
"""

from auto_round import AutoRound

# Configuration
model_name_or_path = "Qwen/Qwen3.5-2B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-2B-W4A16"
scheme = "W4A16"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

# CUDA device selection: single GPU uses device="cuda"
autoround_kwargs = {"device": "cuda"}

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device args: {autoround_kwargs}")

# Create AutoRound instance with disable_opt_rtn to avoid meta offload issues
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    disable_opt_rtn=True,
    **autoround_kwargs,
)

# Quantize and save
print("Starting quantization...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

print(f"Quantization complete! Output: {output_dir}")