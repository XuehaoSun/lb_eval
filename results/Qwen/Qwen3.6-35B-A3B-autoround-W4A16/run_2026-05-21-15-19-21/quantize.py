#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: Qwen/Qwen3.6-35B-A3B
Scheme: W4A16 (RTN mode)
Export format: auto_round
"""

from auto_round import AutoRound

# Configuration
model_name_or_path = "Qwen/Qwen3.6-35B-A3B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3.6-35B-A3B-W4A16"
scheme = "W4A16"
iters = 0          # RTN mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

# Single GPU: use device="cuda"
autoround_device_kwargs = {"device": "cuda"}

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device args: {autoround_device_kwargs}")

# Create AutoRound instance
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    low_gpu_mem_usage=True,   # Lower VRAM usage for large MoE model
    **autoround_device_kwargs,
)

# Quantize and save
print("Starting quantization...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

print(f"Quantization complete! Output: {output_dir}")