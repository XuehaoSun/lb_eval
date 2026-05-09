#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Generated for Qwen/Qwen3-0.6B
Scheme: W4A16 / RTN
Format: auto_round
Device: cuda
"""

from auto_round import AutoRound
import time

start_time = time.time()

# Configuration
model_name_or_path = "Qwen/Qwen3-0.6B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16"
runtime_output_dir = "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16"
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
print(f"Device: cuda")

# Create AutoRound instance - single GPU uses device="cuda"
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

duration = time.time() - start_time
print(f"Quantization complete! Duration: {duration:.2f}s")
print(f"Output: {output_dir}")