#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: Qwen/Qwen3-32B
Scheme: MXFP4
Method: RTN (iters=0)
Export format: auto_round
Device: cuda (1 GPU)
"""

import os
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Configuration
model_name_or_path = "/root/.openclaw/workspace/quantized/Qwen3-32B-cache"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3-32B-MXFP4"
runtime_output_dir = "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-32B-MXFP4"
scheme = "MXFP4"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

start_time = time.time()

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device: cuda")
print(f"Output: {output_dir}")

from auto_round import AutoRound

ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    device="cuda",
)

print("Starting quantization and export...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

end_time = time.time()
duration = end_time - start_time
print(f"Quantization complete in {duration:.2f} seconds!")
print(f"Output directory: {output_dir}")
