#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: XiaomiMiMo/MiMo-V2.5-ASR
Scheme: W4A16 (INT4 weight, FP16 activation)
iters: 0 (RTN fast mode)
Format: auto_round
"""

import os
import time

start_time = time.time()

model_name_or_path = "XiaomiMiMo/MiMo-V2.5-ASR"
output_dir = "/root/.openclaw/workspace/quantized/MiMo-V2.5-ASR-AutoRound-W4A16-RTN"
scheme = "W4A16"
iters = 0  # RTN fast mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

# CUDA device selection: single GPU uses device="cuda"
autoround_kwargs = {"device": "cuda"} if num_gpus <= 1 else {"device_map": "auto"}

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device args: {autoround_kwargs}")

from auto_round import AutoRound

ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    **autoround_kwargs,
)

print("Starting quantization...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

end_time = time.time()
print(f"Quantization complete in {end_time - start_time:.2f}s")
print(f"Output: {output_dir}")