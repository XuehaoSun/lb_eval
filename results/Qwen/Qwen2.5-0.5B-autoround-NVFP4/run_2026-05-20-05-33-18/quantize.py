#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: Qwen/Qwen2.5-0.5B
Scheme: NVFP4 (RTN mode)
Format: auto_round
"""

import os
import sys
import time

print(f"Python: {sys.executable}", flush=True)

from auto_round import AutoRound

model_name_or_path = "Qwen/Qwen2.5-0.5B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen2.5-0.5B-NVFP4"
runtime_dir = "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen2.5-0.5B-NVFP4"
scheme = "NVFP4"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
num_gpus = 1

print(f"Model: {model_name_or_path}", flush=True)
print(f"Scheme: {scheme}", flush=True)
print(f"Iters: {iters} (RTN)", flush=True)
print(f"nsamples: {nsamples}", flush=True)
print(f"Format: {format_str}", flush=True)
print(f"Device: cuda (single GPU)", flush=True)
print(f"Output: {output_dir}", flush=True)

os.makedirs(output_dir, exist_ok=True)

start_time = time.time()

ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    device="cuda",
)

print("Quantizing...", flush=True)
ar.quantize_and_save(output_dir=output_dir, format=format_str)

end_time = time.time()
duration = end_time - start_time

print(f"Quantization complete in {duration:.2f}s", flush=True)
print(f"Output: {output_dir}", flush=True)

# List output files
import pathlib
for f in sorted(pathlib.Path(output_dir).rglob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.relative_to(output_dir)} ({size_mb:.2f} MB)", flush=True)