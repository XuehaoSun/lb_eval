#!/usr/bin/env python3
"""
Auto-Round Quantization Script - W4A16 RTN Mode
"""

import os
import time
import sys

start_time = time.time()

from auto_round import AutoRound

model_name_or_path = "sapientinc/HRM-Text-1B"
output_dir = "/root/.openclaw/workspace/quantized/sapientinc_HRM-Text-1B-W4A16"
scheme = "W4A16"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"

print(f"Loading model: {model_name_or_path}", flush=True)
sys.stdout.flush()

ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    device="cuda",
    trust_remote_code=True,
    disable_opt_rtn=False,  # Keep OPT-RTN for speed
    seqlen=512,  # Reduce sequence length for faster processing
    low_gpu_mem_usage=False,
)

print("Starting quantization...", flush=True)
sys.stdout.flush()

ar.quantize_and_save(output_dir=output_dir, format=format_str)

end_time = time.time()
print(f"Quantization complete! Output: {output_dir}", flush=True)
print(f"Duration: {end_time - start_time:.2f}s", flush=True)