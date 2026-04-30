#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: EbanLee/kobart-summary-v3
Scheme: W4A16
Method: RTN (iters=0)
Format: auto_round
Device: cuda
"""

import os
import sys
import time

# Configuration
model_name_or_path = "EbanLee/kobart-summary-v3"
output_dir = "/root/.openclaw/workspace/quantized/EbanLee_kobart-summary-v3-W4A16"
runtime_output_dir = "/root/.openclaw/workspace/quantized/runs/EbanLee_kobart-summary-v3-W4A16"
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
print(f"Device: cuda (single GPU)")

start_time = time.time()

try:
    from auto_round import AutoRound

    # Single GPU: use device="cuda"
    ar = AutoRound(
        model_name_or_path,
        scheme=scheme,
        iters=iters,
        nsamples=nsamples,
        device="cuda",
        trust_remote_code=True,
    )

    print("Starting quantization...")
    ar.quantize_and_save(output_dir=output_dir, format=format_str)

    elapsed = time.time() - start_time
    print(f"Quantization complete! Output: {output_dir}")
    print(f"Duration: {elapsed:.2f} seconds")

    # List output files
    import glob
    output_files = glob.glob(os.path.join(output_dir, "*"))
    output_files += glob.glob(os.path.join(output_dir, "**", "*"), recursive=True)
    output_files = [f for f in output_files if os.path.isfile(f)]
    print(f"Output files ({len(output_files)}):")
    for f in sorted(output_files):
        size = os.path.getsize(f)
        print(f"  {f} ({size/1024/1024:.2f} MB)")

except Exception as e:
    elapsed = time.time() - start_time
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
