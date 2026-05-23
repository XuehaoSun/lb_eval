#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: Qwen/Qwen3-235B-A22B
Scheme: W4A16 (INT4 weight, FP16 activation)
Method: RTN (iters=0, no calibration training)
Export format: auto_round
Device: cuda (single GPU)
"""

import os
import sys
import time

# Set HF_HOME to /dev/shm (117GB available)
os.environ["HF_HOME"] = "/dev/shm/hf_cache"

# Suppress torch compile warnings
os.environ["TORCH_COMPILE_DISABLE"] = "1"

print("=" * 60)
print("Auto-Round Quantization - Qwen/Qwen3-235B-A22B")
print("=" * 60)

start_time = time.time()

# Configuration
model_name_or_path = "Qwen/Qwen3-235B-A22B"
output_dir = "/root/.openclaw/workspace/quantized/Qwen_Qwen3-235B-A22B-W4A16"
runtime_dir = "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-235B-A22B-W4A16"
scheme = "W4A16"
iters = 0  # RTN mode - no calibration training
nsamples = 128  # used for activation range estimation even in RTN
format_str = "auto_round"
num_gpus = 1

print(f"Model: {model_name_or_path}")
print(f"Output: {output_dir}")
print(f"Scheme: {scheme}")
print(f"Method: RTN (iters={iters})")
print(f"Format: {format_str}")
print(f"Device: cuda")
print(f"GPUs: {num_gpus}")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")
print()

# Create output directory
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Import AutoRound
try:
    from auto_round import AutoRound
    print("AutoRound imported successfully")
except ImportError as e:
    print(f"ERROR: Failed to import AutoRound: {e}")
    sys.exit(1)

# Device selection - use device_map="auto" for better memory handling
autoround_kwargs = {"device_map": "auto"}

# Create AutoRound instance
print("\nInitializing AutoRound...")
try:
    ar = AutoRound(
        model_name_or_path,
        scheme=scheme,
        iters=iters,
        nsamples=nsamples,
        low_gpu_mem_usage=False,  # 235B model needs full memory
        **autoround_kwargs,
    )
    print("AutoRound instance created")
except Exception as e:
    print(f"ERROR: Failed to create AutoRound instance: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Quantize and save
print("\nStarting quantization...")
print("This may take 30-60 minutes for a 235B parameter model...")
sys.stdout.flush()

try:
    ar.quantize_and_save(output_dir=output_dir, format=format_str)
    print("\nQuantization completed successfully!")
except Exception as e:
    print(f"\nERROR: Quantization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

end_time = time.time()
duration = end_time - start_time

print(f"\nTotal time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
print(f"Output saved to: {output_dir}")

# List output files
print("\nOutput files:")
for root, dirs, files in os.walk(output_dir):
    for f in files:
        fp = os.path.join(root, f)
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        print(f"  {os.path.relpath(fp, output_dir)} ({size_mb:.2f} MB)")
