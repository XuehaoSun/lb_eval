#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1
Scheme: W4A16
Method: RTN (iters=0)
Format: auto_round
"""

from auto_round import AutoRound
import sys

# Configuration
model_name_or_path = "jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1"
output_dir = "/root/.openclaw/workspace/quantized/jpacifico_Chocolatine-2-4B-Instruct-DPO-v2.1-W4A16"
scheme = "W4A16"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
seqlen = 512  # Reduce memory

# Use device_map="auto" for better device handling
autoround_kwargs = {"device_map": "auto"}

print(f"Loading model: {model_name_or_path}", flush=True)
print(f"Scheme: {scheme}", flush=True)
print(f"Iters: {iters} (RTN mode)", flush=True)
print(f"nsamples: {nsamples}", flush=True)
print(f"Format: {format_str}", flush=True)
print(f"Device args: {autoround_kwargs}", flush=True)
print(f"seqlen: {seqlen}", flush=True)

ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    seqlen=seqlen,
    disable_opt_rtn=True,  # Pure RTN, faster
    **autoround_kwargs,
)

print("Starting quantization...", flush=True)
ar.quantize_and_save(output_dir=output_dir, format=format_str)

print(f"Quantization complete! Output: {output_dir}", flush=True)