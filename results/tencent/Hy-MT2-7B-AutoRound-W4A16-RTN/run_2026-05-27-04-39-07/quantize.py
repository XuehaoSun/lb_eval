#!/usr/bin/env python3
"""
AutoRound Quantization Script for tencent/Hy-MT2-7B
Scheme: W4A16 | iters=0 (RTN) | format: auto_round
"""

from auto_round import AutoRound

model_name_or_path = "tencent/Hy-MT2-7B"
output_dir = "/root/.openclaw/workspace/quantized/Hy-MT2-7B-AutoRound-W4A16-RTN"
scheme = "W4A16"
iters = 0
nsamples = 128
format_str = "auto_round"
num_gpus = 1

autoround_kwargs = {"device": "cuda"} if num_gpus <= 1 else {"device_map": "auto"}

print(f"Model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters}")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device kwargs: {autoround_kwargs}")

ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    trust_remote_code=True,
    **autoround_kwargs,
)

print("Starting quantization and export...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)
print(f"Done! Output: {output_dir}")