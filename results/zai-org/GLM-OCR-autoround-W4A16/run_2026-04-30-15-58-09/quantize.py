#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: zai-org/GLM-OCR
Scheme: W4A16 (RTN mode, iters=0)
Format: auto_round
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoProcessor
from auto_round import AutoRound

model_name_or_path = "zai-org/GLM-OCR"
output_dir = "/root/.openclaw/workspace/quantized/zai-org_GLM-OCR-W4A16"
os.makedirs(output_dir, exist_ok=True)

scheme = "W4A16"
iters = 0  # RTN mode
nsamples = 128
format_str = "auto_round"
device = "cuda"

print(f"Loading model: {model_name_or_path}")
print(f"Scheme: {scheme}")
print(f"Iters: {iters} (RTN mode)")
print(f"nsamples: {nsamples}")
print(f"Format: {format_str}")
print(f"Device: {device}")

# Load processor to enable MLLM mode
print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    device=device,
    trust_remote_code=True,
    processor=processor,
)

print("Starting quantization (RTN)...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

print(f"Quantization complete! Output: {output_dir}")