#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Model: google-t5/t5-3b
Scheme: W4A16 (RTN)
Format: auto_round
"""

import os
import sys

# Ensure proper CUDA device visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from auto_round import AutoRound
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configuration
model_name_or_path = "google-t5/t5-3b"
output_dir = "/root/.openclaw/workspace/quantized/google-t5_t5-3b-W4A16"
runtime_output_dir = "/root/.openclaw/workspace/quantized/runs/google-t5_t5-3b-W4A16"
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
print(f"Device: cuda")

# T5 is a Seq2Seq model - AutoRound only supports CausalLM by default
# Load model and tokenizer directly using the correct model class
print("Loading T5 model with AutoModelForSeq2SeqLM...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
print("Model loaded successfully")

# Create AutoRound instance with pre-loaded model
# Since T5 is not a CausalLM model, AutoRound's model loading will fail
# We pass the already-loaded model to bypass that
ar = AutoRound(
    model,
    tokenizer=tokenizer,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    device="cuda",
)

# Quantize and save
print("Starting quantization...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

print(f"Quantization complete! Output: {output_dir}")