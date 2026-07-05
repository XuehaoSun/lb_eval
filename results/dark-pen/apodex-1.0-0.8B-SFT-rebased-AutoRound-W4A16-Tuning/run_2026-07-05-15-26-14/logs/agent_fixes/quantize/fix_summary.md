# Fix Summary — apodex-1.0-0.8B-SFT-rebased (Qwen3.5 VL style)

## Root cause
- Model is `Qwen3_5ForConditionalGeneration` (qwen3_5_text + qwen3_5_vision) — genuinely multimodal.
- Its HF repo (`dark-pen/apodex-1.0-0.8B-SFT-rebased`) ships only `video_preprocessor_config.json`
  and is MISSING `preprocessor_config.json` / `processor_config.json`.
- AutoRound correctly routes through `mllm_load_model` → `AutoProcessor.from_pretrained(...)`.
- `transformers.AutoProcessor.from_pretrained` requires `preprocessor_config.json` and raises
  `OSError: Can't load image processor for 'dark-pen/...'`. auto-round 0.13.1's `mllm_load_model`
  wraps the secondary `AutoImageProcessor.from_pretrained` call in try/except but NOT the primary
  `AutoProcessor.from_pretrained` call, so quantization hard-fails before any LM block sees
  calibration data.

## Fix (tier: patch — auto_round 0.13.1)
Two minimal patches in the installed `/root/.venv/lib/python3.12/site-packages/auto_round/`:

1. **`auto_round/utils/model.py`** — wrap `processor = AutoProcessor.from_pretrained(...)` in
   `mllm_load_model` with try/except. On `OSError`/`ValueError`/`KeyError`/`TypeError`, fall
   back to `processor = None` and log a warning. The downstream `mllm_mixin.py` already
   tolerates `processor is None`, and `MLLMMixin.quant_nontext_module=True` semantics ensure
   only the LM backbone (`model.language_model.layers`) is quantized — vision/audio modules
   are skipped automatically.

2. **`auto_round/compressors/mllm/processor.py`** — `qwen3_5` has no registered template in
   auto-round so it falls back to `default` → `HFProcessor`, which `assert processor is not None`.
   Relax the assert: warn-and-continue when `processor is None` (vision inputs already
   unsupported in this code path since `image_processor` was already None due to the existing
   try/except in `mllm_load_model`). Also added a missing `logger = ...` import (it was used
   but undefined, raising `NameError`).

## Result
- `iters=0` (RTN) end-to-end smoke test PASSES in ~31s; produces a valid W4A16 quantized
  model.safetensors (~970 MB) at `/tmp/apodex-out/`.
- `iters=200` (TUNING) constructor PASSES; calibration will use the tokenizer-only
  fallback (`_process_v2` path in HFProcessor for list messages).
- CUDA still works. Patches persist in `/root/.venv` (auto-round is not reinstalled between
  pipeline retries).

## Files saved during smoke (optional reference)
- /tmp/apodex-out/model.safetensors, config.json, tokenizer.json, preprocessor_config.json
  (AutoRound auto-generated a preprocessor_config.json + chat_template.jinja on save, so the
  output is fully self-contained for downstream loading).
