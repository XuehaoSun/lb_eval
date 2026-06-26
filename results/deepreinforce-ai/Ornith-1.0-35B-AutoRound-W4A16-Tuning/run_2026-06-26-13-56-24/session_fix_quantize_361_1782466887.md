# Session: fix_quantize_361_1782466887

- **Session ID:** `fix_quantize_361_1782466887`
- **Timestamp:** 2026-06-26 09:41:31 UTC
- **Working Dir:** `/root/.openclaw/workspace`

## Step 1: Quantization

### [2026-06-26 09:41:31 UTC] USER

You are fixing a failed "quantize" phase in the quantization pipeline.

## Error Output (last 100 lines):
09:40:23 [INFO] Configuring AutoRound...
[38;20m2026-06-26 09:40:23 INFO entry.py L587: Using MLLM mode for multimodal model.[0m
09:40:23 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
09:40:23 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/config.json "HTTP/1.1 200 OK"
09:40:23 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/model_index.json "HTTP/1.1 404 Not Found"
404 Client Error. (Request ID: Root=1-6a3e4907-0226003c595f32de42352c1b;53c4e3f2-ccef-4e41-bb17-0811c6fd006a)

Entry Not Found for url: https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/model_index.json.
09:40:23 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
09:40:23 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/config.json "HTTP/1.1 200 OK"
09:40:23 [INFO] HTTP Request: GET https://huggingface.co/api/models/deepreinforce-ai/Ornith-1.0-35B/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
09:40:23 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
09:40:23 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/config.json "HTTP/1.1 200 OK"
09:40:23 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
09:40:23 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/config.json "HTTP/1.1 200 OK"
09:40:23 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/model.safetensors "HTTP/1.1 404 Not Found"
09:40:23 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/model.safetensors.index.json "HTTP/1.1 307 Temporary Redirect"
09:40:23 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/model.safetensors.index.json "HTTP/1.1 200 OK"
09:40:23 [INFO] HTTP Request: GET https://huggingface.co/api/models/deepreinforce-ai/Ornith-1.0-35B/revision/main "HTTP/1.1 200 OK"
09:41:20 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
09:41:20 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/generation_config.json "HTTP/1.1 200 OK"
09:41:20 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/custom_generate/generate.py "HTTP/1.1 404 Not Found"
09:41:20 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
09:41:20 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/config.json "HTTP/1.1 200 OK"
09:41:20 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
09:41:20 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/tokenizer_config.json "HTTP/1.1 200 OK"
09:41:20 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
09:41:20 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/tokenizer_config.json "HTTP/1.1 200 OK"
09:41:20 [INFO] HTTP Request: GET https://huggingface.co/api/models/deepreinforce-ai/Ornith-1.0-35B/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
09:41:20 [INFO] HTTP Request: GET https://huggingface.co/api/models/deepreinforce-ai/Ornith-1.0-35B/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
09:41:21 [INFO] HTTP Request: GET https://huggingface.co/api/models/deepreinforce-ai/Ornith-1.0-35B "HTTP/1.1 200 OK"
09:41:21 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/processor_config.json "HTTP/1.1 307 Temporary Redirect"
09:41:21 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/processor_config.json "HTTP/1.1 200 OK"
09:41:21 [INFO] HTTP Request: GET https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/processor_config.json "HTTP/1.1 200 OK"
09:41:21 [INFO] HTTP Request: GET https://huggingface.co/api/models/deepreinforce-ai/Ornith-1.0-35B/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
09:41:21 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/processor_config.json "HTTP/1.1 307 Temporary Redirect"
09:41:21 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/processor_config.json "HTTP/1.1 200 OK"
09:41:21 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/chat_template.json "HTTP/1.1 404 Not Found"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/chat_template.jinja "HTTP/1.1 307 Temporary Redirect"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/chat_template.jinja "HTTP/1.1 200 OK"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/audio_tokenizer_config.json "HTTP/1.1 404 Not Found"
09:41:22 [INFO] HTTP Request: GET https://huggingface.co/api/models/deepreinforce-ai/Ornith-1.0-35B/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/processor_config.json "HTTP/1.1 307 Temporary Redirect"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/processor_config.json "HTTP/1.1 200 OK"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/chat_template.json "HTTP/1.1 404 Not Found"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/chat_template.jinja "HTTP/1.1 307 Temporary Redirect"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/chat_template.jinja "HTTP/1.1 200 OK"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/audio_tokenizer_config.json "HTTP/1.1 404 Not Found"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/processor_config.json "HTTP/1.1 307 Temporary Redirect"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/processor_config.json "HTTP/1.1 200 OK"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
09:41:22 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/preprocessor_config.json "HTTP/1.1 200 OK"
09:41:22 [INFO] HTTP Request: GET https://huggingface.co/api/resolve-cache/models/deepreinforce-ai/Ornith-1.0-35B/5df2ed3f675c7beaa490328cc70bb573b65fb660/preprocessor_config.json "HTTP/1.1 200 OK"
09:41:22 [ERROR] Quantization failed: Unrecognized image processor in deepreinforce-ai/Ornith-1.0-35B. Should have a `image_processor_type` key in its preprocessor_config.json of config.json, or one of the following `model_type` keys in its config.json: aria, beit, bit, blip, bridgetower, chameleon, chinese_clip, chmv2, clip, cohere2_vision, conditional_detr, convnext, deepseek_ocr2, deepseek_vl, deepseek_vl_hybrid, deformable_detr, deit, depth_pro, detr, dinov3_vit, dpt, efficientloftr, efficientnet, eomt, ernie4_5_vl_moe, flava, fuyu, gemma3, gemma4, gemma4_unified, glm46v, glm4v, glm_image, glmga, glpn, got_ocr2, grounding-dino, idefics, idefics2, idefics3, imagegpt, janus, layoutlmv2, layoutlmv3, levit, lfm2_vl, lightglue, llama4, llava, llava_next, llava_onevision, mask2former, maskformer, minicpmv4_6, minimax_m3_vl, mllama, mobilenet_v1, mobilenet_v2, mobilevit, nougat, oneformer, ovis2, owlv2, owlvit, paddleocr_vl, perceiver, perception_lm, phi4_multimodal, pi0, pix2struct, pixtral, poolformer, pp_chart2table, pp_doclayout_v2, pp_doclayout_v3, pp_formulanet, pp_lcnet, pp_ocrv5_server_det, pp_ocrv5_server_rec, pp_ocrv6_small_rec, prompt_depth_anything, pvt, qwen2_vl, rf_detr, rt_detr, sam, sam2, sam3, sapiens2, segformer, seggpt, siglip, siglip2, slanext, smolvlm, superglue, superpoint, swin2sr, textnet, tvp, uvdoc, video_llama_3, videomae, vilt, vit, vitmatte, vitpose, yolos, zoedepth, aimv2, aimv2_vision_model, align, altclip, aya_vision, blip-2, clipseg, colpali, colqwen2, convnextv2, cosmos3_omni, cvt, data2vec-vision, deimv2, depth_anything, dinat, dinov2, donut-swin, edgetam, emu3, eomt_dinov3, exaone4_5, florence2, focalnet, gemma3n, git, granite4_vision, groupvit, hiera, ijepa, instructblip, internvl, kosmos-2, kosmos-2.5, layoutxlm, lighton_ocr, llava_next_video, lw_detr, metaclip_2, mgp-str, mistral3, mlcd, mm-grounding-dino, mobilevitv2, omdet-turbo, paligemma, pixio, pp_ocrv5_mobile_det, pp_ocrv5_mobile_rec, pp_ocrv6_medium_det, pp_ocrv6_small_det, pp_ocrv6_tiny_rec, pvt_v2, qianfan_ocr, qwen2_5_omni, qwen2_5_vl, qwen3_5, qwen3_5_moe, qwen3_omni_moe, qwen3_vl, regnet, resnet, sam2_video, sam3_lite_text, sam3_tracker, sam3_tracker_video, sam3_video, sam_hq, shieldgemma2, slanet, swiftformer, swin, swinv2, t5gemma2, t5gemma2_encoder, table-transformer, timesformer, timm_wrapper, trocr, udop, upernet, video_llava, vipllava, vit_mae, vit_msn, vivit, xclip
Traceback (most recent call last):
  File "/root/_work/1/s/auto_quant/phases/quantize.py", line 282, in <module>
    quantize(args)
  File "/root/_work/1/s/auto_quant/phases/quantize.py", line 178, in quantize
    autoround = AutoRound(**ar_kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/auto_round/autoround.py", line 165, in __new__
    return AutoRoundCompatible(**local_args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/auto_round/compressors/entry.py", line 594, in __new__
    compressor = AutoRound(
                 ^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/auto_round/compressors/entry.py", line 252, in __new__
    return _get_compressor_class(model_type, DataDrivenCompressor)(alg_configs, **local_args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/auto_round/compressors/mllm_mixin.py", line 87, in __init__
    super().__init__(*args, **kwargs)
  File "/root/.venv/lib/python3.12/site-packages/auto_round/compressors/data_driven.py", line 93, in __init__
    super().__init__(
  File "/root/.venv/lib/python3.12/site-packages/auto_round/compressors/base.py", line 300, in __init__
    self.model_context = ModelContext(
                         ^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/auto_round/context/base.py", line 39, in __call__
    instance.__init__(*args, **kwargs)
  File "/root/.venv/lib/python3.12/site-packages/auto_round/context/base.py", line 27, in wrapped_init
    original_init(self, *args, **kwargs)
  File "/root/.venv/lib/python3.12/site-packages/auto_round/context/model.py", line 106, in __init__
    self._load_model()
  File "/root/.venv/lib/python3.12/site-packages/auto_round/context/model.py", line 138, in _load_model
    self.model, self.processor, self.tokenizer, self.image_processor = mllm_load_model(
                                                                       ^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/auto_round/utils/model.py", line 757, in mllm_load_model
    processor = AutoProcessor.from_pretrained(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/transformers/models/auto/processing_auto.py", line 327, in from_pretrained
    return processor_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/transformers/processing_utils.py", line 1715, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/transformers/processing_utils.py", line 1844, in _get_arguments_from_pretrained
    sub_processor = auto_processor_class.from_pretrained(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/transformers/models/auto/image_processing_auto.py", line 651, in from_pretrained
    raise ValueError(
ValueError: Unrecognized image processor in deepreinforce-ai/Ornith-1.0-35B. Should have a `image_processor_type` key in its preprocessor_config.json of config.json, or one of the following `model_type` keys in its config.json: aria, beit, bit, blip, bridgetower, chameleon, chinese_clip, chmv2, clip, cohere2_vision, conditional_detr, convnext, deepseek_ocr2, deepseek_vl, deepseek_vl_hybrid, deformable_detr, deit, depth_pro, detr, dinov3_vit, dpt, efficientloftr, efficientnet, eomt, ernie4_5_vl_moe, flava, fuyu, gemma3, gemma4, gemma4_unified, glm46v, glm4v, glm_image, glmga, glpn, got_ocr2, grounding-dino, idefics, idefics2, idefics3, imagegpt, janus, layoutlmv2, layoutlmv3, levit, lfm2_vl, lightglue, llama4, llava, llava_next, llava_onevision, mask2former, maskformer, minicpmv4_6, minimax_m3_vl, mllama, mobilenet_v1, mobilenet_v2, mobilevit, nougat, oneformer, ovis2, owlv2, owlvit, paddleocr_vl, perceiver, perception_lm, phi4_multimodal, pi0, pix2struct, pixtral, poolformer, pp_chart2table, pp_doclayout_v2, pp_doclayout_v3, pp_formulanet, pp_lcnet, pp_ocrv5_server_det, pp_ocrv5_server_rec, pp_ocrv6_small_rec, prompt_depth_anything, pvt, qwen2_vl, rf_detr, rt_detr, sam, sam2, sam3, sapiens2, segformer, seggpt, siglip, siglip2, slanext, smolvlm, superglue, superpoint, swin2sr, textnet, tvp, uvdoc, video_llama_3, videomae, vilt, vit, vitmatte, vitpose, yolos, zoedepth, aimv2, aimv2_vision_model, align, altclip, aya_vision, blip-2, clipseg, colpali, colqwen2, convnextv2, cosmos3_omni, cvt, data2vec-vision, deimv2, depth_anything, dinat, dinov2, donut-swin, edgetam, emu3, eomt_dinov3, exaone4_5, florence2, focalnet, gemma3n, git, granite4_vision, groupvit, hiera, ijepa, instructblip, internvl, kosmos-2, kosmos-2.5, layoutxlm, lighton_ocr, llava_next_video, lw_detr, metaclip_2, mgp-str, mistral3, mlcd, mm-grounding-dino, mobilevitv2, omdet-turbo, paligemma, pixio, pp_ocrv5_mobile_det, pp_ocrv5_mobile_rec, pp_ocrv6_medium_det, pp_ocrv6_small_det, pp_ocrv6_tiny_rec, pvt_v2, qianfan_ocr, qwen2_5_omni, qwen2_5_vl, qwen3_5, qwen3_5_moe, qwen3_omni_moe, qwen3_vl, regnet, resnet, sam2_video, sam3_lite_text, sam3_tracker, sam3_tracker_video, sam3_video, sam_hq, shieldgemma2, slanet, swiftformer, swin, swinv2, t5gemma2, t5gemma2_encoder, table-transformer, timesformer, timm_wrapper, trocr, udop, upernet, video_llava, vipllava, vit_mae, vit_msn, vivit, xclip

## Historical Lessons (from past runs — decide which are relevant):
Lesson 1 [phase=quantize, verified=5x]:
  Error: auto_round error or auto-round related exception
  Solution: If auto-round raises an error (import error, API change, compatibility issue, missing method, etc.), upgrade to the latest main branch: uv pip install --reinstall "auto-round @ git+https://github.com/intel/auto-round.git@main" This often fixes issues with new model architectures or recently added features. After reinstall, verify: python -c "import auto_round; print(auto_round.__version__)"
  Notes: auto-round is actively developed. PyPI releases may lag behind fixes for new models. Always try main branch first before other workarounds.

Lesson 2 [phase=evaluate, verified=3x]:
  Error: RuntimeError: The NVIDIA driver on your system is too old (found version XXXXX)
  Solution: Reinstall PyTorch with a CUDA version matching the NVIDIA driver. Steps: 1) Run nvidia-smi to check driver-supported CUDA version (look for "CUDA Version: X.Y"). 2) Map to PyTorch index-url tag. Available: cu118, cu121, cu124, cu126, cu128, cu130. 3) Reinstall: uv pip install --reinstall torch torchaudio torchvision --index-url https://download.pytorch.org/whl/<cu_tag>. Common mappings: CUDA 11.8 -> cu118, CUDA 12.0~12.3 -> cu121, CUDA 12.4~12.5 -> cu124, CUDA 12.6~12.7 -> cu126, CUDA 12.8~12.9 -> cu128, CUDA 13.0+ -> cu130. Do NOT force CPU-only (device_map=cpu). Do NOT upgrade the NVIDIA driver. After reinstall, verify: python -c "import torch; print(torch.cuda.is_available())" should be True.
  Notes: This is an infrastructure issue caused by pre-installed torch compiled for a newer CUDA than the driver supports. The correct fix is always to reinstall torch with --index-url pointing to the compatible CUDA wheel, never to skip GPU.

Lesson 3 [phase=quantize, verified=1x]:
  Error: 16:28:11 [ERROR] Quantization failed: invalid group reference 1 at position 22
  Solution: **FIX_PLAN:**;1. Check current auto-round version and reinstall from main branch (fixes regex group reference bug);2. Re-run the quantization phase to verify the fix;The bug is in `re.sub(r"\(.*\)", "", source_pattern)` — it strips content inside parentheses, removing the capturing group `(.+)`, but the replacement template still contains `\1` which becomes an invalid group reference. This is an auto-round bug.;--;

Lesson 4 [phase=quantize, verified=1x]:
  Error: 03:50:38 [ERROR] Quantization failed:
  Solution: **FIX_PLAN:**;1. Install missing PIL (pillow) and torchvision packages required by Qwen2VLImageProcessor;2. Re-run the quantization phase to verify the fix;;--;

Lesson 5 [phase=quantize, verified=1x]:
  Error: 21:10:03 [ERROR] Quantization failed:
  Solution: **FIX_PLAN:**;1. Install missing `pillow` and `torchvision` packages required by `Qwen2VLImageProcessor`;2. Verify packages installed correctly;3. Re-run the quantization phase;

Lesson 6 [phase=quantize, verified=1x]:
  Error: 02:07:48 [ERROR] Quantization failed: invalid group reference 1 at position 22
  Solution: **FIX_PLAN:**;1. Reinstall auto-round from main branch to fix the regex group reference bug (Lesson 3);2. Verify the installation;3. Re-run the quantization phase;

Lesson 7 [phase=quantize, verified=1x]:
  Error: 01:07:01 [ERROR] Quantization failed:
  Solution: Both packages are missing. Installing them now:;**FIX_PLAN:**;1. Install missing `pillow` and `torchvision` packages required by `Qwen2VLImageProcessor` — ✅ both installed;2. Verify imports work — ✅ confirmed;3. Re-run quantization phase to verify;

Lesson 8 [phase=quantize, verified=1x]:
  Error: 07:23:54 [ERROR] Quantization failed:
  Solution: **FIX_PLAN:**;1. Install missing `pillow` and `torchvision` packages required by `Qwen2VLImageProcessor`;2. Verify imports work;3. Re-run quantization phase;

Lesson 9 [phase=quantize, verified=1x]:
  Error: 10:03:43 [ERROR] Quantization failed:
  Solution: **FIX_PLAN:**;1. Install missing `pillow` and `torchvision` packages;2. Verify imports work;3. Re-run quantization phase;

Lesson 10 [phase=quantize, verified=1x]:
  Error: 15:52:16 [ERROR] Quantization failed:
  Solution: **FIX_PLAN:**;1. Install missing `pillow` and `torchvision` packages;2. Verify imports work;3. Re-run quantization phase;**FIX_PLAN:**;
Review the lessons above and apply the most relevant fix for the current error.

## Your Task:
1. READ the traceback carefully — identify the EXACT file and line that caused the error
2. Determine if the fault is in: auto-round code, transformers, model's custom code, or environment
3. Output a brief FIX_PLAN (3 lines max) describing what you will do
4. Execute the fix, then the phase will be re-run to verify

## Key Technique: Patching Model Custom Code

If the traceback shows files in `~/.cache/huggingface/modules/transformers_modules/`, that is the
MODEL'S CUSTOM CODE that was downloaded from HuggingFace. **YOU CAN AND SHOULD EDIT THESE FILES.**

Common fixes for model custom code:
- dtype mismatch (`.float()` mixed with bfloat16): Replace `.float()` with `.to(other_tensor.dtype)`
- Missing device: Add `device=hidden_states.device` to tensor creation
- Invalid regex: Fix the regex pattern in the model file
- Missing imports: Add the import or install the package

Example: If you see:
  File "/root/.cache/huggingface/modules/transformers_modules/Org/Model/hash/model.py", line 147
    h = h + torch.matmul(compressed[:, k:k+valid_len, :].float(), proj.t())
  RuntimeError: expected m1 and m2 to have the same dtype

Fix: Edit that file, change `.float()` to `.to(proj.dtype)`

## Constraints:
- Do NOT reinstall or downgrade torch (it will break CUDA)
- Do NOT modify the evaluation tasks or expected output format
- Keep fixes minimal and targeted — change only what's needed
- If you need to install a package, use: pip install <package>
- If unsupported model architecture (multimodal/VL), report and stop
- Working directory: /root/_work/1/s/auto_quant/output/runs/Ornith-1.0-35B-AutoRound-W4A16-Tuning
- Model: deepreinforce-ai/Ornith-1.0-35B

### [2026-06-26 09:41:47 UTC] ASSISTANT

**Thinking**

Let me analyze the error carefully.

The error is:
```
ValueError: Unrecognized image processor in deepreinforce-ai/Ornith-1.0-35B. Should have a `image_processor_type` key in its preprocessor_config.json of config.json, or one of the following `model_type` keys in its config.json: [very long list of model types]
```

The traceback shows:
1. The error originates in `transformers/processing_utils.py` line 1844
2. It's trying to load an `AutoProcessor` for the model
3. The model is `deepreinforce-ai/Ornith-1.0-35B`
4. The error says the image processor is unrecognized

This is happening because:
1. AutoRound is trying to quantize a multimodal model (MLLM mode was detected - "Using MLLM mode for multimodal model")
2. It needs to load the model with its processor
3. The processor loading fails because the model's `preprocessor_config.json` doesn't have an `image_processor_type` key and the `config.json` doesn't have a recognized `model_type`

Looking at the error message more closely - it says the model should have one of the listed `model_type` keys. The model type `Ornith` is not in that list. This is a custom model architecture that auto-round/transformers doesn't recognize.

Looking at the model: `deepreinforce-ai/Ornith-1.0-35B` - this appears to be a custom multimodal model.

The error happens during `AutoProcessor.from_pretrained()` which is called from `mllm_load_model()` in auto-round's utils.

Now, looking at the historical lessons:
- Lesson 1 mentions upgrading auto-round from main branch for compatibility issues with new models


- Other lessons focus on missing PIL/torchvision packages

This doesn't match any of the previous lessons exactly. The error is about an unrecognized image processor, not a missing package. Since the model is multimodal and uses MLLM mode, but Ornith isn't in the recognized model types list, this is likely a compatibility issue between the model's custom architecture and what auto-round/transformers supports. I should try upgrading auto-round from main as a first step, similar to how Lesson 1 approached new model architectures

**Tool call:** `exec`

```json
{
  "command": "cd /root/_work/1/s/auto_quant && pip show auto-round | head -5"
}
```

### [2026-06-26 09:41:47 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

sh: 1: pip: not found

### [2026-06-26 09:41:50 UTC] ASSISTANT

**Thinking**

Let me check with uv or the v

**Tool call:** `exec`

```json
{
  "command": "cd /root/_work/1/s/auto_quant && uv pip show auto-round 2>/dev/null | head -5 || /root/.venv/bin/python -m pip show auto-round 2>/dev/null | head -5"
}
```

### [2026-06-26 09:41:50 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Name: auto-round
Version: 0.13.1
Location: /root/.venv/lib/python3.12/site-packages
Requires: accelerate, datasets, numpy, py-cpuinfo, pydantic, torch, tqdm, transformers
Required-by:

### [2026-06-26 09:41:58 UTC] ASSISTANT

**Thinking**

The error is:
```
ValueError: Unrecognized image processor in deepreinforce-ai/Ornith-1.0-35B. Should have a `image_processor_type` key in its preprocessor_config.json of config.json, or one of the following `model_type` keys in its config.json: ...
```

The model `Ornith-1.0-35B` is a multimodal model but its config doesn't have the proper `image_processor_type` or a recognized `model_type`. Looking at the error, the model_type should be one of the listed types.

Looking at the lesson history, Lesson 1 suggests upgrading auto-round from main branch when there are issues. This seems like a model compatibility issue where auto-round might need to handle this model type better.


The real problem is that transformers' `AutoProcessor` can't find a recognized image processor type in the model's config, which is required for multimodal models. I should check the model config to see what's actually specified and whether it's a supported type or if there's a configuration missing

**Tool call:** `exec`

```json
{
  "command": "cat /root/.cache/huggingface/modules/transformers_modules/deepreinforce-ai/Ornith-1.0-35B/*/config.json 2>/dev/null | head -100 || find /root/.cache/huggingface/modules/transformers_modules/ -name \"config.json\" -path \"*Ornith*\" 2>/dev/null | head -5"
}
```

### [2026-06-26 09:41:58 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-06-26 09:42:03 UTC] ASSISTANT

**Thinking**

The error is happening at `mllm_load_model` which is trying to load a processor. The model `deepreinforce-ai/Ornith-1.0-35B` is a multimodal model and it seems the image processor type is not recognized by transformers.

The traceback shows:
```
File "/root/.venv/lib/python3.12/site-packages/transformers/models/auto/image_processing_auto.py", line 651, in from_pretrained
    raise ValueError(
ValueError: Unrecognized image processor in deepreinforce-ai/Ornith-1.0-35B. Should have a `image_processor_type` key in its preprocessor_config.json of config.json, or one of the following `model_type` keys in its config.json: ...
```

This is an issue with how the model's preprocessor_config.json or config.json is set up. The model type might not be recognized.

Let me check if there's a preprocessor_config.json in the cached model

**Tool call:** `exec`

```json
{
  "command": "find /root/.cache/huggingface -name \"preprocessor_config.json\" -path \"*Ornith*\" 2>/dev/null | head -3"
}
```

### [2026-06-26 09:42:03 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.cache/huggingface/hub/models--deepreinforce-ai--Ornith-1.0-35B/snapshots/5df2ed3f675c7beaa490328cc70bb573b65fb660/preprocessor_config.json

### [2026-06-26 09:42:07 UTC] ASSISTANT

**Thinking**

Let me check the preprocessor_config.json to understand the issue. The error says the preprocessor_config.json doesn't have an `image_processor_type` key and the model_type in config.json isn't recognized.


**Tool call:** `exec`

```json
{
  "command": "cat /root/.cache/huggingface/hub/models--deepreinforce-ai--Ornith-1.0-35B/snapshots/5df2ed3f675c7beaa490328cc70bb573b65fb660/preprocessor_config.json"
}
```

### [2026-06-26 09:42:07 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
    "size": {
        "longest_edge": 16777216,
        "shortest_edge": 65536
    },
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [
        0.5,
        0.5,
        0.5
    ],
    "image_std": [
        0.5,
        0.5,
        0.5
    ],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast"
}

### [2026-06-26 09:42:15 UTC] ASSISTANT

**Thinking**

The preprocessor_config.json has `"image_processor_type": "Qwen2VLImageProcessorFast"`. But the error message says it should have a `image_processor_type` key OR one of the listed model types.

Wait - the error says "Should have a `image_processor_type` key in its preprocessor_config.json of config.json". But the preprocessor_config.json DOES have `image_processor_type: Qwen2VLImageProcessorFast`.

Let me look more carefully. The error is about the image processor not being recognized. `Qwen2VLImageProcessorFast` is listed in the supported list as `qwen2_5_vl`. 

The issue might be that transformers doesn't recognize `Qwen2VLImageProcessorFast` as a valid image processor type. This is likely a version mismatch where the installed transformers doesn't support this newer image processor type that was added later.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING; print('Qwen2VLImageProcessorFast' in IMAGE_PROCESSOR_MAPPING)\""
}
```

### [2026-06-26 09:42:20 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

False

### [2026-06-26 09:42:26 UTC] ASSISTANT

**Thinking**

The image processor type "Qwen2VLImageProcessorFast" is not in the standard transformers IMAGE_PROCESSOR_MAPPING. This is a custom image processor type that the model defines. Let me check if it's in a transformers_modules cache directory - the model might have custom code.


**Tool call:** `exec`

```json
{
  "command": "find /root/.cache/huggingface/modules/transformers_modules/ -type d -name \"*Ornith*\" 2>/dev/null"
}
```

### [2026-06-26 09:42:26 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`



(Command exited with code 1)

### [2026-06-26 09:42:29 UTC] ASSISTANT

**Thinking**

The model uses a custom image processor `Qwen2VLImageProcessorFast` which doesn't exist in the installed transformers version. Let me check what version of transformers is installed and if this processor exists somewhere

**Tool call:** `exec`

```json
{
  "command": "uv pip show transformers | head -3"
}
```

### [2026-06-26 09:42:29 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Using Python 3.12.13 environment at: /root/.venv
Name: transformers
Version: 5.12.1
Location: /root/.venv/lib/python3.12/site-packages

### [2026-06-26 09:42:40 UTC] ASSISTANT

**Thinking**

Transformers 5.12.1 is very recent. The `Qwen2VLImageProcessorFast` might not be available yet in this version, or it might be named differently. Let me check what image processors are

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING; print([k for k in IMAGE_PROCESSOR_MAPPING.keys() if 'qwen' in k.lower() or 'vl' in k.lower()])\""
}
```

### [2026-06-26 09:42:45 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
AttributeError: type object 'Aimv2Config' has no attribute 'lower'

(Command exited with code 1)

### [2026-06-26 09:42:51 UTC] ASSISTANT

**Thinking**

Let me check if `Qwen2VLImageProcessorFast` exists in the transformers source

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessorFast; print('OK')\""
}
```

### [2026-06-26 09:42:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/root/.venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py", line 25, in <module>
    from torchvision.transforms.v2 import functional as tvF
ModuleNotFoundError: No module named 'torchvision'

(Command exited with code 1)

### [2026-06-26 09:43:01 UTC] ASSISTANT

**Thinking**

This is the issue. The transformers version requires torchvision for `Qwen2VLImageProcessorFast`, but torchvision isn't installed. According to the historical lessons, this exact error has happened before (Lessons 4, 5, 7, 8, 9, 10). The fix is to install `pillow` and `torchvision`.

However, I need to check if `torchvision` is actually what's missing here. The error says `No module named 'torchvision'`, and this happens when trying to import `Qwen2VLImageProcessorFast`. Let me install torchvision and pillow



The issue is clear: `Qwen2VLImageProcessorFast` (which the model uses) requires `torchvision` but it's not installed. This is a repeat of Lessons 4, 5, 7, 8, 9, and 10.

**FIX_PLAN:**
1. Install missing `torchvision` and `pillow` packages required by `Qwen2VLImageProcessorFast`
2. Verify imports work
3. Re-run the quantization phase


**Tool call:** `exec`

```json
{
  "command": "uv pip install torchvision pillow 2>&1 | tail -5"
}
```

### [2026-06-26 09:43:01 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Downloaded torchvision
Prepared 2 packages in 152ms
Installed 2 packages in 36ms
 + pillow==12.2.0
 + torchvision==0.27.1

### [2026-06-26 09:43:06 UTC] ASSISTANT

**Thinking**

Let me verify the imports work now

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessorFast; print('Qwen2VLImageProcessorFast OK')\""
}
```

### [2026-06-26 09:43:11 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: cannot import name 'Qwen2VLImageProcessorFast' from 'transformers.models.qwen2_vl.image_processing_qwen2_vl' (/root/.venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py). Did you mean: 'Qwen2VLImageProcessor'?

(Command exited with code 1)

### [2026-06-26 09:43:17 UTC] ASSISTANT

**Thinking**

Hmm, the class is `Qwen2VLImageProcessor` not `Qwen2VLImageProcessorFast`. But the preprocessor_config.json says `"image_processor_type": "Qwen2VLImageProcessorFast"`. This might be a version mismatch issue - the preprocessor_config references a newer image processor type than what's available in the current transformers version.

Let me check what version of transformers has `Qwen2VLImageProcessorFast`.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor; print('Qwen2VLImageProcessor OK')\""
}
```

### [2026-06-26 09:43:22 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Qwen2VLImageProcessor OK

### [2026-06-26 09:43:39 UTC] ASSISTANT

**Thinking**

So `Qwen2VLImageProcessor` exists but `Qwen2VLImageProcessorFast` doesn't. The preprocessor_config.json specifies `image_processor_type: "Qwen2VLImageProcessorFast"` which is not available in transformers 5.12.1.

This means the model was likely trained/created with a newer version of transformers that has `Qwen2VLImageProcessorFast`. 

Options:
1. Upgrade transformers to a newer version that has `Qwen2VLImageProcessorFast`
2. Patch the preprocessor_config.json to use `Qwen2VLImageProcessor` instead

But wait - the traceback shows that transformers itself is raising the error. It says "Should have a `image_processor_type` key in its preprocessor_config.json" — meaning the image_processor_type IS present but it's not recognized.
 
The real issue is that `Qwen2VLImageProcessorFast` was added in a transformers version after 5.12.1, so I need to either upgrade transformers or patch the preprocessor config to use the older `Qwen2VLImageProcessor` instead. upgrading to a newer version that has `Qwen2VLImageProcessorFast`.

Let me check if there's a newer transformers available.

Actually, thinking about this more - the error happens during AutoRound's quantization phase when it tries to load the model. The model's preprocessor_config.json specifies `Qwen2VLImageProcessorFast` which doesn't exist in transformers 5.12.1.

The cleanest fix would be to patch the preprocessor_config.json to use `Qwen2VLImageProcessor` instead of `Qwen2VLImageProcessorFast`, since the latter is just a faster implementation of the same functionality.

Let me patch the preprocessor_config.json file



The `Qwen2VLImageProcessorFast` doesn't exist in transformers 5.12.1 — only `Qwen2VLImageProcessor`. I'll patch the model's preprocessor_config.json to use the available class.


**Tool call:** `edit`

```json
{
  "path": "/root/.cache/huggingface/hub/models--deepreinforce-ai--Ornith-1.0-35B/snapshots/5df2ed3f675c7beaa490328cc70bb573b65fb660/preprocessor_config.json",
  "newText": "\"image_processor_type\": \"Qwen2VLImageProcessor\"",
  "old_text": "\"image_processor_type\": \"Qwen2VLImageProcessorFast\""
}
```

### [2026-06-26 09:43:39 UTC] TOOL RESULT
**Tool:** `edit`

Successfully replaced text in /root/.cache/huggingface/hub/models--deepreinforce-ai--Ornith-1.0-35B/snapshots/5df2ed3f675c7beaa490328cc70bb573b65fb660/preprocessor_config.json.

### [2026-06-26 09:43:42 UTC] ASSISTANT

**Thinking**

Now let me verify the import works.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING; print('Qwen2VLImageProcessor' in IMAGE_PROCESSOR_MAPPING)\""
}
```

### [2026-06-26 09:43:46 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

False

### [2026-06-26 09:43:52 UTC] ASSISTANT

**Thinking**

Hmm, `Qwen2VLImageProcessor` is not in the IMAGE_PROCESSOR_MAPPING. Let me check what's available for Qwen2-VL models. The error message from the traceback said the config.json needs a `model_type` key with one of the listed values. Let me check the model's main config.json.


**Tool call:** `exec`

```json
{
  "command": "cat /root/.cache/huggingface/hub/models--deepreinforce-ai--Ornith-1.0-35B/snapshots/5df2ed3f675c7beaa490328cc70bb573b65fb660/config.json | head -50"
}
```

### [2026-06-26 09:43:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "architectures": [
    "Qwen3_5MoeForConditionalGeneration"
  ],
  "bos_token_id": null,
  "dtype": "bfloat16",
  "eos_token_id": 248046,
  "hidden_size": 2048,
  "image_token_id": 248056,
  "model_type": "qwen3_5_moe",
  "pad_token_id": 248044,
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "attn_output_gate": true,
    "bos_token_id": 248044,
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "layer_types": [
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",

### [2026-06-26 09:43:57 UTC] ASSISTANT

**Thinking**

The model type is `qwen3_5_moe`. This is not a VL model type per se - it's `qwen3_5_moe`. But the preprocessor_config references `Qwen2VLImageProcessorFast` which is a vision-language image processor.

The issue is that `transformers` doesn't have a mapping for `qwen3_5_moe` model type to any image processor. Even though `Qwen2VLImageProcessor` exists, it's not mapped to `qwen3_5_moe`.

Let me check the `IMAGE_PROCESSOR_MAPPING_NAMES` to understand the available mappings

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES; print([k for k in IMAGE_PROCESSOR_MAPPING_NAMES if 'qwen' in k.lower()])\""
}
```

### [2026-06-26 09:44:01 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

['qwen2_vl', 'colqwen2', 'qwen2_5_omni', 'qwen2_5_vl', 'qwen3_5', 'qwen3_5_moe', 'qwen3_omni_moe', 'qwen3_vl']

### [2026-06-26 09:44:06 UTC] ASSISTANT

**Thinking**

`qwen3_5_moe` is in the list. The model's `model_type` is `qwen3_5_moe`. But the error message says `qwen3_5` is not a valid image processor type either.

Let me check if `Qwen2VLImageProcessor` is registered under any model type

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"\nfrom transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES\nfor k, v in IMAGE_PROCESSOR_MAPPING_NAMES.items():\n    if 'qwen' in k.lower():\n        print(k, '->', v)\n\""
}
```

### [2026-06-26 09:44:10 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

qwen2_vl -> {'pil': 'Qwen2VLImageProcessorPil', 'torchvision': 'Qwen2VLImageProcessor'}
colqwen2 -> {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}
qwen2_5_omni -> {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}
qwen2_5_vl -> {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}
qwen3_5 -> {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}
qwen3_5_moe -> {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}
qwen3_omni_moe -> {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}
qwen3_vl -> {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}

### [2026-06-26 09:44:44 UTC] ASSISTANT

**Thinking**

So for `qwen3_5` model type, the `torchvision` variant uses `Qwen2VLImageProcessor`. But the `IMAGE_PROCESSOR_MAPPING` doesn't have a `Qwen2VLImageProcessorFast` entry either. The error message lists supported model types but `qwen3_5` IS in the list! But it says the `image_processor_type` should be one of the supported types or have a `image_processor_type` key...

Wait, the error says: `Should have a image_processor_type key in its preprocessor_config.json of config.json, or one of the following model_type keys in its config.json: ...`
 
Ah, so the issue is that the model's `config.json` has `model_type: qwen3_5_moe` which IS in the supported list, so transformers should be able to find the right image processor automatically. But the problem might be that `IMAGE_PROCESSOR_MAPPING` doesn't have an entry for `qwen3_5_moe` directly, so when it tries to look up the image processor type, it can't find the mapping it needs. I'm seeing the actual image processor mapping for `qwen3_5` uses `Qwen2VLImageProcessor` from torchvision, not the fast variant. The error message is saying the processor type isn't registered in the mapping—`Qwen2VLImageProcessorFast` doesn't exist in `IMAGE_PROCESSOR_MAPPING`, so when `from_pretrained` tries to resolve it, it fails. The model's `config.json` has `model_type: qwen3_5_moe` which is a supported type, so the issue is specifically that the image processor class name being used isn't available in the transformers registry. Looking at the stack trace more carefully, the error originates in `processing_utils.py` when trying to load the processor, not from the model's custom code itself. The traceback shows the failure happens during `AutoProcessor.from_pretrained`, which suggests the issue is with how the processor is being instantiated rather than a problem with the model's architecture.

The real problem seems to be that `Qwen2VLImageProcessorFast` doesn't actually exist in transformers despite being listed in the preprocessor config, and even after changing it to `Qwen2VLImageProcessor`, that class still isn't registered in the image processing mapping for the `qwen3_5_moe` model type. The issue is that `IMAGE_PROCESSOR_MAPPING` only contains base classes like `Qwen2VLImageProcessor`, not `Qwen2VLImageProcessorFast`. Since the preprocessor config references `Qwen2VLImageProcessorFast` and there's no registered factory for it, the system can't instantiate the processor and falls back to iterating through known model types—but `qwen3_5_moe` isn't recognized, so it fails.

Wait, actually `qwen3_5_moe` IS in that list of supported model types the error message provides. Let me check the full list more carefully to see if I'm missing something about which types are actually supported. Looking at the traceback, the error gets raised when `auto_processor_class.from_pretrained()` is called inside `_get_arguments_from_pretrained`. I should check which specific `image_processor_type` values are actually registered in the mapping to understand why `Qwen2VLImageProcessorFast` isn't being resolved.

But actually, I think I see it now—when `image_processor_type` isn't found in `IMAGE_PROCESSOR_MAPPING`, the code falls back to iterating through all available model types and trying to load an image processor for each one. The error message is telling me that none of those attempts succeeded, which suggests the real problem is that `qwen3_5_moe` isn't properly registered as supporting any image processor type. So the real problem is that `qwen3_5_moe` doesn't appear in `IMAGE_PROCESSOR_MAPPING` at all, even though it's listed in the error message's supported types. When the code iterates through all registered image processors trying to find one compatible with `qwen3_5_moe`, it fails because that model type was never added to the mapping in the first place. Let me check what model types are actually registered in `IMAGE_PROCESSOR_MAPPING` to see where the disconnect is. The model type `qwen3_5_moe` IS listed in the error message, yet it's not actually in `IMAGE_PROCESSOR_MAPPING`—this suggests a transformers version mismatch where newer model types were added to error messages but not to the actual mapping. Let me check the transformers version to confirm this hypothesis. The issue is that while `qwen3_5_moe` is recognized as a valid model type in the error message, the actual `IMAGE_PROCESSOR_MAPPING` doesn't have a corresponding entry for it. I need to check whether there's a `Qwen2_5MoeImageProcessor` or similar that should be mapped to this model type, since the mapping seems incomplete compared to what's actually supported. The error occurs because `AutoProcessor._get_arguments_from_pretrained` tries every image processor class in `IMAGE_PROCESSOR_MAPPING` to find one compatible with the model, but none of them actually support `qwen3_5_moe`. So even though the model type is recognized, there's no corresponding image processor registered for it in the mapping. the `image_processor_type` in the preprocessor config to an actual registered class, but none of them support `qwen3_5_moe` anyway. I need to check what the actual `Qwen3_5MoeImageProcessor` implementation looks like to understand what's supported.

Actually, let me reconsider the model architecture. The error mentions `Qwen3_5MoeForConditionalGeneration` in the config, but looking at the supported model types, `qwen3_5_moe` isn't listed—though `qwen3_5` and `qwen3_5_moe` are both there as distinct types. The error is saying `qwen3_5_moe` isn't recognized, so there's a mismatch between what the model claims to be and what the image processor supports. The issue is that `qwen3_5_moe` isn't registered as a model type in transformers, even though the error message lists it as supported. I should try a different approach—maybe add the missing model type registration or find an alternative way to resolve this image processor mismatch without modifying the config directly.

Let me check if auto-round has any built-in handling for this, or if I need to manually register the model type or patch the preprocessor config to use a compatible image processor class instead.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"\nfrom transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES\nprint('qwen3_5' in IMAGE_PROCESSOR_MAPPING_NAMES)\nprint('qwen3_5_moe' in IMAGE_PROCESSOR_MAPPING_NAMES)\n\""
}
```

### [2026-06-26 09:44:48 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

True
True

### [2026-06-26 09:45:11 UTC] ASSISTANT

**Thinking**

Wait, the model_type is "qwen3_5_moe" which is in the list! But the error says the image processor is not found. Let me look at the full error again.

The error says: "Should have a `image_processor_type` key in its preprocessor_config.json of config.json, or one of the following `model_type` keys in its config.json: ..."

So the config.json has `model_type: "qwen3_5_moe"`. And `qwen3_5_moe` IS in the list of supported model types. So why does it fail?

Wait, the error says `image_processor_type` was "Qwen2VLImageProcessorFast" in the preprocessor_config.json. 

The error message is confusing because it seems to be conflating two different checks—one for `image_processor_type` in preprocessor_config.json and another for `model_type` in config.json. Looking at the image_processing_auto.py code at line 651, it's checking whether the processor matches one of the supported types, but the error message appears to be listing both `image_processor_type` options and `model_type` keys in a way that's not clear which one actually failed.

The issue is that even though `qwen3_5_moe` exists in IMAGE_PROCESSOR_MAPPING_NAMES, the AutoProcessor is trying to load a `Qwen2VLImageProcessorFast` from the preprocessor_config, which doesn't exist. The problem seems to be that the preprocessor_config.json has an `image_processor_type` value that isn't actually registered in the processor mapping, so when AutoProcessor tries to resolve it, it fails even though the model_type itself is valid.

I need to check if `Qwen2VLImageProcessorFast` is even in the IMAGE_PROCESSOR_MAPPING or if it's in a different mapping entirely. The error suggests the processor type isn't found in any of the known mappings.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"\nfrom transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES, IMAGE_PROCESSOR_MAPPING\nprint('IMAGE_PROCESSOR_MAPPING_NAMES:', IMAGE_PROCESSOR_MAPPING_NAMES)\nprint()\nprint('IMAGE_PROCESSOR_MAPPING:', dict(IMAGE_PROCESSOR_MAPPING))\n\""
}
```

### [2026-06-26 09:45:16 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

IMAGE_PROCESSOR_MAPPING_NAMES: OrderedDict({'aria': {'pil': 'AriaImageProcessorPil', 'torchvision': 'AriaImageProcessor'}, 'beit': {'pil': 'BeitImageProcessorPil', 'torchvision': 'BeitImageProcessor'}, 'bit': {'pil': 'BitImageProcessorPil', 'torchvision': 'BitImageProcessor'}, 'blip': {'pil': 'BlipImageProcessorPil', 'torchvision': 'BlipImageProcessor'}, 'bridgetower': {'pil': 'BridgeTowerImageProcessorPil', 'torchvision': 'BridgeTowerImageProcessor'}, 'chameleon': {'pil': 'ChameleonImageProcessorPil', 'torchvision': 'ChameleonImageProcessor'}, 'chinese_clip': {'pil': 'ChineseCLIPImageProcessorPil', 'torchvision': 'ChineseCLIPImageProcessor'}, 'chmv2': {'torchvision': 'CHMv2ImageProcessor'}, 'clip': {'pil': 'CLIPImageProcessorPil', 'torchvision': 'CLIPImageProcessor'}, 'cohere2_vision': {'torchvision': 'Cohere2VisionImageProcessor'}, 'conditional_detr': {'pil': 'ConditionalDetrImageProcessorPil', 'torchvision': 'ConditionalDetrImageProcessor'}, 'convnext': {'pil': 'ConvNextImageProcessorPil', 'torchvision': 'ConvNextImageProcessor'}, 'deepseek_ocr2': {'pil': 'DeepseekOcr2ImageProcessorPil', 'torchvision': 'DeepseekOcr2ImageProcessor'}, 'deepseek_vl': {'pil': 'DeepseekVLImageProcessorPil', 'torchvision': 'DeepseekVLImageProcessor'}, 'deepseek_vl_hybrid': {'pil': 'DeepseekVLHybridImageProcessorPil', 'torchvision': 'DeepseekVLHybridImageProcessor'}, 'deformable_detr': {'pil': 'DeformableDetrImageProcessorPil', 'torchvision': 'DeformableDetrImageProcessor'}, 'deit': {'pil': 'DeiTImageProcessorPil', 'torchvision': 'DeiTImageProcessor'}, 'depth_pro': {'torchvision': 'DepthProImageProcessor'}, 'detr': {'pil': 'DetrImageProcessorPil', 'torchvision': 'DetrImageProcessor'}, 'dinov3_vit': {'torchvision': 'DINOv3ViTImageProcessor'}, 'dpt': {'pil': 'DPTImageProcessorPil', 'torchvision': 'DPTImageProcessor'}, 'efficientloftr': {'pil': 'EfficientLoFTRImageProcessorPil', 'torchvision': 'EfficientLoFTRImageProcessor'}, 'efficientnet': {'pil': 'EfficientNetImageProcessorPil', 'torchvision': 'EfficientNetImageProcessor'}, 'eomt': {'pil': 'EomtImageProcessorPil', 'torchvision': 'EomtImageProcessor'}, 'ernie4_5_vl_moe': {'pil': 'Ernie4_5_VLMoeImageProcessorPil', 'torchvision': 'Ernie4_5_VLMoeImageProcessor'}, 'flava': {'pil': 'FlavaImageProcessorPil', 'torchvision': 'FlavaImageProcessor'}, 'fuyu': {'pil': 'FuyuImageProcessorPil', 'torchvision': 'FuyuImageProcessor'}, 'gemma3': {'pil': 'Gemma3ImageProcessorPil', 'torchvision': 'Gemma3ImageProcessor'}, 'gemma4': {'pil': 'Gemma4ImageProcessorPil', 'torchvision': 'Gemma4ImageProcessor'}, 'gemma4_unified': {'torchvision': 'Gemma4UnifiedImageProcessor'}, 'glm46v': {'pil': 'Glm46VImageProcessorPil', 'torchvision': 'Glm46VImageProcessor'}, 'glm4v': {'pil': 'Glm4vImageProcessorPil', 'torchvision': 'Glm4vImageProcessor'}, 'glm_image': {'pil': 'GlmImageImageProcessorPil', 'torchvision': 'GlmImageImageProcessor'}, 'glmga': {'pil': 'GlmgaImageProcessorPil', 'torchvision': 'GlmgaImageProcessor'}, 'glpn': {'pil': 'GLPNImageProcessorPil', 'torchvision': 'GLPNImageProcessor'}, 'got_ocr2': {'pil': 'GotOcr2ImageProcessorPil', 'torchvision': 'GotOcr2ImageProcessor'}, 'grounding-dino': {'pil': 'GroundingDinoImageProcessorPil', 'torchvision': 'GroundingDinoImageProcessor'}, 'idefics': {'pil': 'IdeficsImageProcessorPil', 'torchvision': 'IdeficsImageProcessor'}, 'idefics2': {'pil': 'Idefics2ImageProcessorPil', 'torchvision': 'Idefics2ImageProcessor'}, 'idefics3': {'pil': 'Idefics3ImageProcessorPil', 'torchvision': 'Idefics3ImageProcessor'}, 'imagegpt': {'pil': 'ImageGPTImageProcessorPil', 'torchvision': 'ImageGPTImageProcessor'}, 'janus': {'pil': 'JanusImageProcessorPil', 'torchvision': 'JanusImageProcessor'}, 'layoutlmv2': {'pil': 'LayoutLMv2ImageProcessorPil', 'torchvision': 'LayoutLMv2ImageProcessor'}, 'layoutlmv3': {'pil': 'LayoutLMv3ImageProcessorPil', 'torchvision': 'LayoutLMv3ImageProcessor'}, 'levit': {'pil': 'LevitImageProcessorPil', 'torchvision': 'LevitImageProcessor'}, 'lfm2_vl': {'torchvision': 'Lfm2VlImageProcessor'}, 'lightglue': {'pil': 'LightGlueImageProcessorPil', 'torchvision': 'LightGlueImageProcessor'}, 'llama4': {'torchvision': 'Llama4ImageProcessor'}, 'llava': {'pil': 'LlavaImageProcessorPil', 'torchvision': 'LlavaImageProcessor'}, 'llava_next': {'pil': 'LlavaNextImageProcessorPil', 'torchvision': 'LlavaNextImageProcessor'}, 'llava_onevision': {'pil': 'LlavaOnevisionImageProcessorPil', 'torchvision': 'LlavaOnevisionImageProcessor'}, 'mask2former': {'pil': 'Mask2FormerImageProcessorPil', 'torchvision': 'Mask2FormerImageProcessor'}, 'maskformer': {'pil': 'MaskFormerImageProcessorPil', 'torchvision': 'MaskFormerImageProcessor'}, 'minicpmv4_6': {'pil': 'MiniCPMV4_6ImageProcessorPil', 'torchvision': 'MiniCPMV4_6ImageProcessor'}, 'minimax_m3_vl': {'torchvision': 'MiniMaxM3VLImageProcessor'}, 'mllama': {'pil': 'MllamaImageProcessorPil', 'torchvision': 'MllamaImageProcessor'}, 'mobilenet_v1': {'pil': 'MobileNetV1ImageProcessorPil', 'torchvision': 'MobileNetV1ImageProcessor'}, 'mobilenet_v2': {'pil': 'MobileNetV2ImageProcessorPil', 'torchvision': 'MobileNetV2ImageProcessor'}, 'mobilevit': {'pil': 'MobileViTImageProcessorPil', 'torchvision': 'MobileViTImageProcessor'}, 'nougat': {'pil': 'NougatImageProcessorPil', 'torchvision': 'NougatImageProcessor'}, 'oneformer': {'pil': 'OneFormerImageProcessorPil', 'torchvision': 'OneFormerImageProcessor'}, 'ovis2': {'pil': 'Ovis2ImageProcessorPil', 'torchvision': 'Ovis2ImageProcessor'}, 'owlv2': {'pil': 'Owlv2ImageProcessorPil', 'torchvision': 'Owlv2ImageProcessor'}, 'owlvit': {'pil': 'OwlViTImageProcessorPil', 'torchvision': 'OwlViTImageProcessor'}, 'paddleocr_vl': {'pil': 'PaddleOCRVLImageProcessorPil', 'torchvision': 'PaddleOCRVLImageProcessor'}, 'perceiver': {'pil': 'PerceiverImageProcessorPil', 'torchvision': 'PerceiverImageProcessor'}, 'perception_lm': {'torchvision': 'PerceptionLMImageProcessor'}, 'phi4_multimodal': {'torchvision': 'Phi4MultimodalImageProcessor'}, 'pi0': {'torchvision': 'PI0ImageProcessor'}, 'pix2struct': {'pil': 'Pix2StructImageProcessorPil', 'torchvision': 'Pix2StructImageProcessor'}, 'pixtral': {'pil': 'PixtralImageProcessorPil', 'torchvision': 'PixtralImageProcessor'}, 'poolformer': {'pil': 'PoolFormerImageProcessorPil', 'torchvision': 'PoolFormerImageProcessor'}, 'pp_chart2table': {'pil': 'PPChart2TableImageProcessorPil', 'torchvision': 'PPChart2TableImageProcessor'}, 'pp_doclayout_v2': {'torchvision': 'PPDocLayoutV2ImageProcessor'}, 'pp_doclayout_v3': {'torchvision': 'PPDocLayoutV3ImageProcessor'}, 'pp_formulanet': {'torchvision': 'PPFormulaNetImageProcessor'}, 'pp_lcnet': {'torchvision': 'PPLCNetImageProcessor'}, 'pp_ocrv5_server_det': {'torchvision': 'PPOCRV5ServerDetImageProcessor'}, 'pp_ocrv5_server_rec': {'torchvision': 'PPOCRV5ServerRecImageProcessor'}, 'pp_ocrv6_small_rec': {'torchvision': 'PPOCRV6SmallRecImageProcessor'}, 'prompt_depth_anything': {'pil': 'PromptDepthAnythingImageProcessorPil', 'torchvision': 'PromptDepthAnythingImageProcessor'}, 'pvt': {'pil': 'PvtImageProcessorPil', 'torchvision': 'PvtImageProcessor'}, 'qwen2_vl': {'pil': 'Qwen2VLImageProcessorPil', 'torchvision': 'Qwen2VLImageProcessor'}, 'rf_detr': {'torchvision': 'RfDetrImageProcessor'}, 'rt_detr': {'pil': 'RTDetrImageProcessorPil', 'torchvision': 'RTDetrImageProcessor'}, 'sam': {'pil': 'SamImageProcessorPil', 'torchvision': 'SamImageProcessor'}, 'sam2': {'torchvision': 'Sam2ImageProcessor'}, 'sam3': {'torchvision': 'Sam3ImageProcessor'}, 'sapiens2': {'torchvision': 'Sapiens2ImageProcessor'}, 'segformer': {'pil': 'SegformerImageProcessorPil', 'torchvision': 'SegformerImageProcessor'}, 'seggpt': {'pil': 'SegGptImageProcessorPil', 'torchvision': 'SegGptImageProcessor'}, 'siglip': {'pil': 'SiglipImageProcessorPil', 'torchvision': 'SiglipImageProcessor'}, 'siglip2': {'pil': 'Siglip2ImageProcessorPil', 'torchvision': 'Siglip2ImageProcessor'}, 'slanext': {'torchvision': 'SLANeXtImageProcessor'}, 'smolvlm': {'pil': 'SmolVLMImageProcessorPil', 'torchvision': 'SmolVLMImageProcessor'}, 'superglue': {'pil': 'SuperGlueImageProcessorPil', 'torchvision': 'SuperGlueImageProcessor'}, 'superpoint': {'pil': 'SuperPointImageProcessorPil', 'torchvision': 'SuperPointImageProcessor'}, 'swin2sr': {'pil': 'Swin2SRImageProcessorPil', 'torchvision': 'Swin2SRImageProcessor'}, 'textnet': {'pil': 'TextNetImageProcessorPil', 'torchvision': 'TextNetImageProcessor'}, 'tvp': {'pil': 'TvpImageProcessorPil', 'torchvision': 'TvpImageProcessor'}, 'uvdoc': {'torchvision': 'UVDocImageProcessor'}, 'video_llama_3': {'pil': 'VideoLlama3ImageProcessorPil', 'torchvision': 'VideoLlama3ImageProcessor'}, 'videomae': {'pil': 'VideoMAEImageProcessorPil', 'torchvision': 'VideoMAEImageProcessor'}, 'vilt': {'pil': 'ViltImageProcessorPil', 'torchvision': 'ViltImageProcessor'}, 'vit': {'pil': 'ViTImageProcessorPil', 'torchvision': 'ViTImageProcessor'}, 'vitmatte': {'pil': 'VitMatteImageProcessorPil', 'torchvision': 'VitMatteImageProcessor'}, 'vitpose': {'pil': 'VitPoseImageProcessorPil', 'torchvision': 'VitPoseImageProcessor'}, 'yolos': {'pil': 'YolosImageProcessorPil', 'torchvision': 'YolosImageProcessor'}, 'zoedepth': {'pil': 'ZoeDepthImageProcessorPil', 'torchvision': 'ZoeDepthImageProcessor'}, 'aimv2': {'torchvision': 'CLIPImageProcessor', 'pil': 'CLIPImageProcessorPil'}, 'aimv2_vision_model': {'torchvision': 'CLIPImageProcessor', 'pil': 'CLIPImageProcessorPil'}, 'align': {'torchvision': 'EfficientNetImageProcessor', 'pil': 'EfficientNetImageProcessorPil'}, 'altclip': {'torchvision': 'CLIPImageProcessor', 'pil': 'CLIPImageProcessorPil'}, 'aya_vision': {'torchvision': 'GotOcr2ImageProcessor', 'pil': 'GotOcr2ImageProcessorPil'}, 'blip-2': {'torchvision': 'BlipImageProcessor', 'pil': 'BlipImageProcessorPil'}, 'clipseg': {'torchvision': 'ViTImageProcessor', 'pil': 'ViTImageProcessorPil'}, 'colpali': {'torchvision': 'SiglipImageProcessor', 'pil': 'SiglipImageProcessorPil'}, 'colqwen2': {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}, 'convnextv2': {'torchvision': 'ConvNextImageProcessor', 'pil': 'ConvNextImageProcessorPil'}, 'cosmos3_omni': {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}, 'cvt': {'torchvision': 'ConvNextImageProcessor', 'pil': 'ConvNextImageProcessorPil'}, 'data2vec-vision': {'torchvision': 'BeitImageProcessor', 'pil': 'BeitImageProcessorPil'}, 'deimv2': {'torchvision': 'RTDetrImageProcessor', 'pil': 'RTDetrImageProcessorPil'}, 'depth_anything': {'torchvision': 'DPTImageProcessor', 'pil': 'DPTImageProcessorPil'}, 'dinat': {'torchvision': 'ViTImageProcessor', 'pil': 'ViTImageProcessorPil'}, 'dinov2': {'torchvision': 'BitImageProcessor', 'pil': 'BitImageProcessorPil'}, 'donut-swin': {'torchvision': 'DonutImageProcessor', 'pil': 'DonutImageProcessorPil'}, 'edgetam': {'torchvision': 'Sam2ImageProcessor'}, 'emu3': {'pil': 'Emu3ImageProcessor'}, 'eomt_dinov3': {'torchvision': 'EomtImageProcessor', 'pil': 'EomtImageProcessorPil'}, 'exaone4_5': {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}, 'florence2': {'torchvision': 'CLIPImageProcessor', 'pil': 'CLIPImageProcessorPil'}, 'focalnet': {'torchvision': 'BitImageProcessor', 'pil': 'BitImageProcessorPil'}, 'gemma3n': {'torchvision': 'SiglipImageProcessor', 'pil': 'SiglipImageProcessorPil'}, 'git': {'torchvision': 'CLIPImageProcessor', 'pil': 'CLIPImageProcessorPil'}, 'granite4_vision': {'torchvision': 'LlavaNextImageProcessor', 'pil': 'LlavaNextImageProcessorPil'}, 'groupvit': {'torchvision': 'CLIPImageProcessor', 'pil': 'CLIPImageProcessorPil'}, 'hiera': {'torchvision': 'BitImageProcessor', 'pil': 'BitImageProcessorPil'}, 'ijepa': {'torchvision': 'ViTImageProcessor', 'pil': 'ViTImageProcessorPil'}, 'instructblip': {'torchvision': 'BlipImageProcessor', 'pil': 'BlipImageProcessorPil'}, 'internvl': {'torchvision': 'GotOcr2ImageProcessor', 'pil': 'GotOcr2ImageProcessorPil'}, 'kosmos-2': {'torchvision': 'CLIPImageProcessor', 'pil': 'CLIPImageProcessorPil'}, 'kosmos-2.5': {'torchvision': 'Kosmos2_5ImageProcessor', 'pil': 'Kosmos2_5ImageProcessorPil'}, 'layoutxlm': {'torchvision': 'LayoutLMv2ImageProcessor', 'pil': 'LayoutLMv2ImageProcessorPil'}, 'lighton_ocr': {'torchvision': 'PixtralImageProcessor', 'pil': 'PixtralImageProcessorPil'}, 'llava_next_video': {'torchvision': 'LlavaNextImageProcessor', 'pil': 'LlavaNextImageProcessorPil'}, 'lw_detr': {'torchvision': 'DeformableDetrImageProcessor', 'pil': 'DeformableDetrImageProcessorPil'}, 'metaclip_2': {'torchvision': 'CLIPImageProcessor', 'pil': 'CLIPImageProcessorPil'}, 'mgp-str': {'torchvision': 'ViTImageProcessor', 'pil': 'ViTImageProcessorPil'}, 'mistral3': {'torchvision': 'PixtralImageProcessor', 'pil': 'PixtralImageProcessorPil'}, 'mlcd': {'torchvision': 'CLIPImageProcessor', 'pil': 'CLIPImageProcessorPil'}, 'mm-grounding-dino': {'torchvision': 'GroundingDinoImageProcessor', 'pil': 'GroundingDinoImageProcessorPil'}, 'mobilevitv2': {'torchvision': 'MobileViTImageProcessor', 'pil': 'MobileViTImageProcessorPil'}, 'omdet-turbo': {'torchvision': 'DetrImageProcessor', 'pil': 'DetrImageProcessorPil'}, 'paligemma': {'torchvision': 'SiglipImageProcessor', 'pil': 'SiglipImageProcessorPil'}, 'pixio': {'torchvision': 'BitImageProcessor', 'pil': 'BitImageProcessorPil'}, 'pp_ocrv5_mobile_det': {'torchvision': 'PPOCRV5ServerDetImageProcessor'}, 'pp_ocrv5_mobile_rec': {'torchvision': 'PPOCRV5ServerRecImageProcessor'}, 'pp_ocrv6_medium_det': {'torchvision': 'PPOCRV5ServerDetImageProcessor'}, 'pp_ocrv6_small_det': {'torchvision': 'PPOCRV5ServerDetImageProcessor'}, 'pp_ocrv6_tiny_rec': {'torchvision': 'PPOCRV6SmallRecImageProcessor'}, 'pvt_v2': {'torchvision': 'PvtImageProcessor', 'pil': 'PvtImageProcessorPil'}, 'qianfan_ocr': {'torchvision': 'GotOcr2ImageProcessor', 'pil': 'GotOcr2ImageProcessorPil'}, 'qwen2_5_omni': {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}, 'qwen2_5_vl': {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}, 'qwen3_5': {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}, 'qwen3_5_moe': {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}, 'qwen3_omni_moe': {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}, 'qwen3_vl': {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}, 'regnet': {'torchvision': 'ConvNextImageProcessor', 'pil': 'ConvNextImageProcessorPil'}, 'resnet': {'torchvision': 'ConvNextImageProcessor', 'pil': 'ConvNextImageProcessorPil'}, 'sam2_video': {'torchvision': 'Sam2ImageProcessor'}, 'sam3_lite_text': {'torchvision': 'Sam3ImageProcessor'}, 'sam3_tracker': {'torchvision': 'Sam3ImageProcessor'}, 'sam3_tracker_video': {'torchvision': 'Sam3ImageProcessor'}, 'sam3_video': {'torchvision': 'Sam3ImageProcessor'}, 'sam_hq': {'torchvision': 'SamImageProcessor', 'pil': 'SamImageProcessorPil'}, 'shieldgemma2': {'torchvision': 'Gemma3ImageProcessor', 'pil': 'Gemma3ImageProcessorPil'}, 'slanet': {'torchvision': 'SLANeXtImageProcessor'}, 'swiftformer': {'torchvision': 'ViTImageProcessor', 'pil': 'ViTImageProcessorPil'}, 'swin': {'torchvision': 'ViTImageProcessor', 'pil': 'ViTImageProcessorPil'}, 'swinv2': {'torchvision': 'ViTImageProcessor', 'pil': 'ViTImageProcessorPil'}, 't5gemma2': {'torchvision': 'Gemma3ImageProcessor', 'pil': 'Gemma3ImageProcessorPil'}, 't5gemma2_encoder': {'torchvision': 'Gemma3ImageProcessor', 'pil': 'Gemma3ImageProcessorPil'}, 'table-transformer': {'torchvision': 'DetrImageProcessor', 'pil': 'DetrImageProcessorPil'}, 'timesformer': {'pil': 'VideoMAEImageProcessorPil', 'torchvision': 'VideoMAEImageProcessor'}, 'timm_wrapper': {'pil': 'TimmWrapperImageProcessor'}, 'trocr': {'torchvision': 'ViTImageProcessor', 'pil': 'ViTImageProcessorPil'}, 'udop': {'torchvision': 'LayoutLMv3ImageProcessor', 'pil': 'LayoutLMv3ImageProcessorPil'}, 'upernet': {'torchvision': 'SegformerImageProcessor', 'pil': 'SegformerImageProcessorPil'}, 'video_llava': {'pil': 'VideoLlavaImageProcessor'}, 'vipllava': {'torchvision': 'CLIPImageProcessor', 'pil': 'CLIPImageProcessorPil'}, 'vit_mae': {'torchvision': 'ViTImageProcessor', 'pil': 'ViTImageProcessorPil'}, 'vit_msn': {'torchvision': 'ViTImageProcessor', 'pil': 'ViTImageProcessorPil'}, 'vivit': {'torchvision': 'VivitImageProcessor'}, 'xclip': {'torchvision': 'CLIPImageProcessor', 'pil': 'CLIPImageProcessorPil'}})

IMAGE_PROCESSOR_MAPPING: {<class 'transformers.models.aimv2.configuration_aimv2.Aimv2Config'>: {'torchvision': <class 'transformers.models.clip.image_processing_clip.CLIPImageProcessor'>, 'pil': <class 'transformers.models.clip.image_processing_pil_clip.CLIPImageProcessorPil'>}, <class 'transformers.models.aimv2.configuration_aimv2.Aimv2VisionConfig'>: {'torchvision': <class 'transformers.models.clip.image_processing_clip.CLIPImageProcessor'>, 'pil': <class 'transformers.models.clip.image_processing_pil_clip.CLIPImageProcessorPil'>}, <class 'transformers.models.align.configuration_align.AlignConfig'>: {'torchvision': <class 'transformers.models.efficientnet.image_processing_efficientnet.EfficientNetImageProcessor'>, 'pil': <class 'transformers.models.efficientnet.image_processing_pil_efficientnet.EfficientNetImageProcessorPil'>}, <class 'transformers.models.altclip.configuration_altclip.AltCLIPConfig'>: {'torchvision': <class 'transformers.models.clip.image_processing_clip.CLIPImageProcessor'>, 'pil': <class 'transformers.models.clip.image_processing_pil_clip.CLIPImageProcessorPil'>}, <class 'transformers.models.aria.configuration_aria.AriaConfig'>: {'pil': <class 'transformers.models.aria.image_processing_pil_aria.AriaImageProcessorPil'>, 'torchvision': <class 'transformers.models.aria.image_processing_aria.AriaImageProcessor'>}, <class 'transformers.models.aya_vision.configuration_aya_vision.AyaVisionConfig'>: {'torchvision': <class 'transformers.models.got_ocr2.image_processing_got_ocr2.GotOcr2ImageProcessor'>, 'pil': <class 'transformers.models.got_ocr2.image_processing_pil_got_ocr2.GotOcr2ImageProcessorPil'>}, <class 'transformers.models.beit.configuration_beit.BeitConfig'>: {'pil': <class 'transformers.models.beit.image_processing_pil_beit.BeitImageProcessorPil'>, 'torchvision': <class 'transformers.models.beit.image_processing_beit.BeitImageProcessor'>}, <class 'transformers.models.bit.configuration_bit.BitConfig'>: {'pil': <class 'transformers.models.bit.image_processing_pil_bit.BitImageProcessorPil'>, 'torchvision': <class 'transformers.models.bit.image_processing_bit.BitImageProcessor'>}, <class 'transformers.models.blip.configuration_blip.BlipConfig'>: {'pil': <class 'transformers.models.blip.image_processing_pil_blip.BlipImageProcessorPil'>, 'torchvision': <class 'transformers.models.blip.image_processing_blip.BlipImageProcessor'>}, <class 'transformers.models.blip_2.configuration_blip_2.Blip2Config'>: {'torchvision': <class 'transformers.models.blip.image_processing_blip.BlipImageProcessor'>, 'pil': <class 'transformers.models.blip.image_processing_pil_blip.BlipImageProcessorPil'>}, <class 'transformers.models.bridgetower.configuration_bridgetower.BridgeTowerConfig'>: {'pil': <class 'transformers.models.bridgetower.image_processing_pil_bridgetower.BridgeTowerImageProcessorPil'>, 'torchvision': <class 'transformers.models.bridgetower.image_processing_bridgetower.BridgeTowerImageProcessor'>}, <class 'transformers.models.chameleon.configuration_chameleon.ChameleonConfig'>: {'pil': <class 'transformers.models.chameleon.image_processing_pil_chameleon.ChameleonImageProcessorPil'>, 'torchvision': <class 'transformers.models.chameleon.image_processing_chameleon.ChameleonImageProcessor'>}, <class 'transformers.models.chinese_clip.configuration_chinese_clip.ChineseCLIPConfig'>: {'pil': <class 'transformers.models.chinese_clip.image_processing_pil_chinese_clip.ChineseCLIPImageProcessorPil'>, 'torchvision': <class 'transformers.models.chinese_clip.image_processing_chinese_clip.ChineseCLIPImageProcessor'>}, <class 'transformers.models.chmv2.configuration_chmv2.CHMv2Config'>: {'torchvision': <class 'transformers.models.chmv2.image_processing_chmv2.CHMv2ImageProcessor'>}, <class 'transformers.models.clip.configuration_clip.CLIPConfig'>: {'pil': <class 'transformers.models.clip.image_processing_pil_clip.CLIPImageProcessorPil'>, 'torchvision': <class 'transformers.models.clip.image_processing_clip.CLIPImageProcessor'>}, <class 'transformers.models.clipseg.configuration_clipseg.CLIPSegConfig'>: {'torchvision': <class 'transformers.models.vit.image_processing_vit.ViTImageProcessor'>, 'pil': <class 'transformers.models.vit.image_processing_pil_vit.ViTImageProcessorPil'>}, <class 'transformers.models.cohere2_vision.configuration_cohere2_vision.Cohere2VisionConfig'>: {'torchvision': <class 'transformers.models.cohere2_vision.image_processing_cohere2_vision.Cohere2VisionImageProcessor'>}, <class 'transformers.models.colpali.configuration_colpali.ColPaliConfig'>: {'torchvision': <class 'transformers.models.siglip.image_processing_siglip.SiglipImageProcessor'>, 'pil': <class 'transformers.models.siglip.image_processing_pil_siglip.SiglipImageProcessorPil'>}, <class 'transformers.models.colqwen2.configuration_colqwen2.ColQwen2Config'>: {'torchvision': <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>, 'pil': <class 'transformers.models.qwen2_vl.image_processing_pil_qwen2_vl.Qwen2VLImageProcessorPil'>}, <class 'transformers.models.conditional_detr.configuration_conditional_detr.ConditionalDetrConfig'>: {'pil': <class 'transformers.models.conditional_detr.image_processing_pil_conditional_detr.ConditionalDetrImageProcessorPil'>, 'torchvision': <class 'transformers.models.conditional_detr.image_processing_conditional_detr.ConditionalDetrImageProcessor'>}, <class 'transformers.models.convnext.configuration_convnext.ConvNextConfig'>: {'pil': <class 'transformers.models.convnext.image_processing_pil_convnext.ConvNextImageProcessorPil'>, 'torchvision': <class 'transformers.models.convnext.image_processing_convnext.ConvNextImageProcessor'>}, <class 'transformers.models.convnextv2.configuration_convnextv2.ConvNextV2Config'>: {'torchvision': <class 'transformers.models.convnext.image_processing_convnext.ConvNextImageProcessor'>, 'pil': <class 'transformers.models.convnext.image_processing_pil_convnext.ConvNextImageProcessorPil'>}, <class 'transformers.models.cosmos3_omni.configuration_cosmos3_omni.Cosmos3OmniConfig'>: {'torchvision': <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>, 'pil': <class 'transformers.models.qwen2_vl.image_processing_pil_qwen2_vl.Qwen2VLImageProcessorPil'>}, <class 'transformers.models.cvt.configuration_cvt.CvtConfig'>: {'torchvision': <class 'transformers.models.convnext.image_processing_convnext.ConvNextImageProcessor'>, 'pil': <class 'transformers.models.convnext.image_processing_pil_convnext.ConvNextImageProcessorPil'>}, <class 'transformers.models.data2vec.configuration_data2vec_vision.Data2VecVisionConfig'>: {'torchvision': <class 'transformers.models.beit.image_processing_beit.BeitImageProcessor'>, 'pil': <class 'transformers.models.beit.image_processing_pil_beit.BeitImageProcessorPil'>}, <class 'transformers.models.deepseek_ocr2.configuration_deepseek_ocr2.DeepseekOcr2Config'>: {'pil': <class 'transformers.models.deepseek_ocr2.image_processing_pil_deepseek_ocr2.DeepseekOcr2ImageProcessorPil'>, 'torchvision': <class 'transformers.models.deepseek_ocr2.image_processing_deepseek_ocr2.DeepseekOcr2ImageProcessor'>}, <class 'transformers.models.deepseek_vl.configuration_deepseek_vl.DeepseekVLConfig'>: {'pil': <class 'transformers.models.deepseek_vl.image_processing_pil_deepseek_vl.DeepseekVLImageProcessorPil'>, 'torchvision': <class 'transformers.models.deepseek_vl.image_processing_deepseek_vl.DeepseekVLImageProcessor'>}, <class 'transformers.models.deepseek_vl_hybrid.configuration_deepseek_vl_hybrid.DeepseekVLHybridConfig'>: {'pil': <class 'transformers.models.deepseek_vl_hybrid.image_processing_pil_deepseek_vl_hybrid.DeepseekVLHybridImageProcessorPil'>, 'torchvision': <class 'transformers.models.deepseek_vl_hybrid.image_processing_deepseek_vl_hybrid.DeepseekVLHybridImageProcessor'>}, <class 'transformers.models.deformable_detr.configuration_deformable_detr.DeformableDetrConfig'>: {'pil': <class 'transformers.models.deformable_detr.image_processing_pil_deformable_detr.DeformableDetrImageProcessorPil'>, 'torchvision': <class 'transformers.models.deformable_detr.image_processing_deformable_detr.DeformableDetrImageProcessor'>}, <class 'transformers.models.deimv2.configuration_deimv2.Deimv2Config'>: {'torchvision': <class 'transformers.models.rt_detr.image_processing_rt_detr.RTDetrImageProcessor'>, 'pil': <class 'transformers.models.rt_detr.image_processing_pil_rt_detr.RTDetrImageProcessorPil'>}, <class 'transformers.models.deit.configuration_deit.DeiTConfig'>: {'pil': <class 'transformers.models.deit.image_processing_pil_deit.DeiTImageProcessorPil'>, 'torchvision': <class 'transformers.models.deit.image_processing_deit.DeiTImageProcessor'>}, <class 'transformers.models.depth_anything.configuration_depth_anything.DepthAnythingConfig'>: {'torchvision': <class 'transformers.models.dpt.image_processing_dpt.DPTImageProcessor'>, 'pil': <class 'transformers.models.dpt.image_processing_pil_dpt.DPTImageProcessorPil'>}, <class 'transformers.models.depth_pro.configuration_depth_pro.DepthProConfig'>: {'torchvision': <class 'transformers.models.depth_pro.image_processing_depth_pro.DepthProImageProcessor'>}, <class 'transformers.models.detr.configuration_detr.DetrConfig'>: {'pil': <class 'transformers.models.detr.image_processing_pil_detr.DetrImageProcessorPil'>, 'torchvision': <class 'transformers.models.detr.image_processing_detr.DetrImageProcessor'>}, <class 'transformers.models.dinat.configuration_dinat.DinatConfig'>: {'torchvision': <class 'transformers.models.vit.image_processing_vit.ViTImageProcessor'>, 'pil': <class 'transformers.models.vit.image_processing_pil_vit.ViTImageProcessorPil'>}, <class 'transformers.models.dinov2.configuration_dinov2.Dinov2Config'>: {'torchvision': <class 'transformers.models.bit.image_processing_bit.BitImageProcessor'>, 'pil': <class 'transformers.models.bit.image_processing_pil_bit.BitImageProcessorPil'>}, <class 'transformers.models.dinov3_vit.configuration_dinov3_vit.DINOv3ViTConfig'>: {'torchvision': <class 'transformers.models.dinov3_vit.image_processing_dinov3_vit.DINOv3ViTImageProcessor'>}, <class 'transformers.models.donut.configuration_donut_swin.DonutSwinConfig'>: {'torchvision': <class 'transformers.models.donut.image_processing_donut.DonutImageProcessor'>, 'pil': <class 'transformers.models.donut.image_processing_pil_donut.DonutImageProcessorPil'>}, <class 'transformers.models.dpt.configuration_dpt.DPTConfig'>: {'pil': <class 'transformers.models.dpt.image_processing_pil_dpt.DPTImageProcessorPil'>, 'torchvision': <class 'transformers.models.dpt.image_processing_dpt.DPTImageProcessor'>}, <class 'transformers.models.edgetam.configuration_edgetam.EdgeTamConfig'>: {'torchvision': <class 'transformers.models.sam2.image_processing_sam2.Sam2ImageProcessor'>}, <class 'transformers.models.efficientloftr.configuration_efficientloftr.EfficientLoFTRConfig'>: {'pil': <class 'transformers.models.efficientloftr.image_processing_pil_efficientloftr.EfficientLoFTRImageProcessorPil'>, 'torchvision': <class 'transformers.models.efficientloftr.image_processing_efficientloftr.EfficientLoFTRImageProcessor'>}, <class 'transformers.models.efficientnet.configuration_efficientnet.EfficientNetConfig'>: {'pil': <class 'transformers.models.efficientnet.image_processing_pil_efficientnet.EfficientNetImageProcessorPil'>, 'torchvision': <class 'transformers.models.efficientnet.image_processing_efficientnet.EfficientNetImageProcessor'>}, <class 'transformers.models.emu3.configuration_emu3.Emu3Config'>: {'pil': <class 'transformers.models.emu3.image_processing_emu3.Emu3ImageProcessor'>}, <class 'transformers.models.eomt.configuration_eomt.EomtConfig'>: {'pil': <class 'transformers.models.eomt.image_processing_pil_eomt.EomtImageProcessorPil'>, 'torchvision': <class 'transformers.models.eomt.image_processing_eomt.EomtImageProcessor'>}, <class 'transformers.models.eomt_dinov3.configuration_eomt_dinov3.EomtDinov3Config'>: {'torchvision': <class 'transformers.models.eomt.image_processing_eomt.EomtImageProcessor'>, 'pil': <class 'transformers.models.eomt.image_processing_pil_eomt.EomtImageProcessorPil'>}, <class 'transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe.Ernie4_5_VLMoeConfig'>: {'pil': <class 'transformers.models.ernie4_5_vl_moe.image_processing_pil_ernie4_5_vl_moe.Ernie4_5_VLMoeImageProcessorPil'>, 'torchvision': <class 'transformers.models.ernie4_5_vl_moe.image_processing_ernie4_5_vl_moe.Ernie4_5_VLMoeImageProcessor'>}, <class 'transformers.models.exaone4_5.configuration_exaone4_5.Exaone4_5_Config'>: {'torchvision': <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>, 'pil': <class 'transformers.models.qwen2_vl.image_processing_pil_qwen2_vl.Qwen2VLImageProcessorPil'>}, <class 'transformers.models.flava.configuration_flava.FlavaConfig'>: {'pil': <class 'transformers.models.flava.image_processing_pil_flava.FlavaImageProcessorPil'>, 'torchvision': <class 'transformers.models.flava.image_processing_flava.FlavaImageProcessor'>}, <class 'transformers.models.florence2.configuration_florence2.Florence2Config'>: {'torchvision': <class 'transformers.models.clip.image_processing_clip.CLIPImageProcessor'>, 'pil': <class 'transformers.models.clip.image_processing_pil_clip.CLIPImageProcessorPil'>}, <class 'transformers.models.focalnet.configuration_focalnet.FocalNetConfig'>: {'torchvision': <class 'transformers.models.bit.image_processing_bit.BitImageProcessor'>, 'pil': <class 'transformers.models.bit.image_processing_pil_bit.BitImageProcessorPil'>}, <class 'transformers.models.fuyu.configuration_fuyu.FuyuConfig'>: {'pil': <class 'transformers.models.fuyu.image_processing_pil_fuyu.FuyuImageProcessorPil'>, 'torchvision': <class 'transformers.models.fuyu.image_processing_fuyu.FuyuImageProcessor'>}, <class 'transformers.models.gemma3.configuration_gemma3.Gemma3Config'>: {'pil': <class 'transformers.models.gemma3.image_processing_pil_gemma3.Gemma3ImageProcessorPil'>, 'torchvision': <class 'transformers.models.gemma3.image_processing_gemma3.Gemma3ImageProcessor'>}, <class 'transformers.models.gemma3n.configuration_gemma3n.Gemma3nConfig'>: {'torchvision': <class 'transformers.models.siglip.image_processing_siglip.SiglipImageProcessor'>, 'pil': <class 'transformers.models.siglip.image_processing_pil_siglip.SiglipImageProcessorPil'>}, <class 'transformers.models.gemma4.configuration_gemma4.Gemma4Config'>: {'pil': <class 'transformers.models.gemma4.image_processing_pil_gemma4.Gemma4ImageProcessorPil'>, 'torchvision': <class 'transformers.models.gemma4.image_processing_gemma4.Gemma4ImageProcessor'>}, <class 'transformers.models.gemma4_unified.configuration_gemma4_unified.Gemma4UnifiedConfig'>: {'torchvision': <class 'transformers.models.gemma4_unified.image_processing_gemma4_unified.Gemma4UnifiedImageProcessor'>}, <class 'transformers.models.git.configuration_git.GitConfig'>: {'torchvision': <class 'transformers.models.clip.image_processing_clip.CLIPImageProcessor'>, 'pil': <class 'transformers.models.clip.image_processing_pil_clip.CLIPImageProcessorPil'>}, <class 'transformers.models.glm46v.configuration_glm46v.Glm46VConfig'>: {'pil': <class 'transformers.models.glm46v.image_processing_pil_glm46v.Glm46VImageProcessorPil'>, 'torchvision': <class 'transformers.models.glm46v.image_processing_glm46v.Glm46VImageProcessor'>}, <class 'transformers.models.glm4v.configuration_glm4v.Glm4vConfig'>: {'pil': <class 'transformers.models.glm4v.image_processing_pil_glm4v.Glm4vImageProcessorPil'>, 'torchvision': <class 'transformers.models.glm4v.image_processing_glm4v.Glm4vImageProcessor'>}, <class 'transformers.models.glm_image.configuration_glm_image.GlmImageConfig'>: {'pil': <class 'transformers.models.glm_image.image_processing_pil_glm_image.GlmImageImageProcessorPil'>, 'torchvision': <class 'transformers.models.glm_image.image_processing_glm_image.GlmImageImageProcessor'>}, <class 'transformers.models.glmga.configuration_glmga.GlmgaConfig'>: {'pil': <class 'transformers.models.glmga.image_processing_pil_glmga.GlmgaImageProcessorPil'>, 'torchvision': <class 'transformers.models.glmga.image_processing_glmga.GlmgaImageProcessor'>}, <class 'transformers.models.glpn.configuration_glpn.GLPNConfig'>: {'pil': <class 'transformers.models.glpn.image_processing_pil_glpn.GLPNImageProcessorPil'>, 'torchvision': <class 'transformers.models.glpn.image_processing_glpn.GLPNImageProcessor'>}, <class 'transformers.models.got_ocr2.configuration_got_ocr2.GotOcr2Config'>: {'pil': <class 'transformers.models.got_ocr2.image_processing_pil_got_ocr2.GotOcr2ImageProcessorPil'>, 'torchvision': <class 'transformers.models.got_ocr2.image_processing_got_ocr2.GotOcr2ImageProcessor'>}, <class 'transformers.models.granite4_vision.configuration_granite4_vision.Granite4VisionConfig'>: {'torchvision': <class 'transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor'>, 'pil': <class 'transformers.models.llava_next.image_processing_pil_llava_next.LlavaNextImageProcessorPil'>}, <class 'transformers.models.grounding_dino.configuration_grounding_dino.GroundingDinoConfig'>: {'pil': <class 'transformers.models.grounding_dino.image_processing_pil_grounding_dino.GroundingDinoImageProcessorPil'>, 'torchvision': <class 'transformers.models.grounding_dino.image_processing_grounding_dino.GroundingDinoImageProcessor'>}, <class 'transformers.models.groupvit.configuration_groupvit.GroupViTConfig'>: {'torchvision': <class 'transformers.models.clip.image_processing_clip.CLIPImageProcessor'>, 'pil': <class 'transformers.models.clip.image_processing_pil_clip.CLIPImageProcessorPil'>}, <class 'transformers.models.hiera.configuration_hiera.HieraConfig'>: {'torchvision': <class 'transformers.models.bit.image_processing_bit.BitImageProcessor'>, 'pil': <class 'transformers.models.bit.image_processing_pil_bit.BitImageProcessorPil'>}, <class 'transformers.models.idefics.configuration_idefics.IdeficsConfig'>: {'pil': <class 'transformers.models.idefics.image_processing_pil_idefics.IdeficsImageProcessorPil'>, 'torchvision': <class 'transformers.models.idefics.image_processing_idefics.IdeficsImageProcessor'>}, <class 'transformers.models.idefics2.configuration_idefics2.Idefics2Config'>: {'pil': <class 'transformers.models.idefics2.image_processing_pil_idefics2.Idefics2ImageProcessorPil'>, 'torchvision': <class 'transformers.models.idefics2.image_processing_idefics2.Idefics2ImageProcessor'>}, <class 'transformers.models.idefics3.configuration_idefics3.Idefics3Config'>: {'pil': <class 'transformers.models.idefics3.image_processing_pil_idefics3.Idefics3ImageProcessorPil'>, 'torchvision': <class 'transformers.models.idefics3.image_processing_idefics3.Idefics3ImageProcessor'>}, <class 'transformers.models.ijepa.configuration_ijepa.IJepaConfig'>: {'torchvision': <class 'transformers.models.vit.image_processing_vit.ViTImageProcessor'>, 'pil': <class 'transformers.models.vit.image_processing_pil_vit.ViTImageProcessorPil'>}, <class 'transformers.models.imagegpt.configuration_imagegpt.ImageGPTConfig'>: {'pil': <class 'transformers.models.imagegpt.image_processing_pil_imagegpt.ImageGPTImageProcessorPil'>, 'torchvision': <class 'transformers.models.imagegpt.image_processing_imagegpt.ImageGPTImageProcessor'>}, <class 'transformers.models.instructblip.configuration_instructblip.InstructBlipConfig'>: {'torchvision': <class 'transformers.models.blip.image_processing_blip.BlipImageProcessor'>, 'pil': <class 'transformers.models.blip.image_processing_pil_blip.BlipImageProcessorPil'>}, <class 'transformers.models.internvl.configuration_internvl.InternVLConfig'>: {'torchvision': <class 'transformers.models.got_ocr2.image_processing_got_ocr2.GotOcr2ImageProcessor'>, 'pil': <class 'transformers.models.got_ocr2.image_processing_pil_got_ocr2.GotOcr2ImageProcessorPil'>}, <class 'transformers.models.janus.configuration_janus.JanusConfig'>: {'pil': <class 'transformers.models.janus.image_processing_pil_janus.JanusImageProcessorPil'>, 'torchvision': <class 'transformers.models.janus.image_processing_janus.JanusImageProcessor'>}, <class 'transformers.models.kosmos2.configuration_kosmos2.Kosmos2Config'>: {'torchvision': <class 'transformers.models.clip.image_processing_clip.CLIPImageProcessor'>, 'pil': <class 'transformers.models.clip.image_processing_pil_clip.CLIPImageProcessorPil'>}, <class 'transformers.models.kosmos2_5.configuration_kosmos2_5.Kosmos2_5Config'>: {'torchvision': <class 'transformers.models.kosmos2_5.image_processing_kosmos2_5.Kosmos2_5ImageProcessor'>, 'pil': <class 'transformers.models.kosmos2_5.image_processing_pil_kosmos2_5.Kosmos2_5ImageProcessorPil'>}, <class 'transformers.models.layoutlmv2.configuration_layoutlmv2.LayoutLMv2Config'>: {'pil': <class 'transformers.models.layoutlmv2.image_processing_pil_layoutlmv2.LayoutLMv2ImageProcessorPil'>, 'torchvision': <class 'transformers.models.layoutlmv2.image_processing_layoutlmv2.LayoutLMv2ImageProcessor'>}, <class 'transformers.models.layoutlmv3.configuration_layoutlmv3.LayoutLMv3Config'>: {'pil': <class 'transformers.models.layoutlmv3.image_processing_pil_layoutlmv3.LayoutLMv3ImageProcessorPil'>, 'torchvision': <class 'transformers.models.layoutlmv3.image_processing_layoutlmv3.LayoutLMv3ImageProcessor'>}, <class 'transformers.models.layoutxlm.configuration_layoutxlm.LayoutXLMConfig'>: {'torchvision': <class 'transformers.models.layoutlmv2.image_processing_layoutlmv2.LayoutLMv2ImageProcessor'>, 'pil': <class 'transformers.models.layoutlmv2.image_processing_pil_layoutlmv2.LayoutLMv2ImageProcessorPil'>}, <class 'transformers.models.levit.configuration_levit.LevitConfig'>: {'pil': <class 'transformers.models.levit.image_processing_pil_levit.LevitImageProcessorPil'>, 'torchvision': <class 'transformers.models.levit.image_processing_levit.LevitImageProcessor'>}, <class 'transformers.models.lfm2_vl.configuration_lfm2_vl.Lfm2VlConfig'>: {'torchvision': <class 'transformers.models.lfm2_vl.image_processing_lfm2_vl.Lfm2VlImageProcessor'>}, <class 'transformers.models.lightglue.configuration_lightglue.LightGlueConfig'>: {'pil': <class 'transformers.models.lightglue.image_processing_pil_lightglue.LightGlueImageProcessorPil'>, 'torchvision': <class 'transformers.models.lightglue.image_processing_lightglue.LightGlueImageProcessor'>}, <class 'transformers.models.lighton_ocr.configuration_lighton_ocr.LightOnOcrConfig'>: {'torchvision': <class 'transformers.models.pixtral.image_processing_pixtral.PixtralImageProcessor'>, 'pil': <class 'transformers.models.pixtral.image_processing_pil_pixtral.PixtralImageProcessorPil'>}, <class 'transformers.models.llama4.configuration_llama4.Llama4Config'>: {'torchvision': <class 'transformers.models.llama4.image_processing_llama4.Llama4ImageProcessor'>}, <class 'transformers.models.llava.configuration_llava.LlavaConfig'>: {'pil': <class 'transformers.models.llava.image_processing_pil_llava.LlavaImageProcessorPil'>, 'torchvision': <class 'transformers.models.llava.image_processing_llava.LlavaImageProcessor'>}, <class 'transformers.models.llava_next.configuration_llava_next.LlavaNextConfig'>: {'pil': <class 'transformers.models.llava_next.image_processing_pil_llava_next.LlavaNextImageProcessorPil'>, 'torchvision': <class 'transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor'>}, <class 'transformers.models.llava_next_video.configuration_llava_next_video.LlavaNextVideoConfig'>: {'torchvision': <class 'transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor'>, 'pil': <class 'transformers.models.llava_next.image_processing_pil_llava_next.LlavaNextImageProcessorPil'>}, <class 'transformers.models.llava_onevision.configuration_llava_onevision.LlavaOnevisionConfig'>: {'pil': <class 'transformers.models.llava_onevision.image_processing_pil_llava_onevision.LlavaOnevisionImageProcessorPil'>, 'torchvision': <class 'transformers.models.llava_onevision.image_processing_llava_onevision.LlavaOnevisionImageProcessor'>}, <class 'transformers.models.lw_detr.configuration_lw_detr.LwDetrConfig'>: {'torchvision': <class 'transformers.models.deformable_detr.image_processing_deformable_detr.DeformableDetrImageProcessor'>, 'pil': <class 'transformers.models.deformable_detr.image_processing_pil_deformable_detr.DeformableDetrImageProcessorPil'>}, <class 'transformers.models.mask2former.configuration_mask2former.Mask2FormerConfig'>: {'pil': <class 'transformers.models.mask2former.image_processing_pil_mask2former.Mask2FormerImageProcessorPil'>, 'torchvision': <class 'transformers.models.mask2former.image_processing_mask2former.Mask2FormerImageProcessor'>}, <class 'transformers.models.maskformer.configuration_maskformer.MaskFormerConfig'>: {'pil': <class 'transformers.models.maskformer.image_processing_pil_maskformer.MaskFormerImageProcessorPil'>, 'torchvision': <class 'transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor'>}, <class 'transformers.models.metaclip_2.configuration_metaclip_2.MetaClip2Config'>: {'torchvision': <class 'transformers.models.clip.image_processing_clip.CLIPImageProcessor'>, 'pil': <class 'transformers.models.clip.image_processing_pil_clip.CLIPImageProcessorPil'>}, <class 'transformers.models.mgp_str.configuration_mgp_str.MgpstrConfig'>: {'torchvision': <class 'transformers.models.vit.image_processing_vit.ViTImageProcessor'>, 'pil': <class 'transformers.models.vit.image_processing_pil_vit.ViTImageProcessorPil'>}, <class 'transformers.models.minicpmv4_6.configuration_minicpmv4_6.MiniCPMV4_6Config'>: {'pil': <class 'transformers.models.minicpmv4_6.image_processing_pil_minicpmv4_6.MiniCPMV4_6ImageProcessorPil'>, 'torchvision': <class 'transformers.models.minicpmv4_6.image_processing_minicpmv4_6.MiniCPMV4_6ImageProcessor'>}, <class 'transformers.models.minimax_m3_vl.configuration_minimax_m3_vl.MiniMaxM3VLConfig'>: {'torchvision': <class 'transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl.MiniMaxM3VLImageProcessor'>}, <class 'transformers.models.mistral3.configuration_mistral3.Mistral3Config'>: {'torchvision': <class 'transformers.models.pixtral.image_processing_pixtral.PixtralImageProcessor'>, 'pil': <class 'transformers.models.pixtral.image_processing_pil_pixtral.PixtralImageProcessorPil'>}, <class 'transformers.models.mllama.configuration_mllama.MllamaConfig'>: {'pil': <class 'transformers.models.mllama.image_processing_pil_mllama.MllamaImageProcessorPil'>, 'torchvision': <class 'transformers.models.mllama.image_processing_mllama.MllamaImageProcessor'>}, <class 'transformers.models.mm_grounding_dino.configuration_mm_grounding_dino.MMGroundingDinoConfig'>: {'torchvision': <class 'transformers.models.grounding_dino.image_processing_grounding_dino.GroundingDinoImageProcessor'>, 'pil': <class 'transformers.models.grounding_dino.image_processing_pil_grounding_dino.GroundingDinoImageProcessorPil'>}, <class 'transformers.models.mobilenet_v1.configuration_mobilenet_v1.MobileNetV1Config'>: {'pil': <class 'transformers.models.mobilenet_v1.image_processing_pil_mobilenet_v1.MobileNetV1ImageProcessorPil'>, 'torchvision': <class 'transformers.models.mobilenet_v1.image_processing_mobilenet_v1.MobileNetV1ImageProcessor'>}, <class 'transformers.models.mobilenet_v2.configuration_mobilenet_v2.MobileNetV2Config'>: {'pil': <class 'transformers.models.mobilenet_v2.image_processing_pil_mobilenet_v2.MobileNetV2ImageProcessorPil'>, 'torchvision': <class 'transformers.models.mobilenet_v2.image_processing_mobilenet_v2.MobileNetV2ImageProcessor'>}, <class 'transformers.models.mobilevit.configuration_mobilevit.MobileViTConfig'>: {'pil': <class 'transformers.models.mobilevit.image_processing_pil_mobilevit.MobileViTImageProcessorPil'>, 'torchvision': <class 'transformers.models.mobilevit.image_processing_mobilevit.MobileViTImageProcessor'>}, <class 'transformers.models.mobilevitv2.configuration_mobilevitv2.MobileViTV2Config'>: {'torchvision': <class 'transformers.models.mobilevit.image_processing_mobilevit.MobileViTImageProcessor'>, 'pil': <class 'transformers.models.mobilevit.image_processing_pil_mobilevit.MobileViTImageProcessorPil'>}, <class 'transformers.models.nougat.configuration_nougat.NougatConfig'>: {'pil': <class 'transformers.models.nougat.image_processing_pil_nougat.NougatImageProcessorPil'>, 'torchvision': <class 'transformers.models.nougat.image_processing_nougat.NougatImageProcessor'>}, <class 'transformers.models.omdet_turbo.configuration_omdet_turbo.OmDetTurboConfig'>: {'torchvision': <class 'transformers.models.detr.image_processing_detr.DetrImageProcessor'>, 'pil': <class 'transformers.models.detr.image_processing_pil_detr.DetrImageProcessorPil'>}, <class 'transformers.models.oneformer.configuration_oneformer.OneFormerConfig'>: {'pil': <class 'transformers.models.oneformer.image_processing_pil_oneformer.OneFormerImageProcessorPil'>, 'torchvision': <class 'transformers.models.oneformer.image_processing_oneformer.OneFormerImageProcessor'>}, <class 'transformers.models.ovis2.configuration_ovis2.Ovis2Config'>: {'pil': <class 'transformers.models.ovis2.image_processing_pil_ovis2.Ovis2ImageProcessorPil'>, 'torchvision': <class 'transformers.models.ovis2.image_processing_ovis2.Ovis2ImageProcessor'>}, <class 'transformers.models.owlv2.configuration_owlv2.Owlv2Config'>: {'pil': <class 'transformers.models.owlv2.image_processing_pil_owlv2.Owlv2ImageProcessorPil'>, 'torchvision': <class 'transformers.models.owlv2.image_processing_owlv2.Owlv2ImageProcessor'>}, <class 'transformers.models.owlvit.configuration_owlvit.OwlViTConfig'>: {'pil': <class 'transformers.models.owlvit.image_processing_pil_owlvit.OwlViTImageProcessorPil'>, 'torchvision': <class 'transformers.models.owlvit.image_processing_owlvit.OwlViTImageProcessor'>}, <class 'transformers.models.paddleocr_vl.configuration_paddleocr_vl.PaddleOCRVLConfig'>: {'pil': <class 'transformers.models.paddleocr_vl.image_processing_pil_paddleocr_vl.PaddleOCRVLImageProcessorPil'>, 'torchvision': <class 'transformers.models.paddleocr_vl.image_processing_paddleocr_vl.PaddleOCRVLImageProcessor'>}, <class 'transformers.models.paligemma.configuration_paligemma.PaliGemmaConfig'>: {'torchvision': <class 'transformers.models.siglip.image_processing_siglip.SiglipImageProcessor'>, 'pil': <class 'transformers.models.siglip.image_processing_pil_siglip.SiglipImageProcessorPil'>}, <class 'transformers.models.perceiver.configuration_perceiver.PerceiverConfig'>: {'pil': <class 'transformers.models.perceiver.image_processing_pil_perceiver.PerceiverImageProcessorPil'>, 'torchvision': <class 'transformers.models.perceiver.image_processing_perceiver.PerceiverImageProcessor'>}, <class 'transformers.models.perception_lm.configuration_perception_lm.PerceptionLMConfig'>: {'torchvision': <class 'transformers.models.perception_lm.image_processing_perception_lm.PerceptionLMImageProcessor'>}, <class 'transformers.models.phi4_multimodal.configuration_phi4_multimodal.Phi4MultimodalConfig'>: {'torchvision': <class 'transformers.models.phi4_multimodal.image_processing_phi4_multimodal.Phi4MultimodalImageProcessor'>}, <class 'transformers.models.pi0.configuration_pi0.PI0Config'>: {'torchvision': <class 'transformers.models.pi0.image_processing_pi0.PI0ImageProcessor'>}, <class 'transformers.models.pix2struct.configuration_pix2struct.Pix2StructConfig'>: {'pil': <class 'transformers.models.pix2struct.image_processing_pil_pix2struct.Pix2StructImageProcessorPil'>, 'torchvision': <class 'transformers.models.pix2struct.image_processing_pix2struct.Pix2StructImageProcessor'>}, <class 'transformers.models.pixio.configuration_pixio.PixioConfig'>: {'torchvision': <class 'transformers.models.bit.image_processing_bit.BitImageProcessor'>, 'pil': <class 'transformers.models.bit.image_processing_pil_bit.BitImageProcessorPil'>}, <class 'transformers.models.pixtral.configuration_pixtral.PixtralVisionConfig'>: {'pil': <class 'transformers.models.pixtral.image_processing_pil_pixtral.PixtralImageProcessorPil'>, 'torchvision': <class 'transformers.models.pixtral.image_processing_pixtral.PixtralImageProcessor'>}, <class 'transformers.models.poolformer.configuration_poolformer.PoolFormerConfig'>: {'pil': <class 'transformers.models.poolformer.image_processing_pil_poolformer.PoolFormerImageProcessorPil'>, 'torchvision': <class 'transformers.models.poolformer.image_processing_poolformer.PoolFormerImageProcessor'>}, <class 'transformers.models.pp_chart2table.configuration_pp_chart2table.PPChart2TableConfig'>: {'pil': <class 'transformers.models.pp_chart2table.image_processing_pil_pp_chart2table.PPChart2TableImageProcessorPil'>, 'torchvision': <class 'transformers.models.pp_chart2table.image_processing_pp_chart2table.PPChart2TableImageProcessor'>}, <class 'transformers.models.pp_doclayout_v2.configuration_pp_doclayout_v2.PPDocLayoutV2Config'>: {'torchvision': <class 'transformers.models.pp_doclayout_v2.image_processing_pp_doclayout_v2.PPDocLayoutV2ImageProcessor'>}, <class 'transformers.models.pp_doclayout_v3.configuration_pp_doclayout_v3.PPDocLayoutV3Config'>: {'torchvision': <class 'transformers.models.pp_doclayout_v3.image_processing_pp_doclayout_v3.PPDocLayoutV3ImageProcessor'>}, <class 'transformers.models.pp_formulanet.configuration_pp_formulanet.PPFormulaNetConfig'>: {'torchvision': <class 'transformers.models.pp_formulanet.image_processing_pp_formulanet.PPFormulaNetImageProcessor'>}, <class 'transformers.models.pp_lcnet.configuration_pp_lcnet.PPLCNetConfig'>: {'torchvision': <class 'transformers.models.pp_lcnet.image_processing_pp_lcnet.PPLCNetImageProcessor'>}, <class 'transformers.models.pp_ocrv5_mobile_det.configuration_pp_ocrv5_mobile_det.PPOCRV5MobileDetConfig'>: {'torchvision': <class 'transformers.models.pp_ocrv5_server_det.image_processing_pp_ocrv5_server_det.PPOCRV5ServerDetImageProcessor'>}, <class 'transformers.models.pp_ocrv5_mobile_rec.configuration_pp_ocrv5_mobile_rec.PPOCRV5MobileRecConfig'>: {'torchvision': <class 'transformers.models.pp_ocrv5_server_rec.image_processing_pp_ocrv5_server_rec.PPOCRV5ServerRecImageProcessor'>}, <class 'transformers.models.pp_ocrv5_server_det.configuration_pp_ocrv5_server_det.PPOCRV5ServerDetConfig'>: {'torchvision': <class 'transformers.models.pp_ocrv5_server_det.image_processing_pp_ocrv5_server_det.PPOCRV5ServerDetImageProcessor'>}, <class 'transformers.models.pp_ocrv5_server_rec.configuration_pp_ocrv5_server_rec.PPOCRV5ServerRecConfig'>: {'torchvision': <class 'transformers.models.pp_ocrv5_server_rec.image_processing_pp_ocrv5_server_rec.PPOCRV5ServerRecImageProcessor'>}, <class 'transformers.models.pp_ocrv6_medium_det.configuration_pp_ocrv6_medium_det.PPOCRV6MediumDetConfig'>: {'torchvision': <class 'transformers.models.pp_ocrv5_server_det.image_processing_pp_ocrv5_server_det.PPOCRV5ServerDetImageProcessor'>}, <class 'transformers.models.pp_ocrv6_small_det.configuration_pp_ocrv6_small_det.PPOCRV6SmallDetConfig'>: {'torchvision': <class 'transformers.models.pp_ocrv5_server_det.image_processing_pp_ocrv5_server_det.PPOCRV5ServerDetImageProcessor'>}, <class 'transformers.models.pp_ocrv6_small_rec.configuration_pp_ocrv6_small_rec.PPOCRV6SmallRecConfig'>: {'torchvision': <class 'transformers.models.pp_ocrv6_small_rec.image_processing_pp_ocrv6_small_rec.PPOCRV6SmallRecImageProcessor'>}, <class 'transformers.models.pp_ocrv6_tiny_rec.configuration_pp_ocrv6_tiny_rec.PPOCRV6TinyRecConfig'>: {'torchvision': <class 'transformers.models.pp_ocrv6_small_rec.image_processing_pp_ocrv6_small_rec.PPOCRV6SmallRecImageProcessor'>}, <class 'transformers.models.prompt_depth_anything.configuration_prompt_depth_anything.PromptDepthAnythingConfig'>: {'pil': <class 'transformers.models.prompt_depth_anything.image_processing_pil_prompt_depth_anything.PromptDepthAnythingImageProcessorPil'>, 'torchvision': <class 'transformers.models.prompt_depth_anything.image_processing_prompt_depth_anything.PromptDepthAnythingImageProcessor'>}, <class 'transformers.models.pvt.configuration_pvt.PvtConfig'>: {'pil': <class 'transformers.models.pvt.image_processing_pil_pvt.PvtImageProcessorPil'>, 'torchvision': <class 'transformers.models.pvt.image_processing_pvt.PvtImageProcessor'>}, <class 'transformers.models.pvt_v2.configuration_pvt_v2.PvtV2Config'>: {'torchvision': <class 'transformers.models.pvt.image_processing_pvt.PvtImageProcessor'>, 'pil': <class 'transformers.models.pvt.image_processing_pil_pvt.PvtImageProcessorPil'>}, <class 'transformers.models.qianfan_ocr.configuration_qianfan_ocr.QianfanOCRConfig'>: {'torchvision': <class 'transformers.models.got_ocr2.image_processing_got_ocr2.GotOcr2ImageProcessor'>, 'pil': <class 'transformers.models.got_ocr2.image_processing_pil_got_ocr2.GotOcr2ImageProcessorPil'>}, <class 'transformers.models.qwen2_5_omni.configuration_qwen2_5_omni.Qwen2_5OmniConfig'>: {'torchvision': <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>, 'pil': <class 'transformers.models.qwen2_vl.image_processing_pil_qwen2_vl.Qwen2VLImageProcessorPil'>}, <class 'transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig'>: {'torchvision': <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>, 'pil': <class 'transformers.models.qwen2_vl.image_processing_pil_qwen2_vl.Qwen2VLImageProcessorPil'>}, <class 'transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLConfig'>: {'pil': <class 'transformers.models.qwen2_vl.image_processing_pil_qwen2_vl.Qwen2VLImageProcessorPil'>, 'torchvision': <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>}, <class 'transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5Config'>: {'torchvision': <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>, 'pil': <class 'transformers.models.qwen2_vl.image_processing_pil_qwen2_vl.Qwen2VLImageProcessorPil'>}, <class 'transformers.models.qwen3_5_moe.configuration_qwen3_5_moe.Qwen3_5MoeConfig'>: {'torchvision': <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>, 'pil': <class 'transformers.models.qwen2_vl.image_processing_pil_qwen2_vl.Qwen2VLImageProcessorPil'>}, <class 'transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe.Qwen3OmniMoeConfig'>: {'torchvision': <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>, 'pil': <class 'transformers.models.qwen2_vl.image_processing_pil_qwen2_vl.Qwen2VLImageProcessorPil'>}, <class 'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig'>: {'torchvision': <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>, 'pil': <class 'transformers.models.qwen2_vl.image_processing_pil_qwen2_vl.Qwen2VLImageProcessorPil'>}, <class 'transformers.models.regnet.configuration_regnet.RegNetConfig'>: {'torchvision': <class 'transformers.models.convnext.image_processing_convnext.ConvNextImageProcessor'>, 'pil': <class 'transformers.models.convnext.image_processing_pil_convnext.ConvNextImageProcessorPil'>}, <class 'transformers.models.resnet.configuration_resnet.ResNetConfig'>: {'torchvision': <class 'transformers.models.convnext.image_processing_convnext.ConvNextImageProcessor'>, 'pil': <class 'transformers.models.convnext.image_processing_pil_convnext.ConvNextImageProcessorPil'>}, <class 'transformers.models.rf_detr.configuration_rf_detr.RfDetrConfig'>: {'torchvision': <class 'transformers.models.rf_detr.image_processing_rf_detr.RfDetrImageProcessor'>}, <class 'transformers.models.rt_detr.configuration_rt_detr.RTDetrConfig'>: {'pil': <class 'transformers.models.rt_detr.image_processing_pil_rt_detr.RTDetrImageProcessorPil'>, 'torchvision': <class 'transformers.models.rt_detr.image_processing_rt_detr.RTDetrImageProcessor'>}, <class 'transformers.models.sam.configuration_sam.SamConfig'>: {'pil': <class 'transformers.models.sam.image_processing_pil_sam.SamImageProcessorPil'>, 'torchvision': <class 'transformers.models.sam.image_processing_sam.SamImageProcessor'>}, <class 'transformers.models.sam2.configuration_sam2.Sam2Config'>: {'torchvision': <class 'transformers.models.sam2.image_processing_sam2.Sam2ImageProcessor'>}, <class 'transformers.models.sam2_video.configuration_sam2_video.Sam2VideoConfig'>: {'torchvision': <class 'transformers.models.sam2.image_processing_sam2.Sam2ImageProcessor'>}, <class 'transformers.models.sam3.configuration_sam3.Sam3Config'>: {'torchvision': <class 'transformers.models.sam3.image_processing_sam3.Sam3ImageProcessor'>}, <class 'transformers.models.sam3_lite_text.configuration_sam3_lite_text.Sam3LiteTextConfig'>: {'torchvision': <class 'transformers.models.sam3.image_processing_sam3.Sam3ImageProcessor'>}, <class 'transformers.models.sam3_tracker.configuration_sam3_tracker.Sam3TrackerConfig'>: {'torchvision': <class 'transformers.models.sam3.image_processing_sam3.Sam3ImageProcessor'>}, <class 'transformers.models.sam3_tracker_video.configuration_sam3_tracker_video.Sam3TrackerVideoConfig'>: {'torchvision': <class 'transformers.models.sam3.image_processing_sam3.Sam3ImageProcessor'>}, <class 'transformers.models.sam3_video.configuration_sam3_video.Sam3VideoConfig'>: {'torchvision': <class 'transformers.models.sam3.image_processing_sam3.Sam3ImageProcessor'>}, <class 'transformers.models.sam_hq.configuration_sam_hq.SamHQConfig'>: {'torchvision': <class 'transformers.models.sam.image_processing_sam.SamImageProcessor'>, 'pil': <class 'transformers.models.sam.image_processing_pil_sam.SamImageProcessorPil'>}, <class 'transformers.models.sapiens2.configuration_sapiens2.Sapiens2Config'>: {'torchvision': <class 'transformers.models.sapiens2.image_processing_sapiens2.Sapiens2ImageProcessor'>}, <class 'transformers.models.segformer.configuration_segformer.SegformerConfig'>: {'pil': <class 'transformers.models.segformer.image_processing_pil_segformer.SegformerImageProcessorPil'>, 'torchvision': <class 'transformers.models.segformer.image_processing_segformer.SegformerImageProcessor'>}, <class 'transformers.models.seggpt.configuration_seggpt.SegGptConfig'>: {'pil': <class 'transformers.models.seggpt.image_processing_pil_seggpt.SegGptImageProcessorPil'>, 'torchvision': <class 'transformers.models.seggpt.image_processing_seggpt.SegGptImageProcessor'>}, <class 'transformers.models.shieldgemma2.configuration_shieldgemma2.ShieldGemma2Config'>: {'torchvision': <class 'transformers.models.gemma3.image_processing_gemma3.Gemma3ImageProcessor'>, 'pil': <class 'transformers.models.gemma3.image_processing_pil_gemma3.Gemma3ImageProcessorPil'>}, <class 'transformers.models.siglip.configuration_siglip.SiglipConfig'>: {'pil': <class 'transformers.models.siglip.image_processing_pil_siglip.SiglipImageProcessorPil'>, 'torchvision': <class 'transformers.models.siglip.image_processing_siglip.SiglipImageProcessor'>}, <class 'transformers.models.siglip2.configuration_siglip2.Siglip2Config'>: {'pil': <class 'transformers.models.siglip2.image_processing_pil_siglip2.Siglip2ImageProcessorPil'>, 'torchvision': <class 'transformers.models.siglip2.image_processing_siglip2.Siglip2ImageProcessor'>}, <class 'transformers.models.slanet.configuration_slanet.SLANetConfig'>: {'torchvision': <class 'transformers.models.slanext.image_processing_slanext.SLANeXtImageProcessor'>}, <class 'transformers.models.slanext.configuration_slanext.SLANeXtConfig'>: {'torchvision': <class 'transformers.models.slanext.image_processing_slanext.SLANeXtImageProcessor'>}, <class 'transformers.models.smolvlm.configuration_smolvlm.SmolVLMConfig'>: {'pil': <class 'transformers.models.smolvlm.image_processing_pil_smolvlm.SmolVLMImageProcessorPil'>, 'torchvision': <class 'transformers.models.smolvlm.image_processing_smolvlm.SmolVLMImageProcessor'>}, <class 'transformers.models.superglue.configuration_superglue.SuperGlueConfig'>: {'pil': <class 'transformers.models.superglue.image_processing_pil_superglue.SuperGlueImageProcessorPil'>, 'torchvision': <class 'transformers.models.superglue.image_processing_superglue.SuperGlueImageProcessor'>}, <class 'transformers.models.superpoint.configuration_superpoint.SuperPointConfig'>: {'pil': <class 'transformers.models.superpoint.image_processing_pil_superpoint.SuperPointImageProcessorPil'>, 'torchvision': <class 'transformers.models.superpoint.image_processing_superpoint.SuperPointImageProcessor'>}, <class 'transformers.models.swiftformer.configuration_swiftformer.SwiftFormerConfig'>: {'torchvision': <class 'transformers.models.vit.image_processing_vit.ViTImageProcessor'>, 'pil': <class 'transformers.models.vit.image_processing_pil_vit.ViTImageProcessorPil'>}, <class 'transformers.models.swin.configuration_swin.SwinConfig'>: {'torchvision': <class 'transformers.models.vit.image_processing_vit.ViTImageProcessor'>, 'pil': <class 'transformers.models.vit.image_processing_pil_vit.ViTImageProcessorPil'>}, <class 'transformers.models.swin2sr.configuration_swin2sr.Swin2SRConfig'>: {'pil': <class 'transformers.models.swin2sr.image_processing_pil_swin2sr.Swin2SRImageProcessorPil'>, 'torchvision': <class 'transformers.models.swin2sr.image_processing_swin2sr.Swin2SRImageProcessor'>}, <class 'transformers.models.swinv2.configuration_swinv2.Swinv2Config'>: {'torchvision': <class 'transformers.models.vit.image_processing_vit.ViTImageProcessor'>, 'pil': <class 'transformers.models.vit.image_processing_pil_vit.ViTImageProcessorPil'>}, <class 'transformers.models.t5gemma2.configuration_t5gemma2.T5Gemma2Config'>: {'torchvision': <class 'transformers.models.gemma3.image_processing_gemma3.Gemma3ImageProcessor'>, 'pil': <class 'transformers.models.gemma3.image_processing_pil_gemma3.Gemma3ImageProcessorPil'>}, <class 'transformers.models.t5gemma2.configuration_t5gemma2.T5Gemma2EncoderConfig'>: {'torchvision': <class 'transformers.models.gemma3.image_processing_gemma3.Gemma3ImageProcessor'>, 'pil': <class 'transformers.models.gemma3.image_processing_pil_gemma3.Gemma3ImageProcessorPil'>}, <class 'transformers.models.table_transformer.configuration_table_transformer.TableTransformerConfig'>: {'torchvision': <class 'transformers.models.detr.image_processing_detr.DetrImageProcessor'>, 'pil': <class 'transformers.models.detr.image_processing_pil_detr.DetrImageProcessorPil'>}, <class 'transformers.models.textnet.configuration_textnet.TextNetConfig'>: {'pil': <class 'transformers.models.textnet.image_processing_pil_textnet.TextNetImageProcessorPil'>, 'torchvision': <class 'transformers.models.textnet.image_processing_textnet.TextNetImageProcessor'>}, <class 'transformers.models.timesformer.configuration_timesformer.TimesformerConfig'>: {'pil': <class 'transformers.models.videomae.image_processing_pil_videomae.VideoMAEImageProcessorPil'>, 'torchvision': <class 'transformers.models.videomae.image_processing_videomae.VideoMAEImageProcessor'>}, <class 'transformers.models.timm_wrapper.configuration_timm_wrapper.TimmWrapperConfig'>: {'pil': <class 'transformers.image_processing_timm_wrapper._LazyModule.__getattr__.<locals>.Placeholder'>}, <class 'transformers.models.trocr.configuration_trocr.TrOCRConfig'>: {'torchvision': <class 'transformers.models.vit.image_processing_vit.ViTImageProcessor'>, 'pil': <class 'transformers.models.vit.image_processing_pil_vit.ViTImageProcessorPil'>}, <class 'transformers.models.tvp.configuration_tvp.TvpConfig'>: {'pil': <class 'transformers.models.tvp.image_processing_pil_tvp.TvpImageProcessorPil'>, 'torchvision': <class 'transformers.models.tvp.image_processing_tvp.TvpImageProcessor'>}, <class 'transformers.models.udop.configuration_udop.UdopConfig'>: {'torchvision': <class 'transformers.models.layoutlmv3.image_processing_layoutlmv3.LayoutLMv3ImageProcessor'>, 'pil': <class 'transformers.models.layoutlmv3.image_processing_pil_layoutlmv3.LayoutLMv3ImageProcessorPil'>}, <class 'transformers.models.upernet.configuration_upernet.UperNetConfig'>: {'torchvision': <class 'transformers.models.segformer.image_processing_segformer.SegformerImageProcessor'>, 'pil': <class 'transformers.models.segformer.image_processing_pil_segformer.SegformerImageProcessorPil'>}, <class 'transformers.models.uvdoc.configuration_uvdoc.UVDocConfig'>: {'torchvision': <class 'transformers.models.uvdoc.image_processing_uvdoc.UVDocImageProcessor'>}, <class 'transformers.models.video_llama_3.configuration_video_llama_3.VideoLlama3Config'>: {'pil': <class 'transformers.models.video_llama_3.image_processing_pil_video_llama_3.VideoLlama3ImageProcessorPil'>, 'torchvision': <class 'transformers.models.video_llama_3.image_processing_video_llama_3.VideoLlama3ImageProcessor'>}, <class 'transformers.models.video_llava.configuration_video_llava.VideoLlavaConfig'>: {'pil': <class 'transformers.models.video_llava.image_processing_video_llava.VideoLlavaImageProcessor'>}, <class 'transformers.models.videomae.configuration_videomae.VideoMAEConfig'>: {'pil': <class 'transformers.models.videomae.image_processing_pil_videomae.VideoMAEImageProcessorPil'>, 'torchvision': <class 'transformers.models.videomae.image_processing_videomae.VideoMAEImageProcessor'>}, <class 'transformers.models.vilt.configuration_vilt.ViltConfig'>: {'pil': <class 'transformers.models.vilt.image_processing_pil_vilt.ViltImageProcessorPil'>, 'torchvision': <class 'transformers.models.vilt.image_processing_vilt.ViltImageProcessor'>}, <class 'transformers.models.vipllava.configuration_vipllava.VipLlavaConfig'>: {'torchvision': <class 'transformers.models.clip.image_processing_clip.CLIPImageProcessor'>, 'pil': <class 'transformers.models.clip.image_processing_pil_clip.CLIPImageProcessorPil'>}, <class 'transformers.models.vit.configuration_vit.ViTConfig'>: {'pil': <class 'transformers.models.vit.image_processing_pil_vit.ViTImageProcessorPil'>, 'torchvision': <class 'transformers.models.vit.image_processing_vit.ViTImageProcessor'>}, <class 'transformers.models.vit_mae.configuration_vit_mae.ViTMAEConfig'>: {'torchvision': <class 'transformers.models.vit.image_processing_vit.ViTImageProcessor'>, 'pil': <class 'transformers.models.vit.image_processing_pil_vit.ViTImageProcessorPil'>}, <class 'transformers.models.vit_msn.configuration_vit_msn.ViTMSNConfig'>: {'torchvision': <class 'transformers.models.vit.image_processing_vit.ViTImageProcessor'>, 'pil': <class 'transformers.models.vit.image_processing_pil_vit.ViTImageProcessorPil'>}, <class 'transformers.models.vitmatte.configuration_vitmatte.VitMatteConfig'>: {'pil': <class 'transformers.models.vitmatte.image_processing_pil_vitmatte.VitMatteImageProcessorPil'>, 'torchvision': <class 'transformers.models.vitmatte.image_processing_vitmatte.VitMatteImageProcessor'>}, <class 'transformers.models.vitpose.configuration_vitpose.VitPoseConfig'>: {'pil': <class 'transformers.models.vitpose.image_processing_pil_vitpose.VitPoseImageProcessorPil'>, 'torchvision': <class 'transformers.models.vitpose.image_processing_vitpose.VitPoseImageProcessor'>}, <class 'transformers.models.vivit.configuration_vivit.VivitConfig'>: {'torchvision': <class 'transformers.models.vivit.image_processing_vivit.VivitImageProcessor'>}, <class 'transformers.models.x_clip.configuration_x_clip.XCLIPConfig'>: {'torchvision': <class 'transformers.models.clip.image_processing_clip.CLIPImageProcessor'>, 'pil': <class 'transformers.models.clip.image_processing_pil_clip.CLIPImageProcessorPil'>}, <class 'transformers.models.yolos.configuration_yolos.YolosConfig'>: {'pil': <class 'transformers.models.yolos.image_processing_pil_yolos.YolosImageProcessorPil'>, 'torchvision': <class 'transformers.models.yolos.image_processing_yolos.YolosImageProcessor'>}, <class 'transformers.models.zoedepth.configuration_zoedepth.ZoeDepthConfig'>: {'pil': <class 'transformers.models.zoedepth.image_processing_pil_zoedepth.ZoeDepthImageProcessorPil'>, 'torchvision': <class 'transformers.models.zoedepth.image_processing_zoedepth.ZoeDepthImageProcessor'>}, <class 'transformers.models.mlcd.configuration_mlcd.MLCDVisionConfig'>: {'torchvision': <class 'transformers.models.clip.image_processing_clip.CLIPImageProcessor'>, 'pil': <class 'transformers.models.clip.image_processing_pil_clip.CLIPImageProcessorPil'>}}

### [2026-06-26 09:45:30 UTC] ASSISTANT

**Thinking**

OK so the model's `config.json` has `model_type: "qwen3_5_moe"`, and `IMAGE_PROCESSOR_MAPPING` does have an entry for `Qwen3_5MoeConfig`. The `image_processor_type` was `Qwen2VLImageProcessorFast` which I already changed to `Qwen2VLImageProcessor`.

Let me test if this works now

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"\nfrom transformers.models.auto.image_processing_auto import AutoImageProcessor\nprocessor = AutoImageProcessor.from_pretrained('/root/.cache/huggingface/hub/models--deepreinforce-ai--Ornith-1.0-35B/snapshots/5df2ed3f675c7beaa490328cc70bb573b65fb660/')\nprint('AutoProcessor loaded:', type(processor))\n\""
}
```

### [2026-06-26 09:45:34 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

AutoProcessor loaded: <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>

### [2026-06-26 09:45:44 UTC] ASSISTANT

**Thinking**

The AutoProcessor now loads successfully after:
1. Installing torchvision and pillow
2. Changing `Qwen2VLImageProcessorFast` to `Qwen2VLImageProcessor` in preprocessor_config.json

The fix is verified to work. Now the quantization phase can be re-run.




The fix is verified — `AutoProcessor` now loads successfully. Two changes were applied:

1. Installed `torchvision` and `pillow` (required by `Qwen2VLImageProcessor`)
2. Patched `preprocessor_config.json`: changed `Qwen2VLImageProcessorFast` → `Qwen2VLImageProcessor` (the Fast variant doesn't exist in transformers 5.12.1)

**FIX_PLAN:**
1. ✅ Installed `torchvision` and `pillow` (required by Qwen2VLImageProcessor)
2. ✅ Patched model's `preprocessor_config.json` to use `Qwen2VLImageProcessor` instead of non-existent `Qwen2VLImageProcessorFast`
3. ✅ Verified AutoProcessor loads correctly — ready to re-run quantize
