# Qwen3.5-122B-A10B-NVFP4

> [Sehyo/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4)

This is a quantized version of [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) using the **NVFP4** quantization scheme.

Please use nightly vLLM for support.

## Changelog

- **02/03/2026**: Added MTP (multi-token prediction) weights from source checkpoint, enabling speculative decoding with vLLM.
- **25/02/2026**: Initial upload.

## Calibration

- **Samples**: 512 (256 from each dataset)
- **Datasets**:
  - [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (`train_sft` split)
  - [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) (`chat` split)
- **Max sequence length**: 4096
- **All experts calibrated**: `moe_calibrate_all_experts=True`

## Creation

This model was created using [VLLM's LLM Compressor](https://github.com/vllm-project/llm-compressor) with Qwen3.5 MoE support added via [PR #2383](https://github.com/vllm-project/llm-compressor/pull/2383). The PR adds a custom `CalibrationQwen3MoeSparseMoeBlock` that routes calibration data to all experts during quantization, ensuring every expert receives proper calibration for accurate NVFP4 quantization.
