# Qwen3.5-27B-NVFP4-Opus-GB10

> [natfii/Qwen3.5-27B-NVFP4-Opus-GB10](https://huggingface.co/natfii/Qwen3.5-27B-NVFP4-Opus-GB10)

NVFP4 (W4A4) quantization of [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), calibrated on reasoning datasets and optimized for serving on NVIDIA DGX Spark (GB10, Blackwell SM121).

**~18 GB quantized** — fits comfortably in GB10's 128 GB unified memory with room for 64k context and multi-user batching.

## Quantization Details

| Parameter | Value |
|-----------|-------|
| Method | NVFP4 (W4A4 FP4) |
| Format | nvfp4-pack-quantized |
| Library | llmcompressor 0.10.0+ / compressed-tensors 0.14.0+ |
| Group size | 16 |
| Weight bits | 4-bit float |
| Activation bits | 4-bit float (dynamic per-group at inference) |
| Scale dtype | float8_e4m3fn |
| Weight observer | memoryless_minmax |
| Activation calibration | static_minmax (512 samples) |
| Symmetric | Yes |

### Layers kept in BF16

- `lm_head` — output projection (quantizing is optional via `QUANTIZE_LM_HEAD=1`)
- `linear_attn.conv1d` — small 4D convolution kernels in linear attention layers
- `linear_attn.in_proj_ba` — small projection layers in linear attention

These layers are excluded because FP4 kernel dispatch overhead exceeds the BF16 compute cost for their shapes.

## Calibration

Calibrated on 512 samples (max sequence length 2048, seed 42) from three reasoning datasets:

- [Crownelius/Opus-4.6-Reasoning-3300x](https://huggingface.co/datasets/Crownelius/Opus-4.6-Reasoning-3300x)
- [Jackrong/Qwen3.5-reasoning-700x](https://huggingface.co/datasets/Jackrong/Qwen3.5-reasoning-700x)
- [TeichAI/claude-4.5-opus-high-reasoning-250x](https://huggingface.co/datasets/TeichAI/claude-4.5-opus-high-reasoning-250x)

## Architecture

| Parameter | Value |
|-----------|-------|
| Hidden size | 5120 |
| FFN intermediate size | 17408 |
| Attention heads | 24 |
| KV heads | 4 (GQA) |
| Head dimension | 256 |
| Layers | 64 |
| Attention pattern | 3x linear + 1x full (repeating) |
| Max position embeddings | 262,144 |
| Vocab size | 248,320 |

Qwen3.5-27B is a hybrid attention model alternating linear and full attention layers.

## How to Serve (vLLM on DGX Spark)

```bash
docker run -d \
  --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_NVFP4_GEMM_BACKEND=cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  nvllm:gb10 \
  serve \
  --model natfii/Qwen3.5-27B-NVFP4-Opus-GB10 \
  --served-model-name default \
  --kv-cache-dtype auto \
  --attention-backend triton_attn \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --language-model-only \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-block-size 64 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.80
```

Requires vLLM with NVFP4 support (Blackwell SM120+).

## Hardware

- **Target**: NVIDIA DGX Spark (GB10, Blackwell SM121)
- **Memory**: 128 GB unified, 221 GB/s bandwidth
- **Benchmarks**: Coming soon

## Notes

- The quantization script includes compatibility shims for transformers 5.x (`TORCH_INIT_FUNCTIONS` and `_get_no_split_modules` patches).
- Qwen3.5's hybrid linear attention layers require `--language-model-only`, `--mamba-cache-mode align`, and `--mamba-block-size 64` flags in vLLM. vLLM reuses its Mamba cache infrastructure to handle the linear attention state.
- To also quantize `lm_head` to FP4, re-run quantization with `QUANTIZE_LM_HEAD=1`.
