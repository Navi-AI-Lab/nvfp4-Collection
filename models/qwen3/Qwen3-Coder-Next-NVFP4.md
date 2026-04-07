# Qwen3-Coder-Next-NVFP4

> [GadflyII/Qwen3-Coder-Next-NVFP4](https://huggingface.co/GadflyII/Qwen3-Coder-Next-NVFP4)

NVFP4 quantized version of [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) (80B-A3B).

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | Qwen/Qwen3-Coder-Next |
| **Architecture** | Qwen3NextForCausalLM (Hybrid DeltaNet + Attention + MoE) |
| **Parameters** | 80B total, 3B activated per token |
| **Experts** | 512 total, 10 activated + 1 shared |
| **Layers** | 48 |
| **Context Length** | 262,144 tokens (256K) |
| **Quantization** | NVFP4 (FP4 weights + FP4 activations) |
| **Size** | 45GB (down from ~149GB BF16, 70% reduction) |
| **Format** | compressed-tensors |

## Quantization Details

Quantized using [llmcompressor](https://github.com/vllm-project/llm-compressor) 0.9.0.1.

```
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 2048
DATASET = "HuggingFaceH4/ultrachat_200k" (train_sft)
moe_calibrate_all_experts = True

# Layers kept in BF16
ignore = [
    "lm_head",
    "re:.*mlp.gate$",               # MoE router gates
    "re:.*mlp.shared_expert_gate$", # Shared expert gates
    "re:.*linear_attn.*",           # DeltaNet linear attention
]
```

## Benchmark Results

### MMLU-Pro

| Model | Accuracy | Delta |
|-------|----------|-------|
| BF16 | 52.90% | - |
| **NVFP4** | **51.27%** | **-1.63%** |

### Context Length Testing

Successfully tested up to **128K tokens** with FP8 KV cache (Not enough VRAM to test any higher context).

## Usage with vLLM

Requires vLLM with NVFP4 support (0.16.0+), Transformers 5.0.0+

```bash
vllm serve GadflyII/Qwen3-Coder-Next-NVFP4 \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --kv-cache-dtype fp8
```

## License

Apache 2.0 (same as base model)

## Acknowledgments

- [Qwen Team](https://huggingface.co/Qwen) for the base model
- [RedHatAI](https://huggingface.co/RedHatAI) for the quantization approach reference
- [vLLM Project](https://github.com/vllm-project/vllm) for llmcompressor

---

**Note:** If you have a multi-GPU SM120 Blackwell system (RTX 50/Pro), try the [vLLM fork](https://github.com/Gadflyii/vllm/tree/main) to resolve P2P / TP=2 issues (Pending PR into upstream).
