# ig1/Qwen3.5-27B-NVFP4

## https://huggingface.co/ig1/Qwen3.5-35B-A3B-NVFP4

---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/LICENSE
pipeline_tag: image-text-to-text
datasets:
- HuggingFaceH4/ultrachat_200k
- openai/gsm8k
- sahil2801/CodeAlpaca-20k
- CohereLabs/aya_dataset
base_model:
- Qwen/Qwen3.5-35B-A3B
---

# Qwen3.5-35B-A3B-NVFP4 by IG1

## Quantization

This model has been quantized using [llm-compressor](https://github.com/vllm-project/llm-compressor) v0.10.1.dev31+geb49917e (just after Qwen3.5 support was merged) and transformers v5.3.0.
It is based on the [official example](https://github.com/vllm-project/llm-compressor/blob/36fab734a744b4f46b7b379b7c4b4b3f47fde3bf/docs/key-models/qwen3.5/nvfp4-moe-example.md) with a few modifications (see next section).

### Quantization particularities

The sequence length has been increased from 4096 to 8192 and the number of samples from 256 to 1024.
The 1024 samples come from 4 differents datasets:

* 256 general conversation samples ([UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k))
* 256 math reasoning samples ([GSM8K](https://huggingface.co/datasets/openai/gsm8k))
* 256 code samples ([CodeAlpaca](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k))
* 256 multilingual samples ([Aya](https://huggingface.co/datasets/CohereLabs/aya_dataset))

You can find the quantization script [here](Qwen3.5-35B-A3B_nvfp4.py).

While the quantization needed transformers v5, the original (transformers v4) tokenizer files has been put back for simple execution on current vLLM versions. The transformers v5 tokenizer files produced by llm-compressor can be found in the `transformers_v5` folder.

### About FP8 KV cache

In our testing, the Qwen3.5 Mamba hybrid architecture did not play well with FP8 KV cache:

* **vLLM dynamic FP8 KV cache** (`--kv-cache-dtype fp8_e4m3 --calculate-kv-scales`) appeared to work initially but quality degraded rapidly into gibberish.

* **Static FP8 scales via llm-compressor** (`kv_cache_scheme` in the recipe) corrupted the NVFP4 weight quantization during calibration. Because FP8 is injected into the forward pass during scale computation, layers with mismatched head dimensions (256 for attention vs 128 for linear attention) produced corrupted activations that propagated through the network, poisoning the weight quantization scales. The resulting model output gibberish **even when FP8 KV cache was disabled at inference** — the weights themselves were permanently damaged. Note that static FP8 KV scales stored in a checkpoint are passive metadata and still require explicit activation via `--kv-cache-dtype fp8_e4m3` at vLLM startup to be used; however, the corruption occurred during quantization, not at inference time.

## Qwen3.5 Profiles

Alongside support for dynamic thinking and non-thinking modes, the Qwen team offers [4 sampling parameter profiles](https://huggingface.co/Qwen/Qwen3.5-35B-A3B#using-qwen35-via-the-chat-completions-api):

* Thinking General
* Thinking Coding
* Instruct General
* Instruct Reasoning (we prefer to call it Instruct Creative internally)

Manually configuring these parameters for every AI client can be difficult.
To solve this, we built a lightweight reverse proxy that exposes the 4 profiles as virtual model names.
It handles request transformation on the fly using a single inference server as backend.
View the project on [our GitHub](https://github.com/iguanesolutions/qwen35-rp).

## Inference

We run this model with vLLM, here is a sample execution command:

```bash
docker run --rm --name 'Qwen3.5-35B-A3B-NVFP4' \
  --runtime=nvidia --gpus 'all' --ipc=host \
  -e 'HF_TOKEN' \
  -e 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1' \
  -v '/srv/cache:/root/.cache' \
  -p '127.0.0.1:8000:8000' \
  'vllm/vllm-openai:v0.18.0-cu130' \
  'ig1/Qwen3.5-35B-A3B-NVFP4' \
  --served-model-name 'Qwen3.5-35B-A3B' \
  --reasoning-parser 'qwen3' \
  --enable-auto-tool-choice \
  --tool-call-parser 'qwen3_coder' \
  --max-model-len 'auto' \
  --gpu-memory-utilization '0.65' \
  --kv-cache-memory-bytes '20600M'
```

A few notes about some of the parameters:

* Adapt the `/srv/cache:/root/.cache` mount point to your liking. It contains files you want to keep between multiples run (dynamo bytecode and AOT with torch compile but most importantly the huggingface folder for the model)
* `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` allows for more precise CUDA graph VRAM estimation. It should become the default once vLLM reaches v0.19.0 which at which point you can simply remove it
* `--gpu-memory-utilization 0.65 --kv-cache-memory-bytes 20600M` was set to get 1x the max model len and check the total VRAM consumption once vLLM has been fully started. The `--gpu-memory-utilization` is an upper bound for vLLM start (here on a RTX 6000 Pro Blackwell) before the KV cache is fixed to .
* If you deploy the model into several GPUs using Tensor Parallelism, be sure to check the [official recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) as others flags are needed.

With this config, a vLLM consumed a total of 53,240MiB on a RTX 6000 Pro Blackwell.

---

### RTX 5090 Optimized Deployment

Because of the increase model size from its previous versions (Qwen3-30B-A3B-Thinking-2507 and Qwen3-30B-A3B-Instruct-2507 were 30B not 35B) but also because of its mamba hybrid architecture and its native vision support (layers excluded from the quantization) the final model size is bigger (~ +5GiB) which make its inference by a RTX 5090 with only 32 GiB of RAM challenging.

If you really want/need to, it is still possible by tuning a few parameters (and accepting a lower max model/kv cache size).

> [!TIP]
> On a RTX 5090, we recommend to use the [9B variant](https://huggingface.co/ig1/Qwen3.5-9B-NVFP4#rtx-5090-optimized-deployment)

#### Validated Configuration (Desktop RTX 5090)

```bash
docker run --rm --name 'Qwen3.5-35B-A3B-NVFP4' \
  --runtime=nvidia --gpus 'all' --ipc=host \
  -e 'HF_TOKEN' \
  -e 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1' \
  -v '/srv/cache:/root/.cache' \
  -p '127.0.0.1:8000:8000' \
  'vllm/vllm-openai:v0.18.0-cu130' \
  'ig1/Qwen3.5-35B-A3B-NVFP4' \
  --served-model-name 'Qwen3.5-35B-A3B' \
  --reasoning-parser 'qwen3' \
  --enable-auto-tool-choice \
  --tool-call-parser 'qwen3_coder' \
  --max-model-len 'auto' \
  --limit-mm-per-prompt.video 0 \
  --max-cudagraph-capture-size 32 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization '0.95'
```

**Important:** On a non-headless host (with graphical environment), even `--gpu-memory-utilization 0.95` may cause instability. Use the values below based on your setup.

#### GPU Memory Utilization Guide

| Setup | `--gpu-memory-utilization` | Est. Context | Stability |
|-------|---------------------------|--------------|-----------|
| Desktop Conservative | 0.885 | ~12,500 tokens | ✅ High |
| Desktop Recommended | 0.905 | ~19,500 tokens | ✅ Good |
| Desktop Aggressive | 0.926 | ~28,000 tokens | ⚠️ Monitor |
| Headless/Server | 0.956 | ~40,000 tokens | ✅ Safe |

#### Advanced Optimization Flags

| Flag | Value | Purpose | Impact |
|------|-------|---------|--------|
| `--limit-mm-per-prompt.video` | `0` | Disable video encoder | Saves ~170 MiB |
| `--max-cudagraph-capture-size` | `≤32` | Limit CUDA graph buffers | Saves ~4,400 MiB |
| `--max-num-seqs` | `≤32` | Limit concurrent sequences, can not be higher than max-cudagraph-capture-size | Included above |
| `--max-num-batched-tokens` | `2048` | Reduce activation buffers | Saves ~350 MiB |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `1` | Precise CUDA graph measurement | +~1,500 tokens |

**Total Savings vs Default:** ~4,920 MiB (enables RTX 5090 deployment)

#### Concurrency vs VRAM Trade-off

For advanced users who want to fine-tune parallel request capacity:

| `--max-num-seqs` / `--max-cudagraph-capture-size` | Base VRAM | vs 30 GiB Target | Recommended Use |
|------------------------------------------------|-----------|------------------|-----------------|
| 1 | 30,502 MiB | ✅ -218 MiB | Single request only |
| 8 | 30,510 MiB | ✅ -210 MiB | Single user (best value) |
| 32 | 30,570 MiB | ✅ -150 MiB | **Recommended default** |
| 64 | 30,840 MiB | ⚠️ +120 MiB | Moderate concurrency |
| 128 | 31,236 MiB | ❌ +516 MiB | High concurrency (reduce gpu-memory-utilization) |
| 256 | 31,948 MiB | ❌ +1,228 MiB | **Avoid on desktop RTX 5090** |
