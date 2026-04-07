# NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4

> [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)

## Model Summary

| Property | Value |
|----------|-------|
| **Total Parameters** | 120B (12B active) |
| **Architecture** | LatentMoE - Mamba-2 + MoE + Attention hybrid with Multi-Token Prediction (MTP) |
| **Context Length** | Up to 1M tokens |
| **Minimum GPU Requirement** | 1x B200 OR 1x DGX Spark |
| **Supported Languages** | English, French, German, Italian, Japanese, Spanish, Chinese |
| **Best For** | Agentic workflows, long-context reasoning, high-volume workloads (e.g. IT ticket automation), tool use, RAG |
| **Reasoning Mode** | Configurable on/off via chat template (`enable_thinking=True/False`) |
| **License** | [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/) |
| **Release Date** | March 11, 2026 |

## Quick Start

> Use `temperature=1.0` and `top_p=0.95` across **all tasks and serving backends** — reasoning, tool calling, and general chat alike.

## Model Overview

**Nemotron-3-Super-120B-A12B-NVFP4** is a large language model trained by NVIDIA, designed to deliver strong agentic, reasoning, and conversational capabilities. It's optimized for collaborative agents and high-volume workloads such as IT ticket automation.

The model employs a hybrid **Latent Mixture-of-Experts (LatentMoE)** architecture, utilizing interleaved Mamba-2 and MoE layers, along with select Attention layers. The Super model incorporates **Multi-Token Prediction (MTP)** layers for faster text generation and improved quality, and is trained using **NVFP4** quantization to maximize compute efficiency.

## Model Architecture

- **Architecture Type:** Mamba2-Transformer Hybrid Latent Mixture of Experts (LatentMoE) with Multi-Token Prediction (MTP)
- **Network Architecture:** Nemotron Hybrid LatentMoE
- **Number of model parameters:** 120B Total / 12B Active

## Training Methodology

### Stage 1: Pre-Training
- Pre-trained for over 25T tokens using crawled and synthetic code, math, science, and general knowledge data
- Leveraged NVFP4 quantization for efficiency
- Software: [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

### Stage 2: Supervised Fine-Tuning
- Fine-tuned on synthetic code, math, science, tool calling, instruction following, structured outputs, and general knowledge data
- Designed to support long-range retrieval and multi-document aggregation
- Software: [Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)

### Stage 3: Reinforcement Learning
- Multi-environment RL using asynchronous GRPO (Group Relative Policy Optimization)
- Environments: math, code, science, instruction following, multi-step tool use, multi-turn conversations, structured outputs
- Software: [NeMo RL](https://github.com/NVIDIA-NeMo/RL), [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)

## Quick Start Guide

### vLLM

```bash
pip install vllm==0.18.1

export MODEL_CKPT=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4

vllm serve $MODEL_CKPT \
  --served-model-name nvidia/nemotron-3-super \
  --async-scheduling \
  --dtype auto \
  --max-model-len 262144 \
  --swap-space 0 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-cudagraph-capture-size 128 \
  --enable-chunked-prefill \
  --mamba-ssm-cache-dtype float16 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

#### vLLM on DGX Spark

```bash
VLLM_NVFP4_GEMM_BACKEND=marlin
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
vllm serve $MODEL_CKPT \
  --served-model-name nemotron-3-super \
  --host 0.0.0.0 \
  --port 8000 \
  --async-scheduling \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --max-num-seqs 4 \
  --max-model-len 394000 \
  --attention-backend TRITON_ATTN \
  --mamba_ssm_cache_dtype float32 \
  --moe-backend marlin
```

### SGLang

```bash
pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python'

sglang serve \
  --model-path nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --served-model-name nvidia/nemotron-3-super \
  --trust-remote-code \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_3
```

### TRT-LLM

```bash
cat > ./extra-llm-api-config.yml << EOF
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.9
  mamba_ssm_cache_dtype: float16
  mamba_ssm_stochastic_rounding: true
  mamba_ssm_philox_rounds: 5
moe_config:
   backend: CUTLASS
cuda_graph_config:
    enable_padding: true
    max_batch_size: 8
enable_attention_dp: false
enable_chunked_prefill: true
stream_interval: 1
print_iter_log: true
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 3
  allow_advanced_sampling: true
EOF

TLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
trtllm-serve <nvfp4_ckpt> \
  --host 0.0.0.0 \
  --port 8123 \
  --max_batch_size 8 \
  --tp_size 1 --ep_size 1 \
  --max_num_tokens 8192 \
  --trust_remote_code \
  --reasoning_parser nano-v3 \
  --tool_parser qwen3_coder \
  --extra_llm_api_options extra-llm-api-config.yml \
  --max_seq_len 1048576
```

## API Client Usage

### OpenAI-Compatible Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "nvidia/nemotron-3-super"
```

#### Reasoning ON (default)

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Write a haiku about GPUs"}],
    max_tokens=16000,
    temperature=1.0,
    top_p=0.95,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
)
print(response.choices[0].message.content)
```

#### Reasoning OFF

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What is the capital of Japan?"}],
    max_tokens=16000,
    temperature=1.0,
    top_p=0.95,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)
print(response.choices[0].message.content)
```

#### Low-effort reasoning

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What is the capital of Japan?"}],
    max_tokens=16000,
    temperature=1.0,
    top_p=0.95,
    extra_body={"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}
)
print(response.choices[0].message.content)
```

#### Budget-Controlled Reasoning

```python
from typing import Any, Dict, List
import openai
from transformers import AutoTokenizer


class ThinkingBudgetClient:
    def __init__(self, base_url: str, api_key: str, tokenizer_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        reasoning_budget: int = 512,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Dict[str, Any]:
        assert max_tokens > reasoning_budget, (
            f"reasoning_budget must be less than max_tokens. "
            f"Got {max_tokens=} and {reasoning_budget=}"
        )

        # Step 1: generate the reasoning trace up to the budget
        response = self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=reasoning_budget, **kwargs
        )
        reasoning_content = response.choices[0].message.content
        if "" not in reasoning_content:
            reasoning_content = f"{reasoning_content}.\n\n\n"

        reasoning_tokens_len = len(
            self.tokenizer.encode(reasoning_content, add_special_tokens=False)
        )
        remaining_tokens = max_tokens - reasoning_tokens_len
        assert remaining_tokens > 0, (
            f"No tokens remaining for response ({remaining_tokens=}). "
            "Increase max_tokens or lower reasoning_budget."
        )

        # Step 2: continue from the closed reasoning trace
        messages.append({"role": "assistant", "content": reasoning_content})
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
        response = self.client.completions.create(
            model=model, prompt=prompt, max_tokens=remaining_tokens, **kwargs
        )

        return {
            "reasoning_content": reasoning_content.strip().strip("").strip(),
            "content": response.choices[0].text,
            "finish_reason": response.choices[0].finish_reason,
        }
```

## Key Benchmarks

| Benchmark | Nemotron-3-Super | Nemotron-3-Super FP8 | Nemotron-3-Super NVFP4 |
|-----------|-----------------|----------------------|------------------------|
| **MMLU-Pro** | 83.73 | 83.63 | 83.33 |
| **HMMT Feb25 (with tools)** | 94.73 | 94.38 | 95.36 |
| **GPQA (no tools)** | 79.23 | 79.36 | 79.42 |
| **LiveCodeBench (v6)** | 78.69 | 78.44 | 78.44 |
| **RULER-500 @ 256k** | 96.60 | 96.33 | 96.52 |
| **Arena-Hard-V2** | 73.88 | 76.06 | 76.00 |

## Input / Output

- **Input Type(s):** Text
- **Input Format(s):** String
- **Input Parameters:** One-Dimensional (1D): Sequences
- **Maximum context length:** Up to 1M tokens
- **Output Type(s):** Text
- **Output Format:** String

## Software Integration

- **Runtime Engine(s):** NeMo 25.11.01
- **Supported Hardware:** NVIDIA Ampere (A100), Blackwell, Hopper (H100-80GB)
- **Operating System(s):** Linux

## Training & Evaluation Datasets

**Total Parameters:** 15,573,172,908,990 Tokens
**Total Datasets:** 153
**Data Collection Period:** 2013 to February 24, 2026

### Key Datasets

- **Nemotron-CC-v2 & v2.1:** 9.13T tokens - English web data from Common Crawl
- **Nemotron-CC-Code-v1:** 427.9B tokens - High-quality code from Common Crawl
- **Nemotron-Pretraining-Code-v1 & v2:** 1.09T tokens - Curated GitHub code
- **Nemotron-CC-Math-v1:** 133.3B tokens - Math with LaTeX formatting
- **Nemotron-Pretraining-Specialized-v1:** 336.4B tokens - Synthetic STEM data

### Language Distribution in Post-Training

| Language | Size |
|----------|------|
| English | 13.48M |
| Italian | 53k |
| German | 53k |
| Spanish | 53k |
| French | 53k |
| Japanese | 53k |
| Chinese | 53k |

## License & Terms of Use

This model is governed by the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).

For NIM container deployment, also refer to:
- [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
- [Product-Specific Terms for AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/)

## Citation

```bibtex
@misc{nvidia_nemotron_3_2025,
  title  = {NVIDIA Nemotron 3: Efficient and Open Intelligence},
  author = {{NVIDIA}},
  year   = {2025},
  url    = {https://arxiv.org/abs/2512.20856},
  note   = {White Paper}
}
```

## Additional Resources

- [Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [Nemotron Developer Page](https://developer.nvidia.com/nemotron)
- [Chat with Nemotron 3 Super](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b)
- [Discord Community](https://discord.gg/9xpKQtVvrk)
- [Advanced Deployment Guide](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide)
