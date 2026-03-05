---
layout: post
title: "MoE Expert FFN Backend: experts_implementation"
date: 2026-01-30 19:00:00 +0900
description: "Selecting Expert FFN computation backends (eager, batched_mm, grouped_mm) in HuggingFace Transformers and benchmarking with Solar-Open 100B"
tags: moe experts-implementation huggingface transformers torch-compile grouped-gemm
categories: ml-engineering
lang: en
toc:
  beginning: true
---

## 1. experts_implementation

A [PR (#42697)](https://github.com/huggingface/transformers/pull/42697) that adds support for selecting the Expert FFN computation method in MoE models has been merged into HuggingFace Transformers. Just like how `attn_implementation` allowed you to choose the attention computation backend, you can now hook into the expert computation and run it with the backend of your choice.

---

## 2. eager, batched_mm, grouped_mm

Fundamentally, expert FFN follows the same logic: the router selects top-k experts, performs hidden state projection with expert parameters (`gate_up_proj`, `down_proj`), and then computes a weighted sum using routing weights. The key difference lies in **how the per-expert matrix multiplications are performed**.

### eager: Loop-Based Reference Implementation

This is the most intuitive approach. It iterates through activated experts one by one using a Python loop, selects only the tokens routed to each expert, and performs per-expert projection on those tokens. However, because it uses `torch.where` to select tokens assigned to each expert, it becomes difficult to use with `torch.compile` with the `fullgraph=True` option.

### batched_mm

`batched_mm` duplicates the selected expert's weights for each token, stacks them into a 3D tensor, and performs batched matrix multiplication all at once using `torch.bmm`.

`torch.bmm` stands for Batched Matrix Multiplication — it multiplies pairs of identically-sized matrices all at once.

Since `batched_mm` is compatible with `torch.compile`, it supports `fullgraph`. However, because it copies expert weights, memory usage can more than double compared to eager — making it more advantageous for short sequences or small batch sizes.

### grouped_mm

`grouped_mm` uses `torch._grouped_mm` to support Grouped GEMM. This approach does not copy weights. Instead, it groups tokens by expert and processes all expert projections simultaneously using the Grouped GEMM kernel.

Since weights are not duplicated, this approach is the most memory-efficient, and it particularly excels with long sequences and large batches.

{% include figure.liquid loading="eager" path="assets/img/experts_impl_blog_1.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="80%" zoomable=true caption="Batched Matrix Multiplication (bmm) vs Grouped GEMM Approach" %}

---

## 3. Solar-Open 100B Benchmark

Since the PR showed significant performance differences, I ran benchmarks directly using Upstage's Solar-Open model. Looking at just the Mean Latency, the differences were meaningful. `batched_mm` performed reasonably well with short, small inputs, but overall `grouped_mm` showed the best performance. Compared to eager, even without compile it was about 4x faster on average, and with compile applied the latency difference reached up to 10x.

{% include figure.liquid loading="eager" path="assets/img/solar_latency_summary.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="80%" zoomable=true caption="Solar-Open 100B Latency Comparison (Experts Backend & Torch Compile)" %}

```python
model = AutoModelForCausalLM.from_pretrained(
    "upstage/solar-open-...",
    experts_implementation="grouped_mm",  # or "batched_mm", "eager"
)
```

Note that with `batched_mm` under `batch_size=4`, `seq_len=128` conditions, a memory spike was observed during computation in both compile default and no-compile cases. I left the results as-is without separate modifications for reference.

---

## 4. Wrapping Up

It would be worth considering these backend options for MoE model inference. However, since vLLM has its own `fused_moe` kernel, it cannot be directly used there. I'll cover the `fused_moe` kernel in a future post.
