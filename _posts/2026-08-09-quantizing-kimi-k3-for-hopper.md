---
layout: post
title: "Requantizing Kimi-K3 from MXFP4 to W4AFP8 for H200"
date: 2026-08-09
description: "Requantizing Kimi-K3 from MXFP4 to W4AFP8 for H200 — why the original format was unusable on Hopper, which layers we could and couldn't touch, and the wrong turns we took proving the result was equivalent."
tags: llm inference sglang quantization fp8 int4 moe kv-cache hopper h200 kimi-k3
categories: infrastructure
lang: en
toc:
  beginning: true
---

Kimi-K3 came out and we wanted it on our endpoint the same week. The weights were right there on HuggingFace, 1.5 TB of them, already quantized to MXFP4. We had two nodes of 8×H200 sitting ready. This looked like a download-and-serve afternoon.

It was not. The model loaded fine and then ran at a speed that made no sense, worse in some configurations than we had any right to expect from hardware this size. Working out *why* took us through the MoE runner matrix, a KV cache path that was quietly undoing its own optimization, and eventually to rebuilding the checkpoint ourselves.

This post is about that rebuild: what we changed, what we deliberately left alone, and why.

![Precision map of the Kimi-K3-W4AFP8 checkpoint. Routed experts — 247,296 of 249,694 weight tensors — became INT4 at group size 128 with FP8 activations. The remaining 0.96% is expanded below: 393 attention projections in FP8, everything else left in bf16.](/assets/img/k3-precision-map.png)
*What each of the 2.8T model's weight tensors became. Experts are 99% of the tensors and tolerate 4 bits; attention mostly stayed as it was.*

---

## Part 1: Why the original format didn't work

### MXFP4 is not Blackwell-only, but the kernels are

Kimi-K3 ships in MXFP4 at group size 32. Nothing about that format requires a Blackwell GPU. The problem is that on Hopper, there is almost no execution path that can actually use it.

We went through the MoE runner combinations one at a time:

| Runner combination | What happened on H200 |
|---|---|
| deepep + marlin | `NotImplementedError` — the pair isn't registered at all |
| deepep + triton | OOM. Triton expands mxfp4 to bf16 via `upcast_from_mxfp` |
| deepep + deep_gemm | The fp4 path uses `MXF8F6F4 UMMA`, an SM100-only instruction |
| marlin (W4A16) | Works — dequantizes 4-bit to bf16, then computes in bf16 |

Only the last one runs. And look at what it does: it takes weights that are already compact, unpacks them back to 16-bit, and multiplies them on bf16 tensor cores. The H200's FP8 tensor cores sit idle the whole time.

A profile confirmed the cost. In the decode path, `marlin_moe::Marlin<bf16>` alone accounted for **25.2% of GPU time**.

So the blocker was never the hardware. It was the combination of this weight format and the kernel matrix available for it. On a B200 the same checkpoint would have had better options.

### The KV cache had the same problem

While we were profiling, something else stood out. We had FP8 KV cache turned on, the obvious memory win for a long-context model, and it was making things *slower*.

The profile showed dtype casts eating **29.5% of GPU time**. Not the attention math. The casts.

Here's what was happening. FlashAttention-3 has a fast path for FP8 KV, but it's gated on `head_dim <= 256`. MLA's head dimension is 512 + 64 = **576**, so it fails that check every time. The fallback then does this, once per layer:

```python
get_key_buffer(...).to(q.dtype)   # upcast the entire KV pool to bf16
```

The *entire pool*. 332 million elements of it, on every layer, on every forward pass. We measured the element count at 332,513,280 and the pool is 577,216 × 576 = 332,476,416, a match to within 0.01%.

The part that stings: actual KV utilization at the time was around **2%**. We were converting 100% of the pool to use 2% of it, roughly fifty times more work than necessary.

![Why FP8 KV cache was slower on Hopper. MLA's head dimension of 576 fails FlashAttention-3's fp8 fast path, which is gated on head_dim of 256 or less. The fallback upcasts the entire 332-million-element KV pool to bf16 once per layer, while only 2% of the pool is in use — 50 times more conversion than needed, costing 29.5% of GPU time.](/assets/img/k3-fp8-kv-cast.png)
*One gate check, missed by a factor of 2.25, and nearly a third of GPU time went to moving data between formats.*

That one had a configuration fix. Switching the prefill attention backend to `flashmla` removed the cast entirely, because flashmla delegates ordinary extends to the parent implementation and handles target-verify with an in-kernel `descale_k`. The 29.5% dropped to 2.2%.

But it made the larger point clear. **If you want FP8 KV without paying to undo it, the compute has to be FP8 too.** Which meant the weights had to change.

---

## Part 2: Choosing what to quantize

### W4AFP8: 4-bit storage, 8-bit math

The recipe we landed on stores routed MoE expert weights as INT4 at group size 128 with per-group scales, and pairs them with static per-tensor FP8 (E4M3) activation scales carried in the checkpoint. SGLang's `w4afp8` path collapses those into one activation scale per layer at load time, so it fits that serving stack particularly well.

The decision that mattered wasn't the format. It was the scope.

### Why only the experts

Out of 249,694 weight tensors, we converted 247,296, all of them routed experts. Everything else stayed as it was.

| Module | Tensors | Precision | Reasoning |
|---|---|---|---|
| routed experts | 247,296 | **INT4** + FP8 act | Only a few activate per token; errors average out |
| attention (MLA) | 1,020 | bf16 | **100% exposure** — every token, every layer |
| shared experts | 276 | bf16 | Always active |
| vision tower, lm_head | 172 | bf16 | Small win, real risk |
| norms and the rest | 930 | bf16 | — |

The asymmetry is the whole argument. Kimi-K3 routes each token to 16 of 896 experts, so any given expert sees under 2% of traffic, and an error in one of the 16 that did fire is averaged against the other 15. Attention gets no such averaging: every token passes through every attention layer, and the errors compound rather than cancel.

That's also why experts are the right target economically. They're 99% of the tensors and nearly all of the parameter mass, so you get almost the entire size reduction from the part of the model that tolerates it best.

### Going through MXFP4 beats going through bf16

The conversion error measured **8.11%**. The control, taking the same tensors from bf16 and quantizing straight to INT4, measured **10.8–12.2%**.

Starting from the 4-bit grid is *better* than starting from full precision, which is backwards until you account for Kimi-K3 being a QAT model. The MXFP4 grid isn't a lossy approximation of some truer bf16 value; training put the weights on that grid deliberately. Moving from one 4-bit grid to another is a smaller step than moving down from 16 bits.

We also tested whether calibration-based methods like AWQ would help. They didn't: the difference from plain round-to-nearest wasn't statistically significant (the 95% confidence interval included zero). So we skipped it.

---

## Part 3: Proving it didn't break anything

This is the part that took the longest.

### The bar: 2σ, not "looks about the same"

We set the pass criterion up front: every benchmark had to land within **two standard errors** of the published Kimi-K3 number. Two standard errors of the binomial at that benchmark's sample size, not a flat percentage.

That distinction matters more than it sounds. OCRBench with 2,000 samples has a 2σ of ±1.4 points. AA-LCR with 100 samples has ±7.7. A fixed "within 2%" rule would be far too loose on one and impossibly strict on the other.

We ran all six at full size, no subsampling:

| Benchmark | Ours | Kimi-K3 | Δ | 2σ |
|---|---|---|---|---|
| GPQA-Diamond (198) | 92.9 | 93.5 | −0.6 | ±3.6 |
| MMMU-Pro vision (1730) | 80.3 | 81.6 | −1.3 | ±1.9 |
| OCRBench (2×1000) | 89.5 | 89.0 | +0.5 | ±1.4 |
| SciCode (288 sub-step) | 56.9 | 58.7 | −1.8 | ±5.8 |
| Terminal-Bench 2.1 (89) | 86.5 | 88.3 | −1.8 | ±7.2 |
| AA-LCR (100) | 82.0 | 74.7 | +7.3 | ±7.7 |

All six inside the interval. And one honest caveat that belongs right next to the table: **AA-LCR's +7.3 is not an improvement.** It's LLM-judge scored, and we never ran a control through the same judge. A number that beats the reference by that much, on the benchmark with the loosest error bars, is exactly the kind of result you should distrust from your own harness.

### What the controls actually are

A published number is only a control once you know what it measured, and for several of these the answer takes some digging.

**MMMU-Pro** has three subsets (4-option, 10-option, and vision), all with exactly 1,730 items, so the count doesn't identify them. The reference is **vision**. The card's `81.6 / 83.4` pair is without and with tool augmentation, so a harness without tools compares against 81.6.

**SciCode** runs through the official `gencode.py`, which injects scaffolding into three steps and takes a `with_background` flag that *adds* annotated background rather than replacing anything. The reference has it on. Both settings matter: dropping the scaffolding cascades 22 of 288 sub-steps into failure, and turning background off costs 15.6 points, and neither has anything to do with quantization.

**Truncation is measurement failure, not a wrong answer.** Responses that hit the token limit mid-reasoning are re-run rather than scored incorrect, and counted once.

---

## Part 4: The results, and the second round

With MoE quantized, we benchmarked against the original at identical settings: concurrency 8, ~1,530-token prompts, 256-token outputs:

| Metric | MXFP4 | **W4AFP8** | Change |
|---|---|---|---|
| TTFT p50 | 0.924s | **0.611s** | **−33.9%** |
| TPOT p50 | 23.21ms | **21.32ms** | **−8.1%** |
| per-request TPS p50 | 43.1 | **46.9** | **+8.8%** |
| total throughput | 277.7 tok/s | **327.5 tok/s** | **+17.9%** |
| DSPARK accept length | ~3.4 | 3.41 | unchanged |

Four out of four improved, which is what you'd hope for when the MoE GEMM moves from bf16 tensor cores to FP8 ones.

The unchanged accept length is worth a moment. Speculative decoding works by having a small draft model propose several tokens that the main model then verifies in one pass. If quantization shifted the target model's output distribution at all, draft and target would start disagreeing and that number would drop. It didn't move. That's independent evidence, nothing to do with benchmarks, that the model still behaves like itself.

### Round two: the attention weights

We came back for the memory. Attention weights and the KV cache compete for the same GPU memory, so shrinking one grows the other. We converted 393 attention tensors to FP8: 96% by size, 64.5 GB of 67.4 GB.

Not 100%, and the missing 4% comes down to one code path.

`kimi_k3.py` has code that reaches past the quantization layer and reads `.weight` directly. Normally you'd never notice, because every access goes through the module call, which knows about the storage format. But the FP8 path stores weights **transposed** to satisfy `scaled_mm`, and a direct reader gets a flipped tensor and believes it's the original:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (384x128 and 768x128)
  kimi_k3.py  forget_gate = gemm(bfa[..., :n_fa], self.f_b_proj.weight)
```

So the exclusion list is every projection with a raw `.weight` reader: `kv_b_proj`, `q_b_proj`, `f_b_proj`, `f_a_proj`, `b_proj`. Together they're 2.4 GB left on the table, and recovering them would mean patching the model code, not the checkpoint. Note the readers aren't all reached through `self.`; `self_attn.kv_b_proj.weight` is one of them, so grep by attribute, not by receiver.

### Per-channel, per-tensor, and why it doesn't matter here

Two scale granularities are in play. The runtime path quantizes attention weights **per-channel**; the serialized path used for a baked checkpoint expects **per-tensor**, because `convert_to_channelwise` indexes scales per logical shard. Activations are per-tensor dynamic in both.

Dropping to per-tensor costs almost nothing. On a `q_proj` where the row-wise absolute max varies by **26×**, relative error goes from 2.638% to 2.647%. FP8 has an exponent field, and e4m3's normal range spans roughly 2¹⁵, so a 26× spread is absorbed. That is specific to floating-point formats. The intuition that per-tensor is destructive comes from INT8, where a fixed step size across a wide row makes it so.

One thing to watch when applying the patch: the module list is matched by name, and **dp-attention renames the fused QKV module** from `fused_qkvg_proj` to `qkv_proj` (it turns `do_fuse_qkvbfg` off). A name that doesn't match isn't an error (it silently leaves 45 GB in bf16), so the patch logs what it matched. `qkv_proj: 69` matching the KDA layer count is the signal that it took.

### Where the memory went

The attention conversion on its own bought **+47,488 tokens of KV pool (+5.3%)**. Stacking dp-attention, fixed mamba slots, and a memory-fraction adjustment on top of it took the pool from 888,960 to **2,213,760 tokens, 2.49×**.

That's worth stating precisely, because it's easy to imply the attention change did all of it. It didn't. The single biggest lever was dp-attention, and the reason is structural: MLA can't shard its latent KV within an attention-TP group, so all 16 ranks hold identical copies. Adding tensor parallelism buys you more copies, not more capacity. Splitting into two groups of 8 gives you two independent pools.

![Under TP16, all 16 ranks form one attention group and hold identical copies of the same latent KV, so the pool does not grow with parallelism. With dp-attention set to 2, the ranks split into two groups of 8, each keeping its own KV pool.](/assets/img/k3-dp-attention-kv.png)
*The counterintuitive part: scaling tensor parallelism up buys copies, not capacity. Splitting the group is what grows the pool.*

With the pool out to roughly 2M tokens, the endpoint held steady under a replay of production traffic at 1M tokens per minute, and needle-in-a-haystack passed 12/12 from 8K to 1M at depths of 10, 50 and 90%.

---

## Running it yourself

The checkpoint is public at [`vessl/Kimi-K3-W4AFP8`](https://huggingface.co/vessl/Kimi-K3-W4AFP8). It ships with a patch script, because SGLang's `W4AFp8Config` hardcodes block-wise FP8 for every non-MoE linear and doesn't read `ignored_layers` from config. Upstream work to make that configurable is in flight ([sglang#16643](https://github.com/sgl-project/sglang/pull/16643), [#22806](https://github.com/sgl-project/sglang/pull/22806)); when it lands, the patch goes away.
