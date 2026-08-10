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

It was not. The model loaded fine and then ran at a speed that made no sense — worse, in some configurations, than we had any right to expect from hardware this size. Working out *why* took us through the MoE runner matrix, a KV cache path that was quietly undoing its own optimization, and eventually to rebuilding the checkpoint ourselves.

This post is about that rebuild: what we changed, what we deliberately left alone, and the several times we were confidently wrong along the way.

![Precision map of the Kimi-K3-W4AFP8 checkpoint. Routed experts — 247,296 of 249,694 weight tensors — became INT4 at group size 128 with FP8 activations. The remaining 0.96% is expanded below: 393 attention projections in FP8, everything else left in bf16.](/assets/img/k3-precision-map.png)
*What each of the 2.8T model's weight tensors became. Experts are 99% of the tensors and tolerate 4 bits; attention mostly stayed as it was.*

---

## Part 1 — Why the original format didn't work

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

### The KV cache had the same disease

While we were profiling, something else stood out. We had FP8 KV cache turned on — the obvious memory win for a long-context model — and it was making things *slower*.

The profile showed dtype casts eating **29.5% of GPU time**. Not the attention math. The casts.

Here's what was happening. FlashAttention-3 has a fast path for FP8 KV, but it's gated on `head_dim <= 256`. MLA's head dimension is 512 + 64 = **576**, so it fails that check every time. The fallback then does this, once per layer:

```python
get_key_buffer(...).to(q.dtype)   # upcast the entire KV pool to bf16
```

The *entire pool*. 332 million elements of it, on every layer, on every forward pass. We measured the element count at 332,513,280 and the pool is 577,216 × 576 = 332,476,416 — a match to within 0.01%.

The part that stings: actual KV utilization at the time was around **2%**. We were converting 100% of the pool to use 2% of it — roughly fifty times more work than necessary.

![Why FP8 KV cache was slower on Hopper. MLA's head dimension of 576 fails FlashAttention-3's fp8 fast path, which is gated on head_dim of 256 or less. The fallback upcasts the entire 332-million-element KV pool to bf16 once per layer, while only 2% of the pool is in use — 50 times more conversion than needed, costing 29.5% of GPU time.](/assets/img/k3-fp8-kv-cast.png)
*One gate check, missed by a factor of 2.25, and nearly a third of GPU time went to moving data between formats.*

That one had a configuration fix. Switching the prefill attention backend to `flashmla` removed the cast entirely, because flashmla delegates ordinary extends to the parent implementation and handles target-verify with an in-kernel `descale_k`. The 29.5% dropped to 2.2%.

But it made the larger point clear. **If you want FP8 KV without paying to undo it, the compute has to be FP8 too.** Which meant the weights had to change.

---

## Part 2 — Choosing what to quantize

### W4AFP8: 4-bit storage, 8-bit math

The recipe we landed on stores routed MoE expert weights as INT4 at group size 128 with per-group scales, and pairs them with static per-tensor FP8 (E4M3) activation scales carried in the checkpoint. SGLang's `w4afp8` path collapses those into one activation scale per layer at load time, so it fits that serving stack particularly well.

The interesting decision wasn't the format. It was the scope.

### Why only the experts

Out of 249,694 weight tensors, we converted 247,296 — all of them routed experts. Everything else stayed as it was.

| Module | Tensors | Precision | Reasoning |
|---|---|---|---|
| routed experts | 247,296 | **INT4** + FP8 act | Only a few activate per token; errors average out |
| attention (MLA) | 1,020 | bf16 | **100% exposure** — every token, every layer |
| shared experts | 276 | bf16 | Always active |
| vision tower, lm_head | 172 | bf16 | Small win, real risk |
| norms and the rest | 930 | bf16 | — |

The asymmetry is the whole argument. In a MoE model, a routed expert sees maybe a few percent of tokens, and a quantization error in one expert gets diluted by the fifteen others that weren't picked. Attention has no such mercy — every token passes through every attention layer, and the errors compound rather than cancel.

That's also why experts are the right target economically. They're 99% of the tensors and nearly all of the parameter mass, so you get almost the entire size reduction from the part of the model that tolerates it best.

### The requantization error was smaller than we expected

We measured the conversion error and got **8.11%**. Then we measured a control — taking the same tensors from bf16 and quantizing straight to INT4 — and got **10.8–12.2%**.

Going through MXFP4 first was *better* than starting from full precision. That seemed backwards until we remembered that Kimi-K3 is a QAT model. The MXFP4 grid isn't a lossy approximation of some truer bf16 value; the training process put the weights on that grid deliberately. Moving from one 4-bit grid to another is a smaller step than moving down from 16 bits.

We also tested whether calibration-based methods like AWQ would help. They didn't — the difference from plain round-to-nearest wasn't statistically significant (the 95% confidence interval included zero). So we skipped it.

---

## Part 3 — Proving it didn't break anything

This is the part that took the longest, and where most of our mistakes lived.

### The bar: 2σ, not "looks about the same"

We set the pass criterion up front: every benchmark had to land within **two standard errors** of the published Kimi-K3 number. Two standard errors of the binomial at that benchmark's sample size — not a flat percentage.

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

### Model card numbers are not benchmarks

The single most expensive lesson: **you cannot use a model card figure as a control until you know what it measured.**

MMMU-Pro has three subsets — standard 4-option, standard 10-option, and vision. All three have exactly 1,730 items, so you can't tell them apart by count. We started with 4-option, scored 95%, and briefly felt very good about ourselves. It's the easiest of the three. We moved to 10-option. Also not it. The official number turned out to be **vision**, which we only pinned down by cross-referencing a third-party reproduction that reported both MMMU and GPQA from the same run.

Kimi-K3's card also lists `81.6 / 83.4` for MMMU-Pro. We assumed those were config variants. They're **without and with tool augmentation** — so for a harness without tools, only 81.6 is a valid control.

### You have to read the official eval code

We kept finding differences that no README mentioned.

SciCode's official `gencode.py` injects scaffolding code into three of the steps. If you don't provide it, 22 of 288 sub-steps fail in a cascade, and you'd score 7.6 points lower for reasons that have nothing to do with your model.

Worse, there's a flag called `with_background` whose name reads backwards from what it does. It *adds* annotated background to the prompt rather than replacing anything, and the official control has it **on**. We had turned it off, reasoning that removing extra context was the conservative choice. It wasn't — it was a handicap we'd applied to ourselves, and we nearly attributed the resulting 15.6-point drop to quantization.

The general shape of that mistake: **you can't judge which direction a bias runs until you know exactly what the control did.**

### Two more traps worth naming

**Re-running only the failures is one-directional.** On Terminal-Bench we re-ran failed tasks and took the latest result. Sixteen of thirty flipped — and every single flip was a gain, fourteen up and zero down. Of course it was: we only gave the failures another chance. The fix is stratified correction, re-running a sample of the passes too.

**Truncation is measurement failure, not a wrong answer.** When a response hits the token limit mid-reasoning, scoring it as incorrect understates the model. We treat truncated items separately and re-run them, being careful not to double-count them in both the "correct" and "re-run correct" buckets.

---

## Part 4 — The results, and the second round

With MoE quantized, we benchmarked against the original at identical settings — concurrency 8, ~1,530-token prompts, 256-token outputs:

| Metric | MXFP4 | **W4AFP8** | Change |
|---|---|---|---|
| TTFT p50 | 0.924s | **0.611s** | **−33.9%** |
| TPOT p50 | 23.21ms | **21.32ms** | **−8.1%** |
| per-request TPS p50 | 43.1 | **46.9** | **+8.8%** |
| total throughput | 277.7 tok/s | **327.5 tok/s** | **+17.9%** |
| DSPARK accept length | ~3.4 | 3.41 | unchanged |

Four out of four improved, which is what you'd hope for when the MoE GEMM moves from bf16 tensor cores to FP8 ones.

The unchanged accept length is worth a moment. Speculative decoding works by having a small draft model propose several tokens that the main model then verifies in one pass. If quantization shifted the target model's output distribution at all, draft and target would start disagreeing and that number would drop. It didn't move. That's independent evidence — nothing to do with benchmarks — that the model still behaves like itself.

### Round two: the attention weights

Months later we came back for the memory. Attention weights and the KV cache compete for the same GPU memory, so shrinking one grows the other. We converted 393 attention tensors to FP8 — 96% by size, 64.5 GB of 67.4 GB.

Not 100%, and the missing 4% is the interesting part.

`kimi_k3.py` has code that reaches past the quantization layer and reads `.weight` directly. Normally you'd never notice, because every access goes through the module call, which knows about the storage format. But the FP8 path stores weights **transposed** to satisfy `scaled_mm`, and a direct reader gets a flipped tensor and believes it's the original:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (384x128 and 768x128)
  kimi_k3.py  forget_gate = gemm(bfa[..., :n_fa], self.f_b_proj.weight)
```

We found this by converting everything and watching it crash. Then we audited every raw `.weight` access in the model code and excluded the five projections that had one: `kv_b_proj`, `q_b_proj`, `f_b_proj`, `f_a_proj`, `b_proj`. Together they're 2.4 GB we left on the table, and recovering them would mean patching the model code, not the checkpoint.

Our first audit missed one, incidentally. We grepped for accesses starting with `self.` and didn't catch `self_attn.kv_b_proj.weight`. Don't assume the receiver's name.

### Two things we were wrong about

**"It's per-tensor."** We wrote that in the patch comments and repeated it for weeks. We'd seen `weight_block_size=None` and concluded that meant per-tensor. It doesn't — it means *not block-quantized*, and inside that branch the cutlass path picks **per-channel**. The weights were per-channel the whole time. Only the activations are per-tensor.

Ironically, when we later needed to bake the scales into the checkpoint, the serialized path *does* expect per-tensor, and we measured whether that hurt. It didn't: on a tensor where row-wise amax varied by 26×, relative error went from 2.638% to 2.647%. FP8 has an exponent field, so a 26× spread fits comfortably inside e4m3's ~2¹⁵ normal range. Per-channel scaling is insurance here, not the load-bearing part — that intuition comes from INT8, where fixed step size makes per-tensor genuinely destructive.

**"An allowlist is safe because it fails loudly."** It doesn't fail at all. If a module name doesn't match, nothing gets quantized and nothing errors. You just quietly get the unoptimized model and conclude the optimization was worthless.

We caught this because we'd added a counter that logs what actually matched. When we enabled dp-attention, `do_fuse_qkvbfg` turned off, the fused QKV module got renamed from `fused_qkvg_proj` to `qkv_proj`, and 45 GB — the single largest item on the list — would have silently reverted to bf16. The log line showing `qkv_proj: 69`, matching the KDA layer count exactly, is what told us.

### Where the memory went

The attention conversion on its own bought **+47,488 tokens of KV pool (+5.3%)**. Stacking dp-attention, fixed mamba slots, and a memory-fraction adjustment on top of it took the pool from 888,960 to **2,213,760 tokens — 2.49×**.

That's worth stating precisely, because it's easy to imply the attention change did all of it. It didn't. The single biggest lever was dp-attention, and the reason is structural: MLA can't shard its latent KV within an attention-TP group, so all 16 ranks hold identical copies. Adding tensor parallelism buys you more copies, not more capacity. Splitting into two groups of 8 gives you two independent pools.

![Under TP16, all 16 ranks form one attention group and hold identical copies of the same latent KV, so the pool does not grow with parallelism. With dp-attention set to 2, the ranks split into two groups of 8, each keeping its own KV pool.](/assets/img/k3-dp-attention-kv.png)
*The counterintuitive part: scaling tensor parallelism up buys copies, not capacity. Splitting the group is what grows the pool.*

What we ended up with:

| | |
|---|---|
| KV pool | 2,213,760 tokens |
| longest single request | 1,048,487 tokens (99.99% of the model max) |
| two concurrent 900K requests | 1,800,178 tokens, both completed |
| production traffic replay | 1,189,102 tokens/min, zero errors |
| needle-in-haystack, 8K–1M | 12/12 at depths 10/50/90% |

---

## What we'd tell ourselves at the start

**The bottleneck is rarely the number you're staring at.** FP8 KV cache looked like a memory optimization that had gone wrong. It was actually a dtype cast doing fifty times more work than needed, in a fallback path we didn't know existed, because of a head-dimension check we'd never read.

**Silent failures are the expensive ones.** Every crash we hit cost an afternoon. The two near-misses that would have cost weeks — a 45 GB tensor quietly staying in bf16, and a scale broadcast that would have loaded fine and produced garbage — both failed *quietly*. Now we log what matched, not just what broke.

**Your control needs as much scrutiny as your candidate.** More of our accuracy debugging went into understanding what the published numbers measured than into measuring our own. Subset, tooling, prompt scaffolding, judge — get all four wrong and you'll confidently attribute an evaluation artifact to your quantization.

The checkpoint is public at [`vessl/Kimi-K3-W4AFP8`](https://huggingface.co/vessl/Kimi-K3-W4AFP8). It ships with a patch script, because SGLang's `W4AFp8Config` hardcodes block-wise FP8 for every non-MoE linear and doesn't read `ignored_layers` from config. Upstream work to make that configurable is in flight ([sglang#16643](https://github.com/sgl-project/sglang/pull/16643), [#22806](https://github.com/sgl-project/sglang/pull/22806)); when it lands, the patch goes away.
