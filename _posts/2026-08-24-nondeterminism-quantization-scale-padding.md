---
layout: post
title: "temperature was 0 and the answers kept changing"
date: 2026-08-24
description: "Batch invariance is not the only thing that makes LLM inference nondeterministic. Seven rows of leftover memory, one amax that read them anyway, and a top-k router thin enough at the cut that the last bit of the score decides."
tags: llm inference nondeterminism determinism quantization fp8 int4 moe cuda-graph numerics reproducibility
categories: infrastructure
lang: en
toc:
  beginning: true
---

Temperature zero. Same prompt, same weights, same GPU, same process. Run it twice
and the two completions diverge somewhere around token 40.

The well-known explanation for this is batch invariance. Thinking Machines Lab
[laid it out clearly](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/):
your request gets batched with whatever else arrived at the same moment. Batch size
changes how the reduction splits inside matmul, normalization, and attention.
Floating-point addition isn't associative. So the same input row comes out different
depending on who else was in the batch. Fix the kernels to reduce in a
batch-independent order and the problem goes away.

That cause is real and so is the fix. But we lost the most time to a second one that
batch-invariant kernels do nothing about. Every symptom pointed at the kernels, which
turned out to be fine. It came down to **seven rows of leftover memory nobody had
zeroed, one `amax` that read them anyway, and a router with 24.3% of its tokens on a
knife edge.**

![Two schemes for computing an FP8 activation scale, side by side. With a per-tensor scale, every activation row including the padded ones feeds a single scalar s, and s feeds every output row, so stale bytes in the pad region reach the real tokens. With a per-token scale, each row computes its own scale, and the padded rows' scales are discarded along with the padded rows.](/assets/img/nd-scale-coupling.png)
*The padded rows get thrown away in both schemes. Only the right-hand one keeps them out of the arithmetic first.*

---

## Three things have to line up

Batch invariance is about what order the numbers get summed in. This one is about
which numbers go into the sum at all.

You need all three of these at once:

1. **One `amax` over everything:** a dynamic per-tensor quantization scale is the
   clean case. A single number, computed from the whole buffer, that every output
   row gets multiplied by.
2. **Rows you never filled in:** padding, usually. Bytes sitting in the same buffer as
   your real tokens, left over from whatever ran before them.
3. **A top-k underneath:** an argmax, greedy sampling, anything that turns a
   float into a choice.

Any two of these are harmless. All three and your model gives different answers at
temperature zero, with no batch size involved anywhere. Batch-invariant kernels will
not help, because every sum here already runs in the right order. The sum is just
reading bytes that were never yours.

![The five stages from stale bytes to different output. Bucket padding leaves rows 313 to 319 holding whatever was there last time; amax runs over all 320 rows; the scale shifts so all 313 real rows requantize; the top-k router swaps experts ranked 8 and 9; the token is routed elsewhere and the output diverges. Three of the five stages are the three ingredients.](/assets/img/nd-chain.png)
*A few stale bytes at one end, a different answer at the other. The router is what does the rescaling.*

---

## 1. One number for the whole tensor

Take dynamic per-tensor FP8 activation quantization, the standard setup for 4-bit
weight / 8-bit activation GEMMs:

```
s      = amax(X) / 448          # amax over the WHOLE tensor; 448 = FP8 E4M3 max
X_q    = fp8(X / s)
```

The GEMM consumes `X_q`, and the epilogue multiplies the accumulator back by `s`.

Now look at what `s` reads: all of `X`, every row of it. So what row 0 quantizes to
depends on row 319, and on row 7, and on every other row in the batch. Move the
maximum anywhere in the tensor and all 320 rows shift together.

That is a deliberate trade and a cheap one: a single scalar, a single epilogue
multiply, no extra memory traffic. Most W4A8 kernels offer it as the dynamic path,
and it is what you get whenever the checkpoint ships no calibrated scales.

The bill comes due when part of the buffer isn't yours. Every byte in `X` reaches
every output row, including the bytes in rows you never wrote. Fill all 320 rows with
real tokens and you will never notice. The next section is where you stop filling all
320.

---

## 2. Seven rows nobody zeroed

A CUDA graph is captured once and replayed, which is how servers get their per-step
CPU cost down. Capture records a fixed sequence of launches against fixed buffers at
one fixed shape.

"Fixed" is the operative word. Real requests arrive with arbitrary token counts, so
the server captures a set of graphs at bucketed sizes: 256, 288, 320, and so on. When
313 tokens show up, it runs them through the 320-token graph and pads.

That leaves rows 313 through 319 holding whatever the last request left there. They
live in a persistent buffer that gets reused across steps and across requests. Nobody
zeroes them, because nobody needs to. The outputs of those rows are sliced off and
thrown away, and everything downstream knows to skip them.

Everything except `amax`.

So: seven rows of stale bytes go into `amax`, `s` comes out different, and all 313
real tokens quantize differently. The answer you asked for now depends on the request
before yours.

Padding is necessary but not sufficient. In our sweep the 128 bucket stayed
deterministic at every token count from 121 through 128, padding and all. Whether the
leftover bytes are large enough to move the maximum depends on what happens to be
sitting there.

The eager path does not pad. That is what makes this hard to find. Turn graph capture
off and the bug goes away, so you spend the afternoon on graph capture and on the
kernels that only run underneath it. The scale sits in both paths, behaving
identically.

---

## 3. A router does not round, it picks

The odd part was that the error barely moved. FP8 quantization of these activations
sits at 2.2361e-02 relative error, and it stays there while the padding misbehaves. We
pushed on it: hijack the `amax` with one row 64 times too large and the real rows come
out at 2.2397e-02, which is 1.0x normal. Blow that row up 1024 times and it reaches
3.0121e-02, 1.3x.

So the size of the error is not the problem. The problem is that it lands somewhere
different each run. Every value sits inside an FP8 bin, and moving the scale moves the
bin edges, so a number that rounded down last time rounds up this time.

An error that small should have disappeared into the noise, and instead whole
paragraphs changed. So we went looking for something that could do that: a split-K
reduction accumulating in a different order on each launch, or a race between blocks.
That was the wrong place to look.

It was the router. An MoE layer scores each token against every expert and keeps the
best 8. With a few hundred experts, the scores around 8th and 9th place are packed
tight. The more experts you score, the thinner the margin between the last one you
keep and the first one you drop. We measured that margin on a stand-in rather than the real thing: a
random matrix shaped like the production router, with Gaussian inputs.

**24.3% of rows had their 8th and 9th score within one bf16 ulp of each other.**

![Two panels. Left: one row's expert scores sorted descending, with a dashed cut line after rank 8 separating routed from dropped experts, and ranks 8 and 9 highlighted as nearly equal in height. Right: a 10 by 10 grid of squares where 24 are filled, showing that 24.3 percent of rows have their 8th and 9th score within one bf16 ulp of each other, measured on a synthetic router-shaped matrix with Gaussian inputs.](/assets/img/nd-topk-amplifier.png)
*A router does not round. It picks, and near the cut the margin is thinner than the arithmetic that produced it. Synthetic input, not the deployed router.*

Synthetic scores are not real ones, so read that as an order of magnitude rather than
a measurement of the deployed router. It is enough to make the point. For a quarter of
rows the last bit of the score decides which expert runs, and when it flips, that
token gets a different expert in one of its eight slots. The next layer sees a
different vector. Twenty layers on, different words.

Anything that turns a float into a choice throws away how big the error was and keeps
only which side of the line it landed on. Top-k routing is the loudest case, but
greedy decoding is the same trick with a bigger table: `argmax` over the vocabulary,
near-ties whenever the model isn't sure. So you cannot read the size of the cause off
the size of the damage. Different paragraphs can come from one bit.

---

## Which of the two do you have

Batch variance and this one look identical from outside and need different fixes.
Cheapest test first:

1. **Pin the batch.** Send the same request with the same neighbours every time. If
   determinism comes back, it's batch variance, and batch-invariant kernels are what
   you want. If it doesn't, keep going.
2. **Turn graph capture off.** Eager deterministic, graph not? Then it lives in the
   capture, and the fixed buffers are the first place to look.
3. **Zero the padding.** This is the test that separates them, and it is one memset.
   If zeroing the pad rows makes the divergence stop, the padding is in one of your
   reductions. We ran the intervention on one shape: 1019 of 1019 positions differed
   before, 0 of 1019 after. The observational version is cheaper and we have more of
   it, because any request whose token count lands exactly on a bucket pads with
   nothing. All 40 of those cases came out deterministic, no counterexample.
4. **Read a value out of the middle of the graph.** No Python runs during replay. The
   graph is a recorded list of kernel launches, so you cannot print, breakpoint, or
   hook. What you can do is record a copy into a persistent buffer at *capture* time.
   It becomes one of the recorded launches. So it fires on every replay, and you read
   the buffer from the host afterwards.

### Two measurement mistakes

Neither is specific to this bug.

Our first comparison harness reported 22 of 40 cases as nondeterministic. All 22 came
from the harness. The MoE kernel only writes the rows it actually routes, so every
other row in the output buffer kept whatever was there before, and we were diffing
stale memory. Zeroing both trials first took the re-run to 0 of 30.
**Zero the output on both trials before every comparison**, or you are testing your
allocator instead of your kernel.

Then we measured a fresh deployment against a number taken before the deploy and
reported an 8% prefill regression. It was noise between pods. A same-process A/B, both
kernels loaded and alternating on identical inputs, showed the change was never
slower. A cross-deployment comparison picks up every difference between the two
deployments, and the one you care about is the smallest of them.

---

## Give every row its own scale

The fix that goes after the cause rather than the trigger is to stop sharing:

```
s[i]   = amax(X[i, :]) / 448    # one reduction per row
X_q[i] = fp8(X[i, :] / s[i])
```

One scale per token. Row 0 now depends on row 0 and nothing else. The seven pad rows
still hold garbage, and they still get their own scales. Both leave together when
those rows are sliced off. **There is no longer a path from row 313 to row 0.**

Three practical notes on doing it.

The epilogue now takes a vector where it used to take a scalar. In CUTLASS that is a
row or column broadcast in the fusion callback instead of a scalar multiply. Which one
you want depends on the layout. Ours computes the transpose, `D^T = B^T · A^T`, because
the mainloop takes the weight operand first. That puts the token axis on the kernel's
N dimension, so a row broadcast is what you need. Check yours rather than assuming. Getting it
backwards compiles fine and produces garbage, so check against a reference first.

Keep one kernel rather than two. You probably still need the static per-tensor path
for checkpoints that ship calibrated scales. Instead of compiling a second kernel,
give the broadcast a runtime stride. `Stride<_0, bool, _0>` in CUTLASS: stride 0
re-reads the same scalar for every row, stride 1 walks the vector. One compiled
kernel, both behaviours, picked by a flag at launch.

It also came out faster, which we did not expect. One `amax` over the whole tensor has
to combine across blocks, so it costs either two passes or an atomic. One `amax` per
row is one block per row, one pass, nothing to combine. We fused it into the
quantize-and-scatter kernel, which was already reading those rows. Timed in the same
process with CUDA events, 12 runs a point: 20% less MoE time at 256 rows, 45% less at
512, and +1.2% at 4096. Never slower at any size we tried. Accuracy did not move
either. Relative error came out at 2.2361e-02 both ways, because these rows leave a
normalization with their row maxima already equal: max over min across rows was
exactly 1.0.

The nine shapes that had been diverging came back bitwise identical with graph capture
on. Max absolute difference exactly `0.000e+00`, all nine. The four that had always
been deterministic stayed that way.

### If you can't touch the kernel

Sometimes it isn't yours to change. Then:

- **Mask the padding out of the `amax`:** take the max over the valid token range
  only. This gets the padding out. But row 0 still shares a scale with row 312, so
  batch composition still moves your results.
- **Zero the pad rows on every replay:** cheap, and sensible as a second layer
  regardless. It fixes the trigger and leaves the cause, so the next buffer that gets
  padded brings the bug back.
- **Ship calibrated scales:** sidesteps all of it, at the cost of clipping anything
  the calibration set never saw.

---

## Every statistic over a buffer includes the padding

The quantization scale is just where we happened to hit it. The same thing turns up
wherever one number gets computed across an axis that can carry padding.

| The statistic | The axis it runs over |
|---|---|
| `mean` and `var` in a normalization | the sequence dimension |
| the softmax denominator in attention | the key positions |
| a per-block KV dequantization scale | a page that is only half full |
| the activation range during calibration | whichever batch you calibrated on |
| the loss scale in training | the padded positions in a batch |

We did not find bugs in any of these. They are where to look, and the check is the one
from earlier: force the unused region to a known value and see whether the answer stops
moving.

Then look at what sits below. If it's a top-k or an argmax, you don't have a rounding
difference. You have a different answer.

---

## The short version

- Batch-invariant kernels fix what order a sum runs in. They don't fix a sum that is
  reading rows you never wrote.
- A per-tensor scale ties row 0 to row 319. Pad rows count as rows.
- CUDA graph bucket padding is how those rows get there, and it never shows up in
  eager mode.
- A top-k router is the amplifier. On a synthetic stand-in, a quarter of rows sat
  within one ulp of the cut, so the last bit of the score picked the expert.
- Per-row scales cut the tie and ran faster than the global `amax` they replaced: 45%
  less MoE time at 512 rows, never slower anywhere.
- When you test: zero the output buffers before comparing, and benchmark both versions
  in the same process.
