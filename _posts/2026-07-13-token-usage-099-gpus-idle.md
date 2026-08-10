---
layout: post
title: "token_usage was 0.99 and the GPUs were idle"
date: 2026-07-13
description: "Five diagnoses we got wrong while debugging a disaggregated LLM serving fleet — token_usage decomposition, max-running vs cache, the death-spiral's leading indicator, the O(N) scheduler feedback loop, and why QP follows hardware while CQ follows software."
tags: llm inference sglang mooncake nixl kv-cache disaggregated-serving rdma prefill-decode
categories: infrastructure
lang: en
toc:
  beginning: true
---

The first time our disaggregated serving fleet fell over, everything on the dashboard looked *busy*. KV cache utilization pinned at 0.99. Queues full. GPUs reporting high occupancy. And yet output tokens per second were sliding toward the floor, first-token latency was climbing past ten seconds, and pushing *more* traffic in made throughput go *down*. That last part should have stopped us cold.

We run a large, long-context MoE model in production, served with prefill/decode disaggregation. We split the two halves of inference across separate pools of workers: prefill workers read the prompt and build the KV cache; decode workers take that cache and generate tokens one step at a time. The cache is shipped between them over RDMA: Mooncake for the cache tier, NIXL for the point-to-point hop. Disaggregation is the right shape for this workload. Prefill is compute-bound and bursty, decode is memory-bound and steady, and pinning them to the same GPU means one is always starving the other. But it also multiplies the number of places a request can get stuck, and the number of dashboards that can lie to you about where.

This is a post about the dashboards lying. Over a few weeks we ran into five wrong diagnoses, one after another. Every one was the obvious reading of a real metric. Every one was wrong, and wrong in a way that a single aggregate number had made look correct. Here's the order we hit them in, what each one really turned out to be, and the one habit that finally got us out.



![The shape of the fleet. Client → cache-aware router → prefill (builds KV) → KV transfer over RDMA → decode (emits tokens), with both pools backed by a shared Mooncake cache. The orange hop is the one every finding traces back to.](/assets/img/blog_pd_disagg_1_arch.png)
*The shape of the fleet. Client → cache-aware router → prefill (builds KV) → KV transfer over RDMA → decode (emits tokens), with both pools backed by a shared Mooncake cache. The orange hop is the one every finding traces back to.*



## 1. The metric that lies: `token_usage 0.99` is not saturation

We started where anyone would. The decode pool was the thing falling behind, and its KV cache utilization (`token_usage` in the metrics) sat at 0.99 and never moved. That is what an out-of-memory service looks like. The pool is full, requests queue because there's nowhere to put their KV, so: add decode replicas, spread the load, done.

We added replicas. It barely helped, and under real load it sometimes made things worse. That was the first sign something was off. If the pool were really full, more pools should have been strictly better. So we went and read what `token_usage` actually measures, and the whole diagnosis fell apart.

SGLang preallocates KV up to `mem-fraction-static` at startup. The number you're watching is the fraction of that *reservation* that's currently reserved, and because the allocator hands that reserved memory out right away, it sits near 1.0 essentially all the time, under load and at idle alike. It tells you how much you set aside. It tells you almost nothing about how much work is happening. We'd been reading a number that sits near the top no matter what, as if it tracked the actual work.

The number that means something only appears when you break the pool into the three regions it's actually divided into:

```
kv_used  +  kv_evictable  +  kv_available  =  max_total
   |            |                |
real work    radix cache       free
             (reclaimable)
```

`kv_evictable` is the radix prefix cache. It looks like "used" memory, but it's reclaimable on demand: when admission needs room it evicts from here first. So it's really a cushion, not pressure. And `kv_used`, the part that's really in use, has to be split one level further: requests that are *actively decoding* versus requests that have been admitted, grabbed their KV slot, and are now sitting there *waiting for their cache to arrive over the transfer*. That second group holds memory it can't yield (it isn't evictable), but it isn't doing any work either.

When we finally measured the split and looked, the picture flipped. The radix cache was 22% of the pool. Preallocated-but-waiting requests were **62%**. And the number of requests actually running a forward pass at any given moment was *three or four*. The GPUs weren't saturated. They were *blocked*. Most of the pool was tied up by requests parked on KV slots waiting for transfers that weren't completing, and decode was left with almost nothing to run.



![The same 0.99, taken apart. The dashboard reads 'the pool is full, add replicas.' The decomposition reads 62% blocked, not busy, and adding replicas does nothing for a transfer bottleneck.](/assets/img/blog_pd_disagg_2_tokenusage.png)
*The same 0.99, taken apart. The dashboard reads 'the pool is full, add replicas.' The decomposition reads 62% blocked, not busy, and adding replicas does nothing for a transfer bottleneck.*



That reframes what "saturated" even means here. Real saturation is `kv_available` at zero *with* the running set large: lots of requests decoding, no headroom. What we had was `kv_available` at zero with the running set tiny: a pool full of waiters. Same 0.99. Opposite problem, opposite fix. One wants more decode capacity; the other wants the transfer unblocked and admission throttled so requests stop piling into slots they'll only sit in.

> There's an embarrassing footnote here. Early on, before we instrumented the real split, someone (fine, it was me) eyeballed the situation and *asserted the radix cache was around 83% of the pool* — which would have pointed at a completely different cause. It was 22%. The estimate wasn't just off, it pointed the wrong direction, and we spent real days chasing it. Decomposing a number you can measure is not optional just because you think you can guess it.

## 2. Turning concurrency *up* quietly turns your cache *off*

With the transfer bottleneck identified, the next symptom moved to the front: cache hit rate. Under load it didn't just dip, it fell off a cliff, from the high 80s into the 30s, and for this traffic, cache hit rate is the thing that matters most. Our prompts are long and heavily prefixed: system preambles, few-shot examples, long conversations replayed turn after turn. A warm prefix cache is the difference between reprocessing 40K tokens and reprocessing 400. When hit rate collapses, every prefill gets dramatically more expensive at exactly the moment the fleet has no spare capacity.

The obvious suspect was the router. If hit rate is falling, requests must be landing on workers that don't have their prefix, a routing problem. We spent a while there. It was a dead end, because the real cause wasn't *where* requests were landing. It was a knob we'd turned up for an unrelated reason: `max-running-requests`.

Here's the mechanism, and it isn't obvious from the flag's name. In the decode pool, the admission controller sets its slot budget like this:

```
slot cap = max_running + pre_alloc + 1
```

A bigger `max_running` means a bigger slot cap, which means admission is *allowed to reserve more of the pool*, and it makes that room by evicting the radix cache. The prefix cache and the concurrency budget draw from the same pool of blocks, and `max_running` decides how much the concurrency budget is allowed to take. Raise it, and you're literally spending your cache to buy admission slots.

We ran it as a clean A/B at a fixed RPS of 50, same hardware, same traffic, only the one flag changed:

| max_running | slot cap | radix left | what happened |
|:---:|:---:|:---:|:---|
| 96 | 113 | ~0.3M tokens | cache hit **collapsed to 29–54%**, generation throughput fell with it |
| 50 | 67 | ~0.7M tokens | cache hit held, and decode did **3.7× more generation work** |

Halving the configured concurrency roughly *doubled* the radix cache we got to keep, held the hit rate, and produced nearly four times the useful output, on identical hardware under identical load.



![Same hardware, same RPS 50, one flag changed. Less configured concurrency, far more work done, because the cache it preserved was worth more than the slots it gave up.](/assets/img/blog_pd_disagg_3_maxrunning.png)
*Same hardware, same RPS 50, one flag changed. Less configured concurrency, far more work done, because the cache it preserved was worth more than the slots it gave up.*



This one took the longest to accept, because it goes against the instinct every serving engineer has: throughput problem → raise concurrency. For prefix-light traffic that reflex is fine. For traffic like ours, where the cache does most of the real work, past a point more concurrency actually *hurts* throughput: each extra admitted request evicts cache that would have saved more compute than the request itself contributes. Turning the knob down was one of the biggest throughput wins we found, and on paper it looks exactly like a downgrade.

## 3. The collapse has an early-warning light. It isn't TTFT.

By now we could see the individual failures, but not yet how they chained. Every incident had the same shape: fine, fine, fine, then a fast slide into the spiral. And the thing we were *alerting* on, first-token latency, always fired last, well after the fleet was already on its way down. We wanted the earliest link in the chain: the one that moves before anything a user can feel.

Watching the queues through a controlled overload, the order of cause and effect finally became clear. And the surprising part is that it's a loop. It feeds itself:



![The self-reinforcing cascade. Stage 1 (the decode transfer queue) is the leading indicator; stages 2–3 lag it; TTFT and cache-hit collapse are the symptom that branches off last.](/assets/img/blog_pd_disagg_4_spiral.png)
*The self-reinforcing cascade. Stage 1 (the decode transfer queue) is the leading indicator; stages 2–3 lag it; TTFT and cache-hit collapse are the symptom that branches off last.*



Trace it forward from the top. Transfer throughput falls behind demand, so the decode-side transfer queue backs up. Because decode can't pull cache fast enough, prefill workers sit holding finished KV with nowhere to hand it, and the bootstrap queue waits. Admitted decode requests, meanwhile, are parked on their KV slots waiting for that same transfer. This is the 62% from finding #1. With the pool full of waiters, decode runs its three or four requests and starves. Starved decode finishes requests slowly, so the number of in-flight requests *grows* instead of draining, which pushes even more onto the transfer queue. And it comes back around.

And it doesn't just circle, it *tightens*. As prefill backs up it starts evicting its own prefix cache to keep admitting, so hit rate collapses, so the KV blobs it produces for each request get *larger*, so the transfers it's shipping to an already-behind decode pool get *heavier*. Every trip around the loop makes the next trip worse. That's why it's not a slowdown, it's a collapse. Past a tipping point it's self-sustaining and won't recover just because the incoming rate eases off.

The takeaway is a single sentence: **alert on the decode transfer queue, not on TTFT.** By the time first-token latency has visibly spiked, you're four steps behind a loop that has already closed. The transfer queue backing up is the light that comes on while you can still do something (shed load, throttle admission) before the spiral has a hold.

## 4. The scheduler gets slower at exactly the moment you need it fast

One thing about the spiral still didn't add up. Even after we relieved the queues, above a certain depth the fleet couldn't recover on its own. Drop the incoming rate to something it had handled comfortably an hour earlier and it would just… stay down: GPUs half-idle, queue refusing to drain. A system that's merely overloaded recovers when you remove the load. This one didn't, which meant the load wasn't the whole story.

Rather than keep reasoning about it, we attached py-spy to a decode worker mid-incident and let the flamegraph show where the time was going. It wasn't in the model. It was in `process_decode_queue()`, the bookkeeping the scheduler does on every tick, and it was O(N) in the depth of the queue. Every tick it polls every queued request, and for each one runs a radix-tree prefix match under a lock (`_match_prefix_and_lock`, the expensive piece: a tree traversal that, for our prompt lengths, walks tens of thousands of tokens deep), then hits two gloo all-reduce barriers to keep the tensor-parallel ranks in agreement. Cheap when the queue is short. Brutal when it's long.



![Poll cost against queue depth, measured with py-spy. As the queue grows the per-tick poll goes from ~1 ms to ~12 ms and GPU utilization falls from 90% to 35%: already halved by ~800, and past ~1,200 the loop can't climb out on its own.](/assets/img/blog_pd_disagg_5_scheduler.png)
*Poll cost against queue depth, measured with py-spy. As the queue grows the per-tick poll goes from ~1 ms to ~12 ms and GPU utilization falls from 90% to 35%: already halved by ~800, and past ~1,200 the loop can't climb out on its own.*



There's the missing piece. This is a self-reinforcing loop, and it lives *inside the control plane itself*, separate from the data-path spiral in finding #3. A longer queue makes each scheduler tick slower; a slower tick takes time away from the GPU's forward pass; less GPU time means decode drains the queue more slowly; a slower-draining queue grows longer. The scheduler is competing with the model for the same seconds, and the busier things get, the more it wins, which is exactly backwards from what you want. The load didn't create the collapse. The *structure* did; the load just pushed it past the point where the feedback tips over.

The fixes, once you can see it this way, are almost boring, and they worked. Skip requests that are already confirmed when you poll, so N shrinks to the part that can actually change. Raise the polling interval so most ticks don't pay for the barrier at all. And shorten the stuck-request timeouts so the queue can't reach the depth where the poll cost blows up in the first place. None of these are clever. All of them attack the same thing: keeping N out of the range where the scheduler starts losing its race against the model.

## 5. RDMA tuning refused to follow the theory

The last piece was the transfer bottleneck at the root of the whole cascade. Mooncake moves the KV cache over RDMA, and it exposes the knobs you'd expect: queue pairs per endpoint (`MC_NUM_QP_PER_EP`) and completion queues per context (`MC_NUM_CQ_PER_CTX`). Transfers weren't keeping up, and the standard move for "RDMA isn't keeping up" is more parallelism: more queue pairs to submit work on, more completion queues to collect the results. So we pushed both to 8. QP=8, CQ=8.

It got *worse*. Measurably, repeatably less stable than the defaults we'd started from. The result was confusing enough that we nearly dismissed it as noise, since more channels are supposed to mean more throughput. But it held across runs, so we had to explain it.

The explanation is physical, and it's the kind of thing a tuning guide can't tell you because it depends on your box. Our engines are TP=2. Each has exactly *two* NIC paths to push data through. Configuring eight queue pairs doesn't create eight independent lanes onto two NICs. It puts eight submission streams into contention over the same two pieces of hardware, and the contention costs more than the parallelism buys. Completion queues are a different story: collecting completions is a *software*-side, CPU-side concern, and spreading that polling across more workers does help. We'd been treating two physically different resources as one knob labeled "parallelism."

> **QP follows the hardware — match it to NIC count. CQ follows the software — match it to worker count.**

QP=2 / CQ=8 was the combination that matched the hardware, and it held up. But the deeper lesson was about what the bottleneck had been the whole time. Transfer *latency* was never the problem: a single transfer's p50 was about 9 ms, perfectly healthy. The problem was concurrent channel *count*: too few transfers in flight at once to meet peak demand, so a queue formed, and the queue is what fed the spiral three sections up. Faster individual transfers wouldn't have helped. More *simultaneous* ones would, up to the point the physical NICs allow, and not one queue pair further.

And we could only find that by measuring it in place. There's no way to reason from the docs to "QP=8 is worse on this box." It depends on NIC count, on the TP layout, on the specific hardware. What worked was a canary: a second pod carrying the alternate config, using the same labels so the router split live traffic across it and the baseline, so we could tell the two apart in the metrics. The theory said QP=8 should win. The canary said otherwise, in production, on our hardware. We believed the canary.

## What actually connected all five

Line the mistakes up and they all look the same. Every one was the reasonable reading of a real, correctly-reported number, and every one was wrong because a single summary number had hidden the structure underneath.

- `token_usage 0.99` → "decode is out of memory" → **blocked on transfers, not full**
- cache hit falling → "the router is misrouting" → **admission evicting the cache**
- TTFT spiking → "we're overloaded" → **a queue four steps upstream**
- won't recover → "it just needs time" → **a control-plane feedback loop**
- slow transfers → "add parallelism" → **NIC contention**

The discipline that got us out isn't clever, and that's sort of the point. Don't accept a top-line metric as a diagnosis. Decompose it into the parts it's hiding, the way `token_usage` had to be split into work, cushion, and waiters before it said anything true. Profile the hot path instead of reasoning about it; py-spy found in an afternoon what we'd argued about for a week. And when the theory and the box disagree, run the canary and believe the box.

The expensive mistakes, looking back, were never the wrong fixes. Those are cheap. You try them, they don't work, you move on. The expensive ones were the confident diagnoses: the days spent optimizing hard against a number we hadn't yet taken apart. The fix was free once we could see the problem. Seeing it was the whole job.
