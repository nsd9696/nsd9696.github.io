---
layout: post
title: "Cross-Instance KV Cache Sharing for Disaggregated LLM Serving: Cutting TTFT with Mooncake and LMCache"
date: 2026-03-09
description: "How cross-instance KV cache sharing with Mooncake + LMCache reduces TTFT by 24% in multi-instance disaggregated LLM serving"
tags: gpu vllm llm inference rdma networking
categories: infrastructure
lang: en
toc:
  beginning: true
---

## TL;DR

Disaggregated prefill-decode serving splits LLM inference across multiple instances, but each prefill instance maintains its own isolated KV cache. When the same prompt lands on a different instance, expensive prefill computation repeats from scratch. We implemented **cross-instance KV cache sharing** using Mooncake + LMCache on a multiple prefill-decode vLLM deployment, reducing cross-instance TTFT by **24%** compared to full recomputation.

---

## 1. The Problem: Isolated KV Caches in Multi-Instance Serving

Modern LLM serving at scale requires multiple prefill instances to handle concurrent requests. vLLM's **Automatic Prefix Caching (APC)** efficiently caches KV tensors within a single instance's GPU memory — when the same system prompt arrives again at the same instance, prefill computation is skipped entirely.

But what happens when the **same prompt arrives at a different instance**?

```
Instance A: "You are an expert in..." → computes KV cache → stores in GPU A
Instance B: "You are an expert in..." → GPU B has nothing → recomputes from scratch
```

In a multi-prefill setup, assuming round-robin routing (a load balancing strategy that distributes incoming requests evenly across instances in sequential order), there is a **(N-1)/N chance** that a repeated prompt lands on an instance that doesn't have its KV cache. This means:

- **Wasted GPU compute**: The same prefill computation runs redundantly across instances.
- **Higher TTFT**: Users experience first-token latency equivalent to a cold request, even for prompts the system has seen before.
- **Poor scaling**: Adding more prefill instances actually _reduces_ per-instance cache hit rate.

The core question: **Can we share KV caches across instances so that any instance can serve a cached prompt?**

## 2. What is Cross-Instance KV Cache Sharing?

Cross-instance KV cache sharing creates a **distributed KV cache pool** that spans prefill (and optionally decode) instances. When one instance computes KV tensors for a prompt, those tensors are stored in a shared pool. When another instance receives the same prompt, it fetches pre-computed KV tensors from the pool instead of recomputing them.

The cache hierarchy becomes:

```
Request arrives at Prefill Instance
  1. Check local GPU cache (APC) → fastest (~0.33s)
  2. Check distributed cache pool → fast (~0.60s)
  3. Full recomputation            → slowest (~0.81s)
```

This is analogous to how CPU caches work: L1 (local APC) is fastest but per-core, while L2/L3 (distributed pool) is shared across cores with slightly higher latency.

## 3. Our Solution: Mooncake + LMCache on vLLM

### Architecture Overview

We deployed a **disaggregated serving** setup:

- Multiple prefill instances + multiple decode instances across several nodes
- Each instance with TP=2 (tensor parallelism)
- 16 A100 80GB GPUs total
- NIXL + UCX for prefill-to-decode KV transfer

For cross-instance KV cache sharing, we used vLLM's **MultiConnector**:

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      { "kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both" },
      { "kv_connector": "NixlConnector", "kv_role": "kv_both" }
    ]
  }
}
```

- **LMCacheConnectorV1**: Handles cross-instance prefix caching via Mooncake distributed store
- **NixlConnector**: Handles prefill-to-decode GPU-to-GPU KV transfer via NIXL/UCX

### How MultiConnector Works Internally

On each request, MultiConnector iterates connectors in config order and calls `get_num_new_matched_tokens()` on each. The **first connector that reports a cache hit** is assigned to load KV for that request; all connectors are written to on save.

In our setup, this means LMCacheConnectorV1 (Mooncake) is checked first, and NixlConnector handles prefill-to-decode transfer only when there is no Mooncake hit.

### What is Mooncake?

[Mooncake](https://github.com/kvcache-ai/Mooncake) (mooncake-transfer-engine) is a distributed KV cache pool originally developed by Moonshot AI. It provides:

- **Distributed memory pool**: Aggregates memory across all connected workers
- **Metadata server**: Tracks which KV chunks exist and where they are stored
- **RPC-based coordination**: A master process coordinates chunk registration, lookup, and transfer
- **Chunk-level storage**: KV tensors are split into fixed-size chunks (we used 64 tokens per chunk) for granular caching

### How LMCache Integrates with Mooncake

[LMCache](https://github.com/LMCache/LMCache) is the caching layer that bridges vLLM and Mooncake. The data flow:

1. **On prefill completion**: LMCache stores computed KV chunks to Mooncake via `save_kv_layer` (Batch Put API)
2. **On new request**: vLLM first checks local APC. For tokens not in local APC, LMCache checks Mooncake (Exist + Batch Get API)
3. **On cache hit**: KV tensors are loaded from Mooncake directly into GPU memory, skipping prefill computation

LMCache configuration for Mooncake:

```yaml
local_cpu: False
remote_url: "mooncakestore://<master-ip>:<rpc-port>/"
chunk_size: 256 # default: 256 tokens per chunk
global_segment_size: 3355443200 # default: ~3.125 GB per worker
protocol: "tcp" # default: "tcp"
```

Key parameters:

- **`remote_url`**: Address of the Mooncake master. The `mooncakestore://` scheme tells LMCache to use Mooncake as the distributed backend.
- **`chunk_size`**: Number of tokens per KV cache chunk. Smaller values allow finer-grained cache reuse — shorter prompts can be partially served from cache.
- **`global_segment_size`**: Amount of memory (in bytes) allocated per worker in the Mooncake pool.
- **`protocol`**: Transport protocol used for chunk transfer between workers (`tcp` or `rdma`).



![Cross-Instance KV Cache Sharing architecture with LMCache + Mooncake. Orange path shows cold request flow (full prefill compute), blue dashed path shows cross-instance cache hit flow (KV fetched from Mooncake), and green path shows prefill-to-decode transfer via NixlConnector.](/assets/img/blog_mooncacke_1.png)
*Cross-Instance KV Cache Sharing architecture with LMCache + Mooncake. Orange path shows cold request flow (full prefill compute), blue dashed path shows cross-instance cache hit flow (KV fetched from Mooncake), and green path shows prefill-to-decode transfer via NixlConnector.*



## 4. Cache-Aware Routing

Cross-instance caching alone isn't enough — **routing strategy** determines how often cache hits actually occur.

### vllm-router

[vllm-router](https://github.com/vllm-project/router) is a lightweight, high-performance router built in Rust, developed by the vLLM team. It supports several routing policies:

- **consistent_hash**: Hashes the prompt and routes to a fixed instance. Same prompt always goes to the same instance → local APC handles it → Mooncake is never needed.
- **round_robin**: Distributes requests evenly. Same prompt goes to different instances → Mooncake helps as fallback.
- **power_of_two**: Randomly picks two candidate instances and routes to the one with the shorter queue — low overhead with good load distribution.

vllm-router also natively supports P/D disaggregation, allowing explicit routing of prefill and decode requests to separate instance pools.

**Pros**: Extremely lightweight (Rust-based, minimal overhead), simple to deploy, native P/D disaggregation support, and benchmarks show 25% higher throughput than llm-d in certain workloads.

**Cons**: Routing decisions are stateless with respect to cache — `consistent_hash` maximizes local APC hits but can cause uneven load, while `round_robin` balances load but increases cross-instance misses. No real-time visibility into actual cache state across instances.

### llm-d

[llm-d](https://github.com/llm-d/llm-d) is a Kubernetes-native inference framework that implements **precise prefix-cache aware scheduling** via its External Processing Pod (EPP). vLLM pods publish cache block events (BlockStored, BlockRemoved) via ZMQ, which llm-d aggregates into a global, near-real-time `kvblock.Index`. The EPP's `prefix-cache-scorer` queries this index on every incoming request to route to the instance most likely to have a cache hit:

```yaml
# Example: llm-d's Endpoint Picker configuration
schedulingProfiles:
  - name: prefill
    plugins:
      - pluginRef: queue-scorer # prefer instances with shorter queues
        weight: 1.0
      - pluginRef: prefix-cache-scorer # prefer instances with actual cached blocks
        weight: 1.0
```

**Pros**: Makes routing decisions based on actual real-time cache state across the cluster, not just heuristics. Particularly effective for workloads with shared long prefixes (e.g., RAG, multi-turn conversations). Maximizes local APC hit rate, reducing the need for cross-instance Mooncake fallback.

**Cons**: More complex to operate (requires ZMQ event pipeline, in-memory index, Kubernetes-native deployment). Adds scheduling overhead — benchmarks show higher TTFT than vllm-router in some configurations due to the scoring computation.

### Comparison

| | **vllm-router** | **llm-d** |
| --- | --- | --- |
| Cache awareness | Heuristic (hash/queue-based) | Precise (real-time block index) |
| Deployment | Lightweight, any environment | Kubernetes-native |
| P/D disaggregation | Native support | Supported |
| Operational complexity | Low | High |
| Best for | General serving, high throughput | Workloads with shared long prefixes |

The ideal architecture combines cache-aware routing (to maximize local APC hits) with cross-instance sharing via Mooncake (as a fallback when requests inevitably land on non-optimal instances).

## 5. Related Work

### LMCache + Mooncake (Tencent / LMCache Team)

The [LMCache team's blog](https://blog.lmcache.ai/2025-05-08-mooncake/) describes integrating Mooncake as a distributed backend for cross-process KV sharing. Their setup demonstrated KV cache sharing across multiple vLLM instances, validating the architecture we adopted.

### llm-d Disaggregated PD with KV Cache Sharing

The [llm-d KV Cache blog](https://llm-d.ai/blog/kvcache-wins-you-can-see) demonstrates significant latency improvements through cache-aware scheduling in disaggregated prefill-decode setups. Their EPP router with prefix-cache-scorer achieves high cache hit rates by intelligently routing requests to cache-warm instances.

### vLLM Production Stack

The [vLLM Production Stack](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/sharing-kv-cache.html) documents KV cache sharing across vLLM instances using LMCache with various backends (Redis, Mooncake, etc.), providing the foundational architecture patterns.

### Key Insight from Related Work

A common finding across all implementations: **local GPU APC is always checked first** in vLLM V1. The cache hierarchy is strict — GPU HBM → CPU DRAM → External (Mooncake). This means cross-instance sharing only activates when local APC cannot serve the request, making it a complement to (not replacement for) local caching.

## 6. Benchmark: Measuring the Impact

### Test Design

We designed a controlled benchmark that isolates each caching layer's contribution. Using long system prompts (~3,000 words / ~4,000+ tokens), we sent requests directly to specific prefill instances, bypassing the router:

| Scenario          | Description                      | What's Tested           |
| ----------------- | -------------------------------- | ----------------------- |
| Cold              | First request, no cache anywhere | Baseline prefill cost   |
| Same-instance (APC) | Repeat to same instance        | Local GPU cache         |
| Cross + Mooncake  | Same prompt to different instance | Distributed cache      |
| Cross - No cache  | Unique prompt to different instance | Full recompute (control) |

### Results

| Scenario                                         | Avg TTFT | Speedup vs Cold |
| ------------------------------------------------ | -------- | --------------- |
| **Cold (no cache)**                              | 0.808s   | —               |
| **Same-instance (local APC)**                    | 0.330s   | **2.45x**       |
| **Cross-instance + Mooncake (same node)**        | 0.605s   | **1.34x**       |
| **Cross-instance + Mooncake (different node)**   | 0.599s   | **1.35x**       |
| **Cross-instance - no cache**                    | 0.789s   | 1.02x           |

### Key Takeaways

**Mooncake delivers 24% TTFT reduction** in cross-instance scenarios (0.789s → 0.605s). When a prompt lands on an instance without local cache, Mooncake fetches pre-computed KV tensors instead of recomputing, saving ~0.19s per request.

**Same-node vs different-node: negligible difference** (0.605s vs 0.599s). Even over TCP, cross-node Mooncake transfer is fast enough. This suggests the bottleneck is not network transfer but rather the overhead of chunk lookup and GPU memory operations. Mooncake also supports RDMA by setting `protocol: "rdma"` in the LMCache config (requires RDMA-capable NICs and a compatible Mooncake build), though given the bottleneck is not network-bound, the practical benefit is likely limited except at very high request rates or large chunk sizes.

**Local APC is still significantly faster** (0.330s vs 0.605s). Mooncake adds ~84% overhead compared to local APC, making cache-aware routing critical for maximizing local hits. Mooncake serves as a fallback for requests that inevitably miss local cache.

**Mooncake metrics confirmed actual cross-instance transfer**:

- Batch Put: +12 requests, +110 items (KV chunks stored to pool)
- Batch Get: +12 requests, +156 items (KV chunks fetched cross-instance)
- Memory pool grew by 660 MB during the test

## 7. Conclusion

Cross-instance KV cache sharing with Mooncake + LMCache fills an important gap in multi-instance LLM serving. While local APC handles the common case efficiently, the 24% TTFT improvement in cross-instance scenarios translates to meaningful latency savings at scale — especially as the number of prefill instances grows and per-instance cache hit rates naturally decline.

The optimal setup combines three layers:

1. **Local APC** for same-instance cache hits (fastest)
2. **Cache-aware routing** (e.g., prefix-cache-scorer) to maximize local hits
3. **Distributed KV cache** (Mooncake) as a fallback for cross-instance misses

For production deployments, we recommend starting with cache-aware routing to maximize local APC utilization, then adding Mooncake for the remaining cross-instance misses. The benefit scales with the diversity of prompts and the number of instances — workloads with many shared long prefixes (e.g., RAG with common document contexts) will see the greatest impact.
