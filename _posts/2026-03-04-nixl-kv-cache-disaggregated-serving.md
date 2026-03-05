---
layout: post
title: "NIXL for KV Cache in Disaggregated Serving"
date: 2026-03-04
description: "How NIXL accelerates KV Cache transfer in Prefill/Decode disaggregated LLM serving, its architecture, vLLM integration, and a real-world memory leak debugging story"
tags: gpu networking nixl vllm llm inference rdma
categories: infrastructure
lang: en
toc:
  beginning: true
---

## 1. KV Cache in Disaggregated Serving

In Prefill/Decode (P/D) disaggregated serving, the speed of KV Cache transfer from Prefill to Decode is critical. The per-token KV Cache size is calculated as:

```
Per-token KV Cache = num_layers × 2 (K, V) × num_kv_heads × head_dim × dtype_size
```

For **Llama-3.1-70B** with BF16:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Layers | 80 | Transformer blocks |
| × 2 | K and V | Each layer stores both Key and Value vectors |
| KV Heads | 8 | GQA (Grouped Query Attention) |
| Head Dim | 128 | Vector size per head |
| dtype size | 2 bytes | BF16 precision |

```
Per token = 80 × 2 × 8 × 128 × 2 = 327,680 bytes ≈ 320 KB
```

That's **320 KB per token**, multiplied by the prompt length:

| Prompt Length | KV Cache Size | IB HDR × 1 (200 Gbps, ~25 GB/s) |
|:---:|:---:|:---:|
| 1K tokens | 0.32 GB | ~13 ms |
| 4K tokens | 1.28 GB | ~51 ms |
| 8K tokens | 2.56 GB | ~102 ms |
| 32K tokens | 10.24 GB | ~410 ms |
| 128K tokens | 40.96 GB | ~1,638 ms |

These numbers make it clear: KV Cache transfer speed directly impacts P/D disaggregated serving latency. Several factors make this transfer challenging:

- **GPU-to-GPU memory**: Direct transfer without CPU involvement (GPUDirect RDMA)
- **Non-contiguous memory**: PagedAttention scatters KV Cache across non-contiguous blocks
- **Asynchronous execution**: GPUs must continue processing other requests during transfer
- **Heterogeneous paths**: NVLink, InfiniBand, RoCE, EFA, TCP — the optimal path varies by environment
- **Memory registration**: RDMA requires registering memory with the NIC before transfer, which is expensive

This is exactly what **NIXL** (NVIDIA Inference Xfer Library) is designed to solve — a library for accelerating data transfer in disaggregated serving. It is open-sourced at the `ai-dynamo/nixl` repository. (Note: NCCL is for collective operations like AllReduce and AllGather; NIXL is specifically for point-to-point data movement in serving.)

---

## 2. NIXL Architecture Overview

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog_nixl_1.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="80%" zoomable=true %}
    </div>
</div>
<div class="caption">
    NIXL Architecture — from inference frameworks down to hardware transport layers.
</div>

NIXL's **Transfer Agent** abstracts three core entities:

### 2.1 Memory Section

Memory Section provides a unified abstraction over diverse memory and storage types. HBM (GPU), DRAM (CPU), NVMe SSD, file systems, and object storage are all handled through the same interface.

### 2.2 Transfer Backend Interface

The Transfer Backend Interface abstracts different transport backends from the Agent and selects the most optimal one for each transfer. For example:
- Source is DRAM, destination is VRAM → select **UCX**
- VRAM to parallel file system → select **GPUDirect Storage**

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog_nixl_2.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="80%" zoomable=true %}
    </div>
</div>
<div class="caption">
    NIXL Backend Selection Flow — from memory type identification to final backend selection.
</div>

When a user does not explicitly specify a backend, NIXL's selection engine follows a 4-step process:
1. **Identify memory types** from local and remote descriptors (e.g., VRAM)
2. **Find common backend engines** — intersection of local and remote agent's initialized backends
3. **Verify memory registration** — check if src/dst addresses fall within each candidate's registered ranges
4. **Final selection** — usually only one backend qualifies; if multiple, use preference list

### 2.3 Metadata Handler

The Metadata Handler encapsulates the information needed for data transfer between NIXL Agents running on different workers. It exchanges this information via side channels (HTTP, ZeroMQ, TCP) or centralized metadata services (etcd, Redis).

---

## 3. NIXL Internals

### 3.1 Transfer Agent

A Transfer Agent is the core runtime entity of NIXL. Each Agent is identified by a unique name (e.g., `"prefill_worker_0"`). A single Agent can manage multiple GPUs, and in vLLM's case, one Agent is created per TP (Tensor Parallelism) worker.

An Agent manages:
- **Local memory sections**: Registered VRAM/DRAM/SSD address ranges and per-backend registration state
- **Backend engine instances**: Initialized plugins (UCX, GDS, libfabric, etc.) with their connection info
- **Remote agent metadata cache**: Connection info and memory layout received via `load_remote_md()` — required before sending transfers to a remote Agent
- **In-flight transfer requests**: Status tracking (IN_PROGRESS, DONE, ERROR) of active handles

### 3.2 Memory Section and Descriptors

A **Memory Section** is a collection of memory registered with a NIXL Agent. Each segment is a `(address, length, device_id)` tuple.

| Memory Type | address | device_id | Use Case |
|-------------|---------|-----------|----------|
| VRAM | GPU virtual addr | GPU index | KV Cache blocks |
| DRAM | CPU virtual addr | NUMA node | CPU offload buffer |
| NVMe | file offset | disk index | SSD checkpoint |
| Object Store | bucket/key | S3 endpoint | Object storage |

A **Descriptor List** is the basic unit of transfer requests, coming in two varieties:
- **Registration descriptor**: `(addr, len, devID, optional_str)`
- **Transfer descriptor**: `(addr, len, devID, backend_metadata)`

Importantly, descriptors within a single Descriptor List **can span different GPUs**. A single transfer request can include 3 blocks from GPU 0 + 2 blocks from GPU 1, and NIXL processes them in parallel.

### 3.3 Backend Plugin (South Bound API)

NIXL's backend plugins implement the South Bound API (SB API), and each plugin is loaded on demand.

Currently supported backends (NIXL 0.8.0):

| Backend | Transport Path |
|---------|---------------|
| UCX | RDMA, NVLink, TCP |
| GPUNetIO (DOCA) | GDAKI, CPU bypass |
| Libfabric | EFA, CXI, SRD |
| GDS / GDS_MT | GPUDirect Storage |
| POSIX | AIO, io_uring |
| OBJ | S3 API |
| Mooncake | Third-party transfer engine |

Each plugin implements these core SB API functions:

```
initBackend()      — Initialize backend, establish connections
registerMem()      — Register memory regions (e.g., RDMA MR registration)
deregisterMem()    — Deregister memory
prepXferReq()      — Prepare transfer request (resource allocation)
postXferReq()      — Start transfer (async)
checkXferReq()     — Check transfer status
releaseXferReq()   — Release transfer handle
```

### 3.4 North Bound API (User Interface)

The North Bound API is what inference frameworks (vLLM, etc.) use. It follows a 5-phase lifecycle:

```python
# ─── Phase 1: Backend Initialization ───
agent = NixlAgent("prefill_worker_0", backends=["UCX"])

# ─── Phase 2: Memory Registration ───
# Internally performs RDMA Memory Region registration (pin + physical address mapping).
# Called once at server startup; no need to re-register per transfer.
reg_desc = agent.register_memory(memory_section)

# ─── Phase 3: Metadata Exchange ───
# Serialize local Agent's connection info + registered memory info.
# Send this byte string to the remote Agent via side channel (HTTP, etcd, etc.).
local_md = agent.get_local_md()        # Serialized metadata (bytes)
agent.load_remote_md(remote_agent_md)  # Load remote agent info
# Only after load can you create transfer requests to that remote Agent.

# ─── Phase 4: Create and Execute Transfer ───
# create: backend selection + resource allocation → returns handle (transfer not started yet)
handle = agent.create_xfer_req(local_descs, remote_descs, "decode_worker_0", WRITE)
# post: actually starts the async transfer
agent.post_xfer_req(handle)

# ─── Phase 5: Completion Check (non-blocking) ───
# Poll until transfer completes. GPU is free to do other work in the meantime.
while agent.get_xfer_status(handle) == IN_PROGRESS:
    pass  # Or perform prefill for other requests
# Release handle after completion
agent.release_xfer_req(handle)
```

The key design is that **create and post are separated**. `create` handles backend selection and resource allocation upfront, while `post` triggers the actual data movement. This separation enables pre-allocation of resources and immediate, low-latency transfer initiation.

Combining these three entities, the Transfer Agent accepts **buffer list primitives** (non-contiguous GPU memory address lists from PagedAttention) and returns an **async handle**. All transfers are non-blocking — `post_xfer_req()` returns immediately, and `get_xfer_status()` checks completion — preventing the GPU from sitting idle during transfer.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog_nixl_3.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="80%" zoomable=true %}
    </div>
</div>
<div class="caption">
    NIXL Transfer Request Lifecycle — from composing buffer lists to release.
</div>

**READ vs WRITE**: WRITE means the Prefill Worker pushes its KV Cache directly into the Decode Worker's memory (push model). READ means the Decode Worker pulls from the Prefill Worker's memory (pull model). Both are RDMA one-sided operations requiring no CPU involvement on the remote side. vLLM's NixlConnector uses the WRITE mode.

---

## 4. vLLM NixlConnector — KV Cache Transfer in Practice

```bash
# Prefill instance
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --port 8100 \
  --kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_both"
  }'

# Decode instance
CUDA_VISIBLE_DEVICES=1 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5602 \
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --port 8200 \
  --kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_both"
  }'

# Proxy server (request routing)
python toy_proxy_server.py \
  --port 8192 \
  --prefiller-hosts localhost --prefiller-ports 8100 \
  --decoder-hosts localhost --decoder-ports 8200
```

### 4.1 vLLM's KV Connector Architecture

vLLM's Disaggregated Serving supports various transfer backends through the `KVConnector` abstraction:

```
vLLM KV Connector Plugins:
├── NixlConnector      — NIXL-based, fully async (recommended default)
├── LMCacheConnectorV1 — LMCache library (can use NIXL internally)
├── P2pNcclConnector   — NCCL P2P-based
└── MooncakeConnector  — Mooncake Transfer Engine
```

The NixlConnector is implemented in `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` and operates at two layers:

1. **Scheduler Connector** (EngineCore process):
   - Decides which requests' KV Cache to transfer
   - Transfer scheduling (block mapping, timing)
   - Transfer completion tracking

2. **Worker Connector** (GPU Worker process):
   - Manages the actual NIXL Agent
   - Memory registration and metadata exchange
   - Async transfer execution and status checking

### 4.2 Key Configuration Details

- **`kv_role: "kv_both"`**: A single instance can perform both Prefill and Decode. This provides flexibility — the Proxy server can dynamically decide which instance handles Prefill and which handles Decode.
- **`VLLM_NIXL_SIDE_CHANNEL_PORT`**: Each instance requires a unique port. In TP/DP deployments, each rank needs its own port.

---

## 5. Debugging: UCX Memory Leak in Disaggregated Serving

### 5.1 Symptom

A memory leak was observed that could not be reproduced with a single vLLM instance or a simple UCX test tool. It only occurred when **all** of the following conditions were met:

- vLLM + **NIXL** + **UCX** (InfiniBand) in P/D disaggregated setup
- System memory (RSS) grew linearly at **~400 MB/min** on the **Decode instance only**
- Prefill side was normal → the problem was exclusive to the **receiving side** of KV Cache transfers
- Eventually led to OOM (Out of Memory) after several hours

The fact that it only occurred on the Decode side was the key clue. In P/D disaggregation, the Decode instance receives KV Cache from Prefill via NIXL (→ UCX → InfiniBand RDMA), making the KV Cache transfer path the prime suspect.

### 5.2 Root Cause: UCX's mmap Hooking Mechanism

#### Why UCX hooks mmap

UCX maintains a **Registration Cache (RCache)** for InfiniBand RDMA transfer performance. RDMA requires registering memory with the NIC before transfer, and this registration is expensive. RCache avoids re-registering the same memory regions by caching previously registered mappings.

The problem is: if the application calls `munmap` on a region that's still in the RCache, the physical memory gets reassigned. If UCX later uses the stale cached registration, the NIC would read or overwrite memory that now belongs to something else — a correctness disaster.

So UCX needs to know when the application frees memory. But UCX is just a library — it has no built-in way to know when `munmap` is called. UCX's memory management module (UCM) solves this by **dynamically patching the process's GOT (Global Offset Table)** at runtime, intercepting all `mmap`/`munmap` calls process-wide.

The critical issue: this hooking intercepts **all** mmap/munmap calls in the process, not just UCX-related ones. Python interpreter, PyTorch, and every other library's mmap calls all go through UCX.

```
Normal mmap call path:
  Application → glibc mmap() → kernel syscall

After UCX hooking:
  Application → UCX-hooked mmap() → UCX RCache management → kernel syscall
```

#### Stack trace evidence

**Stack Trace 1**: Python's memory allocator calling mmap, routed through UCX's hook:

```
#0  syscall()
#1  ucm_orig_mmap_syscall()    ← UCX's mmap hook intercept
#4  ucm_mmap()                 ← UCX-intercepted mmap
#5  _PyMem_ArenaAlloc()        ← Python memory manager
```

**Stack Trace 2**: Python calling munmap (memory free) but UCX internally triggering mmap (memory allocation):

```
#0  syscall()
#1  ucm_orig_mmap_syscall()    ← mmap happening INSIDE munmap!
#3  ucm_munmap()               ← Original munmap call
```

`munmap` is supposed to free memory, but during that process, UCX was allocating new memory. This was the source of the leak.

### 5.3 Why mmap Happens Inside munmap

When `munmap` is called, UCX doesn't immediately delete the corresponding RCache entry — it needs to communicate with the NIC, which is expensive. Instead, it puts the entry into an **invalidation queue** to be cleaned up later during `ucp_worker_progress()` calls.

This queue itself is managed by UCX's memory pool (`ucs_mpool`). When entries accumulate and the pool fills up, it calls `mmap` to grow — allocating new memory.

Under normal operation, `ucp_worker_progress()` should drain the invalidation queue. NIXL and vLLM did call this function, but in this specific edge case, the cleanup was not triggered. Combined with `UCX_RCACHE_MAX_UNRELEASED` defaulting to **infinity**, the queue grew without bound.

```
munmap called by Python/PyTorch
  │
  ▼
UCX intercepts → adds RCache entry to invalidation queue
  │
  ▼
Queue grows → ucs_mpool capacity exceeded
  │
  ▼
ucs_mpool_grow() → mmap call (new memory allocation)
  │
  ▼
This new memory also gets hooked by UCX → registered in RCache
  │
  ▼
Queue cleanup never triggers → entries keep accumulating
  │
  ▼
RSS grows linearly (~400 MB/min)
```

### 5.4 The Fix

**Fix 1: Disable mmap hooking**

```bash
export UCX_MEM_MMAP_HOOK_MODE=none
```

This completely disables UCX's mmap/munmap hooking. The memory leak was **immediately resolved**, with **no performance impact**.

Why no performance degradation? vLLM's KV Cache is a **single large contiguous memory block pool** allocated once at server startup. NIXL registers this memory once, and all subsequent transfers use the already-registered region. There's no dynamic memory that needs RCache tracking during steady-state operation.

**Fix 2: Limit invalidation queue size**

```bash
export UCX_RCACHE_MAX_UNRELEASED=1024  # default: infinity
```

This caps the invalidation queue, forcing UCX to start cleanup when the threshold is reached. This preserves hooking while preventing unbounded growth.

**Upstream fixes**: vLLM now includes `UCX_MEM_MMAP_HOOK_MODE=none` as a default setting ([PR #32181](https://github.com/vllm-project/vllm/pull/32181)), and NIXL/UCX decided to change `UCX_RCACHE_MAX_UNRELEASED` from infinity to a finite default value ([NIXL PR #1210](https://github.com/ai-dynamo/nixl/pull/1210)).
