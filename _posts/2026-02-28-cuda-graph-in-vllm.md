---
layout: post
title: "CUDA Graph in vLLM: Eliminating CPU Overhead in LLM Inference"
date: 2026-02-28
description: "How CUDA Graph reduces CPU launch overhead in LLM decode, memory management with Private Pools, and vLLM's graph capture modes"
tags: cuda gpu vllm llm inference optimization
categories: infrastructure
lang: en
toc:
  beginning: true
---

## 1. Why CUDA Graph? — CPU Launch Overhead

Launching a single GPU kernel requires significant preparation on the CPU side. The host driver must load the kernel function, perform various validations, configure grid/block dimensions, allocate memory accordingly, and submit the kernel to a stream. While this overhead is only on the order of milliseconds per kernel, in patterns like LLM decode — where hundreds of short kernels execute in rapid succession — the cumulative CPU-side overhead causes substantial GPU idle time.

```
Traditional Execution (Sequential Kernel Launch):

CPU:  [Launch A][Launch B][Launch C][Launch D][Launch E] ...
GPU:  ──[A]──gap──[B]──gap──[C]──gap──[D]──gap──[E]──
            ↑       ↑       ↑       ↑       ↑
         GPU idle between each kernel launch
```

CUDA Graph solves this by **capturing** the entire sequence of kernel launches and then **replaying** it in one shot via `cudaGraphLaunch()`. Instead of the CPU issuing individual launch commands for each kernel, the driver submits the entire pre-recorded execution plan at once, virtually eliminating the per-kernel launch overhead.

```
CUDA Graph Execution (Single Graph Launch):

CPU:  [Graph Launch]
GPU:  ──[A][B][C][D][E]──
         ↑
      All kernels dispatched at once, no gaps
```

### Node Types in a CUDA Graph

A CUDA Graph is a DAG (Directed Acyclic Graph) where each node represents an operation. CUDA supports several node types:

| Type | API | Parameters |
|------|-----|------------|
| **Kernel** | `cudaGraphAddKernelNode` | Function pointer, grid/block dim, shared memory, kernel args |
| **Memcpy** | `cudaGraphAddMemcpyNode` | src/dst pointer, size, direction (H2D/D2H/D2D) |
| **Memset** | `cudaGraphAddMemsetNode` | dst pointer, value, size |
| **Host** | `cudaGraphAddHostNode` | CPU callback function and user data |
| **MemAlloc** | `cudaGraphAddMemAllocNode` | Allocation size, attributes (CUDA 11.4+) |
| **MemFree** | `cudaGraphAddMemFreeNode` | Target pointer |
| **Child Graph** | `cudaGraphAddChildGraphNode` | Sub-graph |

---

## 2. CUDA Graph DAG in LLM Decode

To visualize what a CUDA Graph DAG looks like for the LLM decode stage, consider the following pipeline. Each box represents a graph node with its operation type, kernel function, launch configuration, and arguments:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog_cuda_graph_1.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="80%" zoomable=true %}
    </div>
</div>
<div class="caption">
    CUDA Graph DAG for a single Transformer decoder layer during inference. Root nodes (Memset and H2D Memcpy) have no dependencies and can execute concurrently. RoPE for Q and K also fork in parallel before joining at the Flash Attention node.
</div>

Key observations:

- **Root nodes execute concurrently**: The Memset (zero output buffer) and Memcpy H2D (token embeddings) nodes have no dependencies and can run in parallel.
- **Fork/Join parallelism**: RoPE for Q and K are independent and execute on parallel branches before synchronizing at the Flash Attention node.
- **The entire pipeline is a single graph launch**: From embedding transfer to logits D2H copy, every operation is captured as one graph and replayed with a single `cudaGraphLaunch()` call.

---

## 3. CUDA Graph Lifecycle

The lifecycle of a CUDA Graph consists of three phases:

1. **Definition**: Create a graph template (`cudaGraph_t`). This is the "blueprint" phase where you define what operations to perform, their parameters, and their dependency relationships.
2. **Instantiation**: Convert the graph template into an executable graph (`cudaGraphExec_t`). During this step, the driver performs snapshot validation, resource allocation, and other preparations.
3. **Execution**: Submit the executable graph to a stream via `cudaGraphLaunch()`.

### Handling Dynamic Parameters

In practice, CUDA Graphs must deal with changing conditions. During decode, the number of requests being processed (batch size) changes, which alters grid dimensions. Parameters like `learning_rate` (training) or `temperature` (inference) can also vary between iterations. Re-capturing and re-instantiating the graph every time would negate the performance benefits.

CUDA provides **graph update** mechanisms for this. `cudaGraphExecUpdate()` allows modifying parameters of an already-instantiated executable graph (e.g., kernel arguments, grid dimensions) without going through the full capture-instantiate cycle again — as long as the graph topology remains the same.

### Graph Construction Methods

#### Explicit API

Create an empty graph with `cudaGraphCreate()` and add nodes one by one with `cudaGraphAddKernelNode()`. This gives fine-grained control over graph structure but is impractical when hundreds of kernels are dynamically determined at runtime.

```c
cudaKernelNodeParams kParams = {0};
kParams.func = (void*)myKernel;       // kernel function pointer
kParams.gridDim = dim3(blocks, 1, 1);
kParams.blockDim = dim3(256, 1, 1);
kParams.sharedMemBytes = 0;
void* args[] = {&d_out, &d_in, &size};
kParams.kernelParams = args;           // pointer addresses copied by value

cudaGraphNode_t node;
cudaGraphAddKernelNode(&node, graph, NULL, 0, &kParams);
```

#### Stream Capture

Stream Capture records all CUDA operations submitted to a stream during a capture region and automatically builds the graph. This is the method used in practice for deep learning frameworks:

```python
# PyTorch stream capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)   # All CUDA ops inside are recorded
g.replay()                         # Replay the captured graph
```

### What Does a Captured Graph Contain?

Each kernel node in a captured graph stores:

1. **Kernel function pointer** (`void* func`): Points to the memory address of the loaded kernel binary — identifies which GPU function to call.
2. **gridDim**: Number of thread blocks.
3. **blockDim**: Number of threads per block.
4. **sharedMemBytes**: Shared memory size.
5. **Arguments**: All kernel arguments, including both scalar values and pointers.

---

## 4. Memory Management in CUDA Graph: Private Pools

### The Problem with PyTorch's Default Allocator

PyTorch uses `CUDACachingAllocator` instead of raw `cudaMalloc()` calls, because the traditional allocation path (CPU -> OS kernel -> GPU driver -> GPU hardware) is inherently slow. The caching allocator retains freed GPU memory in an internal pool rather than returning it to the OS, enabling fast reuse.

However, this creates a problem when combined with CUDA Graphs:

```
1. CUDA Graph Capture:
   with torch.cuda.graph(g):
       temp = model(input)      # temp allocated at 0x...C000
       output = process(temp)   # output allocated at 0x...D000
       del temp                 # 0x...C000 returned to pool

   -> Graph records addresses: 0x...C000 (temp), 0x...D000 (output)

2. After capture, normal code runs:
   something = torch.randn(same_size, device='cuda')
   -> Pool searches for free block -> finds 0x...C000 -> reuses it
   -> something's address = 0x...C000

3. Graph replay:
   g.replay()
   -> Graph writes temp data to 0x...C000
   -> CORRUPTS 'something' tensor!
```

The graph "remembers" specific memory addresses from capture time, but the general-purpose pool doesn't know those addresses are reserved.

### The Solution: Private Pools

To solve this, CUDA Graph capture uses **Private Pools** — dedicated memory pools that are exclusively reserved for graph operations:

```
┌───────────────────────────────────────────┐
│              GPU Memory                    │
│                                            │
│  ┌──────────────────┐  ┌────────────────┐ │
│  │ Global Pool       │  │ Private Pool   │ │
│  │                    │  │ (Graph-only)   │ │
│  │ General tensor     │  │                │ │
│  │ allocation         │  │ External       │ │
│  │ Anyone can reuse   │  │ access blocked │ │
│  │                    │  │ Only graph can │ │
│  │                    │  │ use this pool  │ │
│  └──────────────────┘  └────────────────┘ │
└───────────────────────────────────────────┘
```

With Private Pools:

```
1. Capture:
   temp = model(input)      # Private Pool allocates 0x...C000
   output = process(temp)   # Private Pool allocates 0x...D000
   del temp                 # 0x...C000 returned to Private Pool

2. After capture, normal code:
   something = torch.randn(same_size, device='cuda')
   -> Searches Global Pool only -> Private Pool excluded
   -> Allocates 0x...B000 from Global Pool
   -> 0x...C000 is untouched

3. Graph replay:
   g.replay()
   -> 0x...C000 is still graph-exclusive -> safe to use
```

---

## 5. CUDA Graph in vLLM

### vLLM's Memory Architecture

vLLM uses PagedAttention for efficient KV Cache management. The KV Cache is managed as fixed-size blocks, and the **base pointers** of these blocks are allocated once during model initialization and remain fixed. What changes between iterations is only the **block table** — the mapping of which logical blocks point to which physical blocks. Since the base pointers don't change, they are safe to hardcode in a captured CUDA Graph.

### vLLM's Graph Dispatch Architecture

vLLM caches graphs per batch size. On a cache miss, it captures a new graph; on a cache hit, it updates static buffers and replays.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog_cuda_graph_2.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="80%" zoomable=true %}
    </div>
</div>
<div class="caption">
    vLLM CUDA Graph execution stack. The CudagraphDispatcher analyzes each batch and selects the runtime mode. The CUDAGraphWrapper manages per-batch-size graph caching, capturing new graphs on cache miss and replaying on cache hit.
</div>

### Why Attention Is Problematic for CUDA Graphs

Attention operations in LLMs have several properties that conflict with CUDA Graph's static nature:

1. **Variable sequence lengths**: In prefill, input lengths differ per request, causing the attention kernel's grid dimensions to change.
2. **Dynamic KV Cache indexing**: PagedAttention's block addresses and block tables change between iterations. Capturing the entire attention within a single graph requires padding inputs to remove variability — but this is only supported by certain attention backends.

### Flash-Attention-2 vs Flash-Attention-3

**Flash-Attention-2**:
- Grid dimensions change based on the requested sequence lengths.
- Variable-length requests are packed using a `cu_seqlens` array, and the grid dimension varies with the array length.
- As a result, **prefill with Flash-Attention-2 cannot be graph-captured**.

**Flash-Attention-3**:
- Designed from the ground up with CUDA Graph compatibility in mind.
- Uses a **fixed maximum grid dimension** regardless of actual sequence lengths.
- When the actual workload is smaller than the maximum, the remaining threads simply pad/idle.
- This enables **full graph capture even during prefill**.

### vLLM's CUDA Graph Modes

vLLM provides five compilation modes to handle the tension between graph capture and attention flexibility:

| Mode | Behavior | Best For |
|------|----------|----------|
| **NONE** | Eager execution (no graphs) | Debugging |
| **PIECEWISE** | Captures everything except attention | Most attention backends |
| **FULL** | Captures entire forward pass as one graph | Compatible backends (e.g., FlashAttention-3) |
| **FULL_DECODE_ONLY** | Full graph for decode only, no graph for prefill | P/D disaggregated serving (saves memory) |
| **FULL_AND_PIECEWISE** | Full graph for decode, piecewise for prefill | **vLLM v1 default** — best overall performance |

**PIECEWISE** mode splits the forward pass at attention boundaries. Everything before and after attention is graph-captured, while attention itself runs eagerly. This provides most of the CUDA Graph benefits while maintaining compatibility with backends that have dynamic grid dimensions.

**FULL** mode captures the entire forward pass including attention. This requires an attention backend that supports fixed grid dimensions (like FlashAttention-3).

**FULL_DECODE_ONLY** captures a full graph only for decode steps and runs prefill without any graph. This is ideal for Prefill/Decode disaggregated deployments where the decode instance never needs to handle prefill.

**FULL_AND_PIECEWISE** is the **default in vLLM v1**. It uses full graph capture for decode (where batch size is the only variable and sequence length is always 1) and piecewise capture for prefill (where sequence lengths vary). This combination works with most attention backends and delivers the best overall performance.
