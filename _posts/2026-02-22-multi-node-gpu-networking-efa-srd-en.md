---
layout: post
title: "Multi-Node P/D Disagg vLLM Serving: How EFA Works Compared to InfiniBand?"
date: 2026-02-22
description: "Multi-node GPU communication on AWS EFA, InfiniBand vs EFA comparison, and vLLM P/D Disagg setup"
tags: gpu networking efa infiniband rdma vllm
categories: infrastructure
lang: en
toc:
  beginning: true
---

## 1. Multi-Node GPU Requirements

When GPUs communicate within a single node, NVLink provides hundreds of GB/s of bandwidth. NVLink enables direct point-to-point connections between GPUs, allowing them to communicate without going through the CPU.

The problem arises when you go multi-node — the bottleneck shifts to the inter-node network. This is especially critical for AllReduce in Tensor Parallelism during LLM serving and KV cache transfer in Disaggregated Serving (Prefill/Decode separation). Naturally, TCP/IP can't meet these demands.

### The Limitations of TCP/IP

TCP requires every packet to traverse the kernel network stack. This triggers a chain of system calls, context switches, protocol processing, and buffer copies, making it extremely slow. Unnecessary memcopy operations also occur in the process.

```
TCP/IP Path (Traditional):
GPU → cudaMemcpy → CPU RAM → send() → Kernel → NIC
  → Network →
NIC → Kernel → recv() → CPU RAM → cudaMemcpy → GPU

RDMA + GPUDirect Path:
GPU HBM → NIC (Direct DMA from GPU memory)
  → Network →
NIC → GPU HBM (Direct DMA)
```

RDMA (Remote Direct Memory Access) bypasses the CPU and kernel, allowing the NIC to directly access memory. OS Bypass eliminates system call and context switch overhead. GPUDirect RDMA takes this a step further by enabling the NIC to perform DMA directly to GPU memory (HBM).

### InfiniBand

InfiniBand is what made GPUDirect RDMA possible. With each generation — HDR (A100 standard) and NDR (H100 standard) — per-port bandwidth roughly doubles (200Gbps → 400Gbps).

---

## 2. EFA and SRD

### EFA (Elastic Fabric Adapter)

EFA (Elastic Fabric Adapter) is a high-performance network interface designed by AWS — think of it as something similar to InfiniBand. It's available on specific instance types (p4d, p5, p6, etc.) and provides the OS bypass and RDMA capabilities described above.

### Ethernet vs InfiniBand

The biggest difference with EFA is that it operates on standard Ethernet fabric. InfiniBand is a single-vendor ecosystem from NVIDIA (Mellanox) — switches, NICs, and cables all need to be Mellanox products, with virtually no alternatives. Ethernet equipment, on the other hand, is available from multiple vendors like Broadcom and Intel, making it commodity hardware.

### RDMA Transport Modes

| Mode                                 | Dedicated Connection             | Delivery Guarantee | Order Guarantee   | Description                           |
| ------------------------------------ | -------------------------------- | ------------------ | ----------------- | ------------------------------------- |
| **RC** (Reliable Connection)         | Required (dedicated QP per peer) | O                  | O                 | Default InfiniBand transport          |
| **UD** (Unreliable Datagram)         | Not required (single QP)         | X                  | X                 | Lightest weight                       |
| **RD** (Reliable Datagram)           | Not required (single QP)         | O                  | O                 | Theoretically ideal but unimplemented |
| **SRD** (Scalable Reliable Datagram) | Not required                     | O                  | X (SW reordering) | Designed by AWS                       |

```
RC (Reliable Connection):
Server A ──dedicated link──→ Server B
Server A ──dedicated link──→ Server C
→ Requires a dedicated QP (Queue Pair) per peer

UD (Unreliable Datagram):
Server A ──link──→ can send to anyone
→ No dedicated link needed, one QP for all peers

RD (Reliable Datagram):
Server A ──link──→ can send to anyone
→ No dedicated link + delivery guarantee + order guarantee
```

### Why RD Is Not Actually Used

In practice, InfiniBand doesn't really use RD. The reason comes down to hardware implementation complexity.

- **RC** only needs to track state for one peer per QP, and **UD** doesn't need state tracking at all since it's fire-and-forget.
- But with **RD**, the NIC must track state for every communication peer:
  - "Sent up to packet 3 to Server B, received ACK up to 2"
  - "Sent up to packet 7 to Server C, received ACK up to 5"
  - "Sent packet 1 to Server D, no ACK yet"
  - "Sent up to packet 12 to Server E, received ACK up to 10"
  - ... × every communication peer
- For example, if a single QP communicates with 1,000 servers, the NIC must manage 1,000 states. These need to be stored in the NIC chip's SRAM, which is expensive and small.
- So from NVIDIA's perspective, RC and UD were sufficient, there was no demand for RD, and the cost-benefit didn't make sense — so it was deprecated.

### SRD (Scalable Reliable Datagram)

So what's different about AWS's SRD? **It gives up the ordering guarantee from RD.**

```
RD:   Packet 1 → Packet 2 → Packet 3  (must arrive in order)
      Path: A ━━━━━━━━━━━━━→ B  (single path)

SRD:  Packet 1 ──Path A──→ ┐
      Packet 2 ──Path B──→ ├→ Software reorders after arrival
      Packet 3 ──Path C──→ ┘
```

By giving up ordering, packets can be sprayed across multiple paths. If one path is congested, packets can take another — this is advantageous in Ethernet environments. In environments where thousands of servers share the network, SRD provides stability, which is why it was adopted.

---

## 3. EFA → GPUDirect RDMA

When using EFA, the GPUDirect RDMA flow looks like this:

{% include figure.liquid loading="eager" path="assets/img/efa_blog_1.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="80%" zoomable=true caption="EFA GPUDirect RDMA Flow" %}

Of course, intra-node GPU communication still uses NVLink.

### InfiniBand vs EFA Performance Comparison

So does EFA actually deliver comparable performance to InfiniBand? Here's a summary of benchmark comparisons:

| Metric                        | InfiniBand              | EFA/Ethernet               | Conclusion         |
| ----------------------------- | ----------------------- | -------------------------- | ------------------ |
| **Small message latency**     | ~1 µs                   | ~10 µs                     | IB dominant        |
| **Large transfer bandwidth**  | ~200 Gbps               | ~200 Gbps                  | Similar            |
| **AI Training (large-scale)** | Baseline                | Similar with proper tuning | Minimal gap        |
| **AI Inference (Decode)**     | Favorable               | Avg 1.0166% slower         | IB slightly better |
| **Cost**                      | 1.5–2.5x more expensive | Baseline                   | Ethernet wins      |

> Sources: [WWT - The Battle of AI Networking](https://www.wwt.com/blog/the-battle-of-ai-networking-ethernet-vs-infiniband), [Vitex Tech - InfiniBand vs Ethernet](https://www.vitextech.com/blogs/blog/infiniband-vs-ethernet-for-ai-clusters-effective-gpu-networks-in-2025)

Overall, InfiniBand appears to have better raw performance. However, if you already have your setup on AWS, EFA can be advantageous. That said, considering that recent neo-cloud GPU pricing is significantly cheaper compared to AWS, InfiniBand might be the better choice.

---

## 4. P/D Disagg vLLM Serving on EFA

In my case, I was working on setting up Prefill/Decode Disagg in an A100 environment. In this scenario, EFA is directly involved in the kv_transfer stage.

### KV Cache Transfer Software Stack

{% include figure.liquid loading="eager" path="assets/img/efa_blog_2.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="80%" zoomable=true caption="KV Cache Transfer Software Stack on EFA" %}

Let's look at each component above EFA in detail.

### NIXL (NVIDIA Inference Xfer Library)

NIXL is a dedicated library for GPU-to-GPU memory transfer. The existing NCCL is optimized for collective communication like AllReduce and All-to-All, but it's not well-suited for point-to-point transfers of memory blocks from one specific GPU to another — that's why NIXL was created.

It's used in P/D Disagg for the `Prefill GPU ──RDMA Write──▶ Decode GPU` process, and for migration between Decode instances: `Decode GPU (Server A, overloaded) ──RDMA──▶ Decode GPU (Server B, available)`.

NIXL includes a NIXL Agent that handles memory registration, metadata management (GPU memory addresses, RDMA keys, NIC addresses, etc.), and plugin backends (UCX, libfabric) that perform the actual transfers.

### UCX (Unified Communication X)

UCX is a general-purpose communication framework originally built for InfiniBand. It internally supports protocols including RC, UD, TCP, cuda_ipc (for GPUs on the same node), and cuda_copy (GPU ↔ CPU copy).

On EFA, you can use UCX by setting `UCX_TLS=ib`, since EFA provides an ibverbs-compatible interface.

- [EFA ibverbs implementation (AWS driver)](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.c)
- [rdma-core EFA provider](https://github.com/linux-rdma/rdma-core/blob/master/providers/efa/efa.c)

### libfabric (Open Fabrics Interface)

libfabric is the standard abstraction layer for network hardware vendors. It's composed internally of providers including EFA provider, TCP provider, and SHM provider.

Officially, libfabric appears to be the recommended path, but in practice UCX had better dependency compatibility. With libfabric, I kept running into GPU memory bad address issues, so I ended up deprecating it for my use case.

### aws-ofi-nccl (optional)

Since my setup was P/D Disagg, I didn't need NCCL. However, if your model is large enough to require multi-node setup, or for multi-node training, NCCL is necessary — and in EFA environments, aws-ofi-nccl is required.

NCCL was originally designed for InfiniBand and has a built-in `net_ib` transport that directly calls IB Verbs APIs. aws-ofi-nccl implements NCCL's network plugin API and translates it to libfabric's RDM interface.

### efa-nv-peermem

This is a kernel module that enables the NIC to directly access GPU memory. The standard version is nvidia-peermem, but for EFA, efa-nv-peermem is used instead.

By default, NICs can only read CPU memory. efa-nv-peermem bridges the GPU and NIC, enabling direct access. (It's a GPUDirect RDMA module.)
