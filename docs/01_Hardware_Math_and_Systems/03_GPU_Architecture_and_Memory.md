# 03. GPU Architecture and Memory | GPU 物理架构、内存层级与核心硬件单元

**难度：** Hard | **标签：** `硬件架构`, `GPU`, `内存层级` | **目标人群：** 核心 Infra 与算子开发

在算法工程师的面试中，仅仅懂 PyTorch 是不够的。大语言模型 (LLM) 很大程度上是 **Memory Bound (访存受限)** 与 **Compute Bound (算力受限)** 交织的产物。
如果你不能将软件算法映射到 GPU 的物理硬件上，就很难写出高性能的 Triton/CUDA 算子。

如果先抓住几个数量级直觉，可以先记住下面这张表：

| 代际 | 关键变化 | 代表性指标 |
| --- | --- | --- |
| V100 | 首次引入 Tensor Core | 支持 FP16 MMA，开启了深度学习混合精度时代 |
| A100 | 带宽和片上缓存显著增强 | FP16 Tensor Core 可达 312 TFLOPS，HBM 带宽约 1.5 TB/s 级别 |
| H100 | FP8 + 更强调度与搬运机制 | FP8 Tensor Core 可达 1,979 TFLOPS，HBM 带宽可达 3.35 TB/s，NVLink 可达 900 GB/s |
| Blackwell | 更低精度与更强互连 | 原生 FP4，NVLink 提升到 1.8 TB/s 级别（平台实现有差异） |

本节我们将深入 GPU 的核心物理架构，涵盖计算单元 (Tensor Core)、内存结构 (SRAM vs HBM)、以及它们在现代大模型算法（如 FlashAttention）中的实际应用。

## 本节如何和 Notebook 配合

这一节建议和 [练习页](./03_GPU_Architecture_and_Memory_Practice.md) 一起学：

- 先看本文，建立 GPU 架构、内存层级和访存瓶颈的直觉
- 再做 Notebook，把带宽、显存和 FlashAttention 的收益算出来
- Notebook 里的测试用来确认你不是“看懂了”，而是真的“会算了”

如果你后面要写自己的算子，这一页负责让你知道**为什么要优化**，Notebook 负责让你验证**到底省了多少**。

> **相关阅读**:  
> 本章对应的练习资产：  
> [练习页](./03_GPU_Architecture_and_Memory_Practice.md)  
> [`03_GPU_Architecture_and_Memory_Practice.md`](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory_Practice.md)  



## Q1：简述自 V100 以来 NVIDIA GPU 架构的演进，以及为了适应大模型计算做出了哪些核心改变？

<details>
<summary>点击展开查看解析</summary>

NVIDIA 的 GPU 架构代际演进，本质上是为了适应深度学习（尤其是 Transformer）对**混合精度矩阵计算**和**极高显存带宽**的持续攀升需求。

*   **Volta 架构 (V100 - 2017)**: 
    *   **关键引入**：首次引入了专为深度学习矩阵乘加 (MMA) 设计的 **Tensor Core (张量核心)**，支持 FP16 混合精度计算。
*   **Ampere 架构 (A100 - 2020)**:
    *   **常见能力**：支持了 **TF32 (Tensor Float 32)** 和更广泛的 FP16/BF16。
    *   **架构升级**：提升了 HBM2e (High Bandwidth Memory) 的带宽，并扩大了片上缓存容量（例如 L2 Cache 可达 40MB）。A100 时代的官方规格已经把 FP16 Tensor Core 算力推到 312 TFLOPS 量级，同时把 HBM 带宽推到 1.5 TB/s 级别。它还引入了 MIG (多实例 GPU) 和非对称稀疏化 (Sparse Tensor Core)。
*   **Hopper 架构 (H100 - 2022)**:
    *   **专为 LLM 而生**：引入了原生的 **FP8 数据格式**和 **Transformer Engine**。
    *   **内存与调度**：加入了 Thread Block Cluster 和 TMA (Tensor Memory Accelerator)，允许在不经过寄存器的情况下直接进行 HBM 到 SRAM 的异步数据搬运，进一步缓解了带宽压力。H100 的官方规格把 FP8 Tensor Core 算力推到 1,979 TFLOPS，HBM3 带宽可达 3.35 TB/s，NVLink 带宽可达 900 GB/s。
*   **Blackwell 架构 (B100/B200 - 2024)**:
    *   **针对生成式 AI 的进一步优化**：引入了第二代 Transformer Engine，原生支持更低精度的 **FP4 计算格式**，为单卡推理吞吐提升提供了更高上限。
    *   **通信与互连升级**：第五代 NVLink 双向带宽提升到 1.8 TB/s 级别（不同平台实现会有差异），为大规模模型集群提供了更宽的互连带宽上限。
</details>



## Q2：什么是 Tensor Core？它与普通的 CUDA Core 有何本质区别，为什么能明显加速矩阵计算？

<details>
<summary>点击展开查看解析</summary>

**普通 CUDA Core vs Tensor Core**
*   **CUDA Core (FP32/INT32)**: 每次时钟周期只能执行一个标量的 FMA (Fused Multiply-Add，乘加) 操作：`d = a * b + c`。
*   **Tensor Core (FP16/BF16/FP8)**: 专为矩阵乘法设计。在单个时钟周期内，它可以执行一个完整的 $4 \times 4$ 矩阵的 MMA (Matrix Multiply-Accumulate) 操作：`D = A * B + C`。

**为什么它这么快？**
Tensor Core 利用了半精度 (FP16) 或更低精度 (FP8) 来加速乘法，同时使用单精度 (FP32) 的累加器来保证加法精度。由于 Transformer 的自注意力和 MLP 几乎全是密集的矩阵乘法 (GEMM)，Tensor Core 的算力在这类场景下通常会显著高于普通 CUDA Core（例如 A100 的 FP16 Tensor Core 算力可达 312 TFLOPs）。

更直白地说，Tensor Core 不是“把标量 FMA 做快一点”，而是把一批矩阵乘加打包成更大的 MMA 一次完成，所以它在 GEMM 这种高复用、密集计算任务上特别占优。
</details>



## Q3：请描述 GPU 的内存层级结构 (Memory Hierarchy)，并解释为什么大模型推理通常是 Memory Bound (访存受限) 的？

<details>
<summary>点击展开查看解析</summary>

GPU 的内存结构像一个金字塔，越靠近计算单元的速度越快，但容量越小：

1.  **Registers (寄存器)**：
    *   速度最快（<1 个周期），容量极小（每个线程几十个 32-bit 寄存器）。
    *   如果变量太多发生 **Register Spilling (寄存器溢出)**，数据会被回退到较慢的 Local Memory (物理上位于 HBM)。
2.  **Shared Memory (SRAM / 片上共享内存)**：
    *   速度极快（~19 TB/s），每个 SM (流多处理器) 只有几百 KB。
    *   **很关键**：它是同一个 Block 内所有线程协作、交换数据的主要高速通道。**Triton 的一个重要作用，就是帮你自动化管理 SRAM 的分配和调度。**
3.  **L2 Cache**: 
    *   所有 SM 共享，几十 MB，用于缓冲 HBM 的读写。
4.  **HBM (全局显存 / Global Memory)**:
    *   容量大 (40GB ~ 80GB)，但速度相对极慢 (1.5 TB/s ~ 3 TB/s)。
    *   如果算子的每一次计算都需要去 HBM 走一遭（如 PyTorch 原生的多次小操作），就会触发严重的 **Memory Bound (访存受限)**。

更好记的判断方式是看**算术强度**：

```text
Arithmetic Intensity = FLOPs / Bytes
```

如果一个算子的算术强度很低，就说明它每搬一次数据，只做了很少的计算，通常更容易被 HBM 带宽卡住；如果算术强度足够高，计算单元才更容易跑满。
</details>



## Q4：结合 GPU 的内存结构，解释 FlashAttention 是如何利用 SRAM 解决传统 Attention 的访存瓶颈的？

<details>
<summary>点击展开查看解析</summary>

在标准的自注意力机制中，$S = QK^T$ 产生了一个尺寸为 $N \times N$ 的巨大矩阵。
*   **PyTorch 原生**：计算出 $S$，把它**写回 HBM**；读取 $S$ 计算 Softmax，再**写回 HBM**；读取 Softmax 结果和 $V$，计算出最终结果。这种反复读写 $O(N^2)$ 大小数据的行为，直接导致了显存溢出 (OOM) 和速度极慢。

*   **FlashAttention 的底层逻辑 (Tiling + SRAM)**：
    1.  **切块 (Tiling)**：将巨大的 $Q, K, V$ 切成小块 (Blocks)，使得这些小块**刚好能塞进容量只有几百 KB 的 SRAM 中**。
    2.  **在 SRAM 内完成一切 (Fusion)**：把 $Q_{block}$ 和 $K_{block}$ 加载到 SRAM，利用 Tensor Core 算出 $S_{block}$。
    3.  **在线归约 (Online Softmax)**：在 SRAM 内部直接更新局部最大值和指数和，避免写回 $S$。
    4.  最后再乘以 $V_{block}$，把最终结果写回 HBM。
    
**结论**：把 $O(N^2)$ 的 HBM 读写明显压缩到接近 $O(N)$ 的读写。**FlashAttention 不是减少了计算量，而是通过 SRAM 缓解了 Memory Bound 的影响。**

对学习者来说，最重要的不是死记某个固定 GB 数，而是记住它把 attention 的 IO 模式从“反复搬运大矩阵”改成了“分块在 SRAM 中完成”。FlashAttention-2 再进一步优化了 work partitioning，因此在长序列场景里会更有优势。
</details>



## Q5：在多卡分布式集群中，节点内通信的 PCIe 和 NVLink 有什么区别？

<details>
<summary>点击展开查看解析</summary>

当单卡装不下模型时，我们需要分布式训练。GPU 之间的物理连接方式决定了通信带宽 (Communication Bound)：

*   **PCIe (外围组件互连)**：
    *   传统的插槽，带宽有限 (PCIe Gen4 双向 64 GB/s)。
    *   **拓扑痛点**：跨 GPU 通信通常需要经过 PCIe Switch 甚至 CPU，延迟高、带宽低。
*   **NVLink (NVIDIA 私有互连)**：
    *   专为 GPU-to-GPU 设计的高速通道。
    *   **A100 的 NVLink 3.0**：每条链路 50 GB/s，单卡 12 条，总双向带宽高达 **600 GB/s**。这比 PCIe 快了近 10 倍。
    *   **H100 的 NVLink 4.0**：总双向带宽可达 **900 GB/s**。
    *   **Blackwell / NVLink 5**：带宽进一步提升到 **1.8 TB/s** 级别。
    *   **NVSwitch**：允许同一台物理机内的 8 张 GPU 实现全互连 (All-to-All) 的无阻塞通信，这是跑满 `All-Reduce` 和 `All-Gather` 极限带宽的硬件基础。

</details>



## ⚠️ 常见误区

- `Shared Memory` 比 `L2` 快，不代表可以把所有数据都塞进去；它更适合做局部块内复用。
- `HBM` 带宽已经很高，不代表就不会 `Memory Bound`；在高算力 GPU 上，带宽反而更容易成为瓶颈。
- `FlashAttention` 主要减少的是 HBM 访问，不是把主要计算量“变没了”。
- `NVLink` 很快，但仍然需要正确的通信库、拓扑和并行策略配合，否则并不会自动接近跑满。
