# 第一部分：硬件、数学与系统

## Part 概览

本部分主线覆盖 33 个讨论题（01-33），共同把第零部分的基础能力连接到第二至第五部分的工程实现。其中 `01-10` 是基础主线，`11-20` 是延展主线，`21-33` 是同类扩展主线。正文默认 notebook-first，主线页尽量与 notebook 同页。

Part 1 按 5 个专题组织：`1A`、`1B`、`1C`、`1D`、`1E` 分别承担数值基础、单卡硬件、多卡通信、异构调度和编译生态这五条主线。具体每组怎么读、怎么接后续 Part，由各组导航页分别说明。

```mermaid
flowchart TB
    P1[Part 1]

    subgraph C[内容分层]
        L1[01-10 基础主线]
        L2[11-20 延展主线]
        L3[21-33 扩展主线]
    end

    subgraph T[专题分组]
        G1[1A 数值基础]
        G2[1B 单卡硬件]
        G3[1C 多卡通信]
        G4[1D 执行与编程]
        G5[1E 编译与生态]
    end

    P1 --> C
    P1 --> T
```

## Part 资产总览

本章内容按 5 个主线组组织，后续页面也沿该结构继续扩展。

> 导航说明：侧边栏和组级入口默认收起，先看总览，再点开具体组页。
> 组页是知识包，不需要把整组一次性读完；先抓主线，再按需要查看同组章节页。
> Part 1 不只是知识目录，也是 Part 2-5 的共同前置底座。

| 学习组 | 核心问题 / 主题 | 当前内容映射 | 每组多少节 |
|:---|:---|:---|:---|
| [1A: 数值基础与算力估算](./1A.md) | 先要算什么？ / 数据格式、参数量、FLOPs | [01](./01_Data_Types_and_Precision.ipynb)、[02](./02_LLM_Params_and_FLOPs.ipynb)、[21](./21_Quantization_Theory_and_INT4_INT8.ipynb)、[22](./22_MoE_Parameter_and_Compute.ipynb) | 4 |
| [1B: 单卡硬件与访存优化](./1B.md) | 单卡怎么跑得快？ / GPU 架构、内存层次、Attention 访存 | [03](./03_GPU_Architecture_and_Memory.ipynb)、[04](./04_Attention_Memory_Optimization.ipynb)、[23](./23_TensorCore_Deep_Dive.ipynb)、[24](./24_SRAM_Optimization_Techniques.ipynb)、[25](./25_Sparse_Computation_and_Sparse_Attention.ipynb) | 5 |
| [1C: 分布式通信与显存共享](./1C.md) | 一张卡不够怎么办？ / 通信拓扑、ZeRO、显存切分 | [05](./05_Communication_Topologies.ipynb)、[06](./06_VRAM_Calculation_and_ZeRO.ipynb)、[26](./26_Parallel_Strategy_Decision_Framework.ipynb)、[27](./27_Communication_Scheduling_Optimization.ipynb)、[28](./28_Fault_Tolerance_and_Checkpointing.ipynb) | 5 |
| [1D: 异构调度与算子编程](./1D.md) | 怎么精细控制计算和数据流？ / CPU-GPU 协同、CUDA-Triton | [07](./07_CPU_GPU_Heterogeneous_Scheduling.ipynb)、[08](./08_Programming_Models_CUDA_Triton.ipynb)、[29](./29_CUDA_Stream_Advanced_Scheduling.ipynb)、[30](./30_Dynamic_Shape_Handling.ipynb)、[31](./31_GPU_Virtualization_and_MIG.ipynb) | 5 |
| [1E: 编译优化与硬件生态](./1E.md) | 怎么自动优化和做迁移？ / AI 编译器、芯片现状、TCO | [09](./09_AI_Compilers_and_Graph_Optimization.ipynb)、[10](./10_Domestic_AI_Chips_Overview.ipynb)、[32](./32_TVM_MLIR_Deep_Practice.ipynb)、[33](./33_TCO_and_Cost_Model.ipynb) | 4 |

## 学习路径

Part 1 不只是知识目录，也是 Part 2 到 Part 5 的共同前置。阅读上可以按三层理解：`01-10` 是基础主线，`11-20` 是延展主线，`21-33` 是扩展主线。

```mermaid
flowchart LR
    A[01-10 基础主线] --> B[11-20 延展主线]
    B --> C[21-33 扩展主线]
```

### 推荐顺序

- 快速入门：先看 [1A](./1A.md) → [1B](./1B.md)
- 系统学习：按 [1A](./1A.md) → [1B](./1B.md) → [1C](./1C.md) → [1D](./1D.md) → [1E](./1E.md) 顺序推进

### 面向后续部分

- 基础认知层：先看 [1A](./1A.md)、[1B](./1B.md)，把精度、参数量、GPU 架构和访存直觉先立起来，主要服务 Part 2 / Part 3。
- 执行模型层：先看 [1C](./1C.md)、[1D](./1D.md)、[15](./15_CUDA_Execution_Model.ipynb)、[16](./16_Warp_Block_SharedMemory_Basics.ipynb)、[17](./17_CUDA_Stream_and_Asynchrony.ipynb)、[18](./18_Triton_Block_Model.ipynb)、[19](./19_Operator_Fusion_Introduction.ipynb)、[20](./20_NCCL_and_AllReduce_Basics.ipynb)，理解通信、调度、block / warp / shared memory 和 Triton block model，主要服务 Part 3。
- 优化与选型层：先看 [1E](./1E.md)、[19](./19_Operator_Fusion_Introduction.ipynb)、[32](./32_TVM_MLIR_Deep_Practice.ipynb)、[33](./33_TCO_and_Cost_Model.ipynb)，理解编译优化、算子融合、TCO，以及为什么后面会从 PyTorch 走到 Triton，再走到 CUDA，主要服务 Part 2 / Part 3。

## 环境说明

- 默认按 `CPU-first` 设计
- 少数页面会标注 `GPU optional` 或 `GPU required`
