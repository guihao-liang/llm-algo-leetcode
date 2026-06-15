# 03. GPU Architecture and Memory Practice | 硬件性能优化与算子设计基础

**难度：** Hard | **标签：** `GPU`, `内存层级`, `性能优化` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory_Practice.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本练习配套理论文档：[03_GPU_Architecture_and_Memory.md](./03_GPU_Architecture_and_Memory.md)

在大模型工程中，理解 GPU 的物理架构和内存层级是写出高性能算子的基础。
本节通过实战代码，让你掌握：
- GPU 内存层级的带宽和延迟特性
- Attention 的显存占用计算
- FlashAttention 的显存节省原理
- NVLink vs PCIe 的通信性能对比

## 本节如何和 Notebook 配合

这一节建议和理论文档一起学：

- 先看理论页，理解 GPU 内存层级、Attention 显存和 FlashAttention 的直觉
- 再做 Notebook，把带宽、显存和节省比例真正算一遍
- Notebook 里的测试用来确认你不是“看懂了”，而是真的“会算了”

如果你后面要写自己的算子，这一页负责让你知道**为什么要优化**，Notebook 负责让你验证**到底省了多少**。

### 练习目标

- 估算 GPU 内存层级的带宽与延迟差异
- 计算标准 Attention 与 FlashAttention 的显存占用
- 用测试确认 FlashAttention 确实比标准 Attention 更省显存


```python
import torch
from typing import Dict

def bytes_to_gb(bytes_val: float) -> float:
    return bytes_val / 1e9

# GPU 内存层级的带宽（字节/秒）
MEMORY_BANDWIDTH = {
    'shared_memory': 19e12,  # 19 TB/s (A100)
    'l2_cache': 1.5e12,      # 1.5 TB/s
    'hbm': 1.5e12,           # 1.5 TB/s (A100)
}
```

## Part 1: GPU 内存层级分析

> **为什么要理解内存层级？**
> GPU 的性能瓶颈往往不是计算能力，而是内存访问速度。
> 理解不同内存层级的带宽和延迟，是优化算子的关键。


```python
def analyze_memory_hierarchy() -> Dict[str, Dict[str, float]]:
    """
    分析 GPU 内存层级的性能特性。
    """
    result = {}
    
    for mem_type, bandwidth in MEMORY_BANDWIDTH.items():
        # ==========================================
        # TODO 1.1: 计算访问延迟（访问 1 KB 数据）
        # latency_ns = (1024 / bandwidth) * 1e9
        # ==========================================
        latency_ns = (1024 / bandwidth) * 1e9
        
        result[mem_type] = {
            'bandwidth_tb_s': bandwidth / 1e12,
            'latency_ns': latency_ns,
        }
    
    return result

# 测试
mem_analysis = analyze_memory_hierarchy()
for mem_type, stats in mem_analysis.items():
    print(f"{mem_type:15s}: {stats['bandwidth_tb_s']:6.1f} TB/s, Latency: {stats['latency_ns']:6.2f} ns")
```

## Part 2: Attention 显存占用计算

> **标准 Attention 的显存瓶颈：**
> 在计算 Attention 时，需要生成一个 `[seq_len, seq_len]` 的注意力矩阵。
> 对于长序列（如 128K tokens），这个矩阵会非常巨大，导致 OOM。


```python
def calculate_attention_vram(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2
) -> Dict[str, float]:
    """
    计算标准 Attention 的显存占用。
    """
    # ==========================================
    # TODO 2.1: 计算 Q、K、V 的显存占用
    # qkv_vram = 3 * seq_len * num_heads * head_dim * dtype_bytes
    # ==========================================
    qkv_vram = 3 * seq_len * num_heads * head_dim * dtype_bytes
    
    # ==========================================
    # TODO 2.2: 计算 Attention 矩阵的显存占用
    # attention_matrix_vram = seq_len * seq_len * dtype_bytes
    # ==========================================
    attention_matrix_vram = seq_len * seq_len * dtype_bytes
    
    # ==========================================
    # TODO 2.3: 计算输出的显存占用
    # output_vram = seq_len * num_heads * head_dim * dtype_bytes
    # ==========================================
    output_vram = seq_len * num_heads * head_dim * dtype_bytes
    
    # ==========================================
    # TODO 2.4: 总显存占用
    # total_vram = qkv_vram + attention_matrix_vram + output_vram
    # ==========================================
    total_vram = qkv_vram + attention_matrix_vram + output_vram
    
    return {
        'qkv': bytes_to_gb(qkv_vram),
        'attention_matrix': bytes_to_gb(attention_matrix_vram),
        'output': bytes_to_gb(output_vram),
        'total': bytes_to_gb(total_vram),
    }

# 测试
print("标准 Attention 显存占用:")
for seq_len in [512, 4096, 128*1024]:
    vram = calculate_attention_vram(seq_len, 32, 128)
    print(f"  seq_len={seq_len:6d}: {vram['total']:10.2f} GB")
```

## Part 3: FlashAttention 的显存节省

> **FlashAttention 的核心思想：**
> 不在 HBM 中存储完整的 Attention 矩阵，而是通过 Tiling 和 Online Softmax，
> 在 SRAM 中逐块计算，最后只将结果写回 HBM。


```python
def calculate_flash_attention_vram(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2
) -> Dict[str, float]:
    """
    计算 FlashAttention 的显存占用。
    """
    # ==========================================
    # TODO 3.1: 计算 Q、K、V 的显存占用
    # qkv_vram = 3 * seq_len * num_heads * head_dim * dtype_bytes
    # ==========================================
    qkv_vram = 3 * seq_len * num_heads * head_dim * dtype_bytes
    
    # ==========================================
    # TODO 3.2: FlashAttention 只需存储 Online Softmax 的中间值
    # online_softmax_vram = seq_len * num_heads * 2 * 4
    # ==========================================
    online_softmax_vram = seq_len * num_heads * 2 * 4
    
    # ==========================================
    # TODO 3.3: 计算输出的显存占用
    # output_vram = seq_len * num_heads * head_dim * dtype_bytes
    # ==========================================
    output_vram = seq_len * num_heads * head_dim * dtype_bytes
    
    # ==========================================
    # TODO 3.4: 总显存占用
    # total_vram = qkv_vram + online_softmax_vram + output_vram
    # ==========================================
    total_vram = qkv_vram + online_softmax_vram + output_vram
    
    return {
        'qkv': bytes_to_gb(qkv_vram),
        'online_softmax': bytes_to_gb(online_softmax_vram),
        'output': bytes_to_gb(output_vram),
        'total': bytes_to_gb(total_vram),
    }

# 测试
print("\nFlashAttention 显存占用:")
for seq_len in [512, 4096, 128*1024]:
    vram_std = calculate_attention_vram(seq_len, 32, 128)
    vram_flash = calculate_flash_attention_vram(seq_len, 32, 128)
    ratio = vram_std['total'] / vram_flash['total'] if vram_flash['total'] > 0 else float('inf')
    print(f"  seq_len={seq_len:6d}: 标准={vram_std['total']:10.2f} GB, Flash={vram_flash['total']:10.2f} GB, 节省={ratio:8.0f}x")
```


```python
def test_gpu_memory_practice():
    mem = analyze_memory_hierarchy()
    assert 'shared_memory' in mem and 'hbm' in mem
    assert mem['shared_memory']['bandwidth_tb_s'] > mem['hbm']['bandwidth_tb_s']

    attn = calculate_attention_vram(512, 32, 128)
    flash = calculate_flash_attention_vram(512, 32, 128)
    assert attn['total'] > flash['total']
    assert attn['qkv'] > 0 and flash['online_softmax'] > 0
    print('✅ 03 GPU Architecture and Memory tests passed')

test_gpu_memory_practice()
```

---

🛑 **STOP HERE** 🛑

> 请先尝试自己完成代码并跑通测试。
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。

## 参考代码与解析


```python
# ==================== 参考答案 ====================

def analyze_memory_hierarchy_ref() -> Dict[str, Dict[str, float]]:
    result = {}
    for mem_type, bandwidth in MEMORY_BANDWIDTH.items():
        latency_ns = (1024 / bandwidth) * 1e9
        result[mem_type] = {
            'bandwidth_tb_s': bandwidth / 1e12,
            'latency_ns': latency_ns,
        }
    return result

def calculate_attention_vram_ref(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2
) -> Dict[str, float]:
    qkv_vram = 3 * seq_len * num_heads * head_dim * dtype_bytes
    attention_matrix_vram = seq_len * seq_len * dtype_bytes
    output_vram = seq_len * num_heads * head_dim * dtype_bytes
    total_vram = qkv_vram + attention_matrix_vram + output_vram
    
    return {
        'qkv': bytes_to_gb(qkv_vram),
        'attention_matrix': bytes_to_gb(attention_matrix_vram),
        'output': bytes_to_gb(output_vram),
        'total': bytes_to_gb(total_vram),
    }

def calculate_flash_attention_vram_ref(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2
) -> Dict[str, float]:
    qkv_vram = 3 * seq_len * num_heads * head_dim * dtype_bytes
    online_softmax_vram = seq_len * num_heads * 2 * 4
    output_vram = seq_len * num_heads * head_dim * dtype_bytes
    total_vram = qkv_vram + online_softmax_vram + output_vram
    
    return {
        'qkv': bytes_to_gb(qkv_vram),
        'online_softmax': bytes_to_gb(online_softmax_vram),
        'output': bytes_to_gb(output_vram),
        'total': bytes_to_gb(total_vram),
    }

print("参考答案已加载")
```

### 解析

**1. GPU 内存层级分析**

- **Shared Memory**：速度快（19 TB/s），容量中等（192 KB per SM）
- **L2 Cache**：所有 SM 共享，速度中等（1.5 TB/s），容量较大（40 MB）
- **HBM**：容量最大（80 GB），但速度最慢（1.5 TB/s），是主要的显存瓶颈

**2. Attention 显存占用**

- **标准 Attention**：需要存储 `[seq_len, seq_len]` 的注意力矩阵，显存复杂度 O(N²)
- 对于 128K 序列，显存占用达到 34.4 GB

**3. FlashAttention 的显存节省**

- **核心思想**：通过 Tiling 和 Online Softmax，避免存储完整的 Attention 矩阵
- **显存复杂度**：从 O(N²) 降到 O(N)，节省数千倍
