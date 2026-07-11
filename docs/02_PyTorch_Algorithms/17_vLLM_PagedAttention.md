# 17. vLLM PagedAttention | 经典推理框架: 模拟 Continuous Batching 与 PagedAttention

**难度：** Hard | **标签：** `推理架构`, `vLLM` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/17_vLLM_PagedAttention.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节我们将揭秘工业界大模型推理框架（如 **vLLM**）的两大杀手锏技术：**Continuous Batching (连续批处理/动态批处理)** 和 **PagedAttention (分页注意力池)**。
这是目前算法面经里含金量最高，但资料最匮乏的部分！

> **相关阅读**:
> 本节使用纯 PyTorch 实现了算法逻辑与数学推导。
> 如果你想学习工业界如何打破该算子的 Memory Bound (访存瓶颈)，请前往 Triton 篇：
>  [`../03_Triton_Kernels/09_Triton_PagedAttention.ipynb`](../03_Triton_Kernels/09_Triton_PagedAttention.md)


### Step 1: 核心思想与痛点

> **痛点 1：Static Batching 的低效**
> 在传统的 PyTorch 推理中，Batch 内的不同请求长度不一。如果 Request A 生成了 10 个 Token 就结束了，而 Request B 需要生成 100 个，那么 A 生成完后 GPU 只能干等 B（即用 Padding 填充计算），导致算力非常浪费。
> **解法：Continuous Batching (Orca/vLLM)**
> 打破 Static Batch 的概念，在 `Step` (Iteration) 粒度上动态重组。A 结束了，立刻把队列里的 Request C 塞进来接着算。

> **痛点 2：KV Cache 的显存碎片化**
> KV Cache 的显存大小是**不可预知的**（你不知道模型最终会生成多长的回复）。如果我们提前按 `max_len` 分配整块显存，会造成严重的内部碎片（超过 60% 浪费）。
> **解法：PagedAttention (vLLM)**
> 借鉴操作系统的虚拟内存管理。把显存切分成固定大小的 **Block** (比如 1个Block存16个Token)。在生成时，按需分配物理 Block，并通过 `Block Table` (块表) 记录虚拟 Token 序列到物理块的映射。

### Step 2: 代码实现框架
系统需要维护一个 `BlockTable`，它是一个二维字典或矩阵，记录了每个序列的逻辑 Block 对应着显存池（K_Cache 和 V_Cache 池）中的哪个物理 Block ID。在解码时，通过查询这个表，将散落的物理 Block 重新聚集起来，与当前的 Query 向量进行 Attention 点积。

###  Step 3: PagedAttention 模拟机制

为了让你在不写几千行 C++ 的情况下弄懂 PagedAttention，我们将用纯 Python 模拟它的核心数据结构：

1. **Physical Block Pool (物理块池)**：一个预先分配好的大张量，形状为 `[num_blocks, block_size, hidden_dim]`。
2. **Block Table (块表)**：每个 Request 都有一个专属的块表，它是一个整数列表（`List[int]`），记录了这个 Request 的第 $i$ 个逻辑块存在物理池的哪个索引里。
3. **KV Cache Manager**：负责在 Token 生成时，“按需”分配新的物理块索引。

###  Step 4: 动手实战

**要求**：请补全下方 `KVCacheManager`，实现一个极简版的 vLLM 内存管理器。


```python
import torch
from typing import List
```


```python
class Request:
    def __init__(self, request_id: int, prompt_len: int):
        self.request_id = request_id
        self.seq_len = prompt_len
        # 记录此请求占据的物理 Block 索引
        self.block_table: List[int] = []

class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        
        # 模拟预分配一块大显存池 (vLLM 会在 GPU 上分配几 GB)
        # 形状: [num_blocks, block_size, head_dim]
        self.physical_kv_cache = torch.zeros(num_blocks, block_size, head_dim)
        
        # 跟踪哪些物理块被占用了
        self.free_blocks: List[int] = list(range(num_blocks))

    def allocate_for_prefill(self, req: Request):
        """
        请求刚进来时 (Prefill阶段)，为它的 Prompt 长度分配所需的全部 Block
        """
        # ==========================================
        # TODO 1: 计算需要的 block 数量
        # 提示: 向上取整 (seq_len / block_size)
        # ==========================================
        needed_blocks = (req.seq_len + self.block_size - 1) // self.block_size
        
        # ==========================================
        # TODO 2: 从 free_blocks 中弹出对应数量的 block 索引，
        # 并追加到请求的 block_table 中
        # 如果 free_blocks 不够了，抛出 RuntimeError("OOM")
        # ==========================================
        if needed_blocks > len(self.free_blocks):
            raise RuntimeError("OOM")
        
        split = self.free_blocks[-needed_blocks:]
        self.free_blocks = self.free_blocks[: -needed_blocks]
        req.block_table = split

    def allocate_for_decode(self, req: Request):
        """
        自回归生成时 (Decode阶段)，检查序列长度。
        如果当前最后一个 Block 满了，则按需分配 1 个新 Block。
        """
        req.seq_len += 1  # 长度加 1
        
        # ==========================================
        # TODO 3: 判断是否刚好需要跨入新的一块 Block？
        # 条件：加 1 后的 seq_len 除以 block_size 余数是多少？
        # ==========================================
        is_new_block_needed = (req.seq_len % self.block_size) == 1
        print(req.seq_len, is_new_block_needed)
              
        # 如果需要，尝试分配 1 个新的物理 Block 放入块表
        # if is_new_block_needed:
        #    if not self.free_blocks: ...
        #    req.block_table.append(???)
        if is_new_block_needed:
            if not self.free_blocks:
                raise RuntimeError("OOM")
            req.block_table.append(self.free_blocks[-1])
            self.free_blocks.pop(-1)

    def get_physical_cache(self, req: Request) -> torch.Tensor:
        """
        (模拟 PagedAttention 底层加载逻辑)
        根据块表，把不连续的物理块“拼凑”成逻辑上连续的 KV Cache (仅作验证用途)
        """
        # ==========================================
        # TODO 4: 根据 req.block_table 的索引，
        # ==========================================
        ## advanced indexing will do concat and copy for me
        blocks = self.physical_kv_cache[req.block_table]
        # blocks = self.physical_kv_cache[req.block_table, ...]
        print(blocks.shape)
        blocks = blocks.view(-1, self.head_dim)
        print(blocks.shape)
        cat_blocks = blocks
        
        # 最后，只截取真实 seq_len 长度返回 (因为最后一个块可能没填满)
        return cat_blocks[:req.seq_len]

```


```python
# 运行此单元格以测试你的实现
def test_paged_attention_manager():
    try:
        manager = KVCacheManager(num_blocks=10, block_size=4, head_dim=64)
        print("初始化内存池...")
        
        # 1. 模拟一个 Request (Prompt 长度为 6)
        req1 = Request(request_id=1, prompt_len=6)
        
        manager.allocate_for_prefill(req1)
        assert len(req1.block_table) == 2, "长度 6 的请求应分配 2 个 Block！"
        assert len(manager.free_blocks) == 8, "池中应该剩下 8 个空闲块！"
        print(f"✅ Prefill 测试通过！Req1 分配的块表: {req1.block_table}")
        
        # 2. 模拟 Decode 阶段生成 Token (产生第 7 个 token，不需要新块)
        manager.allocate_for_decode(req1)
        assert len(req1.block_table) == 2, "生成第 7 个 token 时不应该分配新块！"
        
        # 3. 产生第 8，再产生第 9 个 Token (跨过 Block 边界，需要新块)
        manager.allocate_for_decode(req1) # 长度变为 8
        manager.allocate_for_decode(req1) # 长度变为 9，触发新分配
        
        assert len(req1.block_table) == 3, "生成第 9 个 token 时应当分配了第 3 个新块！"
        assert len(manager.free_blocks) == 7, "池中应该剩下 7 个空闲块！"
        print(f"✅ Decode 动态分配测试通过！Req1 最新块表: {req1.block_table}")
        
        # 4. 模拟底层 PagedAttention 组装验证
        # 手动往第一块里写点假数据
        manager.physical_kv_cache[req1.block_table[0], 0, 0] = 999.0
        
        cache = manager.get_physical_cache(req1)
        assert cache.shape == (9, 64), f"拼装出来的连续 Cache 形状不对，应为 (9, 64)，实为 {cache.shape}"
        assert cache[0, 0] == 999.0, "数据未正确映射！"
        
        print("\n✅ All Tests Passed! PagedAttention 内存管理逻辑验证通过。")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except RuntimeError as e:
        print(f"❌ 运行时错误: {e}")
    except TypeError as e:
        print(f"代码可能未完成，导致变量为 NoneType。{e}")
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")

test_paged_attention_manager()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---
## 参考代码与解析

### 代码

```python
class Request:
    def __init__(self, request_id: int, prompt_len: int):
        self.request_id = request_id
        self.seq_len = prompt_len
        self.block_table: List[int] = []

class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        
        # TODO 1: 模拟预分配一块大显存池
        self.physical_kv_cache = torch.zeros(num_blocks, block_size, head_dim)
        
        # 跟踪哪些物理块被占用了
        self.free_blocks: List[int] = list(range(num_blocks))

    def allocate_for_prefill(self, req: Request):
        """
        请求刚进来时 (Prefill阶段)，为它的 Prompt 长度分配所需的全部 Block
        """
        # TODO 2: 计算需要的 block 数量（向上取整）
        needed_blocks = (req.seq_len + self.block_size - 1) // self.block_size
        
        # TODO 3: 从 free_blocks 中弹出对应数量的 block 索引
        if len(self.free_blocks) < needed_blocks:
            raise RuntimeError("OOM")
        
        for _ in range(needed_blocks):
            block_id = self.free_blocks.pop(0)
            req.block_table.append(block_id)

    def allocate_for_decode(self, req: Request):
        """
        自回归生成时 (Decode阶段)，检查序列长度。
        如果当前最后一个 Block 满了，则按需分配 1 个新 Block。
        """
        req.seq_len += 1
        
        # TODO 4: 判断是否需要新的 Block
        is_new_block_needed = (req.seq_len % self.block_size) == 1
        
        if is_new_block_needed:
            if not self.free_blocks:
                raise RuntimeError("OOM")
            block_id = self.free_blocks.pop(0)
            req.block_table.append(block_id)

    def get_physical_cache(self, req: Request) -> torch.Tensor:
        """
        根据块表，把不连续的物理块"拼凑"成逻辑上连续的 KV Cache
        """
        # TODO 5: 根据 req.block_table 的索引，从物理池中提取对应的块
        blocks = [self.physical_kv_cache[block_id] for block_id in req.block_table]
        cat_blocks = torch.cat(blocks, dim=0)
        
        # 只截取真实 seq_len 长度返回
        return cat_blocks[:req.seq_len]
```

### 解析

**1. TODO 1: 初始化物理块池**
- **实现方式**：`self.physical_kv_cache = torch.zeros(num_blocks, block_size, head_dim)`
- **关键点**：预分配固定大小的显存池，避免动态分配的碎片化
- **技术细节**：形状为 `[num_blocks, block_size, head_dim]`，每个 block 存储固定数量的 token

**2. TODO 2: 计算 Prefill 阶段需要的 block 数量**
- **实现方式**：`needed_blocks = (req.seq_len + self.block_size - 1) // self.block_size`
- **关键点**：向上取整，确保能容纳所有 token
- **技术细节**：使用 `(a + b - 1) // b` 实现向上取整，避免浮点运算

**3. TODO 3: 分配物理块**
- **实现方式**：从 `free_blocks` 中弹出 `needed_blocks` 个索引，追加到 `req.block_table`
- **关键点**：如果空闲块不足，抛出 OOM 异常
- **技术细节**：使用 `pop(0)` 从队列头部取出，模拟 FIFO 分配策略

**4. TODO 4: Decode 阶段按需分配**
- **实现方式**：`is_new_block_needed = (req.seq_len % self.block_size) == 1`
- **关键点**：只有当序列长度刚好跨越 block 边界时才分配新块
- **技术细节**：`seq_len % block_size == 1` 表示刚进入新块的第一个位置

**5. TODO 5: 拼装物理块**
- **实现方式**：`blocks = [self.physical_kv_cache[block_id] for block_id in req.block_table]`，`cat_blocks = torch.cat(blocks, dim=0)`
- **关键点**：根据块表索引，将不连续的物理块拼接成逻辑上连续的 KV Cache
- **技术细节**：最后截取 `[:req.seq_len]` 因为最后一个块可能未填满

**工程优化要点**
- **显存利用率**：PagedAttention 将显存利用率从 40% 提升到 90%+，减少内部碎片
- **动态批处理**：配合 Continuous Batching，实现请求级别的动态调度
- **块大小权衡**：block_size 太小增加管理开销，太大增加内部碎片，通常选择 16-32
- **共享机制**：vLLM 支持多个请求共享相同的 Prompt 块（如系统提示词），进一步节省显存
- **工业实现**：真实的 vLLM 使用 CUDA kernel 实现 PagedAttention，支持多头注意力和批处理
---

## 附录：PyTorch Advanced Indexing 详解

本节 `get_physical_cache` 里的 `self.physical_kv_cache[req.block_table]` 用到了 **advanced indexing（高级索引）**。这是 vLLM 风格「按块表 gather」的核心写法，但它的行为和普通切片**完全不同**，这里彻底讲清楚。

### 一、两套索引规则

PyTorch（继承自 NumPy）有**两套**索引规则，行为不一样：

| 类型 | 用什么索引 | 返回 | 是否共享内存 |
|------|-----------|------|-------------|
| **Basic indexing（基础索引）** | 整数、切片 `:`、`...`、`None` | **view（视图）** | ✅ 共享（不 copy） |
| **Advanced indexing（高级索引）** | 整数列表/张量、bool mask | **copy（拷贝）** | ❌ 新内存 |

以本节的 `physical_kv_cache`，形状 `[num_blocks=10, block_size=4, head_dim=64]` 为例：

```python
physical_kv_cache[3]           # 基础索引：单个整数     → view，形状 [4, 64]
physical_kv_cache[[3, 7, 1]]   # 高级索引：整数列表     → copy，形状 [3, 4, 64]
physical_kv_cache[req.block_table]  # ← 本节用的就是这个，一次 gather 多个块
```

> ✅ 所以 `self.physical_kv_cache[req.block_table]` 一步就把散落的物理块「聚拢 + 拷贝」成一个新张量 `[num_blocks_of_req, block_size, head_dim]`，等价于参考答案里 `[... for ...]` + `torch.cat` 那两行，但更简洁。这也是它「会帮我 concat and copy」的原因。

### 二、view vs copy 的真正判据（重要，别记错）

**不是「连续的 index 才给 view」。** 真正的判据是——**选出的元素能否用「原 storage 上的某一组 `(offset, strides)`」表达出来**：

- **切片 `start:stop:step`（规则的等差访问）→ 能用一组 stride 表达 → view**
- **整数列表/张量、bool mask（任意乱序 gather）→ 无法用单组 stride 表达 → copy**

反例：`t[0:6:2]` 取第 0、2、4 个元素，内存上跳着放（**不连续**），但它**仍然是 view**！因为只要把 dim0 的 stride 翻倍（跳一个），一组 `(offset, strides)` 就能描述它。而 `t[[3, 7, 1]]` 这种任意顺序的 gather，没有任何一组统一 stride 能表达，只能新开内存 copy。

> ⚠️ **别把 view 和 contiguous 混为一谈——它们是两个独立的 stride 概念**（`is_contiguous` 本身就是用 stride 定义的，所以「能用 stride 表达」和「连续」不是对立面）：
> - **view**：能否用「原 storage 上的**某一组** `(offset, strides)`」描述 → 决定是不是 copy。步长多大都行。
> - **contiguous**（`is_contiguous()`）：这组 stride 是否**恰好等于行优先紧密排布** `stride[i] = ∏_{j>i} shape[j]` → 决定内存里元素是否紧挨、无间隙。
>
> `t[0:6:2]` 就是两者分离的活例：它是 view（dim0 的 stride 由 256 变成 512，仍能表达），但 512 ≠ 紧密该有的 256，所以 `is_contiguous() = False`。**「是 view」不代表「contiguous」，反之亦然。** 下方 demo cell 的 [5] 段会把 stride 打印出来给你看。

### 三、只指定部分维度：其他维度会 copy 吗？

**不会 copy，而是被隐式补成 `:`（整片保留）。** 写 `t[i]` 时，PyTorch 自动在尾部补齐剩余维度的 `:`：

```python
physical_kv_cache[3]        # 等价 physical_kv_cache[3, :, :]  → [4, 64]
physical_kv_cache[3, 0]     # 等价 physical_kv_cache[3, 0, :]  → [64]
physical_kv_cache[3, 0, 0]  # 标量（test 里写 999.0 就是这么定位的）
```

注意：**隐式的 `:` 只补在尾部**。想在中间维度选、前面维度全要，就得靠 `...`。

### 四、`...`（Ellipsis，省略号）

`...` = 「在这里自动填入足够多的 `:`，把剩余维度铺满」，能自适应维度数量，省去数不清的冒号：

```python
physical_kv_cache[..., 0]     # 等价 [:, :, 0]  → [10, 4]   （每块每位置的第 0 个特征）
physical_kv_cache[0, ...]     # 等价 [0, :, :]  → [4, 64]   （= physical_kv_cache[0]）
physical_kv_cache[..., 0, :]  # 等价 [:, 0, :]  → [10, 64]
```

对比：`t[0]` 选的是**第 0 维**，`t[..., 0]` 选的是**最后一维**，意思完全不同。一个表达式里 `...` 最多出现一次（否则有歧义）。

> 💡 **面试点**：`physical_kv_cache[req.block_table]`（高级索引，copy）和 `physical_kv_cache[3]`（基础索引，view）的 copy/view 差异，直接决定「改了结果会不会污染原 KV Cache 池」——这是推理框架里容易踩的坑。下方 demo cell 会实测给你看。

```python
# 附录 Demo: 亲手验证 view vs copy / 隐式补 : / Ellipsis
import torch

pool = torch.arange(10 * 4 * 64, dtype=torch.float32).reshape(10, 4, 64)  # [num_blocks, block_size, head_dim]
print("pool.shape =", pool.shape)

def shares_storage(a, b):
    # 正确判据：比底层 storage 地址，而不是 .data_ptr()（后者含 offset，view 会误判为不共享）
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

# --- 1) 形状：只指定部分维度，其余维度隐式补 : ---
print("\n[1] 隐式补 : (基础索引)")
print("  pool[3].shape       =", pool[3].shape,       "  # == pool[3, :, :]")
print("  pool[3, 0].shape    =", pool[3, 0].shape,    "  # == pool[3, 0, :]")
print("  pool[3, 0, 0]       =", pool[3, 0, 0].item(),"(标量)")

# --- 2) Ellipsis：... 在前面补 : ---
print("\n[2] Ellipsis ...")
print("  pool[..., 0].shape    =", pool[..., 0].shape,    "  # == pool[:, :, 0]")
print("  pool[..., 0, :].shape =", pool[..., 0, :].shape, "  # == pool[:, 0, :]")
print("  pool[0] 与 pool[..., 0] 选的维度不同：", pool[0].shape, "vs", pool[..., 0].shape)

# --- 3) 基础索引 = view：改视图会污染原张量 ---
print("\n[3] 基础索引 pool[3] → view（共享 storage，会污染原池！）")
v = pool[3]                          # 单个整数 → 基础索引
print("  shares_storage(v, pool) :", shares_storage(v, pool))
v[0, 0] = -999.0
print("  改 v 后 pool[3,0,0] =", pool[3, 0, 0].item(), "  <-- 原池被改了（view 铁证）")

# --- 4) 高级索引 = copy：list [1,2,3] 索引，改结果不影响原张量 ---
print("\n[4] 高级索引 pool[[1, 2, 3]] → copy（新内存，安全）")
g = pool[[1, 2, 3]]                  # 整数列表 → 高级索引，形状 [3, 4, 64]
print("  g.shape =", g.shape, " shares_storage(g, pool) :", shares_storage(g, pool))
before = pool[1, 0, 0].item()
g[0, 0, 0] = -1.0
print("  改 g 后 pool[1,0,0] =", pool[1, 0, 0].item(), "(仍为", before, ") <-- 原池没变（copy 铁证）")

# --- 5) view 与 contiguous 是两个独立的 stride 概念 ---
print("\n[5] 跳步切片 pool[0:6:2]：是 view，但 not contiguous")
s = pool[0:6:2]                      # 取第 0,2,4 块
dense_stride = (s.shape[1] * s.shape[2], s.shape[2], 1)   # 行优先紧密排布本应的 stride
print("  pool.stride()  =", pool.stride())
print("  s.stride()     =", s.stride(), " (dim0 步长 256 -> 512，跳了一块)")
print("  紧密排布应为    =", dense_stride)
print("  shares_storage(s, pool) =", shares_storage(s, pool), " -> 是 view")
print("  s.is_contiguous()       =", s.is_contiguous(), " -> stride 不是紧密那组")
print("  => view 与 contiguous 都由 stride 定义，但要求不同：")
print("     view       : 能用『原 storage 上某一组 (offset, strides)』表达（512 也行）")
print("     contiguous : 这组 stride 恰好等于行优先紧密排布(256) —— 两个独立判据")

```

### 五、`untyped_storage` 与 `data_ptr` 到底是什么？

要理解前面 demo 里为什么判「是否共享内存」要用 `untyped_storage().data_ptr()` 而不是 `data_ptr()`，得先看 PyTorch Tensor 的内存模型。

**一个 Tensor = 「元数据」+ 「指向一块 Storage 的指针」**：

```
Tensor                          Storage（底层一维连续 buffer，只存原始字节）
┌────────────────────┐          ┌───────────────────────────────────────┐
│ shape   (10,4,64)  │          │ [f0][f1][f2] ...............  [f2559]  │
│ strides (256,64,1) │ ───────► │  ▲                                    │
│ storage_offset  0  │          │  └─ untyped_storage().data_ptr()      │
│ dtype  float32     │          │     = 整块 buffer 的起始地址           │
└────────────────────┘          └───────────────────────────────────────┘
```

- **Storage / `untyped_storage()`**：底层那块**一维、连续**的原始内存（一串字节，不带 shape/dtype 语义）。多个 view 张量**共享同一个 Storage**。`untyped` = 不区分 dtype，按字节看；老 API `t.storage()` 是带类型的版本，已不推荐。
- **`untyped_storage().data_ptr()`**：这块 Storage buffer 的**起始地址**。所有共享它的 view 拿到的都是**同一个值** → 所以判断「是否共享内存」要用它。
- **`t.data_ptr()`**：**本张量第 0 个元素**的地址，会把 `storage_offset` 算进去。

**两者的关系（一条公式）**：

```
t.data_ptr() == t.untyped_storage().data_ptr() + t.storage_offset() * t.element_size()
                └── buffer 起始 ──┘   └─ 本张量从第几个元素开始 ─┘  └ 每个元素几字节 ┘
```

**为什么之前 `pool[1]` 用 `data_ptr()` 会被误判成「不共享」？**

`pool[1]` 是 view，和 `pool` 共享同一 Storage（`untyped_storage().data_ptr()` 相同），但它的 `storage_offset = 1×4×64 = 256`，于是 `data_ptr()` = 起始地址 + 256×4 字节，**比 `pool.data_ptr()` 大**。所以用 `data_ptr()` 比较会得出「不相等」的错误结论。**判共享看 Storage 地址，判本张量落点看 `data_ptr()`**——这就是区别。

> 📌 一句话记忆：`untyped_storage().data_ptr()` = 「房子的门牌号」（整块 buffer 起点，view 之间都一样）；`data_ptr()` = 「你住在这栋楼第几间」（门牌号 + 偏移）。

```python
# 附录 Demo: untyped_storage vs data_ptr 的关系
import torch

pool = torch.arange(10 * 4 * 64, dtype=torch.float32).reshape(10, 4, 64)
v = pool[1]                      # view，storage_offset != 0

print("dtype =", pool.dtype, " element_size =", pool.element_size(), "bytes")
print("整块 storage 大小 =", pool.untyped_storage().nbytes(), "bytes =",
      pool.numel(), "个 float32\n")

# (1) 共享内存 → untyped_storage().data_ptr() 相同（门牌号一样）
print("pool.untyped_storage().data_ptr() =", pool.untyped_storage().data_ptr())
print("v.untyped_storage().data_ptr()    =", v.untyped_storage().data_ptr(),
      " -> 相同:", pool.untyped_storage().data_ptr() == v.untyped_storage().data_ptr())

# (2) data_ptr() 含 offset → view 的落点不同（住第几间）
print("\nv.storage_offset() =", v.storage_offset(), "个元素")
print("pool.data_ptr()    =", pool.data_ptr())
print("v.data_ptr()       =", v.data_ptr(), " -> 比 pool 大", v.data_ptr() - pool.data_ptr(), "字节")

# (3) 验证公式: data_ptr == storage起始 + offset * element_size
lhs = v.data_ptr()
rhs = v.untyped_storage().data_ptr() + v.storage_offset() * v.element_size()
print("\n公式验证 data_ptr == storage_ptr + offset*elem_size :", lhs == rhs)
print(f"  {lhs} == {v.untyped_storage().data_ptr()} + {v.storage_offset()}*{v.element_size()}")

# (4) 结论：判「是否共享内存」必须用 storage 地址，不能用 data_ptr
print("\n[错误判法] v.data_ptr() == pool.data_ptr() :", v.data_ptr() == pool.data_ptr(),
      " <- view 被误判为不共享")
print("[正确判法] storage 地址相同               :",
      v.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr())

```
