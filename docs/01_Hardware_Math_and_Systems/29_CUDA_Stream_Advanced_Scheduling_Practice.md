# 29. CUDA Stream Advanced Scheduling Practice | 1D-03. CUDA Stream Advanced Scheduling - 计算练习

**难度：** Hard | **标签：** `CUDA`, `Stream`, `Event`, `Graph` | **目标人群：** 想把推理和训练流程调得更细的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling_Practice.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本练习配套理论文档：[29_CUDA_Stream_Advanced_Scheduling.md](./29_CUDA_Stream_Advanced_Scheduling.md)

这页不是要让你真的在这个 notebook 里跑 CUDA，而是先把高级调度的核心判断做成可验证练习：
- 什么时候并行有意义
- Event 应该怎么串依赖
- CUDA Graph 为什么只适合稳定路径

## 本节如何和 Notebook 配合

这一节建议和理论文档一起学：

- 先看理论页，建立 Stream / Event / Graph 的职责分工
- 再做 Notebook，把调度、同步和回放的收益真正算出来
- Notebook 里的测试用来确认你不是“看懂了”，而是真的“会判断了”

### 练习目标

- 判断一个任务拆成多 Stream 后是否真的有重叠收益
- 用 Event 描述跨 Stream 的依赖关系
- 判断 CUDA Graph 在什么场景值得用、什么场景不值得用


```python
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class TaskStage:
    name: str
    duration_us: float
    stream: str
    depends_on: Tuple[str, ...] = ()


def sequential_time_us(stages: List[TaskStage]) -> float:
    """假设所有阶段串行执行的总时间。"""
    return sum(stage.duration_us for stage in stages)


def pipelined_time_us(stages: List[TaskStage]) -> float:
    """非常粗略地估算流水线调度后的总时间。

    这里我们把同一条 Stream 的阶段视为顺序执行，
    不同 Stream 若无依赖且属于不同职责，可以产生重叠。
    """
    stream_totals: Dict[str, float] = {}
    for stage in stages:
        stream_totals.setdefault(stage.stream, 0.0)
        stream_totals[stage.stream] += stage.duration_us
    return max(stream_totals.values()) if stream_totals else 0.0


def overlap_ratio(sequential_us: float, pipelined_us: float) -> float:
    if sequential_us == 0:
        return 0.0
    return 1 - pipelined_us / sequential_us

```

## Part 1: Stream 调度与并行收益

### 练习 1.1: 比较串行和流水线时间

先看同一批任务在串行执行和按职责拆 Stream 后，时间差有多大。

```python
def test_stream_pipeline_basic():
    stages = [
        TaskStage('H2D', 12, 'copy'),
        TaskStage('Kernel', 40, 'compute'),
        TaskStage('D2H', 10, 'copy'),
        TaskStage('Post', 8, 'post'),
    ]

    seq = sequential_time_us(stages)
    pipe = pipelined_time_us(stages)

    assert seq == 70, seq
    assert pipe == 40, pipe
    assert 0 < overlap_ratio(seq, pipe) < 1
    print('✅ Stream 流水线测试通过')


test_stream_pipeline_basic()

stages = [
    TaskStage('H2D', 12, 'copy'),
    TaskStage('Kernel', 40, 'compute'),
    TaskStage('D2H', 10, 'copy'),
    TaskStage('Post', 8, 'post'),
]
print('串行时间:', sequential_time_us(stages), 'us')
print('流水线时间:', pipelined_time_us(stages), 'us')
print('重叠收益:', f"{overlap_ratio(sequential_time_us(stages), pipelined_time_us(stages)):.1%}")

```

### 练习 1.2: 判断什么时候值得拆 Stream

不是所有任务都值得拆成很多 Stream。你要先看职责是否独立，再看是否真的能重叠。

```python
def should_split_stream(copy_us: float, compute_us: float, post_us: float, threshold: float = 0.2) -> bool:
    """如果搬运和后处理占比足够大，拆 Stream 往往更值得。"""
    total = copy_us + compute_us + post_us
    if total == 0:
        return False
    overhead = (copy_us + post_us) / total
    return overhead >= threshold


assert should_split_stream(20, 40, 15) is True
assert should_split_stream(2, 60, 1) is False
print('✅ Stream 拆分判断函数测试通过')

```

## Part 2: Event 同步与依赖图

### 练习 2.1: 用 Event 描述跨 Stream 依赖

Event 的关键作用不是“阻塞全部”，而是只同步真正有依赖关系的阶段。

```python
def build_dependency_edges(stages: List[TaskStage]) -> List[Tuple[str, str]]:
    edges = []
    for stage in stages:
        for dep in stage.depends_on:
            edges.append((dep, stage.name))
    return edges


def has_cross_stream_dependency(stages: List[TaskStage]) -> bool:
    name_to_stage = {s.name: s for s in stages}
    for stage in stages:
        for dep in stage.depends_on:
            if name_to_stage[dep].stream != stage.stream:
                return True
    return False

```


```python
def test_event_dependencies():
    stages = [
        TaskStage('H2D', 12, 'copy'),
        TaskStage('Kernel', 40, 'compute', depends_on=('H2D',)),
        TaskStage('D2H', 10, 'copy', depends_on=('Kernel',)),
        TaskStage('Post', 8, 'post', depends_on=('D2H',)),
    ]

    edges = build_dependency_edges(stages)
    assert ('H2D', 'Kernel') in edges
    assert ('Kernel', 'D2H') in edges
    assert ('D2H', 'Post') in edges
    assert has_cross_stream_dependency(stages) is True
    print('✅ Event 依赖测试通过')


test_event_dependencies()

```

### 练习 2.2: 判断 Event 是否过密

Event 应该只放在真正有依赖的地方；如果同步点过多，流水线收益会被吃掉。

```python
def event_density(num_events: int, num_stages: int) -> float:
    return num_events / num_stages if num_stages else 0.0

assert abs(event_density(3, 4) - 0.75) < 1e-9
assert event_density(0, 4) == 0.0
print('✅ Event 密度测试通过')

```

## Part 3: CUDA Graph 适用性判断

### 练习 3.1: 判断是否适合 Graph

Graph 适合路径稳定的任务，不适合形状频繁变化的任务。

```python
def graph_suitability(is_fixed_shape: bool, is_repeated_path: bool, has_many_branches: bool) -> bool:
    """非常简化的 CUDA Graph 适用性判断。"""
    return is_fixed_shape and is_repeated_path and not has_many_branches

assert graph_suitability(True, True, False) is True
assert graph_suitability(True, False, False) is False
assert graph_suitability(False, True, False) is False
assert graph_suitability(True, True, True) is False
print('✅ Graph 适用性测试通过')

```


```python
cases = [
    ('离线批处理推理', True, True, False),
    ('变长在线推理', False, True, False),
    ('动态分支很多的控制流', True, True, True),
]

for name, fixed_shape, repeated_path, many_branches in cases:
    print(f'{name:<18s}:', '适合 Graph' if graph_suitability(fixed_shape, repeated_path, many_branches) else '不太适合 Graph')

```

## Part 4: 典型流水线总结

### 练习 4.1: 选择职责划分

把任务按 `H2D / Kernel / D2H + Post` 拆开，通常比把所有任务都放进一条 Stream 更容易形成重叠。

```python
def recommended_pipeline(copy_us: float, compute_us: float, post_us: float) -> Dict[str, str]:
    if not should_split_stream(copy_us, compute_us, post_us):
        return {'decision': 'single_stream', 'reason': '重叠收益有限'}
    return {
        'decision': 'split_streams',
        'reason': '搬运 / 计算 / 收尾职责可分离，适合用 Event 串依赖',
    }

print(recommended_pipeline(12, 40, 10))
print(recommended_pipeline(2, 60, 1))

```

🛑 **STOP HERE** 🛑

先确保你能独立完成上面的练习，再看参考答案。

## 参考代码与解析

下面给出一版更精简的参考实现，帮助你对照自己写的版本。


```python
# ==================== 参考答案 ====================

def sequential_time_us_ref(stages: List[TaskStage]) -> float:
    return sum(stage.duration_us for stage in stages)


def graph_suitability_ref(is_fixed_shape: bool, is_repeated_path: bool, has_many_branches: bool) -> bool:
    return is_fixed_shape and is_repeated_path and not has_many_branches

print('参考结论：')
print('- Stream 负责并行调度')
print('- Event 负责局部同步')
print('- Graph 负责稳定路径回放')

```

### 解析

**1. Stream 调度**

- 只有职责可分离、依赖关系清楚时，拆 Stream 才有意义
- 并行收益来自重叠，不来自“Stream 数量多”本身

**2. Event 同步**

- Event 用来连接不同 Stream 的依赖
- 不是越多越好，太密会压缩重叠空间

**3. CUDA Graph**

- Graph 适合固定路径、重复执行的场景
- 动态 shape 和复杂分支会让 Graph 的收益下降

## 关联阅读

这一页和 1D-01 的异构调度、1D-02 的 CUDA / Triton 编程模型是连着的，建议一起看：

- `1D-01` CPU/GPU 异构调度
- `1D-02` CUDA/Triton 编程模型
- `1D-04` 动态 Shape 处理

## 配合练习

这一页后面的练习可以按三步来设计：

- 先做一个 Stream 优先级的小实验，看高 / 低优先级任务的调度差异
- 再做 Event 同步，观察不同 Stream 之间的依赖是怎么被串起来的
- 最后尝试 CUDA Graph 捕获和回放，比较捕获与回放的耗时差异
