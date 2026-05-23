# CUDA IQ2_XS 量化优化技术文档

## 1. 概述

IQ2_XS 量化是将 FP4 权重转换为 2.3125 bit/weight 超低比特格式的过程。预量化 33792 个专家权重（DeepSeek V4 Flash）的初始速度仅 0.3 experts/s（CPU），经过多轮优化达到 23.7 experts/s（CUDA GPU），提升约 79 倍。

优化历程概览：

| 阶段 | 速度 | 累计提升 |
|------|------|---------|
| CPU 基线 | 0.3 e/s | 1x |
| CUDA 1 thread/block | 1.1 e/s | 3.7x |
| CUDA 16 threads/block | 2.1 e/s | 7x |
| 常量内存 + 批量启动 | 4.4 e/s | 14.7x |
| 双缓冲流水线 | 8.8 e/s | 29.3x |
| 批量处理 | 9.4 e/s | 31.3x |
| 向量化输出解析 | 23.7 e/s | 79x |

---

## 2. IQ2_XS 量化算法

### 2.1 算法流程

每个 super-block 包含 256 个 float32 元素，量化后输出 74 字节的 `block_iq2_xs` 结构。

```
输入: 256 个 float32 元素 (super-block)
  │
  ├─ 步骤1: 符号提取
  │    16 个子块，每个 16 元素
  │    提取符号使所有值为正
  │
  ├─ 步骤2: 19 候选搜索
  │    对每个子块遍历 19 个候选 scale
  │    找最优量化参数 (最小化 MSE)
  │
  ├─ 步骤3: Grid 查找
  │    将量化后的 L 值编码为 uint16
  │    查 kmap 表获取 grid 索引
  │
  ├─ 步骤4: Off-grid 修正
  │    对 kmap 查不到的编码
  │    搜索最近邻 grid 点
  │
  └─ 步骤5: 编码输出
       block_iq2_xs (74 字节):
         d(2B) + qs[32](64B) + scales[8](8B)
```

**关键公式：**

- 子块量化误差：`error = Σ(w_i - d * scale * grid[qs][i])²`
- 最优 scale 选择：遍历 19 个候选，取最小 MSE
- Grid 编码：`L = (l0, l1, l2)` → `uint16 key = l0 | (l1 << 4) | (l2 << 8)` → `kmap[key] → grid_index`

### 2.2 关键数据结构

| 结构 | 大小 | 说明 |
|------|------|------|
| `block_iq2_xs` | 74 字节/super-block | 量化输出：d(2B) + qs[32](64B) + scales[8](8B) |
| `iq2xs_grid[512]` | 512 × 8 字节 | 512 个 grid 点，常量内存 |
| `kmap[65536]` | 65536 × uint16 | L 值编码 → grid 索引映射 |
| `kneighbors` | 变长 | off-grid 点的最近邻列表 |

**`block_iq2_xs` 内存布局：**

```
偏移    大小    字段       说明
0       2B      d          super-block 缩放因子 (float16)
2       64B     qs[32]     16 个子块 × 2 个 uint16 grid 索引
66      8B      scales[8]  16 个子块的 scale (4-bit 打包，2 个/字节)
总计:   74B
```

**`iq2xs_grid` 结构：**

每个 grid 点包含 8 个 int8 值，表示量化后的 L 值组合。512 个 grid 点覆盖了 2-bit 量化空间中所有常见模式。

---

## 3. CUDA Kernel 优化历程

### 3.1 版本1: 1 thread/block (1.1 e/s)

**设计：** 每个 block 用 32 线程（一个 warp），但只有 thread 0 执行实际量化工作。

```cuda
__global__ void iq2xs_quantize_kernel_v1(const float* src, void* dst, int n_block) {
    int ib = blockIdx.x;  // super-block 索引
    if (ib >= n_block) return;

    // 只有 thread 0 工作
    if (threadIdx.x != 0) return;

    // 处理 256 个元素...
}
```

**问题：**
- 31/32 线程空闲，occupancy 约 0.07%
- grid 查找表在全局内存，延迟高
- 单线程串行处理 16 个子块

**性能：** 1.1 experts/s

### 3.2 版本2: 16 threads/block (2.1 e/s)

**设计：** 每个 thread 独立处理一个 16 元素子块，16 个子块并行。

```cuda
__global__ void iq2xs_quantize_kernel_v2(const float* src, void* dst, int n_block) {
    int ib = blockIdx.x;       // super-block 索引
    int isub = threadIdx.x;    // 子块索引 (0-15)
    if (ib >= n_block || isub >= 16) return;

    // 每个 thread 处理 16 个元素
    float scale = find_best_scale(src, ib, isub);
    uint16_t qs = find_grid_index(src, ib, isub, scale);

    // sigma2 计算: 16 线程并行 + warp shuffle 归约
    float sigma2 = compute_sigma2(src, ib, isub, scale);

    // 写入共享内存
    s_scale[isub] = scale;
    s_qs[isub] = qs;
    s_sigma2[isub] = sigma2;

    __syncthreads();

    // thread 0 汇总结果并写入全局内存
    if (isub == 0) {
        // 从共享内存读取所有子块结果
        // 找 max_scale, 写 block_iq2_xs
    }
}
```

**关键优化点：**

1. **sigma2 计算：** 16 线程各自计算子块的 sigma2，通过 warp shuffle 归约求和
2. **max_scale 计算：** 共享内存汇聚 16 个子块的 scale，thread 0 找最大值
3. **最终编码：** thread 0 从共享内存读取所有子块结果，组装 `block_iq2_xs` 写入全局内存

**寄存器使用：** 121 个/thread，0 spill

**性能：** 2.1 experts/s，提升 1.9x

### 3.3 常量内存优化

**改动：** 将 `iq2xs_grid[512]`（4KB）从全局内存移至 `__constant__` 内存。

```cuda
__constant__ int8_t iq2xs_grid_const[512][8];  // 4KB 常量内存

// 初始化时上传
cudaMemcpyToSymbol(iq2xs_grid_const, iq2xs_grid, 512 * 8);
```

**收益：**
- 4KB 只读广播缓存，延迟比全局内存低
- 所有线程读取同一 grid 点时，一次内存事务广播给整个 warp
- 节省一次 `cudaMalloc` + `cudaMemcpy`

**注意：** 常量内存上限 64KB，4KB 的 grid 表完全在限制内。

### 3.4 批量 Kernel 启动

**改动：** FFI 函数从逐行启动改为一次性启动所有 block。

```cuda
// 之前: 逐行启动 (N 次 kernel launch)
for (int i = 0; i < n_rows; i++) {
    iq2xs_quantize_kernel<<<1, 16, 0, stream>>>(
        src + i * 256, dst + i * 74, 1);
}

// 之后: 批量启动 (1 次 kernel launch)
iq2xs_quantize_kernel<<<n_rows, 16, 0, stream>>>(
    src, dst, n_rows);
```

**收益：** 消除 N 次 kernel launch overhead（每次约 5-10μs），对大批量效果显著。

---

## 4. 预量化流水线优化

### 4.1 初始流水线 (4.4 e/s)

```
CPU: 读FP4 → numpy解码 → f16→f32 → 上传GPU
                                        ↓
GPU:                              CUDA量化
                                        ↓
CPU:                        下载结果 → Python循环解析 → 写归档
```

**瓶颈：** GPU 量化时 CPU 空闲，反之亦然。流水线未重叠。

### 4.2 双缓冲流水线 (8.8 e/s)

**设计：** 使用两个 CUDA stream，双缓冲交替执行。

```
stream_xfer:   [准备 buffer A]              [准备 buffer B]              [准备 buffer C]
stream_compute:              [GPU 量化 A]                [GPU 量化 B]                [GPU 量化 C]
```

**关键改动：**

1. **双 CUDA stream：**
   - `stream_xfer`：负责 CPU→GPU 数据传输
   - `stream_compute`：负责 GPU 量化计算

2. **双缓冲交替：** 准备 buffer A 时 GPU 量化 buffer B

3. **GPU FP4 解码：** FP4→float32 从 CPU numpy 改为 GPU torch 查表
   ```python
   # 之前: CPU numpy 解码
   fp4_weights = np.frombuffer(data, dtype=np.uint8)
   decoded = np.zeros(len(fp4_weights) * 2, dtype=np.float16)
   # ... 逐元素查表 ...

   # 之后: GPU torch 查表
   fp4_tensor = torch.from_numpy(fp4_weights).cuda()
   decoded = dequant_table[fp4_tensor]  # GPU 端查表，零 CPU 开销
   ```

**性能：** 8.8 experts/s，提升 2x

### 4.3 批量处理 (9.4 e/s)

**设计：** 按形状分组，一次 kernel 调用处理多个专家。

```python
# 按形状分组
# w1/w3: shape (7168, 18432) → 可合并
# w2:    shape (18432, 7168) → 单独

# 一次 kernel 调用处理 32 个专家
n_blocks_per_expert = 7168 * 18432 // 256  # = 516096
total_blocks = n_blocks_per_expert * 32     # = 16515072

iq2xs_quantize_ffi(src_batch, dst_batch, 1, total_blocks, stream)
```

**其他优化：**

- **safetensors 预加载：** 一次性加载 shard 内所有张量到 CPU
- **元数据预读取：** `get_slice().get_shape()` 获取元数据，不加载数据

**性能：** 9.4 experts/s

### 4.4 向量化输出解析 (23.7 e/s) — 关键突破

**问题：** Python 逐块循环解析 32 个专家需 7530ms。

**原因分析：**
- 每个专家 114688 个 super-block
- 32 个专家 = 367 万次 `np.frombuffer` 调用
- Python 循环开销 + numpy 小数组创建开销

**解决方案：** numpy stride trick 向量化解析

```python
def _parse_iq2xs_batch(output_cpu: np.ndarray, n_blocks: int) -> dict:
    """向量化解析 IQ2_XS 批量输出"""
    raw = output_cpu.reshape(n_blocks, 74)

    # d: float16 缩放因子
    d_all = raw[:, :2].copy().view(np.float16).reshape(n_blocks)

    # qs: 32 个 uint16 grid 索引
    qs_all = raw[:, 2:66].copy().view(np.uint16).reshape(n_blocks, 32)

    # scales: 8 字节 (16 个 4-bit scale 打包)
    scales_all = raw[:, 66:74].copy().reshape(n_blocks, 8)

    return {
        'd': d_all,
        'qs': qs_all,
        'scales': scales_all,
    }
```

**效果：** 7530ms → 76ms，约 100 倍加速

**原理：**
- `reshape(n_blocks, 74)` 将连续内存视为 (N, 74) 矩阵
- 列切片 `raw[:, :2]` 直接映射到 d 字段，零拷贝
- `.copy().view(np.float16)` 将 2 字节 uint8 重新解释为 float16
- 避免了逐块循环和大量小数组分配

**性能：** 23.7 experts/s，提升 2.5x

---

## 5. 性能分析

### 5.1 GPU 利用率

| 版本 | GPU 利用率 | 原因 |
|------|-----------|------|
| 1 thread/block | ~10% | 121 寄存器，occupancy 0.07%，31/32 线程空闲 |
| 16 threads/block | ~20% | 121 寄存器，occupancy ~1%，SM 调度受限 |
| 双缓冲流水线 | ~40% | CPU 准备时间 > GPU 量化时间，GPU 等待 |
| 批量+向量化 | ~95% | GPU 量化饱和，CPU 解析极快 |

### 5.2 寄存器使用分析

| 配置 | 寄存器数 | spill stores | spill loads | 性能 |
|------|---------|-------------|------------|------|
| 默认 (121 regs) | 121 | 0 | 0 | 基准 |
| maxrregcount=96 | 96 | 192 | 258 | 下降约 30% |
| maxrregcount=128 | 128 | 0 | 0 | 与默认持平 |

**结论：** 121 寄存器、0 spill 是最优平衡点。减少寄存器会导致大量 spill，反而降低性能。

**Occupancy 计算：**
- SM 最大寄存器数：65536
- 121 寄存器/thread × 16 threads/block = 1936 寄存器/block
- 最大 block 数/SM：65536 / 1936 ≈ 33 → 受硬件限制为 32
- 实际 occupancy：16 threads × 32 blocks / 2048 max = 25%

### 5.3 各阶段耗时 (32 个 w1 专家, batch)

| 阶段 | 耗时 | 占比 |
|------|------|------|
| CPU 准备 (torch.stack) | ~12ms | 0.8% |
| GPU 上传 + FP4 解码 | ~200ms | 12.5% |
| CUDA 量化 | ~1300ms | 81.3% |
| 向量化解析 | ~60ms | 3.8% |
| 写归档 | ~50ms | 3.1% |
| **总计** | **~1622ms** | **100%** |

**瓶颈分析：** CUDA 量化占 81%，已接近 GPU 计算极限。进一步优化需从算法层面入手。

---

## 6. 代码架构

### 6.1 CUDA Kernel (`csrc/iq2_xs_quantize.cu`)

```
csrc/iq2_xs_quantize.cu
├── iq2xs_quantize_kernel       # 核心量化 kernel
│   ├── 16 threads/block
│   ├── 常量内存 grid 查找
│   └── warp shuffle 归约
├── iq2xs_quantize_ffi          # FFI 入口，批量启动 kernel
├── iq2xs_quantize_init         # 初始化常量内存和 GPU 查找表
├── nearest_int_cuda            # 设备端快速取整
└── iq2_find_best_neighbour_cuda # 设备端最近邻搜索
```

**关键函数说明：**

- `iq2xs_quantize_kernel`：每个 block 处理一个 super-block（256 元素），16 个线程各处理一个子块
- `iq2xs_quantize_ffi`：Python 调用入口，接收源数据指针、输出指针、元素数和 CUDA stream
- `iq2xs_quantize_init`：初始化时上传 `iq2xs_grid` 到常量内存，上传 `kmap` 和 `kneighbors` 到全局内存
- `nearest_int_cuda`：设备端 `round()` 替代，避免 `__saturate` 分支
- `iq2_find_best_neighbour_cuda`：off-grid 编码的最近邻搜索，遍历 kneighbors 列表

### 6.2 预量化流水线 (`inference/prequant_iq2xs.py`)

```
inference/prequant_iq2xs.py
├── prequant_iq2xs              # 主入口，CUDA/CPU 路径选择
├── load_batch_cpu              # CPU 端批量加载
├── prepare_batch               # GPU 端上传 + FP4 解码
├── launch_quantize             # 异步启动 GPU 量化
├── consume_batch               # 等待完成 + 向量化解析 + 写归档
└── _parse_iq2xs_batch          # numpy 向量化输出解析
```

**流水线执行流程：**

```python
# 伪代码
def prequant_iq2xs(model_path, output_path, use_gpu):
    for shard in model_shards:
        tensors = load_shard(shard)          # 预加载整个 shard

        for batch in group_by_shape(tensors): # 按形状分组
            # 双缓冲流水线
            buf_a = prepare_batch(batch[0])   # 准备第一批
            launch_quantize(buf_a)            # 异步启动

            for i in range(1, len(batch)):
                buf_b = prepare_batch(batch[i])  # 准备下一批
                sync_stream()                     # 等待上一批完成
                result = consume_batch(buf_a)     # 解析 + 写归档
                launch_quantize(buf_b)            # 启动新一批
                buf_a = buf_b                     # 交换缓冲区

            sync_stream()
            consume_batch(buf_a)               # 处理最后一批
```

### 6.3 Python 封装 (`inference/iq2xs_cuda_quant.py`)

```
inference/iq2xs_cuda_quant.py
├── iq2xs_quantize_cuda         # 高层 Python API
├── is_cuda_quantize_available   # 检测 .so 是否可用
└── build_cuda_quantize          # JIT 编译
```

**API 使用示例：**

```python
from iq2xs_cuda_quant import iq2xs_quantize_cuda, is_cuda_quantize_available

if is_cuda_quantize_available():
    # src: float32 tensor on GPU, shape (N * 256,)
    # dst: pre-allocated output on GPU, shape (N * 74,)
    iq2xs_quantize_cuda(src, dst, N, stream=0)
```

---

## 7. 进一步优化方向

### 7.1 多 Stream 流水线

CPU 解码 FP4 与 GPU 量化重叠。当前 CPU 准备极快（12ms），收益有限。

```
stream_decode: [FP4 解码 A] [FP4 解码 B] [FP4 解码 C]
stream_quant:              [量化 A]      [量化 B]      [量化 C]
```

### 7.2 FP4 解码融合

将 FP4 解码合并到量化 kernel 中，避免中间 float32 张量。

```cuda
// 当前: 两步
// 1. FP4 → float32 (torch 查表)
// 2. float32 → IQ2_XS (CUDA kernel)

// 优化: 融合为一步
// FP4 → IQ2_XS (单个 kernel)
__global__ void fp4_to_iq2xs_kernel(const uint8_t* fp4_src, void* iq2xs_dst, ...) {
    // 在 kernel 内部完成 FP4 解码 + IQ2_XS 量化
    float values[16];  // 寄存器中解码
    // ... 直接量化 ...
}
```

**预期收益：** 减少 200ms 的 GPU 上传 + FP4 解码时间，节省显存带宽。

### 7.3 异步 I/O

在 GPU 量化当前 shard 时，后台线程读取下一个 shard。

```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    future_shard = executor.submit(load_shard, shard_paths[0])

    for i in range(len(shard_paths)):
        tensors = future_shard.result()
        if i + 1 < len(shard_paths):
            future_shard = executor.submit(load_shard, shard_paths[i + 1])

        # GPU 量化当前 shard (与下一个 shard 的 I/O 重叠)
        process_shard(tensors)
```

### 7.4 多 GPU

多卡并行量化不同 shard，线性加速。

```python
# 每张 GPU 处理不同的 shard
for gpu_id, shard in enumerate(shard_paths):
    torch.cuda.set_device(gpu_id)
    process_shard(shard)
```

**预期收益：** N 卡 → 约 N 倍加速（受限于 I/O 带宽）。

---

## 附录 A: 编译与构建

### JIT 编译

```python
from iq2xs_cuda_quant import build_cuda_quantize

# 自动检测 CUDA 路径，编译 .so
so_path = build_cuda_quantize()
print(f"Compiled: {so_path}")
```

### 手动编译

```bash
nvcc -shared -o iq2xs_quantize.so \
    csrc/iq2_xs_quantize.cu \
    -Xcompiler -fPIC \
    -O3 \
    --use_fast_math \
    -arch=sm_90
```

### 环境要求

- CUDA >= 12.0
- Python >= 3.10
- PyTorch >= 2.0
- NumPy >= 1.24
- GPU: SM >= 80 (Ampere+)

---

## 附录 B: 常见问题

### Q1: 为什么不减少寄存器来提高 occupancy？

121 寄存器已经是 0 spill 的最小值。强制减少到 96 会导致 192 次 spill stores 和 258 次 spill loads，本地内存访问延迟远超寄存器，性能反而下降约 30%。

### Q2: 为什么向量化解析能加速 100 倍？

Python 逐块循环的主要开销不在计算，而在：
1. `np.frombuffer` 每次创建新数组对象的 Python 开销
2. 367 万次循环的 Python 解释器开销
3. 大量小数组的内存分配/释放

向量化解析通过 numpy 的 C 层面批量操作，避免了所有 Python 循环开销。

### Q3: 常量内存 vs 全局内存，何时选择常量内存？

常量内存适合以下场景：
- 数据量小（< 64KB）
- 只读
- warp 内所有线程读取相同地址（广播）

IQ2_XS 的 grid 表（4KB，只读，warp 内同子块读同 grid 点）完美符合这些条件。

### Q4: 双缓冲流水线的 CPU 端准备为什么只有 12ms？

因为 FP4 解码已移至 GPU（torch 查表），CPU 端只需 `torch.stack` 合并多个专家的张量，这是一个轻量的元数据操作。
