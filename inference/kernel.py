"""
基于 TileLang 的 DeepSeek V4 GPU 内核实现

本文件包含 DeepSeek V4 推理所需的核心 GPU 算子，使用 TileLang 编写并编译为 CUDA 内核。
主要涵盖：
  - FP8/FP4 量化与反量化内核（act_quant_kernel, fp4_quant_kernel）
  - FP8/FP4 GEMM 内核（fp8_gemm_kernel, fp4_gemm_kernel）
  - 稀疏注意力内核（sparse_attn_kernel），支持在线 softmax 与 FlashAttention 风格计算
  - HC-Split Sinkhorn 迭代归一化内核（hc_split_sinkhorn_kernel）

数据类型说明：
  - FP8 (float8_e4m3): 1 符号位 + 4 指数位 + 3 尾数位，动态范围约 ±448
  - FP4 (float4_e2m1fn): 1 符号位 + 2 指数位 + 1 尾数位，动态范围约 ±6.0
  - FE8M0 (float8_e8m0fnu): 8 位指数无符号，专用于缩放因子（power-of-2 尺度）
  - BF16: 半精度浮点，推理主要激活数据类型
  - FP32: 单精度浮点，用于内部累加与缩放计算

硬件假设：
  - NVIDIA RTX 5060 Ti 16GB 显存
  - PCIe 5.0 x8 带宽约 16 GB/s
  - 共享内存与寄存器按 TileLang 自动分配策略管理
"""
import torch
import tilelang
import tilelang.language as T
from typing import Tuple, Optional, Dict, Any


# 设置 TileLang 日志级别为 WARNING，减少编译期冗余输出
tilelang.set_log_level("WARNING")

# TileLang 编译优化开关配置
pass_configs = {
    # 禁用 warp-specialized：RTX 5060 Ti 为消费级 GPU，warp 特化收益有限且增加复杂度
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    # 禁用 TMA (Tensor Memory Accelerator)：RTX 系列不支持 Hopper 的 TMA 硬件
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

# ===================== 数据类型常量 =====================
# FP8 E4M3: DeepSeek V4 激活量化主要格式，1符号+4指数+3尾数，最大值 448.0
FP8 = "float8_e4m3"
# FP4 E2M1FN: 路由专家权重量化格式，1符号+2指数+1尾数，最大值 6.0
# 存储时 2 个 FP4 值打包为 1 个 int8 字节
FP4 = "float4_e2m1fn"
# FE8M0: 纯 8 位指数无符号格式，用于表示 power-of-2 缩放因子
# 格式特点: 仅指数无尾数，天然支持通过位操作快速计算 2^x
FE8M0 = "float8_e8m0fnu"
# BF16: 激活与主权重存储格式，推理主数据类型
BF16 = "bfloat16"
# FP32: 内部累加、缩放计算与数值稳定使用
FP32 = "float32"
# INT32: 索引张量数据类型（如 top-k 索引）
INT32 = "int32"


def fast_log2_ceil(x):
    """
    通过 IEEE 754 位操作快速计算 ceil(log2(x))。

    原理：
      1. 将 float32 重新解释为 uint32，直接读取指数域 (bits[30:23])
      2. 指数域偏移量为 127，实际指数 = exp_x - 127
      3. 若尾数域非零，说明 x 不是精确的 2 的幂，ceil 需要 +1
      4. 避免调用昂贵的 log2/ceil 内部函数，完全通过位运算完成

    参数:
        x: 输入浮点值（通常为块内绝对值最大值 amax）
    返回:
        ceil(log2(x)) 的 int32 结果
    """
    # 将 float32 位模式重新解释为 uint32，以便直接操作指数和尾数域
    bits_x = T.reinterpret("uint32", x)
    # 提取指数域：右移 23 位后与 0xFF 掩码，得到 8 位无符号指数
    exp_x = (bits_x >> 23) & 0xFF
    # 提取尾数域：低 23 位，用于判断是否为精确 2 的幂
    man_bits = bits_x & ((1 << 23) - 1)
    # 实际指数 = 指数域 - 127；若尾数非零则 +1 实现 ceil 效果
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    """
    通过 IEEE 754 位操作快速计算 2^x（x 为整数）。

    原理：
      1. FP32 的指数域编码为 x + 127（127 为偏移量 bias）
      2. 将计算后的指数左移 23 位，尾数域保持为 0
      3. 重新解释为 float32，即得到精确的 2^x 值
      4. 完全避免指数函数调用，仅通过位运算完成

    参数:
        x: 整数指数（通常为 fast_log2_ceil 的输出）
    返回:
        2^x 的 float32 结果
    """
    # 计算 IEEE 754 指数域位模式：(x + 127) 左移 23 位，尾数为 0
    bits_x = (x + 127) << 23
    # 将位模式重新解释为 float32
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    """
    将 amax * fp8_max_inv 向上取整到最近的 2 的幂次（power-of-2 scale）。

    这是 MXFP 格式的核心要求：缩放因子必须是 2 的整数幂，
    从而可以通过简单的位操作实现快速量化和反量化。

    参数:
        amax: 块内绝对值最大值
        fp8_max_inv: 目标格式最大值的倒数（如 FP8 为 1/448）
    返回:
        power-of-2 缩放因子
    """
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(
    N, block_size=128, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32,
    round_scale=False, inplace=False
):
    """
    块级 FP8 量化 TileLang 内核。

    功能：
      将输入张量 X[M, N] 按块（block）进行 FP8 量化，输出量化后的张量 Y 和每块缩放因子 S。
      当 inplace=True 时，执行融合 quant+dequant 操作，输出反量化回 BF16 的结果。

    量化策略：
      - 块大小 group_size（默认 128）：每 group_size 列共享一个缩放因子
      - 块行大小 blk_m（32）：每 32 行分配一个 CUDA block，线程数 128
      - 缩放因子计算：s = amax / fp8_max，其中 amax 为块内绝对值最大值
      - round_scale=True 时，缩放因子向上取整为 2 的幂次（MXFP 格式要求）

    内存层次：
      - x_shared: 共享内存，缓存输入块 [blk_m, group_size]
      - x_local: 寄存器片段，用于 reduce_absmax 计算每行 amax
      - amax_local/s_local: 寄存器片段，存储每行最大值和缩放因子
      - y_local/y_shared: 寄存器片段→共享内存，存储量化结果

    流水线：
      - num_stages=2 启用双缓冲流水线（round_scale 或 inplace 时关闭以简化逻辑）

    参数:
        N: 输入张量列数（最后一维大小）
        block_size: 量化块大小，默认 128（即每 128 列一个缩放因子）
        in_dtype: 输入数据类型，默认 BF16
        out_dtype: 输出数据类型，默认 FP8（inplace 时强制为 in_dtype）
        scale_dtype: 缩放因子存储类型，默认 FP32（可配置为 FE8M0）
        round_scale: 是否将缩放因子取整为 2 的幂次
        inplace: 是否执行融合 quant+dequant 回 BF16
    """
    M = T.symbolic("M")
    # FP8 E4M3 格式的有效范围：[-448.0, 448.0]
    fp8_min = -448.0
    fp8_max = 448.0
    # FP8 最大值的倒数，用于将 amax 归一化为缩放因子
    fp8_max_inv = 1 / fp8_max
    # 流水线阶段数：round_scale 或 inplace 时逻辑较复杂，关闭流水线（num_stages=0）
    # 否则启用双缓冲流水线（num_stages=2）隐藏全局内存读取延迟
    num_stages = 0 if round_scale or inplace else 2
    # blk_m: 每个 CUDA block 处理的行数，32 行/block 是权衡寄存器使用和并行度的经验值
    blk_m = 32
    # group_size: 量化粒度，每 group_size 列共享一个缩放因子
    group_size = block_size
    # 内部计算使用 FP32 保证数值精度；scale_dtype 仅控制输出存储格式
    compute_dtype = FP32
    # inplace 模式下输出类型与输入相同（BF16），否则为 FP8
    out_dtype = in_dtype if inplace else out_dtype

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        # 启动配置：
        #   - grid_x = ceil(M / blk_m): 沿行方向划分的块数
        #   - grid_y = ceil(N / group_size): 沿列方向（量化块）划分的块数
        #   - 每个 block 128 线程，对应 warp 级并行处理 32 行
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            # 共享内存分配：缓存从全局内存读取的输入块 [32 行 x 128 列]
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            # 寄存器片段：将共享内存数据加载到寄存器进行逐元素操作
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            # 寄存器片段：存储每行的绝对值最大值（dim=1 表示沿列方向 reduce）
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            # 寄存器片段：存储计算得到的缩放因子
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            # 寄存器片段：存储量化后的输出值
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            # 共享内存：量化结果写回共享内存，再批量拷贝到全局内存
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            # Pipelined 循环：利用 TileLang 的自动流水线调度隐藏内存延迟
            for _ in T.Pipelined(1, num_stages=num_stages):
                # 1. 从全局内存 X 加载当前 block 的数据到共享内存 x_shared
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                # 2. 从共享内存加载到寄存器片段（触发编译器优化数据布局）
                T.copy(x_shared, x_local)
                # 3. 计算每行的绝对值最大值：amax_local[i] = max(|x_local[i, :]|)
                T.reduce_absmax(x_local, amax_local, dim=1)
                # 4. 逐行计算缩放因子，防止除零加入最小值 1e-4
                for i in T.Parallel(blk_m):
                    # 防止 amax 过小导致数值不稳定，设置下限 1e-4
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        # MXFP 格式：缩放因子必须是 2 的幂次，通过 fast_round_scale 取整
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        # 标准 FP8 量化：缩放因子 = amax / fp8_max
                        s_local[i] = amax_local[i] * fp8_max_inv
                # 5. 量化（或融合 quant+dequant）
                if inplace:
                    # 融合 quant+dequant：
                    #   quant = clamp(x / s, fp8_min, fp8_max)  → 先量化到 FP8 范围
                    #   dequant = quant * s  → 反量化回原始尺度
                    #   最终 cast 回 BF16
                    # 这种融合操作常用于训练时的激活检查点，减少内存占用
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(compute_dtype, T.Cast(out_dtype, T.clamp(
                                x_local[i, j] / s_local[i], fp8_min, fp8_max
                            ))) * s_local[i],
                        )
                else:
                    # 标准量化：仅将数值 clamp 到 FP8 范围，除以缩放因子
                    # 输出为 FP8 类型，后续需配合缩放因子反量化
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(
                            x_local[i, j] / s_local[i], fp8_min, fp8_max
                        )
                # 6. 将缩放因子写回全局内存 S
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                # 7. 将量化结果从寄存器拷贝到共享内存，再批量写回全局内存 Y
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32, inplace: bool = False,
) -> torch.Tensor:
    """
    块级 FP8 量化的 Python 封装函数。

    功能：
      对输入张量 x 进行 FP8 块级量化，返回量化后的张量和缩放因子。
      当 inplace=True 时，执行融合 quant+dequant，直接返回 BF16 张量。

    参数:
        x: 输入张量，形状为 [..., N]，N 必须能被 block_size 整除
        block_size: 量化块大小，默认 128
        scale_fmt: 缩放因子格式，若设置则使用 power-of-2 缩放（MXFP）
        scale_dtype: 缩放因子 PyTorch 数据类型，默认 float32
        inplace: 是否执行融合 quant+dequant 回 BF16

    返回:
        inplace=True:  返回反量化后的 BF16 张量（修改 x 原地）
        inplace=False: 返回 (y, s) 元组，y 为 FP8 量化张量，s 为缩放因子
    """
    N = x.size(-1)
    # 确保列数能被块大小整除，这是块级量化的基本要求
    assert N % block_size == 0
    # 根据 PyTorch 的 scale_dtype 选择对应的 TileLang 数据类型
    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
    # 确保输入连续存储，利于全局内存合并访问
    z = x.contiguous()
    # inplace 时输出与输入同类型（BF16）；否则分配 FP8 张量
    y = torch.empty_like(z) if inplace else torch.empty_like(z, dtype=torch.float8_e4m3fn)
    # 缩放因子张量：每 block_size 列一个缩放因子
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=scale_dtype)
    # 编译/加载 TileLang 内核，传入量化参数
    kernel = act_quant_kernel(
        N, block_size, scale_dtype=tl_dtype,
        round_scale=scale_fmt is not None, inplace=inplace,
    )
    # 执行内核：将输入展平为 2D [batch, N] 以匹配内核签名
    kernel(z.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    if inplace:
        # inplace 模式：将结果拷贝回原张量，返回原张量引用
        x.copy_(y)
        return x
    return y, s


@tilelang.jit(pass_configs=pass_configs)
def fp4_quant_kernel(
    N, block_size=32, in_dtype=BF16, scale_dtype=FE8M0, inplace=False
):
    """
    块级 FP4 量化 TileLang 内核。

    功能：
      将输入张量 X[M, N] 按块进行 FP4 量化，输出量化张量 Y 和每块缩放因子 S。
      FP4 格式（e2m1fn）动态范围极小（±6.0），因此采用更细的量化粒度（默认 32）。
      缩放因子强制为 power-of-2（通过 fast_round_scale），这是 FP4 权重存储的标准要求。

    与 FP8 量化的关键差异：
      - 块大小更小（默认 32 vs 128），因为 FP4 动态范围仅 ±6.0，需要更细粒度控制量化误差
      - 缩放因子强制使用 FE8M0（power-of-2），不可配置为非幂次格式
      - 输出存储：2 个 FP4 值打包为 1 字节（int8），沿 K 维度打包

    内存层次：
      与 act_quant_kernel 相同：共享内存缓存输入块，寄存器片段执行 reduce 和逐元素操作。

    参数:
        N: 输入张量列数
        block_size: 量化块大小，默认 32（FP4 需要更细粒度）
        in_dtype: 输入数据类型，默认 BF16
        scale_dtype: 缩放因子类型，默认 FE8M0（power-of-2 专用格式）
        inplace: 是否执行融合 quant+dequant 回 BF16
    """
    M = T.symbolic("M")
    # FP4 E2M1FN 格式的有效范围：[-6.0, 6.0]
    # 1 符号位 + 2 指数位 + 1 尾数位，可表示值：±0.0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max
    # blk_m: 每个 CUDA block 处理 32 行，与 FP8 量化保持一致
    blk_m = 32
    # group_size: FP4 量化粒度，默认 32 列/块（比 FP8 的 128 更细）
    group_size = block_size
    # 内部计算使用 FP32 保证数值精度
    compute_dtype = FP32
    # inplace 时输出回 BF16，否则输出 FP4（打包格式）
    out_dtype = in_dtype if inplace else FP4

    @T.prim_func
    def fp4_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        # 启动配置与 FP8 量化相同：
        #   grid = [ceil(M/blk_m), ceil(N/group_size)]，每 block 128 线程
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            # 共享内存：缓存输入块 [32 行 x 32 列]，比 FP8 的 [32x128] 小 4 倍
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            # 寄存器片段：从共享内存加载后进行逐元素操作
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            # 寄存器片段：存储每行绝对值最大值
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            # 寄存器片段：存储 power-of-2 缩放因子
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            # 寄存器片段：存储量化结果
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            # 共享内存：量化结果中转写回全局内存
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            # 启用双缓冲流水线（num_stages=2），隐藏全局内存读取延迟
            for _ in T.Pipelined(1, num_stages=2):
                # 1. 从全局内存加载输入块到共享内存
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                # 2. 从共享内存加载到寄存器片段
                T.copy(x_shared, x_local)
                # 3. 计算每行绝对值最大值
                T.reduce_absmax(x_local, amax_local, dim=1)
                # 4. 逐行计算缩放因子
                for i in T.Parallel(blk_m):
                    # 设置下限为 6 * 2^-126，防止 subnormal 数值问题
                    # 2^-126 是 FP32 最小正规格化数，确保缩放因子不会导致下溢
                    amax_local[i] = T.max(amax_local[i], 6 * (2**-126))
                    # FP4 强制使用 power-of-2 缩放因子，通过 fast_round_scale 实现
                    # fast_round_scale = 2^ceil(log2(amax / fp4_max))
                    s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)
                # 5. 量化（或融合 quant+dequant）
                if inplace:
                    # 融合 quant+dequant：先量化到 FP4 范围，再反量化回 BF16
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(compute_dtype, T.Cast(FP4, T.clamp(
                                x_local[i, j] / s_local[i], -fp4_max, fp4_max
                            ))) * s_local[i],
                        )
                else:
                    # 标准 FP4 量化：clamp 到 FP4 范围并除以缩放因子
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(
                            x_local[i, j] / s_local[i], -fp4_max, fp4_max
                        )
                # 6. 将缩放因子写回全局内存（FE8M0 格式存储）
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                # 7. 量化结果经共享内存中转，批量写回全局内存
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return fp4_quant_kernel_


def fp4_act_quant(
    x: torch.Tensor, block_size: int = 32, inplace: bool = False,
) -> torch.Tensor:
    """
    块级 FP4 量化的 Python 封装函数。

    功能：
      对输入张量 x 进行 FP4 块级量化，返回量化后的打包张量和 FE8M0 缩放因子。
      FP4 权重以 float4_e2m1fn_x2 格式存储（2 个 FP4 打包为 1 字节）。

    参数:
        x: 输入张量，形状为 [..., N]，N 必须能被 block_size 整除
        block_size: 量化块大小，默认 32
        inplace: 是否执行融合 quant+dequant 回 BF16

    返回:
        inplace=True:  返回反量化后的 BF16 张量（修改 x 原地）
        inplace=False: 返回 (y, s) 元组
            - y: FP4 量化张量，形状 [..., N//2]，dtype=torch.float4_e2m1fn_x2
            - s: 缩放因子，形状 [..., N//block_size]，dtype=torch.float8_e8m0fnu
    """
    N = x.size(-1)
    # 确保列数能被块大小整除
    assert N % block_size == 0
    # 确保输入连续存储
    z = x.contiguous()
    # inplace 时输出与输入同类型；否则分配 FP4 打包张量（N//2 因为 2 个 FP4/字节）
    y = torch.empty_like(z) if inplace else z.new_empty(*z.shape[:-1], N // 2, dtype=torch.float4_e2m1fn_x2)
    # 缩放因子：每 block_size 列一个 FE8M0 缩放因子
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=torch.float8_e8m0fnu)
    # 编译/加载 FP4 量化内核
    kernel = fp4_quant_kernel(N, block_size, inplace=inplace)
    # 执行内核：FP4 输出张量最后一维为 N//2（打包后），需展平匹配
    kernel(z.view(-1, N), y.view(-1, y.size(-1)), s.view(-1, N // block_size))
    if inplace:
        x.copy_(y)
        return x
    return y, s


@tilelang.jit(pass_configs=pass_configs)
def fp8_gemm_kernel(N, K, out_dtype=BF16, accum_dtype=FP32, scale_dtype=FP32):
    """
    FP8 GEMM TileLang 内核：C[M, N] = A[M, K] @ B[N, K]^T。

    功能：
      执行带有块级缩放因子的 FP8 矩阵乘法。
      A 和 B 均为 FP8 量化张量，每 128 列（K 维度）共享一个缩放因子。
      计算时先将 FP8 乘积累加到 FP32，再乘以对应的缩放因子，最后输出 BF16 或 FP32。

    分块策略（Tiling）：
      - block_M = 32: 每个 CUDA block 处理 C 的 32 行
      - block_N = 128: 每个 CUDA block 处理 C 的 128 列
      - block_K = 128: K 维度分块大小，与 group_size 一致简化缩放因子索引
      - 每个 block 128 线程，通过 warp 级并行高效利用 Tensor Core

    流水线优化：
      - num_stages=4: 四缓冲流水线，深度隐藏全局内存读取延迟
      - T.use_swizzle(panel_size=10): L2 缓存 swizzle 优化，改善缓存局部性

    缩放因子应用：
      - A 的缩放因子：每 block_M 行、每 group_size 列一个，形状 [M, K//128]
      - B 的缩放因子：每 block_N 行、每 group_size 列一个，形状 [N//128, K//128]
      - 每个 K 块的缩放乘积：scale_a[i] * scale_b[j]，在寄存器中计算

    精度策略：
      - C_local: 当前 K 块的 FP8 乘积累加器（会被每轮清空）
      - C_local_accum: 带缩放的全局累加器，保持 FP32 精度直到最后写回
      - 分离两个累加器避免精度损失（2x accumulation precision）

    参数:
        N: 输出列数（B 矩阵的行数）
        K: 缩减维度（A 的列数 = B 的列数）
        out_dtype: 输出数据类型，BF16 或 FP32
        accum_dtype: 累加数据类型，默认 FP32
        scale_dtype: 缩放因子存储类型，FP32 或 FE8M0
    """
    # 输出类型只能是 BF16 或 FP32（FP8 累加后需要更高精度输出）
    assert out_dtype in [BF16, FP32]

    M = T.symbolic("M")
    # group_size: 缩放因子粒度，128 列共享一个缩放因子（与 block_K 一致）
    group_size = 128
    # block_M: 每个 CUDA block 处理的输出矩阵行数
    block_M = 32
    # block_N: 每个 CUDA block 处理的输出矩阵列数
    block_N = 128
    # block_K: K 维度分块大小，与 group_size 一致以简化缩放因子索引计算
    block_K = 128

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP8],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), scale_dtype],
        scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), scale_dtype],
    ):
        # 启动配置：
        #   - grid_x = ceil(N / block_N): 沿输出列方向划分的块数
        #   - grid_y = ceil(M / block_M): 沿输出行方向划分的块数
        #   - 每个 block 128 线程，4 个 warp
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            # 共享内存：缓存 A 的子块 [32 x 128]（FP8）
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            # 共享内存：缓存 B 的子块 [128 x 128]（FP8）
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            # 共享内存：缓存输出子块 [32 x 128]（BF16/FP32）
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            # 共享内存：缓存每行的缩放因子乘积 [32]（FP32）
            Scale_C_shared = T.alloc_shared((block_M), FP32)
            # 寄存器片段：当前 K 块的乘积累加器（每轮清空）
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            # 寄存器片段：全局累加器，存储带缩放的最终结果
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            # L2 缓存 swizzle 优化：panel_size=10 改善跨 block 的缓存局部性
            # 通过重排内存访问模式，减少 L2 缓存冲突和未命中
            T.use_swizzle(panel_size=10)
            # 初始化两个累加器为 0
            T.clear(C_local)
            T.clear(C_local_accum)

            # K 维度迭代次数：沿 K 轴分块循环
            K_iters = T.ceildiv(K, block_K)
            # 四缓冲流水线：在读取下一个 K 块的同时计算当前块，深度隐藏内存延迟
            for k in T.Pipelined(K_iters, num_stages=4):
                # 1. 从全局内存加载 A 的子块 [block_M, block_K] 到共享内存
                T.copy(A[by * block_M, k * block_K], A_shared)
                # 2. 从全局内存加载 B 的子块 [block_N, block_K] 到共享内存
                T.copy(B[bx * block_N, k * block_K], B_shared)
                # 3. 计算缩放因子：
                #    - B 的缩放因子：每个 block_N 组（128 列）在 K 维度上每 128 列一个
                #    - 索引：bx * block_N // group_size = 当前 block 对应的 B 行组
                #            k = 当前 K 块索引（因为 block_K == group_size）
                Scale_B = T.Cast(FP32, scales_b[bx * block_N // group_size, k])
                #    - A 的缩放因子：每行一个，与 B 缩放因子相乘得到最终每行缩放
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = T.Cast(FP32, scales_a[by * block_M + i, k]) * Scale_B

                # 4. 执行 FP8 x FP8 GEMM：C_local += A_shared @ B_shared^T
                #    transpose_B=True 表示 B 以转置形式参与计算（B 实际存储为 [N, K]）
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # 5. 将当前 K 块的乘积乘以缩放因子，累加到全局累加器
                #    分离 C_local 和 C_local_accum 的原因：
                #    - C_local 每轮被清空，仅用于原始 FP8 乘积累加
                #    - C_local_accum 存储缩放后的最终结果，避免重复缩放
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                # 6. 清空当前 K 块累加器，为下一轮做准备
                T.clear(C_local)
            # 7. 将最终结果从寄存器拷贝到共享内存，再批量写回全局内存
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp8_gemm_kernel_


def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    FP8 GEMM 的 Python 封装函数：C[M, N] = A[M, K] @ B[N, K]^T。

    功能：
      执行带有每 128 块 FP8 缩放因子的矩阵乘法。
      A 和 B 均为 FP8 量化张量，缩放因子分别按行-块和列-块存储。

    参数:
        a: 左矩阵 A，形状 [..., K]，dtype=torch.float8_e4m3fn
        a_s: A 的缩放因子，形状 [..., K//128]
        b: 右矩阵 B，形状 [N, K]，dtype=torch.float8_e4m3fn（以转置形式参与计算）
        b_s: B 的缩放因子，形状 [N//128, K//128]
        scale_dtype: 缩放因子的 PyTorch 数据类型，默认 float32

    返回:
        c: 输出矩阵，形状 [..., N]，dtype=torch.get_default_dtype()（通常为 BF16）
    """
    # 输入张量必须连续，以确保全局内存合并访问
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scaling factor tensors must be contiguous"
    )
    # 根据 PyTorch scale_dtype 选择 TileLang 对应类型
    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
    K = a.size(-1)
    # M = 总元素数 // K，将输入展平为 2D 矩阵
    M = a.numel() // K
    N = b.size(0)
    # 分配输出张量，使用 PyTorch 默认数据类型（通常为 BF16）
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    # 编译/加载 FP8 GEMM 内核
    kernel = fp8_gemm_kernel(N, K, scale_dtype=tl_dtype)
    # 执行内核：展平 A 为 [M, K]，B 保持 [N, K]，输出展平为 [M, N]
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c


# ===================== IQ2_XS → FP8 反量化内核 =====================
# TileLang 反量化 IQ2_XS → float32（预编译固定形状 kernel）
# CUDA FP8 量化（通过 PyTorch CUDA 操作）
# 优化：w1/w2/w3 形状固定，只需编译 3 个 kernel

# 专家权重固定形状
_EXPERT_SHAPES = {
    "w1": (2048, 7168),
    "w2": (7168, 2048),
    "w3": (2048, 7168),
}

# kernel 缓存：按形状名称缓存



@tilelang.jit(pass_configs=pass_configs)
def sparse_attn_kernel(h: int, d: int, scale=None):
    """
    稀疏多头注意力 TileLang 内核（FlashAttention 风格在线 softmax）。

    功能：
      对每个 (batch, seq_pos) 位置，根据 top-k 索引从 KV 缓存中仅收集需要的 KV 位置，
      计算缩放点积注意力。使用数值稳定的在线 softmax（running max/sum），
      并包含可学习的 attn_sink 偏置项。

    共享内存优化（RTX 5060 Ti 限制 99KB/block）：
      - d_sub = d // 2 = 256：沿 head_dim 分两次处理
      - q_shared: h * d_sub * 2 = 64 * 256 * 2 = 32768 (32KB)
      - kv_shared: block * d_sub * 2 = 64 * 256 * 2 = 32768 (32KB)
      - o_shared: h * d_sub * 2 = 32768 (32KB)
      - acc_s_cast: h * block * 2 = 64 * 64 * 2 = 8192 (8KB)
      - 总计: 106496 (104KB) — 仍超 99KB，需要进一步优化
      - 方案: block=32, h=64, d_sub=256
        q=32KB + kv=16KB + o=32KB + acc=4KB = 84KB ✓

    参数:
        h: 注意力头数（heads）
        d: 每头维度（head dimension）
        scale: 注意力缩放因子，默认 1/sqrt(d)
    """
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    if scale is None:
        scale = (1.0 / d) ** 0.5

    num_stages = 2
    threads = 256
    block = 32
    d_sub = d // 2
    num_blocks = tilelang.cdiv(topk, block)

    @T.prim_func
    def sparse_attn_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        o: T.Tensor[(b, m, h, d), BF16],
        attn_sink: T.Tensor[(h,), FP32],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
    ):
        with T.Kernel(m, b, threads=threads) as (bx, by):
            q_shared = T.alloc_shared((h, d_sub), BF16)
            kv_shared = T.alloc_shared((block, d_sub), BF16)
            o_shared = T.alloc_shared((h, d_sub), BF16)
            acc_s_cast = T.alloc_shared((h, block), BF16)

            idxs = T.alloc_fragment(block, INT32)
            acc_s = T.alloc_fragment((h, block), FP32)
            acc_o = T.alloc_fragment((h, d_sub), FP32)
            scores_max = T.alloc_fragment(h, FP32)
            scores_max_prev = T.alloc_fragment(h, FP32)
            scores_scale = T.alloc_fragment(h, FP32)
            scores_sum = T.alloc_fragment(h, FP32)
            sum_exp = T.alloc_fragment(h, FP32)

            # 沿 d 维度分两次处理
            for d_half in T.serial(2):
                d_start = d_half * d_sub

                T.clear(acc_o)
                T.clear(sum_exp)
                T.fill(scores_max, -T.infinity(FP32))

                # 加载当前 d_half 的 query
                for i, j in T.Parallel(h, d_sub):
                    q_shared[i, j] = q[by, bx, i, d_start + j]

                for t in T.Pipelined(num_blocks, num_stages=num_stages):
                    for i in T.Parallel(block):
                        idxs[i] = T.if_then_else(t * block + i < topk, topk_idxs[by, bx, t * block + i], -1)
                    for i, j in T.Parallel(block, d_sub):
                        kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], d_start + j], 0)
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] *= scale

                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(h):
                        scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(h):
                        sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]

                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(h, d_sub):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(h):
                    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
                for i, j in T.Parallel(h, d_sub):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, o_shared)
                for i, j in T.Parallel(h, d_sub):
                    o[by, bx, i, d_start + j] = o_shared[i, j]

    return sparse_attn_kernel_


def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """
    稀疏多头注意力的 Python 封装函数。

    功能：
      执行基于 top-k 索引的稀疏注意力计算。
      仅对 top-k 个重要的 KV 位置计算注意力，大幅降低长序列的内存和计算开销。
      若头数小于 16，自动填充到 16 以匹配内核的 warp 效率要求。

    参数:
        q: Query 张量，形状 [batch, seq_len, heads, head_dim]
        kv: KV 缓存张量，形状 [batch, kv_len, head_dim]（所有头共享同一组 KV）
        attn_sink: 注意力汇聚偏置，形状 [heads]，可学习的每个头偏置
        topk_idxs: top-k 索引，形状 [batch, seq_len, topk]，每个 query 位置关注的 KV 索引
        softmax_scale: 注意力缩放因子，通常为 1/sqrt(head_dim)

    返回:
        o: 注意力输出，形状 [batch, seq_len, heads, head_dim]
    """
    b, s, h, d = q.size()
    # 若头数小于 16，填充零头以匹配内核效率（内核假设至少 1 个 warp 处理 heads）
    # 填充后在内核执行完毕再截断，避免修改内核逻辑
    if h < 16:
        q = torch.cat([q, q.new_zeros(b, s, 16 - h, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(16 - h)])
    # 分配与 q 同形状（可能已填充）的输出张量
    o = torch.empty_like(q)
    # 编译/加载稀疏注意力内核，传入实际头数、维度和缩放因子
    kernel = sparse_attn_kernel(q.size(2), d, softmax_scale)
    kernel(q, kv, o, attn_sink, topk_idxs)
    # 若之前填充了头数，截断回原始头数
    if h < 16:
        o = o.narrow(2, 0, h).contiguous()
    return o


@tilelang.jit(pass_configs=pass_configs)
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    """
    HC-Split Sinkhorn 迭代归一化 TileLang 内核。

    功能：
      对 MoE（混合专家）路由的混合权重进行 Sinkhorn 迭代归一化，
      生成 pre-routing、post-routing 和 combination 三个矩阵。
      这是 DeepSeek V4 中专家路由的关键步骤，确保专家负载均衡。

    Sinkhorn 算法原理：
      Sinkhorn 迭代是一种将非负矩阵转换为双随机矩阵（doubly stochastic）的算法。
      通过交替进行行归一化和列归一化，使矩阵的每行和每列之和都接近 1。
      在 MoE 路由中，这确保每个专家接收的 token 数量大致均衡。

      迭代步骤（对 comb 矩阵 [hc, hc]）：
        1. 初始化：comb = softmax(comb, dim=-1) + eps（行方向 softmax）
        2. 列归一化：comb = comb / (sum(comb, dim=-2) + eps)
        3. 重复 sinkhorn_iters-1 次：
           a. 行归一化：comb = comb / (sum(comb, dim=-1) + eps)
           b. 列归一化：comb = comb / (sum(comb, dim=-2) + eps)

    输入 mixes 的布局：
      mixes 形状为 [n, (2+hc)*hc]，包含三部分：
        - pre 部分: mixes[:, 0:hc]        → 经 sigmoid 得到 pre-routing 权重
        - post 部分: mixes[:, hc:2*hc]    → 经 sigmoid 得到 post-routing 权重
        - comb 部分: mixes[:, 2*hc:]      → 经 Sinkhorn 迭代得到 combination 矩阵

    参数:
        hc: 专家分组数（hyper-connection 维度）
        sinkhorn_iters: Sinkhorn 迭代次数，默认 20
        eps: 数值稳定性的极小值，防止除零
    """
    n = T.symbolic("n")
    # mix_hc: mixes 张量的特征维度 = (2 + hc) * hc
    # 其中 hc 是 pre/post 的维度，hc*hc 是 comb 矩阵的展平维度
    mix_hc = (2 + hc) * hc
    # 每个 block 64 线程，足够并行处理 hc 维度的操作
    threads = 64

    @T.prim_func
    def hc_split_sinkhorn_kernel_(
        mixes: T.Tensor[(n, mix_hc), FP32],
        hc_scale: T.Tensor[(3,), FP32],
        hc_base: T.Tensor[(mix_hc,), FP32],
        pre: T.Tensor[(n, hc), FP32],
        post: T.Tensor[(n, hc), FP32],
        comb: T.Tensor[(n, hc, hc), FP32],
    ):
        # 启动配置：每个 sample 一个 block，64 线程处理该 sample 的所有计算
        with T.Kernel(n, threads=threads) as i:
            # 共享内存：缓存当前 sample 的 mixes 向量 [mix_hc]
            mixes_shared = T.alloc_shared(mix_hc, FP32)
            # 寄存器片段：存储 comb 矩阵 [hc, hc]
            comb_frag = T.alloc_fragment((hc, hc), FP32)
            # 从全局内存加载 mixes 到共享内存
            T.copy(mixes[i, :], mixes_shared)

            # 1. 计算 pre-routing 权重：
            #    pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps
            #    sigmoid 将值映射到 (0, 1)，+eps 保证最小值防止后续除零
            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            # 2. 计算 post-routing 权重：
            #    post[j] = 2 * sigmoid(mixes[j+hc] * scale[1] + base[j+hc])
            #    乘以 2 将范围扩展到 (0, 2)，用于调整专家组合的强度
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
            # 3. 初始化 comb 矩阵（未归一化）：
            #    comb[j, k] = mixes[j*hc+k+2*hc] * scale[2] + base[j*hc+k+2*hc]
            #    从 mixes 的第三部分提取，经线性变换后准备 Sinkhorn 归一化
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = mixes_shared[j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]

            # 分配行和与列和的寄存器片段
            row_sum = T.alloc_fragment(hc, FP32)
            col_sum = T.alloc_fragment(hc, FP32)

            # 4. 初始化 comb：行方向 softmax + eps
            #    先计算每行最大值（数值稳定），再做 safe exp 和归一化
            row_max = T.alloc_fragment(hc, FP32)
            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                # 减去行最大值防止 exp 溢出（safe softmax）
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                # 行归一化后加 eps，确保所有元素为正（Sinkhorn 收敛要求）
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            # 5. 列归一化：comb = comb / (sum(comb, dim=-2) + eps)
            #    使每列之和接近 1，这是双随机矩阵的列约束
            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            # 6. Sinkhorn 迭代：交替行归一化和列归一化
            #    每次迭代使矩阵更接近双随机矩阵（行和列和均为 1）
            for _ in T.serial(sinkhorn_iters - 1):
                # 行归一化：comb = comb / (sum(comb, dim=-1) + eps)
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                # 列归一化：comb = comb / (sum(comb, dim=-2) + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            # 7. 将归一化后的 comb 矩阵写回全局内存
            T.copy(comb_frag, comb[i, :, :])

    return hc_split_sinkhorn_kernel_


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    """
    HC-Split Sinkhorn 迭代归一化的 Python 封装函数。

    功能：
      对 MoE 路由的混合权重执行 Sinkhorn 迭代归一化，
      返回 pre-routing 权重、post-routing 权重和 combination 矩阵。
      这是 DeepSeek V4 专家路由负载均衡的关键步骤。

    参数:
        mixes: 混合权重张量，形状 [batch, seq_len, (2+hc)*hc]
        hc_scale: 三部分（pre/post/comb）的缩放因子，形状 [3]
        hc_base: 三部分（pre/post/comb）的偏置，形状 [(2+hc)*hc]
        hc_mult: 专家分组数 hc，默认 4
        sinkhorn_iters: Sinkhorn 迭代次数，默认 20
        eps: 数值稳定性极小值，默认 1e-6

    返回:
        pre:  pre-routing 权重，形状 [batch, seq_len, hc]
        post: post-routing 权重，形状 [batch, seq_len, hc]
        comb: combination 矩阵，形状 [batch, seq_len, hc, hc]
    """
    b, s, _ = mixes.size()
    # 分配输出张量
    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)
    # 编译/加载 Sinkhorn 内核
    kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    # 执行内核：将 mixes 展平为 [batch*seq_len, (2+hc)*hc]
    kernel(mixes.view(-1, (2 + hc_mult) * hc_mult), hc_scale, hc_base,
           pre.view(-1, hc_mult), post.view(-1, hc_mult), comb.view(-1, hc_mult, hc_mult))
    return pre, post, comb


@tilelang.jit(pass_configs=pass_configs)
def fp4_gemm_kernel(N, K, out_dtype=BF16, accum_dtype=FP32, scale_dtype=FP32):
    """
    FP8 激活 × FP4 权重 GEMM TileLang 内核。

    功能：
      计算 C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T。
      这是 DeepSeek V4 MoE 路由专家计算的核心算子：
        - 激活 A 使用 FP8 E4M3 格式，每 128 列一个缩放因子
        - 权重 B 使用 FP4 E2M1FN 格式，每 32 列一个 E8M0 缩放因子
        - FP4 权重在运行时解包为 FP8，然后通过 FP8×FP8 GEMM 计算

    存储格式说明：
      - B 的物理存储形状为 [N, K//2]，dtype=float4_e2m1fn_x2
        （2 个 FP4 值打包为 1 字节，沿 K 维度打包）
      - B 的逻辑形状为 [N, K]，每个元素是一个 FP4 值
      - 加载时通过 T.Cast(FP32) → T.Cast(FP8) 解包并提升精度

    分块策略（Tiling）：
      - block_M = 32: 每个 CUDA block 处理 C 的 32 行
      - block_N = 128: 每个 CUDA block 处理 C 的 128 列
      - block_K = 32: K 维度分块大小，与 weight_group_size 一致
                      这样每个 K 块对应一个权重缩放因子，简化索引
      - n_sub = 4: 每个激活缩放因子覆盖 4 个 K 块（128/32=4）

    流水线优化：
      - num_stages=2: 双缓冲流水线，在 FP4→FP8 类型转换时隐藏延迟
      - T.use_swizzle(panel_size=10): L2 缓存 swizzle 优化

    缩放因子处理：
      - 权重缩放因子：每 block_N 行、每 block_K（32）列一个
        索引直接为 k（因为 block_K == weight_group_size）
      - 激活缩放因子：每 block_M 行、每 act_group_size（128）列一个
        索引为 k // n_sub（每 4 个 K 块共享一个激活缩放因子）

    参数:
        N: 输出列数（B 矩阵的行数）
        K: 缩减维度（A 的列数 = B 的逻辑列数）
        out_dtype: 输出数据类型，默认 BF16
        accum_dtype: 累加数据类型，默认 FP32
        scale_dtype: 缩放因子存储类型，FP32 或 FE8M0
    """
    M = T.symbolic("M")
    # act_group_size: 激活量化粒度，128 列共享一个 FP8 缩放因子
    act_group_size = 128
    # weight_group_size: 权重量化粒度，32 列共享一个 FP4 缩放因子
    # 与 block_K 一致，确保每个 K 块有独立的权重缩放因子
    weight_group_size = 32
    # block_M: 每个 CUDA block 处理的输出行数
    block_M = 32
    # block_N: 每个 CUDA block 处理的输出列数
    block_N = 128
    # block_K: K 维度分块大小，与 weight_group_size 一致简化缩放因子索引
    block_K = 32
    # n_sub: 每个激活缩放因子覆盖的 K 块数 = 128 / 32 = 4
    # 用于计算激活缩放因子的索引：k // n_sub
    n_sub = act_group_size // block_K  # 4 sub-blocks per act scale group

    @T.prim_func
    def fp4_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP4],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, act_group_size)), scale_dtype],
        scales_b: T.Tensor[(N, T.ceildiv(K, weight_group_size)), scale_dtype],
    ):
        # 启动配置：与 fp8_gemm_kernel 相同
        #   grid_x = ceil(N / block_N), grid_y = ceil(M / block_M)
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            # 共享内存：缓存 A 的子块 [32 x 32]（FP8）
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            # 共享内存：缓存 B 的 FP4 子块 [128 x 32]（FP4 打包格式）
            B_fp4_shared = T.alloc_shared((block_N, block_K), FP4)
            # 共享内存：缓存 B 解包后的 FP8 子块 [128 x 32]
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            # 共享内存：缓存输出子块 [32 x 128]
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            # 寄存器片段：当前 K 块的乘积累加器
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            # 寄存器片段：全局累加器，存储带缩放的最终结果
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)
            # 寄存器片段：每行的激活缩放因子 [32]
            scale_a_frag = T.alloc_fragment((block_M,), FP32)
            # 寄存器片段：每列的权重缩放因子 [128]
            scale_b_frag = T.alloc_fragment((block_N,), FP32)

            # L2 缓存 swizzle 优化
            T.use_swizzle(panel_size=10)
            # 初始化累加器
            T.clear(C_local)
            T.clear(C_local_accum)

            # K 维度迭代次数
            K_iters = T.ceildiv(K, block_K)
            # 双缓冲流水线：在加载下一个 K 块的同时计算当前块
            for k in T.Pipelined(K_iters, num_stages=2):
                # 1. 从全局内存加载 A 的 FP8 子块到共享内存
                T.copy(A[by * block_M, k * block_K], A_shared)
                # 2. 从全局内存加载 B 的 FP4 打包子块到共享内存
                T.copy(B[bx * block_N, k * block_K], B_fp4_shared)
                # 3. FP4 → FP8 解包与精度提升：
                #    FP4 无法直接 cast 到 FP8（C++ 重载歧义），
                #    必须先通过 FP32 中转：FP4 → FP32 → FP8
                for i, j in T.Parallel(block_N, block_K):
                    B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))

                # 4. 加载权重缩放因子：
                #    每 block_K（32）列一个权重缩放因子
                #    索引：bx * block_N + i = 当前 block 对应的 B 行
                #          k = 当前 K 块索引（block_K == weight_group_size）
                for i in T.Parallel(block_N):
                    scale_b_frag[i] = T.Cast(FP32, scales_b[bx * block_N + i, k])

                # 5. 加载激活缩放因子：
                #    每 act_group_size（128）列一个激活缩放因子
                #    由于 block_K = 32，每 4 个 K 块共享一个激活缩放因子
                #    索引：k // n_sub = 当前 K 块对应的激活缩放因子组
                for i in T.Parallel(block_M):
                    scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k // n_sub])

                # 6. 执行 FP8 × FP8 GEMM：C_local += A_shared @ B_shared^T
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                # 7. 将当前 K 块的乘积乘以激活和权重缩放因子，累加到全局累加器
                #    缩放因子分离的原因：
                #    - 激活和权重使用不同的量化粒度（128 vs 32）
                #    - 分离缩放允许独立调整两种量化策略
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                # 8. 清空当前 K 块累加器，为下一轮做准备
                T.clear(C_local)

            # 9. 将最终结果写回全局内存
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp4_gemm_kernel_


def fp4_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    FP8 激活 × FP4 权重 GEMM 的 Python 封装函数。

    功能：
      计算 C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T。
      A 使用每 128 块的 FP8 激活缩放；B 使用每 32 块的 FP4 E8M0 权重量化。
      B 以 float4_e2m1fn_x2 格式存储（2 个 FP4 打包为 1 字节，沿 K 维度）。

    参数:
        a: 激活矩阵 A，形状 [..., K]，dtype=torch.float8_e4m3fn
        a_s: A 的激活缩放因子，形状 [..., K//128]
        b: 权重矩阵 B，形状 [N, K//2]，dtype=torch.float4_e2m1fn_x2（打包存储）
        b_s: B 的权重缩放因子，形状 [N, K//32]，dtype=torch.float8_e8m0fnu
        scale_dtype: 缩放因子的 PyTorch 数据类型，默认 float32

    返回:
        c: 输出矩阵，形状 [..., N]，dtype=torch.get_default_dtype()（通常为 BF16）
    """
    # 输入张量必须连续，确保全局内存合并访问
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scaling factor tensors must be contiguous"
    )
    # 根据 PyTorch scale_dtype 选择 TileLang 对应类型
    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
    K = a.size(-1)
    # M = 总元素数 // K，将输入展平为 2D 矩阵
    M = a.numel() // K
    N = b.size(0)
    # 分配输出张量
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    # 编译/加载 FP4 GEMM 内核
    kernel = fp4_gemm_kernel(N, K, scale_dtype=tl_dtype)
    # 执行内核
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c
