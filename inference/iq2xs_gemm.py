"""原生 CUDA IQ2_XS GEMM kernel

直接计算 y = x @ W^T，其中 W 从 IQ2_XS 反量化得到。
避免中间的 FP8 量化步骤，直接输出 BF16 结果。

使用 PyTorch CUDA extension 编译。
"""
import os
import torch
from torch.utils.cpp_extension import load

# CUDA kernel 源码
CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// FP4 解码表（与官方一致）
__device__ __constant__ float FP4_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// IQ2_XS → float 反量化并累加到输出
// 每个 thread 处理输出的一个元素
// grid: [M, N]（输出形状）
// block: [BLOCK_K]（K 维度分块）
__global__ void iq2xs_gemm_kernel(
    const c10::BFloat16* __restrict__ x,      // [M, K] 输入矩阵
    const uint8_t* __restrict__ indices,      // [N, K_QK, QK] IQ2_XS 索引
    const half* __restrict__ scales,          // [N, K_QK] 每块缩放因子
    const half* __restrict__ offsets,         // [N, K_QK] 每块偏移量
    c10::BFloat16* __restrict__ y,            // [M, N] 输出矩阵
    int M, int N, int K, int QK
) {
    // 当前 thread 处理的输出位置
    int m = blockIdx.y;  // 输出行
    int n = blockIdx.x;  // 输出列
    int tid = threadIdx.x;  // K 维度分块内的 thread ID

    if (m >= M || n >= N) return;

    // 累加器
    float accum = 0.0f;

    // K 维度分块迭代
    // 每个 thread 处理 K / blockDim.x 个元素
    int k_per_thread = (K + blockDim.x - 1) / blockDim.x;
    int k_start = tid * k_per_thread;
    int k_end = min(k_start + k_per_thread, K);

    for (int k = k_start; k < k_end; ++k) {
        // 读取 x[m, k]
        float x_val = static_cast<float>(x[m * K + k]);

        // 计算 IQ2_XS 块索引
        int block_id = k / QK;
        int local_id = k % QK;

        // 读取 scale 和 offset
        float scale = __half2float(scales[n * ((K + QK - 1) / QK) + block_id]);
        float offset = __half2float(offsets[n * ((K + QK - 1) / QK) + block_id]);

        // 反量化: value = index * scale + offset
        uint8_t idx = indices[n * ((K + QK - 1) / QK) * QK + block_id * QK + local_id];
        float w_val = (float)idx * scale + offset;

        // 累加
        accum += x_val * w_val;
    }

    // 块内归约（沿 K 维度）
    __shared__ float shared_accum[256];
    shared_accum[tid] = accum;
    __syncthreads();

    // 树形归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_accum[tid] += shared_accum[tid + s];
        }
        __syncthreads();
    }

    // 写入输出
    if (tid == 0) {
        y[m * N + n] = static_cast<c10::BFloat16>(shared_accum[0]);
    }
}


torch::Tensor iq2xs_gemm_cuda(
    torch::Tensor x,
    torch::Tensor indices,
    torch::Tensor scales,
    torch::Tensor offsets
) {
    // 输入形状
    int M = x.size(0);
    int K = x.size(1);
    int N = indices.size(0);
    int QK = 256;

    // 输出张量
    auto y = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(x.device()));

    // 启动 kernel
    int threads = 256;
    dim3 blocks(N, M);
    iq2xs_gemm_kernel<<<blocks, threads>>>(
        x.data_ptr<c10::BFloat16>(),
        indices.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<torch::Half>()),
        reinterpret_cast<const half*>(offsets.data_ptr<torch::Half>()),
        y.data_ptr<c10::BFloat16>(),
        M, N, K, QK
    );

    return y;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor iq2xs_gemm_cuda(
    torch::Tensor x,
    torch::Tensor indices,
    torch::Tensor scales,
    torch::Tensor offsets
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("iq2xs_gemm", &iq2xs_gemm_cuda, "IQ2_XS GEMM (CUDA)");
}
"""


def compile_iq2xs_gemm():
    """编译 IQ2_XS GEMM CUDA extension。"""
    module = load(
        name="iq2xs_gemm_ext",
        sources=[],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cxxflags=["-O3"],
        verbose=True,
    )
    return module


# 尝试编译（如果失败则返回 None）
_iq2xs_gemm_module = None

try:
    # 直接编译源码
    import tempfile
    import subprocess

    # 创建临时目录
    tmp_dir = tempfile.mkdtemp()

    # 写入源文件
    with open(os.path.join(tmp_dir, "iq2xs_gemm.cu"), "w") as f:
        f.write(CUDA_SOURCE)
    with open(os.path.join(tmp_dir, "iq2xs_gemm.cpp"), "w") as f:
        f.write(CPP_SOURCE)

    # 编译
    _iq2xs_gemm_module = load(
        name="iq2xs_gemm_ext",
        sources=[
            os.path.join(tmp_dir, "iq2xs_gemm.cpp"),
            os.path.join(tmp_dir, "iq2xs_gemm.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90,code=sm_90"],
        verbose=False,
    )
except Exception as e:
    print(f"[WARN] IQ2_XS GEMM CUDA extension 编译失败: {e}")
    _iq2xs_gemm_module = None


def iq2xs_gemm(
    x: torch.Tensor,
    indices: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """IQ2_XS GEMM：y = x @ W^T，其中 W 从 IQ2_XS 反量化得到。

    参数:
        x: [M, K] BF16 输入矩阵
        indices: [N, n_blocks, 256] uint8，IQ2_XS 量化索引
        scales: [N, n_blocks] FP16，每块缩放因子
        offsets: [N, n_blocks] FP16，每块偏移量

    返回:
        y: [M, N] BF16 输出矩阵
    """
    if _iq2xs_gemm_module is None:
        raise RuntimeError("IQ2_XS GEMM CUDA extension 未编译")

    # 展平 indices/scales/offsets
    N = indices.size(0)
    n_blocks = indices.size(1)
    QK = indices.size(2)

    indices_flat = indices.reshape(N, n_blocks * QK)
    scales_flat = scales.reshape(N, n_blocks)
    offsets_flat = offsets.reshape(N, n_blocks)

    return _iq2xs_gemm_module.iq2xs_gemm(x, indices_flat, scales_flat, offsets_flat)


if __name__ == "__main__":
    import time

    if _iq2xs_gemm_module is None:
        print("CUDA extension 未编译，跳过测试")
        exit(0)

    print("=" * 60)
    print("IQ2_XS GEMM CUDA kernel 测试")
    print("=" * 60)

    # 测试参数
    M, N, K = 1, 2048, 7168
    QK = 256
    n_blocks = (K + QK - 1) // QK

    # 准备输入
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    indices = torch.randint(0, 256, (N, n_blocks, QK), dtype=torch.uint8, device='cuda')
    scales = torch.randn(N, n_blocks, dtype=torch.float16, device='cuda')
    offsets = torch.randn(N, n_blocks, dtype=torch.float16, device='cuda')

    # 预热
    y = iq2xs_gemm(x, indices, scales, offsets)
    torch.cuda.synchronize()

    # 测量
    n_iter = 100
    start = time.perf_counter()
    for _ in range(n_iter):
        y = iq2xs_gemm(x, indices, scales, offsets)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iter * 1000

    print(f"输入形状: x=[{M}, {K}], W=[{N}, {K}]")
    print(f"输出形状: y=[{M}, {N}]")
    print(f"GEMM 时间: {elapsed:.3f} ms")

    # 对比 FP4 GEMM
    from kernel import fp4_gemm

    x_fp8 = x.to(torch.float8_e4m3fn)
    x_scale = torch.ones(M, K // 128, dtype=torch.float8_e8m0fnu, device='cuda')
    w_fp4 = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8, device='cuda')
    w_scale = torch.ones(N, K // 32, dtype=torch.float8_e8m0fnu, device='cuda')

    # 预热
    y_fp4 = fp4_gemm(x_fp8, x_scale, w_fp4, w_scale, scale_dtype=torch.float8_e8m0fnu)
    torch.cuda.synchronize()

    # 测量
    start = time.perf_counter()
    for _ in range(n_iter):
        y_fp4 = fp4_gemm(x_fp8, x_scale, w_fp4, w_scale, scale_dtype=torch.float8_e8m0fnu)
    torch.cuda.synchronize()
    fp4_time = (time.perf_counter() - start) / n_iter * 1000

    print(f"\nFP4 GEMM 时间: {fp4_time:.3f} ms")
    print(f"IQ2_XS GEMM vs FP4 GEMM: {elapsed/fp4_time:.2f}x")
