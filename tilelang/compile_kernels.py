"""
TileLang kernel AOT compiler for ds4rs.

Compiles all kernels needed for Phase A (simplified block, no MoE/CSA)
and outputs .so files to tilelang/build/.

Usage:
    docker exec ds4rs-dev python /workspace/tilelang/compile_kernels.py [--batch A1|A2|A3|B1|all]
"""

import sys
import os
import argparse
import time
import torch
import numpy as np
import json

sys.path.insert(0, "/models/inference")

import tilelang
import tilelang.language as T

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
BF16 = "bfloat16"
FP32 = "float32"
FE8M0 = "float8_e8m0fnu"
INT32 = "int32"

BUILD_DIR = "/workspace/tilelang/build"
os.makedirs(BUILD_DIR, exist_ok=True)


def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp_max_inv))


# ============================================================
# K1: act_quant_kernel
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def act_quant_kernel(N, block_size=128, in_dtype=BF16, out_dtype=FP8,
                     scale_dtype=FP32, round_scale=False, inplace=False):
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale or inplace else 2
    blk_m = 32
    group_size = block_size
    compute_dtype = FP32
    out_dtype_actual = in_dtype if inplace else out_dtype

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype_actual],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (pid_m, pid_n):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype_actual)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype_actual)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype_actual,
                            T.Cast(compute_dtype, T.Cast(out_dtype, T.clamp(
                                x_local[i, j] / s_local[i], fp8_min, fp8_max
                            ))) * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(
                            x_local[i, j] / s_local[i], fp8_min, fp8_max
                        )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


# ============================================================
# K3: fp8_gemm_kernel
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def fp8_gemm_kernel(N, K, out_dtype=BF16, accum_dtype=FP32, scale_dtype=FP32):
    M = T.symbolic("M")
    group_size = 128
    block_M = 32
    block_N = 128
    block_K = 128

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP8],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), scale_dtype],
        scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            Scale_C_shared = T.alloc_shared((block_M), FP32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=4):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                Scale_B = T.Cast(FP32, scales_b[bx * block_N // group_size, k])
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = T.Cast(FP32, scales_a[by * block_M + i, k]) * Scale_B

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                T.clear(C_local)
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp8_gemm_kernel_


# ============================================================
# K5: sparse_attn_kernel
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def sparse_attn_kernel(h, d, scale=None, head_group_size=16):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    if scale is None:
        scale = (1.0 / d) ** 0.5

    num_stages = 1
    threads = 256
    block = 64
    num_blocks = tilelang.cdiv(topk, block)
    hg = head_group_size
    num_head_groups = h // hg

    @T.prim_func
    def sparse_attn_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        o: T.Tensor[(b, m, h, d), BF16],
        attn_sink: T.Tensor[(h,), FP32],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
    ):
        with T.Kernel(m, b, threads=threads) as (bx, by):
            q_shared = T.alloc_shared((hg, d), BF16)
            kv_shared = T.alloc_shared((block, d), BF16)
            acc_s_cast = T.alloc_shared((hg, block), BF16)

            idxs = T.alloc_fragment(block, INT32)
            acc_s = T.alloc_fragment((hg, block), FP32)
            acc_o = T.alloc_fragment((hg, d), FP32)
            scores_max = T.alloc_fragment(hg, FP32)
            scores_max_prev = T.alloc_fragment(hg, FP32)
            scores_scale = T.alloc_fragment(hg, FP32)
            scores_sum = T.alloc_fragment(hg, FP32)
            sum_exp = T.alloc_fragment(hg, FP32)

            for hg_idx in T.serial(num_head_groups):
                T.clear(acc_o)
                T.clear(sum_exp)
                T.fill(scores_max, -T.infinity(FP32))
                for i, j in T.Parallel(hg, d):
                    q_shared[i, j] = q[by, bx, hg_idx * hg + i, j]

                for t in T.Pipelined(num_blocks, num_stages=num_stages):
                    for i in T.Parallel(block):
                        idxs[i] = T.if_then_else(t * block + i < topk, topk_idxs[by, bx, t * block + i], -1)
                    for i, j in T.Parallel(block, d):
                        kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                    for i, j in T.Parallel(hg, block):
                        acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(hg, block):
                        acc_s[i, j] *= scale
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(hg):
                        scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(hg, block):
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(hg):
                        sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(hg, d):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(hg):
                    sum_exp[i] += T.exp(attn_sink[hg_idx * hg + i] - scores_max[i])
                for i, j in T.Parallel(hg, d):
                    acc_o[i, j] /= sum_exp[i]
                for i, j in T.Parallel(hg, d):
                    o[by, bx, hg_idx * hg + i, j] = T.Cast(BF16, acc_o[i, j])

    return sparse_attn_kernel_


# ============================================================
# K6: hc_split_sinkhorn_kernel
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def hc_split_sinkhorn_kernel(hc, sinkhorn_iters, eps):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
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
        with T.Kernel(n, threads=threads) as i:
            mixes_shared = T.alloc_shared(mix_hc, FP32)
            comb_frag = T.alloc_fragment((hc, hc), FP32)
            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = mixes_shared[j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]

            row_sum = T.alloc_fragment(hc, FP32)
            col_sum = T.alloc_fragment(hc, FP32)
            row_max = T.alloc_fragment(hc, FP32)

            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])

    return hc_split_sinkhorn_kernel_


# ============================================================
# K7: rmsnorm_kernel
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def rmsnorm_kernel(N, has_weight=True):
    M = T.symbolic("M")
    blk_m = 8 if N > 2048 else 32
    threads = 128

    @T.prim_func
    def rmsnorm_kernel_(
        X: T.Tensor[(M, N), BF16],
        W: T.Tensor[(N,), FP32],
        Y: T.Tensor[(M, N), BF16],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as (pid_m,):
            x_shared = T.alloc_shared((blk_m, N), BF16)
            x_local = T.alloc_fragment((blk_m, N), BF16)
            sq_local = T.alloc_fragment((blk_m, N), FP32)
            sq_sum_local = T.alloc_fragment((blk_m,), FP32)
            rsqrt_local = T.alloc_fragment((blk_m,), FP32)
            y_local = T.alloc_fragment((blk_m, N), BF16)

            T.copy(X[pid_m * blk_m, 0], x_shared)
            T.copy(x_shared, x_local)
            for i, j in T.Parallel(blk_m, N):
                sq_local[i, j] = T.Cast(FP32, x_local[i, j]) * T.Cast(FP32, x_local[i, j])
            T.reduce_sum(sq_local, sq_sum_local, dim=1)
            for i in T.Parallel(blk_m):
                rsqrt_local[i] = T.rsqrt(sq_sum_local[i] / N + 1e-6)
            for i, j in T.Parallel(blk_m, N):
                if has_weight:
                    y_local[i, j] = T.Cast(BF16, T.Cast(FP32, x_local[i, j]) * rsqrt_local[i] * W[j])
                else:
                    y_local[i, j] = T.Cast(BF16, T.Cast(FP32, x_local[i, j]) * rsqrt_local[i])
            T.copy(y_local, x_shared)
            T.copy(x_shared, Y[pid_m * blk_m, 0])

    return rmsnorm_kernel_


# ============================================================
# K8b: rmsnorm_f32_kernel (FP32 -> FP32, no weight)
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def rmsnorm_f32_kernel(N):
    M = T.symbolic("M")
    blk_m = 8 if N > 2048 else 32
    threads = 128

    @T.prim_func
    def rmsnorm_f32_kernel_(
        X: T.Tensor[(M, N), FP32],
        Y: T.Tensor[(M, N), FP32],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as (pid_m,):
            x_shared = T.alloc_shared((blk_m, N), FP32)
            sq_local = T.alloc_fragment((blk_m, N), FP32)
            sq_sum_local = T.alloc_fragment((blk_m,), FP32)
            rsqrt_local = T.alloc_fragment((blk_m,), FP32)
            y_local = T.alloc_fragment((blk_m, N), FP32)

            T.copy(X[pid_m * blk_m, 0], x_shared)
            for i, j in T.Parallel(blk_m, N):
                sq_local[i, j] = x_shared[i, j] * x_shared[i, j]
            T.reduce_sum(sq_local, sq_sum_local, dim=1)
            for i in T.Parallel(blk_m):
                rsqrt_local[i] = T.rsqrt(sq_sum_local[i] / N + 1e-6)
            for i, j in T.Parallel(blk_m, N):
                y_local[i, j] = x_shared[i, j] * rsqrt_local[i]
            T.copy(y_local, x_shared)
            T.copy(x_shared, Y[pid_m * blk_m, 0])

    return rmsnorm_f32_kernel_


# ============================================================
# K9: swiglu_kernel
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def swiglu_kernel(N, swiglu_limit=10.0):
    M = T.symbolic("M")
    blk_m = 32
    blk_n = 128
    threads = 128

    @T.prim_func
    def swiglu_kernel_(
        Gate: T.Tensor[(M, N), BF16],
        Up: T.Tensor[(M, N), BF16],
        Y: T.Tensor[(M, N), BF16],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, blk_n), threads=threads) as (pid_m, pid_n):
            gate_shared = T.alloc_shared((blk_m, blk_n), BF16)
            up_shared = T.alloc_shared((blk_m, blk_n), BF16)
            y_shared = T.alloc_shared((blk_m, blk_n), BF16)

            T.copy(Gate[pid_m * blk_m, pid_n * blk_n], gate_shared)
            T.copy(Up[pid_m * blk_m, pid_n * blk_n], up_shared)

            for i, j in T.Parallel(blk_m, blk_n):
                g = T.Cast(FP32, gate_shared[i, j])
                u = T.Cast(FP32, up_shared[i, j])
                if swiglu_limit > 0:
                    g = T.clamp(g, -swiglu_limit, swiglu_limit)
                    u = T.clamp(u, -swiglu_limit, swiglu_limit)
                silu_g = g / (1.0 + T.exp(-g))
                y_shared[i, j] = T.Cast(BF16, silu_g * u)

            T.copy(y_shared, Y[pid_m * blk_m, pid_n * blk_n])

    return swiglu_kernel_


# ============================================================
# K2: fp4_quant_kernel (block-wise FP4 quantization)
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def fp4_quant_kernel(N, block_size=32, inplace=False):
    M = T.symbolic("M")
    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max
    blk_m = 32
    group_size = block_size
    in_dtype = BF16
    out_dtype = in_dtype if inplace else FP4
    compute_dtype = FP32

    @T.prim_func
    def fp4_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), FE8M0],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (pid_m, pid_n):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=2):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 6 * (2 ** -126))
                    s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(compute_dtype, T.Cast(FP4, T.clamp(
                                x_local[i, j] / s_local[i], -fp4_max, fp4_max
                            ))) * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(
                            x_local[i, j] / s_local[i], -fp4_max, fp4_max
                        )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(FE8M0, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return fp4_quant_kernel_


# ============================================================
# K4: fp4_gemm_kernel (FP8 act x FP4 weight GEMM)
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def fp4_gemm_kernel(N, K, scale_dtype=FP32):
    M = T.symbolic("M")
    act_group_size = 128
    weight_group_size = 32
    block_M = 32
    block_N = 128
    block_K = 32
    n_sub = act_group_size // block_K

    @T.prim_func
    def fp4_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP4],
        C: T.Tensor[(M, N), BF16],
        scales_a: T.Tensor[(M, T.ceildiv(K, act_group_size)), scale_dtype],
        scales_b: T.Tensor[(N, T.ceildiv(K, weight_group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_fp4_shared = T.alloc_shared((block_N, block_K), FP4)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), BF16)
            C_local = T.alloc_fragment((block_M, block_N), FP32)
            C_local_accum = T.alloc_fragment((block_M, block_N), FP32)
            scale_a_frag = T.alloc_fragment((block_M,), FP32)
            scale_b_frag = T.alloc_fragment((block_N,), FP32)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_fp4_shared)
                for i, j in T.Parallel(block_N, block_K):
                    B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))

                for i in T.Parallel(block_N):
                    scale_b_frag[i] = T.Cast(FP32, scales_b[bx * block_N + i, k])

                for i in T.Parallel(block_M):
                    scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k // n_sub])

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp4_gemm_kernel_


# ============================================================
# K8: rotary_emb_kernel (RoPE forward/inverse)
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def rotary_emb_kernel(D, Total_D, inverse=False):
    M = T.symbolic("M")
    H = T.symbolic("H")
    half_d = D // 2
    nope_d = Total_D - D
    threads = 128

    sign = -1.0 if inverse else 1.0

    @T.prim_func
    def rotary_emb_kernel_(
        X_nope: T.Tensor[(M, H, nope_d), BF16],
        X_rope: T.Tensor[(M, H, D), BF16],
        Freqs: T.Tensor[(M, half_d), FP32],
        Y_nope: T.Tensor[(M, H, nope_d), BF16],
        Y_rope: T.Tensor[(M, H, D), BF16],
    ):
        with T.Kernel(M, H, threads=threads) as (row, h):
            T.copy(X_nope[row, h, 0], Y_nope[row, h, 0])

            freq_shared = T.alloc_shared(half_d, FP32)
            T.copy(Freqs[row, 0], freq_shared)

            x0 = T.alloc_fragment(half_d, FP32)
            x1 = T.alloc_fragment(half_d, FP32)

            for k in T.Parallel(half_d):
                x0[k] = T.Cast(FP32, X_rope[row, h, k])
                x1[k] = T.Cast(FP32, X_rope[row, h, half_d + k])

            for k in T.Parallel(half_d):
                cos_f = freq_shared[k]
                sin_f = T.sqrt(1.0 - cos_f * cos_f)
                y0 = x0[k] * cos_f - sign * x1[k] * sin_f
                y1 = sign * x0[k] * sin_f + x1[k] * cos_f
                Y_rope[row, h, k] = T.Cast(BF16, y0)
                Y_rope[row, h, half_d + k] = T.Cast(BF16, y1)

    return rotary_emb_kernel_


# ============================================================
# K8b: rope_interleaved_kernel (RoPE with interleaved layout)
#   Takes full head tensor (M, H, Total_D) and applies RoPE
#   to the last D dimensions in interleaved format.
#   Pairs at (nope_d + 2k, nope_d + 2k+1) are rotated together.
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def rope_interleaved_kernel(D, inverse=False):
    M = T.symbolic("M")
    H = T.symbolic("H")
    half_d = D // 2
    threads = 128

    sign = -1.0 if inverse else 1.0

    @T.prim_func
    def rope_interleaved_kernel_(
        X_rope: T.Tensor[(M, H, D), BF16],
        Cos: T.Tensor[(M, half_d), FP32],
        Sin: T.Tensor[(M, half_d), FP32],
        Y_rope: T.Tensor[(M, H, D), BF16],
    ):
        with T.Kernel(M, H, threads=threads) as (row, h):
            cos_shared = T.alloc_shared(half_d, FP32)
            sin_shared = T.alloc_shared(half_d, FP32)
            T.copy(Cos[row, 0], cos_shared)
            T.copy(Sin[row, 0], sin_shared)

            x0 = T.alloc_fragment(half_d, FP32)
            x1 = T.alloc_fragment(half_d, FP32)

            for k in T.Parallel(half_d):
                x0[k] = T.Cast(FP32, X_rope[row, h, 2 * k])
                x1[k] = T.Cast(FP32, X_rope[row, h, 2 * k + 1])

            for k in T.Parallel(half_d):
                cos_f = cos_shared[k]
                sin_f = sin_shared[k]
                y0 = x0[k] * cos_f - sign * x1[k] * sin_f
                y1 = sign * x0[k] * sin_f + x1[k] * cos_f
                Y_rope[row, h, 2 * k] = T.Cast(BF16, y0)
                Y_rope[row, h, 2 * k + 1] = T.Cast(BF16, y1)

    return rope_interleaved_kernel_


# ============================================================
# cast_bf16_f32_kernel (BF16 <-> FP32 type conversion)
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def cast_bf16_f32_kernel(N, src_dtype=BF16, dst_dtype=FP32):
    M = T.symbolic("M")
    blk_m = 32
    blk_n = 128
    threads = 128

    @T.prim_func
    def cast_bf16_f32_kernel_(
        X: T.Tensor[(M, N), src_dtype],
        Y: T.Tensor[(M, N), dst_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, blk_n), threads=threads) as (pid_m, pid_n):
            x_shared = T.alloc_shared((blk_m, blk_n), src_dtype)
            y_shared = T.alloc_shared((blk_m, blk_n), dst_dtype)

            T.copy(X[pid_m * blk_m, pid_n * blk_n], x_shared)

            for i, j in T.Parallel(blk_m, blk_n):
                y_shared[i, j] = T.Cast(dst_dtype, T.Cast(FP32, x_shared[i, j]))

            T.copy(y_shared, Y[pid_m * blk_m, pid_n * blk_n])

    return cast_bf16_f32_kernel_


# ============================================================
# K14: scatter_add_kernel (MoE expert output weighted merge)
#   For each token i in [0, n_tokens):
#     dst[token_ids[i], d] += weights[i] * src[i, d]   for d in [0, D)
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def scatter_add_kernel(D):
    N = T.symbolic("N")
    M = T.symbolic("M")
    threads = 128

    @T.prim_func
    def scatter_add_kernel_(
        Src: T.Tensor[(N, D), BF16],
        Weights: T.Tensor[(N), FP32],
        TokenIds: T.Tensor[(N), INT32],
        Dst: T.Tensor[(M, D), BF16],
    ):
        with T.Kernel(N, threads=threads) as (i):
            row_id = TokenIds[i]
            w = Weights[i]
            for d in T.Parallel(D):
                val = T.Cast(FP32, Src[i, d]) * w
                Dst[row_id, d] = T.Cast(BF16, T.Cast(FP32, Dst[row_id, d]) + val)

    return scatter_add_kernel_


# ============================================================
# K15: sigmoid_kernel (element-wise sigmoid for HC mixing)
#   out[i] = 1 / (1 + exp(-(in[i] * scale + base[i])))
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def sigmoid_kernel(N):
    threads = 128

    @T.prim_func
    def sigmoid_kernel_(
        In: T.Tensor[(N), FP32],
        Scale: T.Tensor[(1), FP32],
        Base: T.Tensor[(N), FP32],
        Out: T.Tensor[(N), FP32],
    ):
        with T.Kernel(N, threads=threads) as (i):
            s = Scale[0]
            b = Base[i]
            v = In[i] * s + b
            Out[i] = 1.0 / (1.0 + T.exp(-v))

    return sigmoid_kernel_


# ============================================================
# K16: hc_sigmoid_kernel (HC head-level sigmoid with broadcast)
#   For each row i in [0, N):
#     for j in [0, HC):
#       Out[i, j] = BF16(sigmoid(In[i, j] * Scale[0] + Base[j]) + eps)
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def hc_sigmoid_kernel(HC, eps=1e-6):
    N = T.symbolic("N")
    threads = 64

    @T.prim_func
    def hc_sigmoid_kernel_(
        In: T.Tensor[(N, HC), FP32],
        Scale: T.Tensor[(1), FP32],
        Base: T.Tensor[(HC), FP32],
        Out: T.Tensor[(N, HC), BF16],
    ):
        with T.Kernel(N, threads=threads) as (i):
            s = Scale[0]
            for j in T.Parallel(HC):
                v = In[i, j] * s + Base[j]
                Out[i, j] = T.Cast(BF16, 1.0 / (1.0 + T.exp(-v)) + eps)

    return hc_sigmoid_kernel_


# ============================================================
# K17: moe_route_kernel (MoE gate: activation + topk + normalize)
#   One block per row. Serial top-k scan in shared memory.
#   Eliminates D2H/H2D round-trip for route_scores.
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def moe_route_kernel(N, topk, score_func="sigmoid", has_bias=True, route_scale=1.0):
    M = T.symbolic("M")
    threads = 128

    @T.prim_func
    def moe_route_kernel_(
        Scores: T.Tensor[(M, N), FP32],
        Bias: T.Tensor[(N,), FP32],
        TopkWeights: T.Tensor[(M, topk), FP32],
        TopkIndices: T.Tensor[(M, topk), INT32],
    ):
        with T.Kernel(M, threads=threads) as (row,):
            activated = T.alloc_shared(N, FP32)
            original = T.alloc_shared(N, FP32)
            select = T.alloc_shared(N, FP32)
            best_val = T.alloc_fragment(1, FP32)
            best_idx = T.alloc_fragment(1, INT32)
            w_sum = T.alloc_fragment(1, FP32)

            for i in T.Parallel(N):
                v = Scores[row, i]
                if score_func == "sigmoid":
                    activated[i] = 1.0 / (1.0 + T.exp(-v))
                elif score_func == "sqrt_softplus":
                    sp = T.if_then_else(v > 20.0, v, T.log(1.0 + T.exp(v)))
                    activated[i] = T.sqrt(sp)
                else:
                    activated[i] = v

            if score_func == "softmax":
                best_val[0] = -1e9
                for i in T.serial(N):
                    best_val[0] = T.max(best_val[0], activated[i])
                w_sum[0] = 0.0
                for i in T.serial(N):
                    activated[i] = T.exp(activated[i] - best_val[0])
                    w_sum[0] += activated[i]
                for i in T.serial(N):
                    activated[i] = activated[i] / w_sum[0]

            for i in T.Parallel(N):
                original[i] = activated[i]

            for i in T.Parallel(N):
                if has_bias:
                    select[i] = activated[i] + Bias[i]
                else:
                    select[i] = activated[i]

            for k in T.serial(topk):
                best_val[0] = -1e9
                best_idx[0] = 0
                for i in T.serial(N):
                    if select[i] > best_val[0]:
                        best_val[0] = select[i]
                        best_idx[0] = i
                TopkIndices[row, k] = best_idx[0]
                TopkWeights[row, k] = original[best_idx[0]]
                select[best_idx[0]] = -1e9

            if score_func != "softmax":
                w_sum[0] = 0.0
                for k in T.serial(topk):
                    w_sum[0] += TopkWeights[row, k]
                for k in T.serial(topk):
                    TopkWeights[row, k] = TopkWeights[row, k] / w_sum[0] * route_scale

    return moe_route_kernel_


# ============================================================
# K18: indexer_score_kernel (Indexer: q@kv^T weighted sum + topk)
#   Computes attention scores between query and compressed KV cache,
#   applies ReLU + weighted sum across heads, then topk selection.
#   One block per token. Serial top-k scan.
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def indexer_score_kernel(n_heads, head_dim, index_topk):
    M = T.symbolic("M")
    N = T.symbolic("N")
    threads = 128

    @T.prim_func
    def indexer_score_kernel_(
        Q: T.Tensor[(M, n_heads, head_dim), FP32],
        KV: T.Tensor[(1, N, head_dim), FP32],
        Weights: T.Tensor[(M, n_heads), FP32],
        TopkIndices: T.Tensor[(M, index_topk), INT32],
    ):
        with T.Kernel(M, threads=threads) as (tok,):
            scores = T.alloc_shared(N, FP32)
            select = T.alloc_shared(N, FP32)
            best_val = T.alloc_fragment(1, FP32)
            best_idx = T.alloc_fragment(1, INT32)

            for t in T.Parallel(N):
                scores[t] = 0.0

            for h in T.serial(n_heads):
                for t in T.serial(N):
                    dot: T.FP32 = 0.0
                    for d in T.serial(head_dim):
                        dot += Q[tok, h, d] * KV[0, t, d]
                    dot = T.if_then_else(dot > 0.0, dot, 0.0)
                    scores[t] += Weights[tok, h] * dot

            for t in T.Parallel(N):
                select[t] = scores[t]

            for k in T.serial(index_topk):
                best_val[0] = -1e9
                best_idx[0] = 0
                for t in T.serial(N):
                    if select[t] > best_val[0]:
                        best_val[0] = select[t]
                        best_idx[0] = t
                TopkIndices[tok, k] = best_idx[0]
                select[best_idx[0]] = -1e9

    return indexer_score_kernel_


# ============================================================
# K19: compressor_pool_kernel (Compressor: softmax gating pool)
#   For each token, computes softmax over gate scores and
#   weighted sum of KV projections across the compress group.
#   One block per token.
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def compressor_pool_kernel(head_dim, coff):
    M = T.symbolic("M")
    threads = 128

    @T.prim_func
    def compressor_pool_kernel_(
        KV: T.Tensor[(M, coff, head_dim), FP32],
        Gate: T.Tensor[(M, coff, head_dim), FP32],
        Out: T.Tensor[(M, head_dim), FP32],
    ):
        with T.Kernel(M, threads=threads) as (tok,):
            for d in T.Parallel(head_dim):
                max_val: T.FP32 = -1e9
                for c in T.serial(coff):
                    max_val = T.max(max_val, Gate[tok, c, d])
                sum_exp: T.FP32 = 0.0
                for c in T.serial(coff):
                    sum_exp += T.exp(Gate[tok, c, d] - max_val)
                acc: T.FP32 = 0.0
                for c in T.serial(coff):
                    w = T.exp(Gate[tok, c, d] - max_val) / sum_exp
                    acc += w * KV[tok, c, d]
                Out[tok, d] = acc

    return compressor_pool_kernel_


# ============================================================
# K20: rmsnorm_f32_weighted_kernel (FP32 RMSNorm with weight)
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def rmsnorm_f32_weighted_kernel(N):
    M = T.symbolic("M")
    blk_m = 8 if N > 2048 else 32
    threads = 128

    @T.prim_func
    def rmsnorm_f32_weighted_kernel_(
        X: T.Tensor[(M, N), FP32],
        W: T.Tensor[(N,), FP32],
        Y: T.Tensor[(M, N), FP32],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as (pid_m,):
            x_shared = T.alloc_shared((blk_m, N), FP32)
            w_shared = T.alloc_shared((N,), FP32)
            sq_local = T.alloc_fragment((blk_m, N), FP32)
            sq_sum_local = T.alloc_fragment((blk_m,), FP32)
            rsqrt_local = T.alloc_fragment((blk_m,), FP32)
            y_local = T.alloc_fragment((blk_m, N), FP32)

            T.copy(X[pid_m * blk_m, 0], x_shared)
            T.copy(W[0], w_shared)
            for i, j in T.Parallel(blk_m, N):
                sq_local[i, j] = x_shared[i, j] * x_shared[i, j]
            T.reduce_sum(sq_local, sq_sum_local, dim=1)
            for i in T.Parallel(blk_m):
                rsqrt_local[i] = T.rsqrt(sq_sum_local[i] / N + 1e-6)
            for i, j in T.Parallel(blk_m, N):
                y_local[i, j] = x_shared[i, j] * rsqrt_local[i] * w_shared[j]
            T.copy(y_local, x_shared)
            T.copy(x_shared, Y[pid_m * blk_m, 0])

    return rmsnorm_f32_weighted_kernel_


# ============================================================
# K21: compressor_rope_f32_kernel (RoPE for compressor, FP32)
#   Applies interleaved RoPE to last rd dims of (M, d) tensor.
#   First (d-rd) dims are copied unchanged.
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def compressor_rope_f32_kernel(d, rd):
    M = T.symbolic("M")
    half_rd = rd // 2
    rope_start = d - rd
    threads = 128

    @T.prim_func
    def compressor_rope_f32_kernel_(
        X: T.Tensor[(M, d), FP32],
        Cos: T.Tensor[(M, half_rd), FP32],
        Sin: T.Tensor[(M, half_rd), FP32],
        Y: T.Tensor[(M, d), FP32],
    ):
        with T.Kernel(M, threads=threads) as (row,):
            x_shared = T.alloc_shared((d,), FP32)
            cos_shared = T.alloc_shared(half_rd, FP32)
            sin_shared = T.alloc_shared(half_rd, FP32)
            y_shared = T.alloc_shared((d,), FP32)

            T.copy(X[row, 0], x_shared)
            T.copy(Cos[row, 0], cos_shared)
            T.copy(Sin[row, 0], sin_shared)

            for j in T.Parallel(d):
                if j < rope_start:
                    y_shared[j] = x_shared[j]

            for k in T.Parallel(half_rd):
                idx1 = rope_start + 2 * k
                idx2 = rope_start + 2 * k + 1
                c = cos_shared[k]
                s = sin_shared[k]
                v1 = x_shared[idx1]
                v2 = x_shared[idx2]
                y_shared[idx1] = v1 * c - v2 * s
                y_shared[idx2] = v1 * s + v2 * c

            T.copy(y_shared, Y[row, 0])

    return compressor_rope_f32_kernel_


# ============================================================
# K22: fp4_qdq_f32_kernel (FP4 quantize-dequantize in FP32)
#   Block-wise FP4 QDQ: quantize to FP4 then dequantize back,
#   keeping FP32 precision throughout. Inplace-style output.
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def fp4_qdq_f32_kernel(N, block_size=32):
    M = T.symbolic("M")
    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max
    blk_m = 32
    group_size = block_size

    @T.prim_func
    def fp4_qdq_f32_kernel_(
        X: T.Tensor[(M, N), FP32],
        Y: T.Tensor[(M, N), FP32],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (pid_m, pid_n):
            x_shared = T.alloc_shared((blk_m, group_size), FP32)
            y_shared = T.alloc_shared((blk_m, group_size), FP32)
            amax_local = T.alloc_fragment((blk_m,), FP32)
            s_local = T.alloc_fragment((blk_m,), FP32)

            T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
            T.reduce_absmax(x_shared, amax_local, dim=1)
            for i in T.Parallel(blk_m):
                amax_local[i] = T.max(amax_local[i], 6 * (2 ** -126))
                s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)
            for i, j in T.Parallel(blk_m, group_size):
                y_shared[i, j] = T.Cast(FP32, T.Cast(FP4, T.clamp(
                    x_shared[i, j] / s_local[i], -fp4_max, fp4_max
                ))) * s_local[i]
            T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return fp4_qdq_f32_kernel_


# ============================================================
# K23: indexer_causal_adjust_kernel (Apply causal mask + offset to topk indices)
#   For each row, if raw index >= causal_limit, set to -1;
#   otherwise add offset. Eliminates D2H for causal masking.
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def indexer_causal_adjust_kernel(topk):
    M = T.symbolic("M")
    threads = 128

    @T.prim_func
    def indexer_causal_adjust_kernel_(
        Indices: T.Tensor[(M, topk), INT32],
        CausalLimit: T.Tensor[(M,), INT32],
        Offset: T.Tensor[(1,), INT32],
        Out: T.Tensor[(M, topk), INT32],
    ):
        with T.Kernel(M, threads=threads) as (row,):
            limit = CausalLimit[row]
            off = Offset[0]
            for k in T.Parallel(topk):
                raw = Indices[row, k]
                if raw >= limit:
                    Out[row, k] = -1
                else:
                    Out[row, k] = raw + off

    return indexer_causal_adjust_kernel_


# ============================================================
# K24: scale_f32_kernel (Element-wise scale: Y = X * scale[0])
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def scale_f32_kernel(N):
    M = T.symbolic("M")
    threads = 128

    @T.prim_func
    def scale_f32_kernel_(
        X: T.Tensor[(M, N), FP32],
        Scale: T.Tensor[(1,), FP32],
        Y: T.Tensor[(M, N), FP32],
    ):
        with T.Kernel(T.ceildiv(M * N, threads), threads=threads) as (idx,):
            for i in T.Pipelined(T.ceildiv(M * N, T.ceildiv(M * N, threads)), stride=T.ceildiv(M * N, T.ceildiv(M * N, threads))):
                flat = idx * T.ceildiv(M * N, T.ceildiv(M * N, threads)) + i
                if flat < M * N:
                    row = flat // N
                    col = flat % N
                    Y[row, col] = X[row, col] * Scale[0]

    return scale_f32_kernel_


# ============================================================
# K25: compressor_group_kernel (Gather+slice+ape for Compressor prefill grouping)
#   Eliminates D2H of GEMM results by doing grouping on GPU.
#   Takes precomputed row indices and column offsets from CPU.
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def compressor_group_kernel(d, out_dim, pool_size):
    M = T.symbolic("M")
    N = T.symbolic("N")
    ratio = T.symbolic("ratio")
    threads = 128

    @T.prim_func
    def compressor_group_kernel_(
        Src: T.Tensor[(N, out_dim), FP32],
        Ape: T.Tensor[(ratio, out_dim), FP32],
        RowIdx: T.Tensor[(M, pool_size), INT32],
        ColOff: T.Tensor[(pool_size,), INT32],
        IsScore: T.Tensor[(1,), INT32],
        Dst: T.Tensor[(M, pool_size, d), FP32],
    ):
        with T.Kernel(M, threads=threads) as (m,):
            for r in T.Serial(pool_size):
                row = RowIdx[m, r]
                col = ColOff[r]
                is_score = IsScore[0]
                if row >= 0:
                    for dd in T.Parallel(d):
                        v = Src[row, col + dd]
                        a = Ape[r % ratio, col + dd]
                        Dst[m, r, dd] = v + a * is_score
                else:
                    for dd in T.Parallel(d):
                        Dst[m, r, dd] = T.if_then_else(is_score, -1e9, 0.0)

    return compressor_group_kernel_


# ============================================================
# K26: moe_gather_kernel (Gather rows from source by GPU indices)
#   For each output row i, copy src[indices[i], :] to dst[i, :]
#   Eliminates CPU row index construction + H2D for gather
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def moe_gather_kernel(D):
    N = T.symbolic("N")
    M = T.symbolic("M")
    threads = 128

    @T.prim_func
    def moe_gather_kernel_(
        Src: T.Tensor[(M, D), BF16],
        Indices: T.Tensor[(N,), INT32],
        Dst: T.Tensor[(N, D), BF16],
    ):
        with T.Kernel(N, threads=threads) as (i,):
            src_row = Indices[i]
            for d in T.Parallel(D):
                Dst[i, d] = Src[src_row, d]

    return moe_gather_kernel_


# ============================================================
# K27: moe_extract_weights_kernel (Extract per-expert weights and token_ids from gate output)
#   Given gate indices [total, topk] and weights [total, topk],
#   for a specific expert_id, extract (weight, token_id) pairs
#   where indices == expert_id.
#   Output: out_weights[count], out_token_ids[count], count
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def moe_extract_weights_kernel(topk):
    Total = T.symbolic("Total")

    @T.prim_func
    def moe_extract_weights_kernel_(
        GateIndices: T.Tensor[(Total, topk), INT32],
        GateWeights: T.Tensor[(Total, topk), FP32],
        ExpertId: T.Tensor[(1,), INT32],
        OutWeights: T.Tensor[(Total,), FP32],
        OutTokenIds: T.Tensor[(Total,), INT32],
        OutCount: T.Tensor[(1,), INT32],
    ):
        with T.Kernel(1, threads=1) as (_):
            eid = ExpertId[0]
            cnt = 0
            for t in T.Serial(Total):
                for k in T.Serial(topk):
                    if GateIndices[t, k] == eid:
                        OutWeights[cnt] = GateWeights[t, k]
                        OutTokenIds[cnt] = T.Cast(INT32, t)
                        cnt = cnt + 1
            OutCount[0] = cnt

    return moe_extract_weights_kernel_


# ============================================================
# K28: moe_expert_count_kernel (Count tokens per expert from gate indices)
#   Eliminates D2H of gate indices for expert_counts construction
# ============================================================

@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def moe_expert_count_kernel(topk, n_experts):
    Total = T.symbolic("Total")

    @T.prim_func
    def moe_expert_count_kernel_(
        Indices: T.Tensor[(Total, topk), INT32],
        Counts: T.Tensor[(n_experts,), INT32],
    ):
        with T.Kernel(1, threads=128) as (_):
            for i in T.Parallel(n_experts):
                Counts[i] = 0
            for t in T.Serial(Total):
                for k in T.Serial(topk):
                    eid = Indices[t, k]
                    if eid >= 0 and eid < n_experts:
                        Counts[eid] = Counts[eid] + 1

    return moe_expert_count_kernel_


# ============================================================
# Kernel instance definitions for DS-V4 Flash
# ============================================================

KERNEL_INSTANCES = {
    "act_quant_N4096_bs128": lambda: act_quant_kernel(N=4096, block_size=128, round_scale=True, scale_dtype=FE8M0),
    "act_quant_N8192_bs128": lambda: act_quant_kernel(N=8192, block_size=128, round_scale=True, scale_dtype=FE8M0),
    "act_quant_N2048_bs128": lambda: act_quant_kernel(N=2048, block_size=128, round_scale=True, scale_dtype=FE8M0),
    "act_quant_N1024_bs128": lambda: act_quant_kernel(N=1024, block_size=128, round_scale=True, scale_dtype=FE8M0),
    "act_quant_N448_bs64_inplace": lambda: act_quant_kernel(N=448, block_size=64, round_scale=True, inplace=True),

    "fp8_gemm_N32768_K1024": lambda: fp8_gemm_kernel(N=32768, K=1024, scale_dtype=FE8M0),
    "fp8_gemm_N512_K4096": lambda: fp8_gemm_kernel(N=512, K=4096, scale_dtype=FE8M0),
    "fp8_gemm_N1024_K4096": lambda: fp8_gemm_kernel(N=1024, K=4096, scale_dtype=FE8M0),
    "fp8_gemm_N4096_K8192": lambda: fp8_gemm_kernel(N=4096, K=8192, scale_dtype=FE8M0),
    "fp8_gemm_N2048_K4096": lambda: fp8_gemm_kernel(N=2048, K=4096, scale_dtype=FE8M0),
    "fp8_gemm_N4096_K2048": lambda: fp8_gemm_kernel(N=4096, K=2048, scale_dtype=FE8M0),
    "fp8_gemm_N8192_K1024": lambda: fp8_gemm_kernel(N=8192, K=1024, scale_dtype=FE8M0),

    "sparse_attn_h64_d512": lambda: sparse_attn_kernel(h=64, d=512, head_group_size=16),
    "hc_sinkhorn_hc4_it20": lambda: hc_split_sinkhorn_kernel(hc=4, sinkhorn_iters=20, eps=1e-6),

    "rmsnorm_N4096": lambda: rmsnorm_kernel(N=4096, has_weight=True),
    "rmsnorm_N1024": lambda: rmsnorm_kernel(N=1024, has_weight=True),
    "rmsnorm_N512": lambda: rmsnorm_kernel(N=512, has_weight=True),
    "rmsnorm_f32_N4096": lambda: rmsnorm_f32_kernel(N=4096),
    "rmsnorm_f32_N7168": lambda: rmsnorm_f32_kernel(N=7168),
    "rmsnorm_no_weight_N1024": lambda: rmsnorm_kernel(N=1024, has_weight=False),

    "swiglu_N2048": lambda: swiglu_kernel(N=2048, swiglu_limit=10.0),

    "fp4_gemm_N2048_K4096": lambda: fp4_gemm_kernel(N=2048, K=4096, scale_dtype=FE8M0),
    "fp4_gemm_N4096_K2048": lambda: fp4_gemm_kernel(N=4096, K=2048, scale_dtype=FE8M0),

    "rope_forward_D64_TD512": lambda: rotary_emb_kernel(D=64, Total_D=512, inverse=False),
    "rope_inverse_D64_TD512": lambda: rotary_emb_kernel(D=64, Total_D=512, inverse=True),

    "rope_interleaved_fwd_D64": lambda: rope_interleaved_kernel(D=64, inverse=False),
    "rope_interleaved_inv_D64": lambda: rope_interleaved_kernel(D=64, inverse=True),

    "cast_bf16_to_f32_N4096": lambda: cast_bf16_f32_kernel(N=4096, src_dtype=BF16, dst_dtype=FP32),
    "cast_bf16_to_f32_N16384": lambda: cast_bf16_f32_kernel(N=16384, src_dtype=BF16, dst_dtype=FP32),
    "cast_f32_to_bf16_N4096": lambda: cast_bf16_f32_kernel(N=4096, src_dtype=FP32, dst_dtype=BF16),
    "cast_f32_to_bf16_N16384": lambda: cast_bf16_f32_kernel(N=16384, src_dtype=FP32, dst_dtype=BF16),

    "scatter_add_D4096": lambda: scatter_add_kernel(D=4096),
    "scatter_add_D7168": lambda: scatter_add_kernel(D=7168),

    "sigmoid_N4": lambda: sigmoid_kernel(N=4),
    "sigmoid_dynamic": lambda: sigmoid_kernel(N=T.symbolic("N")),

    "hc_sigmoid_hc4": lambda: hc_sigmoid_kernel(HC=4, eps=1e-6),

    "moe_route_sqrtsp_N256_topk6": lambda: moe_route_kernel(N=256, topk=6, score_func="sqrt_softplus", has_bias=True, route_scale=1.5),

    "indexer_score_h64_d128_topk512": lambda: indexer_score_kernel(n_heads=64, head_dim=128, index_topk=512),
    "compressor_pool_d128_c8": lambda: compressor_pool_kernel(head_dim=128, coff=8),
    "compressor_pool_d512_c4": lambda: compressor_pool_kernel(head_dim=512, coff=4),

    "rmsnorm_f32_weighted_N128": lambda: rmsnorm_f32_weighted_kernel(N=128),
    "compressor_rope_f32_d128_rd64": lambda: compressor_rope_f32_kernel(d=128, rd=64),
    "fp4_qdq_f32_N128_bs32": lambda: fp4_qdq_f32_kernel(N=128, block_size=32),
    "cast_f32_to_bf16_N128": lambda: cast_bf16_f32_kernel(N=128, src_dtype=FP32, dst_dtype=BF16),

    "indexer_causal_adjust_topk512": lambda: indexer_causal_adjust_kernel(topk=512),

    "scale_f32_N4096": lambda: scale_f32_kernel(N=4096),

    "compressor_group_d128_od1024_ps8": lambda: compressor_group_kernel(d=128, out_dim=1024, pool_size=8),
    "compressor_group_d128_od1024_ps16": lambda: compressor_group_kernel(d=128, out_dim=1024, pool_size=16),

    "moe_gather_D4096": lambda: moe_gather_kernel(D=4096),
    "moe_extract_weights_topk6": lambda: moe_extract_weights_kernel(topk=6),

    "moe_expert_count_topk6_ne256": lambda: moe_expert_count_kernel(topk=6, n_experts=256),
}

BATCH_A1 = [k for k in KERNEL_INSTANCES if k.startswith("act_quant_") or k.startswith("fp8_gemm_") or k.startswith("sparse_attn_") or k.startswith("hc_sinkhorn_") or k.startswith("hc_sigmoid_")]
BATCH_A2 = [k for k in KERNEL_INSTANCES if k.startswith("rmsnorm_") or k.startswith("swiglu_") or k.startswith("rope_")]
BATCH_A3 = [k for k in KERNEL_INSTANCES if k.startswith("cast_")]
BATCH_B1 = [k for k in KERNEL_INSTANCES if k.startswith("fp4_gemm_")]


def make_dummy_inputs(name):
    device = "cuda"
    M = 32
    if name.startswith("act_quant_"):
        N = int(name.split("_N")[1].split("_")[0])
        bs = int(name.split("_bs")[1].split("_")[0]) if "_bs" in name else 128
        X = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        Y = torch.empty(M, N, dtype=torch.float8_e4m3fn if "inplace" not in name else torch.bfloat16, device=device)
        s_dtype = torch.float8_e8m0fnu if "inplace" not in name else torch.float32
        S = torch.empty(M, N // bs, dtype=s_dtype, device=device)
        return [X, Y, S]
    elif name.startswith("fp8_gemm_"):
        parts = name.replace("fp8_gemm_N", "").split("_K")
        N = int(parts[0])
        K = int(parts[1])
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
        B = torch.randn(N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
        C = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        Sa = torch.ones(M, K // 128, dtype=torch.float8_e8m0fnu, device=device)
        Sb = torch.ones(N // 128, K // 128, dtype=torch.float8_e8m0fnu, device=device)
        return [A, B, C, Sa, Sb]
    elif name.startswith("sparse_attn_"):
        h, d = 64, 512
        b, m, n, topk = 1, 4, 128, 32
        q = torch.randn(b, m, h, d, dtype=torch.bfloat16, device=device)
        kv = torch.randn(b, n, d, dtype=torch.bfloat16, device=device)
        o = torch.empty(b, m, h, d, dtype=torch.bfloat16, device=device)
        sink = torch.zeros(h, dtype=torch.float32, device=device)
        idxs = torch.randint(0, n, (b, m, topk), dtype=torch.int32, device=device)
        return [q, kv, o, sink, idxs]
    elif name.startswith("hc_sinkhorn_"):
        hc = 4
        mix_hc = (2 + hc) * hc
        n = 32
        mixes = torch.randn(n, mix_hc, dtype=torch.float32, device=device)
        hc_scale = torch.ones(3, dtype=torch.float32, device=device)
        hc_base = torch.zeros(mix_hc, dtype=torch.float32, device=device)
        pre = torch.empty(n, hc, dtype=torch.float32, device=device)
        post = torch.empty(n, hc, dtype=torch.float32, device=device)
        comb = torch.empty(n, hc, hc, dtype=torch.float32, device=device)
        return [mixes, hc_scale, hc_base, pre, post, comb]
    elif name.startswith("rmsnorm_f32_"):
        N = int(name.split("_N")[1].split("_")[0])
        X = torch.randn(M, N, dtype=torch.float32, device=device)
        Y = torch.empty(M, N, dtype=torch.float32, device=device)
        return [X, Y]
    elif name.startswith("rmsnorm_"):
        N = int(name.split("_N")[1].split("_")[0])
        X = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        W = torch.ones(N, dtype=torch.float32, device=device)
        Y = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        return [X, W, Y]
    elif name.startswith("swiglu_"):
        N = int(name.split("_N")[1].split("_")[0])
        Gate = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        Up = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        Y = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        return [Gate, Up, Y]
    elif name.startswith("fp4_gemm_"):
        return None
    elif name.startswith("rope_interleaved_"):
        D = int(name.split("_D")[1].split("_")[0])
        H = 64
        half_d = D // 2
        X = torch.randn(M, H, D, dtype=torch.bfloat16, device=device)
        Cos = torch.ones(M, half_d, dtype=torch.float32, device=device)
        Sin = torch.zeros(M, half_d, dtype=torch.float32, device=device)
        Y = torch.empty(M, H, D, dtype=torch.bfloat16, device=device)
        return [X, Cos, Sin, Y]
    elif name.startswith("rope_"):
        D = int(name.split("_D")[1].split("_")[0])
        Total_D = int(name.split("_TD")[1])
        H = 64
        half_d = D // 2
        nope_d = Total_D - D
        X_nope = torch.randn(M, H, nope_d, dtype=torch.bfloat16, device=device)
        X_rope = torch.randn(M, H, D, dtype=torch.bfloat16, device=device)
        Freqs = torch.ones(M, half_d, dtype=torch.float32, device=device)
        Y_nope = torch.empty(M, H, nope_d, dtype=torch.bfloat16, device=device)
        Y_rope = torch.empty(M, H, D, dtype=torch.bfloat16, device=device)
        return [X_nope, X_rope, Freqs, Y_nope, Y_rope]
    elif name.startswith("cast_"):
        parts = name.replace("cast_", "").split("_to_")
        src_part = parts[0]
        dst_part = parts[1]
        N = int(dst_part.split("_N")[1])
        if src_part.startswith("bf16"):
            X = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            Y = torch.empty(M, N, dtype=torch.float32, device=device)
        else:
            X = torch.randn(M, N, dtype=torch.float32, device=device)
            Y = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        return [X, Y]
    elif name.startswith("scatter_add_"):
        D = int(name.split("_D")[1])
        N = 16
        M2 = 32
        Src = torch.randn(N, D, dtype=torch.bfloat16, device=device)
        Weights = torch.ones(N, dtype=torch.float32, device=device)
        TokenIds = torch.randint(0, M2, (N,), dtype=torch.int32, device=device)
        Dst = torch.zeros(M2, D, dtype=torch.bfloat16, device=device)
        return [Src, Weights, TokenIds, Dst]
    elif name.startswith("sigmoid_") and not name.startswith("sigmoid_dynamic"):
        n = 16
        In = torch.randn(n, dtype=torch.float32, device=device)
        Scale = torch.ones(1, dtype=torch.float32, device=device)
        Base = torch.randn(n, dtype=torch.float32, device=device)
        Out = torch.zeros(n, dtype=torch.float32, device=device)
        return [In, Scale, Base, Out]
    elif name.startswith("hc_sigmoid_"):
        hc = int(name.split("_hc")[1].split("_")[0])
        n = 32
        In = torch.randn(n, hc, dtype=torch.float32, device=device)
        Scale = torch.ones(1, dtype=torch.float32, device=device)
        Base = torch.randn(hc, dtype=torch.float32, device=device)
        Out = torch.zeros(n, hc, dtype=torch.bfloat16, device=device)
        return [In, Scale, Base, Out]
    elif name.startswith("sigmoid_dynamic"):
        n = 16
        In = torch.randn(n, dtype=torch.float32, device=device)
        Scale = torch.ones(1, dtype=torch.float32, device=device)
        Base = torch.randn(n, dtype=torch.float32, device=device)
        Out = torch.zeros(n, dtype=torch.float32, device=device)
        return [In, Scale, Base, Out]
    elif name.startswith("moe_route_"):
        N = 256
        topk = 8
        Scores = torch.randn(M, N, dtype=torch.float32, device=device)
        Bias = torch.randn(N, dtype=torch.float32, device=device)
        TopkWeights = torch.zeros(M, topk, dtype=torch.float32, device=device)
        TopkIndices = torch.zeros(M, topk, dtype=torch.int32, device=device)
        return [Scores, Bias, TopkWeights, TopkIndices]
    elif name.startswith("indexer_score_"):
        n_heads = 64
        head_dim = 128
        index_topk = 512
        N = 256
        Q = torch.randn(M, n_heads, head_dim, dtype=torch.float32, device=device)
        KV = torch.randn(1, N, head_dim, dtype=torch.float32, device=device)
        Weights = torch.randn(M, n_heads, dtype=torch.float32, device=device)
        TopkIndices = torch.zeros(M, index_topk, dtype=torch.int32, device=device)
        return [Q, KV, Weights, TopkIndices]
    elif name.startswith("compressor_pool_"):
        parts = name.split("_")
        head_dim = int(parts[2].lstrip("d"))
        coff = int(parts[3].lstrip("c"))
        KV = torch.randn(M, coff, head_dim, dtype=torch.float32, device=device)
        Gate = torch.randn(M, coff, head_dim, dtype=torch.float32, device=device)
        Out = torch.zeros(M, head_dim, dtype=torch.float32, device=device)
        return [KV, Gate, Out]
    elif name.startswith("rmsnorm_f32_weighted_"):
        N = int(name.split("_N")[1])
        X = torch.randn(M, N, dtype=torch.float32, device=device)
        W = torch.ones(N, dtype=torch.float32, device=device)
        Y = torch.empty(M, N, dtype=torch.float32, device=device)
        return [X, W, Y]
    elif name.startswith("compressor_rope_f32_"):
        parts = name.split("_")
        d = int(parts[3].lstrip("d"))
        rd = int(parts[4].lstrip("rd"))
        half_rd = rd // 2
        X = torch.randn(M, d, dtype=torch.float32, device=device)
        Cos = torch.ones(M, half_rd, dtype=torch.float32, device=device)
        Sin = torch.zeros(M, half_rd, dtype=torch.float32, device=device)
        Y = torch.empty(M, d, dtype=torch.float32, device=device)
        return [X, Cos, Sin, Y]
    elif name.startswith("fp4_qdq_f32_"):
        N = int(name.split("_N")[1].split("_")[0])
        X = torch.randn(M, N, dtype=torch.float32, device=device)
        Y = torch.empty(M, N, dtype=torch.float32, device=device)
        return [X, Y]
    elif name.startswith("indexer_causal_adjust_"):
        topk = 512
        Indices = torch.randint(0, 256, (M, topk), dtype=torch.int32, device=device)
        CausalLimit = torch.full((M,), 256, dtype=torch.int32, device=device)
        Offset = torch.zeros(1, dtype=torch.int32, device=device)
        Out = torch.zeros(M, topk, dtype=torch.int32, device=device)
        return [Indices, CausalLimit, Offset, Out]
    elif name.startswith("scale_f32_"):
        N = int(name.split("_N")[1].split("_")[0])
        X = torch.randn(M, N, dtype=torch.float32, device=device)
        Scale = torch.ones(1, dtype=torch.float32, device=device)
        Y = torch.empty(M, N, dtype=torch.float32, device=device)
        return [X, Scale, Y]
    elif name.startswith("compressor_group_"):
        parts = name.split("_")
        d = int(parts[2][1:])
        out_dim = int(parts[3][2:])
        pool_size = int(parts[4][2:])
        ratio = pool_size // 2 if pool_size > 8 else pool_size
        N = M * ratio * 2
        Src = torch.randn(N, out_dim, dtype=torch.float32, device=device)
        Ape = torch.randn(ratio, out_dim, dtype=torch.float32, device=device)
        RowIdx = torch.randint(0, N, (M, pool_size), dtype=torch.int32, device=device)
        ColOff = torch.zeros(pool_size, dtype=torch.int32, device=device)
        IsScore = torch.ones(1, dtype=torch.int32, device=device)
        Dst = torch.zeros(M, pool_size, d, dtype=torch.float32, device=device)
        return [Src, Ape, RowIdx, ColOff, IsScore, Dst]
    elif name.startswith("moe_gather_"):
        D = int(name.split("_D")[1].split("_")[0])
        N = M
        Total = M * 2
        Src = torch.randn(Total, D, dtype=torch.bfloat16, device=device)
        Indices = torch.randint(0, Total, (N,), dtype=torch.int32, device=device)
        Dst = torch.zeros(N, D, dtype=torch.bfloat16, device=device)
        return [Src, Indices, Dst]
    elif name.startswith("moe_extract_weights_"):
        topk = int(name.split("_topk")[1])
        Total = M
        GateIndices = torch.randint(0, 256, (Total, topk), dtype=torch.int32, device=device)
        GateWeights = torch.randn(Total, topk, dtype=torch.float32, device=device)
        ExpertId = torch.tensor([0], dtype=torch.int32, device=device)
        OutWeights = torch.zeros(Total, dtype=torch.float32, device=device)
        OutTokenIds = torch.zeros(Total, dtype=torch.int32, device=device)
        OutCount = torch.zeros(1, dtype=torch.int32, device=device)
        return [GateIndices, GateWeights, ExpertId, OutWeights, OutTokenIds, OutCount]
    elif name.startswith("moe_expert_count_"):
        parts = name.split("_")
        topk = int(parts[3][4:])
        n_experts = int(parts[4][2:])
        Total = M
        Indices = torch.randint(0, n_experts, (Total, topk), dtype=torch.int32, device=device)
        Counts = torch.zeros(n_experts, dtype=torch.int32, device=device)
        return [Indices, Counts]
    else:
        return None


def compile_kernel(name, kernel_fn):
    so_path = os.path.join(BUILD_DIR, f"{name}.so")
    if os.path.exists(so_path):
        print(f"  [SKIP] {name} (already exists)")
        return so_path

    print(f"  [COMPILE] {name} ...", end=" ", flush=True)
    t0 = time.time()
    try:
        kernel = kernel_fn()
        inputs = make_dummy_inputs(name)
        if inputs is None:
            if kernel.artifact is not None and kernel.artifact.rt_mod is not None:
                kernel.export_library(so_path)
                elapsed = time.time() - t0
                so_size = os.path.getsize(so_path)
                print(f"OK ({elapsed:.1f}s, {so_size/1024:.0f}KB, no-smoke)")
                return so_path
            print("FAILED (no dummy inputs and no artifact)")
            return None

        if "sparse_attn" in name or "rmsnorm_N4096" in name:
            try:
                import ctypes
                cudart = ctypes.CDLL("libcudart.so")
                cudart.cudaFuncSetAttribute.restype = ctypes.c_int
                cudart.cudaFuncSetAttribute.argtypes = [
                    ctypes.c_void_p, ctypes.c_int, ctypes.c_int
                ]
            except Exception:
                pass

        kernel(*inputs)

        if kernel.artifact is not None and kernel.artifact.rt_mod is not None:
            kernel.export_library(so_path)
        elif hasattr(kernel.adapter, "libpath") and kernel.adapter.libpath:
            import shutil
            shutil.copy2(kernel.adapter.libpath, so_path)
        else:
            print("FAILED (no .so artifact)")
            return None
        elapsed = time.time() - t0
        so_size = os.path.getsize(so_path)
        print(f"OK ({elapsed:.1f}s, {so_size/1024:.0f}KB)")
        return so_path
    except Exception as e:
        print(f"FAILED ({e})")
        return None


def main():
    parser = argparse.ArgumentParser(description="Compile TileLang kernels for ds4rs")
    parser.add_argument("--batch", choices=["A1", "A2", "B1", "all"], default="all")
    args = parser.parse_args()

    if args.batch == "A1":
        names = BATCH_A1
    elif args.batch == "A2":
        names = BATCH_A2
    elif args.batch == "B1":
        names = BATCH_B1
    else:
        names = list(KERNEL_INSTANCES.keys())

    print(f"Compiling {len(names)} kernels (batch={args.batch}) ...")
    results = {"ok": 0, "skip": 0, "fail": 0}

    for name in names:
        so_path = compile_kernel(name, KERNEL_INSTANCES[name])
        if so_path is None:
            results["fail"] += 1
        else:
            results["ok"] += 1

    print(f"\nDone! ok={results['ok']} skip={results['skip']} fail={results['fail']}")
    print(f"Output: {BUILD_DIR}/")

    manifest = {}
    for name in KERNEL_INSTANCES:
        so_path = os.path.join(BUILD_DIR, f"{name}.so")
        if os.path.exists(so_path):
            func_name = _get_func_name(name)
            manifest[name] = {"so": f"{name}.so", "func": func_name}

    manifest_path = os.path.join(BUILD_DIR, "kernels.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path} ({len(manifest)} entries)")


def _get_func_name(name):
    if name.startswith("act_quant_"):
        return "act_quant_kernel_"
    elif name.startswith("fp8_gemm_"):
        return "fp8_gemm_kernel_"
    elif name.startswith("sparse_attn_"):
        return "sparse_attn_kernel_"
    elif name.startswith("hc_sinkhorn_"):
        return "hc_split_sinkhorn_kernel_"
    elif name.startswith("rmsnorm_f32_weighted_"):
        return "rmsnorm_f32_weighted_kernel_"
    elif name.startswith("rmsnorm_f32_"):
        return "rmsnorm_f32_kernel_"
    elif name.startswith("rmsnorm_"):
        return "rmsnorm_kernel_"
    elif name.startswith("swiglu_"):
        return "swiglu_kernel_"
    elif name.startswith("fp4_quant_"):
        return "fp4_quant_kernel_"
    elif name.startswith("fp4_gemm_"):
        return "fp4_gemm_kernel_"
    elif name.startswith("rope_interleaved_"):
        return "rope_interleaved_kernel_"
    elif name.startswith("scatter_add_"):
        return "scatter_add_kernel_"
    elif name.startswith("sigmoid_"):
        return "sigmoid_kernel_"
    elif name.startswith("hc_sigmoid_"):
        return "hc_sigmoid_kernel_"
    elif name.startswith("moe_route_"):
        return "moe_route_kernel_"
    elif name.startswith("indexer_score_"):
        return "indexer_score_kernel_"
    elif name.startswith("compressor_pool_"):
        return "compressor_pool_kernel_"
    elif name.startswith("compressor_rope_f32_"):
        return "compressor_rope_f32_kernel_"
    elif name.startswith("fp4_qdq_f32_"):
        return "fp4_qdq_f32_kernel_"
    elif name.startswith("indexer_causal_adjust_"):
        return "indexer_causal_adjust_kernel_"
    elif name.startswith("scale_f32_"):
        return "scale_f32_kernel_"
    elif name.startswith("compressor_group_"):
        return "compressor_group_kernel_"
    elif name.startswith("moe_gather_"):
        return "moe_gather_kernel_"
    elif name.startswith("moe_extract_weights_"):
        return "moe_extract_weights_kernel_"
    elif name.startswith("moe_expert_count_"):
        return "moe_expert_count_kernel_"
    else:
        return name + "_"


if __name__ == "__main__":
    main()
