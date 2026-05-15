import sys
import os
import shutil
import numpy as np
import torch
import tilelang
import tilelang.language as T

sys.path.insert(0, "/models/inference")
from kernel import act_quant

BUILD_DIR = "/workspace/tilelang/build"
REF_DIR = os.path.join(BUILD_DIR, "poc_ref")
SO_PATH = os.path.join(BUILD_DIR, "fp8_gemm_N4096_K4096.so")

os.makedirs(REF_DIR, exist_ok=True)

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"

M_DIM, N_DIM, K_DIM = 32, 4096, 4096
BLOCK_SIZE = 128


@tilelang.jit(pass_configs=pass_configs, execution_backend="tvm_ffi")
def fp8_gemm_kernel_so(N, K, out_dtype=BF16, accum_dtype=FP32, scale_dtype=FP32):
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


print(f"[1/5] Creating random input tensors (M={M_DIM}, N={N_DIM}, K={K_DIM}) ...")
torch.manual_seed(42)
torch.set_default_dtype(torch.bfloat16)

a_bf16 = torch.randn(M_DIM, K_DIM, dtype=torch.bfloat16, device="cuda")
b_bf16 = torch.randn(N_DIM, K_DIM, dtype=torch.bfloat16, device="cuda")

a_fp8, a_s = act_quant(a_bf16, block_size=BLOCK_SIZE)
print(f"  a_fp8: {a_fp8.shape}  a_s: {a_s.shape}")


def quantize_weight_fp8(w_bf16, block_size=128):
    N, K = w_bf16.shape
    assert N % block_size == 0 and K % block_size == 0
    num_n_blocks = N // block_size
    num_k_blocks = K // block_size
    w_reshaped = w_bf16.float().view(num_n_blocks, block_size, num_k_blocks, block_size)
    amax = w_reshaped.abs().amax(dim=(1, 3))
    amax = amax.clamp(min=1e-4)
    scale = amax / 448.0
    scale_expanded = scale.unsqueeze(1).unsqueeze(3)
    w_quantized = (w_reshaped / scale_expanded).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    w_fp8 = w_quantized.view(N, K)
    return w_fp8, scale


b_fp8, b_s = quantize_weight_fp8(b_bf16, block_size=BLOCK_SIZE)
print(f"  b_fp8: {b_fp8.shape}  b_s: {b_s.shape}")

print("[2/5] Compiling fp8_gemm_kernel with tvm_ffi backend ...")
kernel = fp8_gemm_kernel_so(N=N_DIM, K=K_DIM)

c_bf16 = a_bf16.new_empty(M_DIM, N_DIM, dtype=torch.bfloat16)
kernel(a_fp8.view(M_DIM, K_DIM), b_fp8, c_bf16.view(M_DIM, N_DIM), a_s.view(M_DIM, -1), b_s)
print(f"  Kernel executed. c_bf16: {c_bf16.shape} {c_bf16.dtype}")

print("[3/5] Exporting .so ->", SO_PATH)
if kernel.artifact is not None and kernel.artifact.rt_mod is not None:
    kernel.export_library(SO_PATH)
    print("  Exported via export_library()")
elif hasattr(kernel.adapter, "libpath") and kernel.adapter.libpath:
    shutil.copy2(kernel.adapter.libpath, SO_PATH)
    print(f"  Copied from cache: {kernel.adapter.libpath}")
else:
    raise RuntimeError("No .so available: artifact.rt_mod is None and adapter.libpath is None")

so_size = os.path.getsize(SO_PATH)
print(f"  .so size: {so_size} bytes ({so_size / 1024:.1f} KB)")

print("[4/5] Saving reference data ->", REF_DIR)


def save_tensor(name, tensor):
    path = os.path.join(REF_DIR, f"{name}.npy")
    if tensor.dtype == torch.float8_e4m3fn:
        np.save(path, tensor.view(torch.uint8).cpu().numpy())
    elif tensor.dtype == torch.bfloat16:
        np.save(path, tensor.view(torch.float16).cpu().numpy())
    else:
        np.save(path, tensor.cpu().numpy())
    print(f"  saved {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")


save_tensor("a_bf16", a_bf16)
save_tensor("a_fp8", a_fp8)
save_tensor("a_s", a_s)
save_tensor("b_bf16", b_bf16)
save_tensor("b_fp8", b_fp8)
save_tensor("b_s", b_s)
save_tensor("c_bf16", c_bf16)

print("[5/5] Writing metadata ...")
with open(os.path.join(REF_DIR, "metadata.txt"), "w") as f:
    f.write(f"M={M_DIM}\nN={N_DIM}\nK={K_DIM}\n")
    f.write(f"a_bf16: shape={list(a_bf16.shape)} dtype=bfloat16\n")
    f.write(f"a_fp8:  shape={list(a_fp8.shape)} dtype=float8_e4m3fn (saved as uint8)\n")
    f.write(f"a_s:    shape={list(a_s.shape)} dtype={a_s.dtype}\n")
    f.write(f"b_bf16: shape={list(b_bf16.shape)} dtype=bfloat16\n")
    f.write(f"b_fp8:  shape={list(b_fp8.shape)} dtype=float8_e4m3fn (saved as uint8)\n")
    f.write(f"b_s:    shape={list(b_s.shape)} dtype={b_s.dtype}\n")
    f.write(f"c_bf16: shape={list(c_bf16.shape)} dtype=bfloat16\n")
    f.write(f"block_size=128\n")
    f.write(f"seed=42\n")
    f.write(f"note: a quantized via act_quant (per-row block=128 on K), b quantized via per-128x128 block\n")

print("\nDone!")
print(f"  .so: {SO_PATH}")
print(f"  ref: {REF_DIR}")
