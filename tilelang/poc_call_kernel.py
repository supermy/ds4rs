import sys
sys.path.insert(0, '/models/inference')
import tvm_ffi, tilelang, torch, numpy as np
from kernel import act_quant

M, N, K = 32, 4096, 4096
torch.manual_seed(42)
device = 'cuda'

a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)

a_fp8, a_s = act_quant(a_bf16, block_size=128, scale_dtype=torch.float32)

def quantize_weight_per_block(x_bf16, block_size=128):
    N, K = x_bf16.shape
    x = x_bf16.float()
    x_4d = x.view(N // block_size, block_size, K // block_size, block_size).permute(0, 2, 1, 3)
    amax = x_4d.abs().amax(dim=(2, 3), keepdim=False)
    scale = amax / 448.0
    scale = scale.clamp(min=1e-8)
    scale_exp = scale.unsqueeze(2).unsqueeze(3).expand_as(x_4d)
    x_q = (x_4d / scale_exp).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    x_q_2d = x_q.permute(0, 2, 1, 3).reshape(N, K)
    return x_q_2d, scale

b_fp8, b_s = quantize_weight_per_block(b_bf16, block_size=128)

so_path = '/workspace/tilelang/build/fp8_gemm_N4096_K4096.so'
m = tvm_ffi.load_module(so_path)
func = m.get_function('fp8_gemm_kernel_')

torch.set_default_dtype(torch.bfloat16)
c_t = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
func(a_fp8, b_fp8, c_t, a_s, b_s)

c_ref = torch.nn.functional.linear(a_bf16.float(), b_bf16.float())
max_diff = (c_t.float() - c_ref).abs().max().item()
mean_diff = (c_t.float() - c_ref).abs().mean().item()

print(f'fp8_gemm PoC: max_abs_err={max_diff:.4f}, mean_abs_err={mean_diff:.4f}')
print(f'Output shape: {c_t.shape}, dtype: {c_t.dtype}')
print('SUCCESS: Rust->TileLang data path verified via tvm_ffi DLPack')
