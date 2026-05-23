"""单独测试 fp8_gemm 的数值正确性"""
import sys
sys.path.insert(0, "/models/inference")
import torch
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(42)

from kernel import act_quant, fp8_gemm

# 测试1: scale_dtype=FP32 (默认)
print("=== Test 1: scale_dtype=FP32 ===")
M, N, K = 4, 4096, 4096
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
w_scale = torch.ones(N // 128, K // 128, dtype=torch.float32, device="cuda")

x_fp8, x_scale = act_quant(x, 128, None, torch.float32)
out = fp8_gemm(x_fp8, x_scale, w, w_scale, torch.float32)
print(f"  output stats: mean={out.float().mean():.4f}, min={out.float().min():.4f}, max={out.float().max():.4f}")

# 测试2: scale_dtype=FE8M0
print("\n=== Test 2: scale_dtype=FE8M0 ===")
x2 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5
w2 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
w2_scale = torch.ones(N // 128, K // 128, dtype=torch.float8_e8m0fnu, device="cuda")

x2_fp8, x2_scale = act_quant(x2, 128, "ue8m0", torch.float8_e8m0fnu)
out2 = fp8_gemm(x2_fp8, x2_scale, w2, w2_scale, torch.float8_e8m0fnu)
print(f"  output stats: mean={out2.float().mean():.4f}, min={out2.float().min():.4f}, max={out2.float().max():.4f}")

# 测试3: 用真实权重
print("\n=== Test 3: Real weights (wo_b) ===")
from model import Transformer, ModelArgs
args = ModelArgs()
args.ckpt_dir = "/models"
args.scale_dtype = "fp8"
model = Transformer(args).cuda()

wo_b = model.layers[0].attn.wo_b
print(f"  wo_b weight shape: {wo_b.weight.shape}, dtype: {wo_b.weight.dtype}")
print(f"  wo_b scale shape: {wo_b.weight.scale.shape}, dtype: {wo_b.weight.scale.dtype}")
print(f"  wo_b scale sample: {wo_b.weight.scale[:2,:2]}")

# 手动调用 fp8_gemm
x3 = torch.randn(1, 8192, dtype=torch.bfloat16, device="cuda") * 0.5
x3_fp8, x3_scale = act_quant(x3, 128, "ue8m0", torch.float8_e8m0fnu)
print(f"  x3_scale shape: {x3_scale.shape}, dtype: {x3_scale.dtype}")
print(f"  x3_scale sample: {x3_scale[:2,:2]}")

out3 = fp8_gemm(x3_fp8, x3_scale, wo_b.weight, wo_b.weight.scale, torch.float8_e8m0fnu)
print(f"  output stats: mean={out3.float().mean():.4f}, min={out3.float().min():.4f}, max={out3.float().max():.4f}")
