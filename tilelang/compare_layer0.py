"""对比官方 Python 推理和 Rust 推理的数值差异

用法: docker exec ds4rs-dev python3 /workspace/tilelang/compare_layer0.py
"""
import sys
sys.path.insert(0, "/models/inference")
import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(42)

from model import Transformer, ModelArgs
from kernel import act_quant, fp8_gemm

args = ModelArgs()
args.ckpt_dir = "/models"
args.scale_dtype = "fp8"

model = Transformer(args).cuda()

# 测试 wo_b FP8 GEMM
print("--- Testing wo_b FP8 GEMM ---")
wo_b = model.layers[0].attn.wo_b
print(f"wo_b weight shape: {wo_b.weight.shape}, dtype: {wo_b.weight.dtype}")
print(f"wo_b scale shape: {wo_b.weight.scale.shape}, dtype: {wo_b.weight.scale.dtype}")

n_groups = 8
o_lora_rank = 1024
total = 1
wo_a_input = torch.randn(total, n_groups * o_lora_rank, dtype=torch.bfloat16, device="cuda") * 0.5
print(f"wo_a_input stats: mean={wo_a_input.float().mean():.4f}, max={wo_a_input.float().abs().max():.4f}")

with torch.no_grad():
    wo_b_out = wo_b(wo_a_input)
    print(f"wo_b output stats: mean={wo_b_out.float().mean():.4f}, min={wo_b_out.float().min():.4f}, max={wo_b_out.float().max():.4f}")
    print(f"wo_b output std: {wo_b_out.float().std():.4f}")

# 测试 shared expert FFN
print("\n--- Testing Shared Expert FFN ---")
shared_expert = model.layers[0].ffn.shared_experts
ffn_input = torch.randn(1, 4096, dtype=torch.bfloat16, device="cuda") * 0.3
print(f"ffn_input stats: mean={ffn_input.float().mean():.4f}, max={ffn_input.float().abs().max():.4f}")

with torch.no_grad():
    ffn_out = shared_expert(ffn_input)
    print(f"shared_ffn output stats: mean={ffn_out.float().mean():.4f}, min={ffn_out.float().min():.4f}, max={ffn_out.float().max():.4f}")
    print(f"shared_ffn output std: {ffn_out.float().std():.4f}")

# 测试完整层
print("\n--- Testing Full Layer 0 ---")
hc = 4
dim = 4096
x = torch.randn(1, 1, hc, dim, dtype=torch.bfloat16, device="cuda") * 0.1
print(f"input stats: mean={x.float().mean():.4f}, max={x.float().abs().max():.4f}")

with torch.no_grad():
    try:
        out = model.layers[0](x, start_pos=0, input_ids=None)
        print(f"Layer 0 output shape: {out.shape}")
        print(f"Layer 0 output stats: mean={out.float().mean():.4f}, min={out.float().min():.4f}, max={out.float().max():.4f}")
        print(f"Layer 0 output std: {out.float().std():.4f}")
    except Exception as e:
        print(f"Layer 0 forward failed: {e}")
        import traceback
        traceback.print_exc()

# 测试多层累积
print("\n--- Testing Multi-Layer Accumulation ---")
x = torch.randn(1, 1, hc, dim, dtype=torch.bfloat16, device="cuda") * 0.1
with torch.no_grad():
    for i, layer in enumerate(model.layers):
        try:
            x = layer(x, start_pos=0, input_ids=None)
            stats = f"mean={x.float().mean():.4f}, min={x.float().min():.4f}, max={x.float().max():.4f}, std={x.float().std():.4f}"
            print(f"L{i:02d}: {stats}")
            if x.float().abs().max() > 100:
                print(f"  *** DIVERGENCE at L{i}! ***")
                break
        except Exception as e:
            print(f"L{i:02d}: FAILED ({e})")
            break
