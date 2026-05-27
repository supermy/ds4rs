"""Q2_K GEMM TileLang kernel 正确性验证。

从 experts_iq2xxs_q2k.gguf 读取 L0E0 的 w2 权重（Q2_K 格式），
构造简单输入 x=[1,0,0,...]，分别用 CPU FFN (Rust) 和 GPU FFN (TileLang q2k_gemm) 计算，
对比两者的输出。

Q2_K GGUF 格式 (block_q2_K, 84 bytes / 256 elements):
  - scales[16]: 4-bit packed (低 4 位 = scale, 高 4 位 = min)
  - qs[64]: 2-bit 量化值打包
  - d: fp16 super-block scale
  - dmin: fp16 minimum scale

llama.cpp 反量化公式:
  value = d * (sc & 0xF) * ((int8_t)((q[l] >> shift) & 3)) - min * (sc >> 4)

TileLang q2k_gemm 反量化公式:
  value = d * scale_val * (quant_2bit - 1.5) + dmin

用法:
  docker exec ds4rs-dev python /workspace/tests/verify_q2k_gemm.py
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tilelang'))

from gguf import GGUFReader

GGUF_PATH = os.environ.get("GGUF_PATH", "/workspace/gguf/experts_iq2xxs_q2k.gguf")
QK_K = 256


# ============================================================================
# GGUF 解析
# ============================================================================
def parse_q2k_block(data_bytes, n_blocks):
    """解析 Q2_K: scales(16B) + qs(64B) + d(fp16,2B) + dmin(fp16,2B) = 84B/block"""
    raw = np.frombuffer(data_bytes, dtype=np.uint8)
    blocks = raw.reshape(n_blocks, 84)
    scales = blocks[:, 0:16].reshape(n_blocks, 16).copy()
    qs = blocks[:, 16:80].reshape(n_blocks, 64).copy()
    d = blocks[:, 80:82].copy().view(np.float16).astype(np.float32).ravel()
    dmin = blocks[:, 82:84].copy().view(np.float16).astype(np.float32).ravel()
    return d, dmin, scales, qs


def load_w2_from_gguf(reader, layer_id=0, expert_id=0):
    """从 GGUF 加载 w2 (down_proj) 权重。"""
    tensor_name = f"layers.{layer_id}.experts.{expert_id}.w2"
    found = None
    for t in reader.tensors:
        if t.name == tensor_name:
            found = t
            break
    if found is None:
        raise ValueError(f"Tensor {tensor_name} not found")

    data = found.data.ravel().tobytes()
    ne0, ne1 = [int(x) for x in found.shape]
    n_blocks = ne0 * ne1 // 256
    type_name = found.tensor_type.name

    print(f"  Tensor: {tensor_name}")
    print(f"  Type: {type_name}")
    print(f"  GGUF shape: ne0={ne0}, ne1={ne1}")
    print(f"  n_blocks: {n_blocks}")

    assert type_name == 'Q2_K', f"Expected Q2_K, got {type_name}"

    d, dmin, scales, qs = parse_q2k_block(data, n_blocks)
    # GGUF shape [ne0, ne1] 中 ne0=in_features, ne1=out_features
    logical_shape = (ne1, ne0)  # (N, K) = (out_features, in_features)
    return d, dmin, scales, qs, logical_shape


# ============================================================================
# Q2_K 反量化 (llama.cpp 标准，参考实现)
# ============================================================================
def dequantize_q2k_reference(d_flat, dmin_flat, scales_flat, qs_flat, ne0, ne1):
    """llama.cpp 标准反量化。

    反量化公式 (per element):
      value = d * (sc & 0xF) * quant_2bit - dmin * (sc >> 4)

    其中 quant_2bit = (int8_t)((q[l] >> shift) & 3) = 0, 1, 2, 3 (无偏移)
    """
    N = ne1
    K = ne0
    n_blocks_per_row = K // QK_K

    d_2d = d_flat.reshape(N, n_blocks_per_row)
    dmin_2d = dmin_flat.reshape(N, n_blocks_per_row)
    scales_3d = scales_flat.reshape(N, n_blocks_per_row, 16)
    qs_3d = qs_flat.reshape(N, n_blocks_per_row, 64)

    weight = np.zeros((N, K), dtype=np.float32)

    for row in range(N):
        for blk in range(n_blocks_per_row):
            d_val = d_2d[row, blk]
            dmin_val = dmin_2d[row, blk]
            sc = scales_3d[row, blk]  # [16]
            qs_blk = qs_3d[row, blk]  # [64]

            for k_half in range(2):  # 2 halves of 128 elements
                q2_base = k_half * 32
                shift = 0
                is_idx = k_half * 8  # starting scale index

                for _j in range(4):
                    # First sub-block
                    scale_4bit = int(sc[is_idx]) & 0xF
                    min_4bit = int(sc[is_idx]) >> 4
                    is_idx += 1

                    for l in range(16):
                        global_k = blk * QK_K + k_half * 128 + _j * 32 + l
                        quant_2bit = (int(qs_blk[q2_base + l]) >> shift) & 3
                        weight[row, global_k] = d_val * scale_4bit * quant_2bit - dmin_val * min_4bit

                    # Second sub-block
                    scale_4bit = int(sc[is_idx]) & 0xF
                    min_4bit = int(sc[is_idx]) >> 4
                    is_idx += 1

                    for l in range(16):
                        global_k = blk * QK_K + k_half * 128 + _j * 32 + 16 + l
                        quant_2bit = (int(qs_blk[q2_base + 16 + l]) >> shift) & 3
                        weight[row, global_k] = d_val * scale_4bit * quant_2bit - dmin_val * min_4bit

                    shift += 2

    return weight


def dequantize_q2k_tilelang_sim(d_flat, dmin_flat, scales_flat, qs_flat, ne0, ne1):
    """模拟 TileLang q2k_gemm kernel 的反量化逻辑。

    TileLang kernel 的反量化公式:
      ib16 = local_k // 16
      local_in_16 = local_k % 16
      byte_idx = local_in_16 // 4
      bit_offset = (local_in_16 % 4) * 2
      quant_2bit = (qs[byte_idx] >> bit_offset) & 3
      quant_val = quant_2bit - 1.5
      value = d * scales[ib16] * quant_val + dmin
    """
    N = ne1
    K = ne0
    n_blocks_per_row = K // QK_K

    d_2d = d_flat.reshape(N, n_blocks_per_row)
    dmin_2d = dmin_flat.reshape(N, n_blocks_per_row)
    scales_3d = scales_flat.reshape(N, n_blocks_per_row, 16)
    qs_3d = qs_flat.reshape(N, n_blocks_per_row, 64)

    weight = np.zeros((N, K), dtype=np.float32)

    for row in range(N):
        for blk in range(n_blocks_per_row):
            d_val = d_2d[row, blk]
            dmin_val = dmin_2d[row, blk]
            sc = scales_3d[row, blk]  # [16]
            qs_blk = qs_3d[row, blk]  # [64]

            for local_k in range(QK_K):
                global_k = blk * QK_K + local_k

                ib16 = local_k // 16
                local_in_16 = local_k % 16

                byte_idx = local_in_16 // 4
                bit_offset = (local_in_16 % 4) * 2

                qs_val = int(qs_blk[byte_idx])
                quant_2bit = (qs_val >> bit_offset) & 3
                quant_val = float(quant_2bit) - 1.5

                scale_val = float(sc[ib16])

                weight[row, global_k] = d_val * scale_val * quant_val + dmin_val

    return weight


# ============================================================================
# 主验证逻辑
# ============================================================================
def main():
    print("=" * 70)
    print("Q2_K GEMM TileLang kernel 正确性验证")
    print("=" * 70)

    # 1. 读取 GGUF
    print("\n[1] 读取 GGUF 文件...")
    reader = GGUFReader(GGUF_PATH)
    d, dmin, scales, qs, logical_shape = load_w2_from_gguf(reader, layer_id=0, expert_id=0)
    N, K = logical_shape  # (out_features, in_features)
    n_blocks_per_row = K // QK_K
    print(f"  逻辑形状: ({N}, {K}), n_blocks_per_row={n_blocks_per_row}")

    # 2. 构造简单输入 x=[1,0,0,...]
    print("\n[2] 构造输入 x=[1,0,0,...]")
    x_np = np.zeros(K, dtype=np.float32)
    x_np[0] = 1.0
    print(f"  x shape=({K},), x[0]={x_np[0]}, x[1]={x_np[1]}")

    # 3. CPU FFN (Rust Q2KWeight.matvec)
    print("\n[3] CPU FFN (Rust Q2KWeight.matvec)...")
    from ds4rs import Q2KWeight, is_avx512_supported, init_tables, is_tables_initialized

    if not is_tables_initialized():
        from iq2xs_gemm_tilelang import IQ2XS_GRID_U64, KSIGNS_IQ2XS
        init_tables(list(IQ2XS_GRID_U64), list(KSIGNS_IQ2XS))
        print(f"  AVX-512: {is_avx512_supported()}")

    down_w = Q2KWeight.from_numpy(d.copy(), dmin.copy(), scales.reshape(-1).copy(), qs.reshape(-1).copy(), logical_shape)
    cpu_out = down_w.matvec(x_np)
    cpu_out = np.array(cpu_out, dtype=np.float32)
    print(f"  CPU output: shape={cpu_out.shape}, range=[{cpu_out.min():.4f}, {cpu_out.max():.4f}]")
    print(f"  CPU mean={cpu_out.mean():.6f}, std={cpu_out.std():.6f}")
    print(f"  CPU first 10: {cpu_out[:10]}")

    # 4. GPU FFN (TileLang q2k_gemm)
    print("\n[4] GPU FFN (TileLang q2k_gemm)...")
    device = 'cuda'

    # 准备 GPU 输入
    x_gpu = torch.from_numpy(x_np).to(device, dtype=torch.bfloat16).unsqueeze(0)  # [1, K]

    # 准备 Q2_K 权重 tensor (直接传 GGUF 解析的数据)
    d_gpu = torch.from_numpy(d.reshape(N, n_blocks_per_row).astype(np.float16).copy()).to(device)
    dmin_gpu = torch.from_numpy(dmin.reshape(N, n_blocks_per_row).astype(np.float16).copy()).to(device)
    scales_gpu = torch.from_numpy(scales.reshape(N, n_blocks_per_row, 16).copy()).to(device)
    qs_gpu = torch.from_numpy(qs.reshape(N, n_blocks_per_row, 64).copy()).to(device)

    print(f"  d_gpu: {d_gpu.shape}, {d_gpu.dtype}")
    print(f"  dmin_gpu: {dmin_gpu.shape}, {dmin_gpu.dtype}")
    print(f"  scales_gpu: {scales_gpu.shape}, {scales_gpu.dtype}")
    print(f"  qs_gpu: {qs_gpu.shape}, {qs_gpu.dtype}")

    from mixed_quant_gemm import q2k_gemm
    gpu_out = q2k_gemm(x_gpu, d_gpu, dmin_gpu, scales_gpu, qs_gpu)
    gpu_out = gpu_out.squeeze(0).float().cpu().numpy()
    print(f"  GPU output: shape={gpu_out.shape}, range=[{gpu_out.min():.4f}, {gpu_out.max():.4f}]")
    print(f"  GPU mean={gpu_out.mean():.6f}, std={gpu_out.std():.6f}")
    print(f"  GPU first 10: {gpu_out[:10]}")

    # 5. 对比
    print("\n[5] 对比 CPU vs GPU...")
    diff = np.abs(cpu_out - gpu_out)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"  max_diff = {max_diff:.6f}")
    print(f"  mean_diff = {mean_diff:.6f}")
    print(f"  median_diff = {np.median(diff):.6f}")

    # 前 20 个输出对比
    print(f"\n  前 20 个输出对比:")
    print(f"  {'Idx':>4s}  {'CPU':>12s}  {'GPU':>12s}  {'Diff':>12s}")
    for i in range(min(20, len(cpu_out))):
        d_val = abs(cpu_out[i] - gpu_out[i])
        print(f"  {i:4d}  {cpu_out[i]:12.6f}  {gpu_out[i]:12.6f}  {d_val:12.6f}")

    # 6. 参考反量化对比
    print("\n[6] llama.cpp 标准反量化 vs TileLang 模拟反量化 (逐元素分析)...")
    ne0, ne1 = K, N  # GGUF convention: ne0=in_features, ne1=out_features

    # 只分析第一行
    ref_weight_row0 = dequantize_q2k_reference(d, dmin, scales, qs, ne0, ne1)[0]  # [K]
    tl_weight_row0 = dequantize_q2k_tilelang_sim(d, dmin, scales, qs, ne0, ne1)[0]  # [K]

    # x=[1,0,...] → matvec 结果 = 权重矩阵第一列
    # CPU matvec(x) = W @ x = W[:, 0] (第一列)
    # 但 Q2_K 权重是行优先，W[row, :] = 一行
    # matvec(x) for x=[1,0,...] = W 的第一列
    # 但我们的反量化给出的是 W[row, :]，所以需要取 W[:, 0] = 第一列

    # 用反量化权重做 matmul
    ref_weight = dequantize_q2k_reference(d, dmin, scales, qs, ne0, ne1)  # [N, K]
    tl_weight = dequantize_q2k_tilelang_sim(d, dmin, scales, qs, ne0, ne1)  # [N, K]

    ref_matvec = ref_weight @ x_np  # [N]
    tl_matvec = tl_weight @ x_np  # [N]

    print(f"  参考反量化 matvec: range=[{ref_matvec.min():.4f}, {ref_matvec.max():.4f}]")
    print(f"  TileLang模拟 matvec: range=[{tl_matvec.min():.4f}, {tl_matvec.max():.4f}]")

    # 对比参考反量化 vs CPU Rust
    diff_ref_cpu = np.abs(ref_matvec - cpu_out)
    print(f"\n  参考反量化 vs CPU Rust: max_diff={np.max(diff_ref_cpu):.6f}")

    # 对比 TileLang 模拟 vs GPU
    diff_tl_gpu = np.abs(tl_matvec - gpu_out)
    print(f"  TileLang模拟 vs GPU: max_diff={np.max(diff_tl_gpu):.6f}")

    # 7. 详细格式差异分析
    print("\n[7] Q2_K 格式差异分析...")
    print("=" * 70)

    # 分析第一行的第一个 block
    d_2d = d.reshape(N, n_blocks_per_row)
    dmin_2d = dmin.reshape(N, n_blocks_per_row)
    scales_3d = scales.reshape(N, n_blocks_per_row, 16)
    qs_3d = qs.reshape(N, n_blocks_per_row, 64)

    row, blk = 0, 0
    d_val = d_2d[row, blk]
    dmin_val = dmin_2d[row, blk]
    sc = scales_3d[row, blk]
    qs_blk = qs_3d[row, blk]

    print(f"\n  第一行第一个 block:")
    print(f"  d = {d_val:.6f}")
    print(f"  dmin = {dmin_val:.6f}")
    print(f"  scales (raw uint8): {sc}")
    print(f"  scales (low 4-bit): {[s & 0xF for s in sc]}")
    print(f"  scales (high 4-bit = min): {[s >> 4 for s in sc]}")
    print(f"  qs (first 16 bytes): {qs_blk[:16]}")

    # 逐元素对比反量化
    print(f"\n  逐元素对比 (block 0, 前 32 个元素):")
    print(f"  {'k':>3s}  {'llama.cpp':>12s}  {'TileLang':>12s}  {'Diff':>12s}  {'qs_byte':>8s}  {'shift':>6s}  {'ib16':>5s}")

    for local_k in range(32):
        global_k = blk * QK_K + local_k

        # llama.cpp 方式
        k_half = local_k // 128
        pos_in_half = local_k % 128
        _j = pos_in_half // 32
        pos_in_j = pos_in_half % 32
        q2_base = k_half * 32
        shift = _j * 2

        is_idx = k_half * 8 + _j * 2
        if pos_in_j >= 16:
            is_idx += 1
            l = pos_in_j - 16
            quant_llama = (int(qs_blk[q2_base + 16 + l]) >> shift) & 3
        else:
            l = pos_in_j
            quant_llama = (int(qs_blk[q2_base + l]) >> shift) & 3

        scale_4bit = int(sc[is_idx]) & 0xF
        min_4bit = int(sc[is_idx]) >> 4
        val_llama = d_val * scale_4bit * quant_llama - dmin_val * min_4bit

        # TileLang 方式
        ib16 = local_k // 16
        local_in_16 = local_k % 16
        byte_idx = local_in_16 // 4
        bit_offset = (local_in_16 % 4) * 2
        quant_tl = (int(qs_blk[byte_idx]) >> bit_offset) & 3
        quant_val_tl = float(quant_tl) - 1.5
        scale_val_tl = float(sc[ib16])
        val_tl = d_val * scale_val_tl * quant_val_tl + dmin_val

        diff_val = abs(val_llama - val_tl)
        print(f"  {local_k:3d}  {val_llama:12.6f}  {val_tl:12.6f}  {diff_val:12.6f}  "
              f"qs[{byte_idx:2d}]  shift={bit_offset}  ib16={ib16}")

    # 8. 总结
    print("\n" + "=" * 70)
    print("验证结果总结")
    print("=" * 70)

    if max_diff < 0.1:
        print(f"\n✓ CPU vs GPU 一致 (max_diff={max_diff:.6f} < 0.1)")
    else:
        print(f"\n✗ CPU vs GPU 不一致 (max_diff={max_diff:.6f})")
        print(f"\n格式差异分析:")
        print(f"  1. scales 格式:")
        print(f"     - GGUF/llama.cpp: 4-bit packed (低4位=scale, 高4位=min)")
        print(f"     - TileLang: 直接使用 uint8 值作为 scale")
        print(f"  2. qs 索引方式:")
        print(f"     - GGUF/llama.cpp: q[l] >> shift, l=0..15 或 16..31, shift=0,2,4,6")
        print(f"       同一 qs 字节被 4 个子块共享（通过不同 shift）")
        print(f"     - TileLang: byte_idx = local_in_16 // 4, bit_offset = (local_in_16 % 4) * 2")
        print(f"       每 4 个连续元素打包到一个 qs 字节")
        print(f"  3. 反量化公式:")
        print(f"     - GGUF/llama.cpp: d * (sc & 0xF) * quant - dmin * (sc >> 4)")
        print(f"       quant = 0,1,2,3 (无偏移)")
        print(f"     - TileLang: d * scale_val * (quant - 1.5) + dmin")
        print(f"       quant 中心化到 -1.5,-0.5,0.5,1.5")

    return max_diff < 0.1


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
