use crate::dtype::DType;
use crate::tensor::CpuTensor;
use anyhow::Result;

pub fn f32_to_fp8_e4m3(v: f32) -> u8 {
    let fp8_max = 448.0f32;
    if v != v || v.is_infinite() {
        return if v.is_sign_positive() { 0x7F } else { 0xFF };
    }
    let clamped = v.clamp(-fp8_max, fp8_max);
    let bits = clamped.to_bits();
    let sign = ((bits >> 31) & 1) as u8;
    let exp_f32 = ((bits >> 23) & 0xFF) as i32;
    let mant_f32 = (bits & 0x7FFFFF) as u32;

    if exp_f32 == 0 && mant_f32 == 0 {
        return sign << 7;
    }

    let exp_fp8 = exp_f32 - 127 + 7;
    if exp_fp8 <= 0 {
        return sign << 7;
    }
    if exp_fp8 >= 16 {
        let max_mant = 0b110;
        return (sign << 7) | (0b1111 << 3) | max_mant;
    }

    let mant_fp8 = ((mant_f32 >> 20) & 0x7) as u8;
    (sign << 7) | ((exp_fp8 as u8) << 3) | mant_fp8
}

pub fn dequant_fp8_e4m3_to_f32(data: &[u8], scales: &[u8], shape: &[usize]) -> Result<Vec<f32>> {
    let n_elements: usize = shape.iter().product();
    let block_size = 128;
    let last_dim = shape.last().copied().unwrap_or(1);
    let n_blocks_col = (last_dim + block_size - 1) / block_size;

    let mut out = vec![0.0f32; n_elements];
    for i in 0..n_elements {
        let row = i / last_dim;
        let col = i % last_dim;
        let block_row = row / block_size;
        let block_col = col / block_size;
        let scale_idx = block_row * n_blocks_col + block_col;
        let scale_f32 = if scale_idx < scales.len() { e8m0_to_f32(scales[scale_idx]) } else { 1.0 };
        out[i] = fp8_e4m3_to_f32(data[i]) * scale_f32;
    }
    Ok(out)
}

pub fn dequant_fp8_e4m3_to_bf16(data: &[u8], scales: &[u8], shape: &[usize]) -> Result<CpuTensor> {
    let n_elements: usize = shape.iter().product();
    let block_size = 128;

    let last_dim = shape.last().copied().unwrap_or(1);
    let n_blocks_col = (last_dim + block_size - 1) / block_size;
    let n_rows = n_elements / last_dim;
    let _n_blocks_row = (n_rows + block_size - 1) / block_size;

    let mut out = vec![0u16; n_elements];

    for i in 0..n_elements {
        let row = i / last_dim;
        let col = i % last_dim;
        let block_row = row / block_size;
        let block_col = col / block_size;

        let fp8_val = data[i];
        let f32_val = fp8_e4m3_to_f32(fp8_val);

        let scale_idx = block_row * n_blocks_col + block_col;
        let scale_f32 = if scale_idx < scales.len() {
            e8m0_to_f32(scales[scale_idx])
        } else {
            1.0
        };

        let dequant_val = f32_val * scale_f32;
        out[i] = half::bf16::from_f32(dequant_val).to_bits();
    }

    Ok(CpuTensor::new(
        bytemuck::cast_slice(&out).to_vec(),
        shape.to_vec(),
        DType::BF16,
    ))
}

pub fn dequant_fp4_e2m1_to_bf16(
    packed_data: &[u8],
    scales: &[u8],
    packed_shape: &[usize],
    _logical_k: usize,
) -> Result<CpuTensor> {
    let out_dim = packed_shape[0];
    let packed_k = packed_shape[1];
    let logical_cols = packed_k * 2;

    let n_rows = out_dim;
    let group_size = 32;
    let n_groups_per_row = logical_cols / group_size;

    let mut out = vec![0u16; n_rows * logical_cols];

    for row in 0..n_rows {
        for packed_col in 0..packed_k {
            let byte_val = packed_data[row * packed_k + packed_col];
            let lo = byte_val & 0x0F;
            let hi = (byte_val >> 4) & 0x0F;

            let logical_col_lo = packed_col * 2;
            let logical_col_hi = packed_col * 2 + 1;

            let f32_lo = fp4_e2m1_to_f32(lo);
            let f32_hi = fp4_e2m1_to_f32(hi);

            let group_lo = logical_col_lo / group_size;
            let group_hi = logical_col_hi / group_size;

            let scale_lo = get_e8m0_scale(scales, row, group_lo, n_groups_per_row);
            let scale_hi = get_e8m0_scale(scales, row, group_hi, n_groups_per_row);

            if logical_col_lo < logical_cols {
                out[row * logical_cols + logical_col_lo] =
                    half::bf16::from_f32(f32_lo * scale_lo).to_bits();
            }
            if logical_col_hi < logical_cols {
                out[row * logical_cols + logical_col_hi] =
                    half::bf16::from_f32(f32_hi * scale_hi).to_bits();
            }
        }
    }

    Ok(CpuTensor::new(
        bytemuck::cast_slice(&out).to_vec(),
        vec![n_rows, logical_cols],
        DType::BF16,
    ))
}

fn get_e8m0_scale(scales: &[u8], row: usize, group: usize, n_groups_per_row: usize) -> f32 {
    let idx = row * n_groups_per_row + group;
    if idx >= scales.len() {
        return 1.0;
    }
    e8m0_to_f32(scales[idx])
}

pub fn e8m0_to_f32(bits: u8) -> f32 {
    if bits == 0 {
        return 0.0;
    }
    let exponent = (bits as u32) << 23;
    f32::from_bits(exponent)
}

pub fn e8m0_to_f32_scale(data: &[u8], shape: &[usize]) -> Result<Vec<u8>, anyhow::Error> {
    let n: usize = shape.iter().product();
    let mut out = vec![0.0f32; n];
    let src: &[u8] = bytemuck::cast_slice(data);
    for i in 0..n {
        out[i] = e8m0_to_f32(src[i]);
    }
    Ok(bytemuck::cast_slice(&out).to_vec())
}

pub fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = ((bits >> 3) & 0x0F) as i32;
    let mant = (bits & 0x07) as i32;

    let bias = 7;
    let val = if exp == 0 && mant == 0 {
        0.0f32
    } else if exp == 0 {
        let m = mant as f32 / 8.0;
        m * 2.0f32.powi(1 - bias)
    } else {
        let m = 1.0 + mant as f32 / 8.0;
        m * 2.0f32.powi(exp - bias)
    };

    if sign == 1 { -val } else { val }
}

pub fn fp4_e2m1_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 3) & 1;
    let exp = ((bits >> 1) & 0x03) as i32;
    let mant = (bits & 0x01) as i32;

    let bias = 1;
    let val = if exp == 0 && mant == 0 {
        0.0f32
    } else if exp == 0 {
        let m = mant as f32 * 0.5;
        m * 2.0f32.powi(1 - bias)
    } else {
        let m = 1.0 + mant as f32 * 0.5;
        m * 2.0f32.powi(exp - bias)
    };

    if sign == 1 { -val } else { val }
}

pub fn bf16_to_f32(data: &[u8]) -> Vec<f32> {
    let bf16_slice: &[half::bf16] = bytemuck::cast_slice(data);
    bf16_slice.iter().map(|x| x.to_f32()).collect()
}

pub fn f32_to_bf16(data: &[f32]) -> Vec<u8> {
    let bf16: Vec<half::bf16> = data.iter().map(|x| half::bf16::from_f32(*x)).collect();
    bytemuck::cast_slice(&bf16).to_vec()
}

pub fn rmsnorm_bf16(x_bf16: &[half::bf16], weight_bf16: Option<&[half::bf16]>, last_dim: usize, eps: f32) -> Vec<half::bf16> {
    let n_rows = x_bf16.len() / last_dim;
    let mut out = vec![half::bf16::from_f32(0.0); x_bf16.len()];
    for r in 0..n_rows {
        let base = r * last_dim;
        let sq_sum: f32 = (0..last_dim).map(|d| x_bf16[base + d].to_f32().powi(2)).sum();
        let inv_norm = 1.0 / (sq_sum / last_dim as f32 + eps).sqrt();
        for d in 0..last_dim {
            let w = weight_bf16.map_or(1.0f32, |w| w[d].to_f32());
            out[base + d] = half::bf16::from_f32(x_bf16[base + d].to_f32() * inv_norm * w);
        }
    }
    out
}

pub fn rmsnorm_f32(data: &[f32], last_dim: usize, eps: f32) -> Vec<f32> {
    let n_rows = data.len() / last_dim;
    let mut out = vec![0.0f32; data.len()];
    for r in 0..n_rows {
        let base = r * last_dim;
        let sq_sum: f32 = (0..last_dim).map(|d| data[base + d].powi(2)).sum();
        let inv_norm = 1.0 / (sq_sum / last_dim as f32 + eps).sqrt();
        for d in 0..last_dim {
            out[base + d] = data[base + d] * inv_norm;
        }
    }
    out
}

pub fn hadamard_transform(data: &mut [f32], dim: usize) {
    let n = dim;
    assert!(n > 0 && (n & (n - 1)) == 0, "dim must be power of 2");
    let n_rows = data.len() / dim;
    for r in 0..n_rows {
        let base = r * dim;
        let mut h = 1usize;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in 0..h {
                    let a = data[base + i + j];
                    let b = data[base + i + j + h];
                    data[base + i + j] = a + b;
                    data[base + i + j + h] = a - b;
                }
            }
            h *= 2;
        }
        let scale = (n as f32).recip().sqrt();
        for i in 0..n {
            data[base + i] *= scale;
        }
    }
}

pub fn f32_to_fp4_e2m1(v: f32) -> u8 {
    let fp4_max = 6.0f32;
    if v != v {
        return 0;
    }
    let clamped = v.clamp(-fp4_max, fp4_max);
    let sign = if clamped < 0.0 { 1u8 } else { 0u8 };
    let abs_val = clamped.abs();

    if abs_val == 0.0 {
        return sign << 3;
    }

    let bits = abs_val.to_bits();
    let exp_f32 = ((bits >> 23) & 0xFF) as i32;
    let mant_f32 = (bits & 0x7FFFFF) as u32;

    if exp_f32 == 0 {
        return sign << 3;
    }

    let unbiased_exp = exp_f32 - 127;
    let biased_fp4 = unbiased_exp + 1;

    if biased_fp4 < 0 {
        return sign << 3;
    }

    if biased_fp4 == 0 {
        let full_mant = mant_f32 | 0x800000;
        let mant_bit = if full_mant >= 0x400000 { 1u8 } else { 0u8 };
        return (sign << 3) | mant_bit;
    }

    if biased_fp4 >= 4 {
        return (sign << 3) | 0b111;
    }

    let mant_fp4 = ((mant_f32 >> 22) & 1) as u8;
    (sign << 3) | ((biased_fp4 as u8) << 1) | mant_fp4
}

pub fn fp4_e2m1_qdq(v: f32, scale: f32) -> f32 {
    let q = (v / scale).clamp(-6.0, 6.0);
    let fp4_bits = f32_to_fp4_e2m1(q);
    fp4_e2m1_to_f32(fp4_bits) * scale
}

pub fn fast_round_scale(amax: f32, max_val_inv: f32) -> f32 {
    if amax <= 0.0 || amax != amax {
        return 1.0;
    }
    let product = amax * max_val_inv;
    let bits = product.to_bits();
    let exp_bits = ((bits >> 23) & 0xFF) as i32;
    let mant_bits = bits & 0x7FFFFF;
    let log2_ceil = exp_bits - 127 + if mant_bits != 0 { 1 } else { 0 };
    let pow2_bits = ((log2_ceil + 127) as u32) << 23;
    f32::from_bits(pow2_bits)
}

pub fn fp4_act_quant_qdq(data: &mut [f32], dim: usize, block_size: usize) {
    let fp4_max = 6.0f32;
    let fp4_max_inv = 1.0 / fp4_max;
    let n_rows = data.len() / dim;

    for r in 0..n_rows {
        let base = r * dim;
        for block_start in (0..dim).step_by(block_size) {
            let block_end = (block_start + block_size).min(dim);
            let mut amax = 0.0f32;
            for d in block_start..block_end {
                amax = amax.max(data[base + d].abs());
            }
            let scale = fast_round_scale(amax, fp4_max_inv);
            let safe_scale = if scale > 0.0 { scale } else { fp4_max / 6.0 };
            for d in block_start..block_end {
                let q = (data[base + d] / safe_scale).clamp(-fp4_max, fp4_max);
                let fp4_bits = f32_to_fp4_e2m1(q);
                data[base + d] = fp4_e2m1_to_f32(fp4_bits) * safe_scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_identity() {
        let dim = 4usize;
        let mut data = vec![1.0f32, 0.0, 0.0, 0.0];
        hadamard_transform(&mut data, dim);
        let scale = (dim as f32).recip().sqrt();
        for i in 0..dim {
            let expected = scale;
            assert!(
                (data[i] - expected).abs() < 1e-5,
                "hadamard of [1,0,0,0]: data[{}] = {}, expected {}",
                i, data[i], expected
            );
        }
    }

    #[test]
    fn test_hadamard_inverse() {
        let dim = 8usize;
        let original: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0).sin()).collect();
        let mut data = original.clone();
        hadamard_transform(&mut data, dim);
        hadamard_transform(&mut data, dim);
        for i in 0..dim {
            assert!(
                (data[i] - original[i]).abs() < 1e-4,
                "hadamard^2 should be identity: data[{}] = {}, expected {}",
                i, data[i], original[i]
            );
        }
    }

    #[test]
    fn test_hadamard_orthogonal() {
        let dim = 8usize;
        let mut v1 = vec![0.0f32; dim];
        let mut v2 = vec![0.0f32; dim];
        v1[0] = 1.0;
        v2[1] = 1.0;
        hadamard_transform(&mut v1, dim);
        hadamard_transform(&mut v2, dim);
        let dot: f32 = (0..dim).map(|i| v1[i] * v2[i]).sum();
        assert!(dot.abs() < 1e-4, "orthogonal: dot = {}", dot);
    }

    #[test]
    fn test_fp4_e2m1_roundtrip() {
        let test_vals = [0.0f32, 1.0, 2.0, 3.0, 4.0, 6.0, -1.0, -6.0, 0.5, 1.5];
        for &v in &test_vals {
            let bits = f32_to_fp4_e2m1(v);
            let deq = fp4_e2m1_to_f32(bits);
            let rel_err = if v.abs() > 1e-6 { (deq - v).abs() / v.abs() } else { deq.abs() };
            assert!(
                rel_err < 0.35,
                "fp4 roundtrip: v={}, bits={}, deq={}, rel_err={}",
                v, bits, deq, rel_err
            );
        }
    }

    #[test]
    fn test_fp4_qdq_block() {
        let dim = 128usize;
        let block_size = 32usize;
        let mut data: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32 * 0.5).sin() * 3.0).collect();
        hadamard_transform(&mut data, dim);
        let original = data.clone();
        fp4_act_quant_qdq(&mut data, dim, block_size);
        let mut total_sq = 0.0f32;
        let mut diff_sq = 0.0f32;
        for i in 0..dim {
            total_sq += original[i] * original[i];
            diff_sq += (data[i] - original[i]) * (data[i] - original[i]);
        }
        let cos_sim = if total_sq > 0.0 { diff_sq / total_sq } else { 1.0 };
        assert!(cos_sim < 0.3, "fp4 qdq: relative MSE = {} (should be < 0.3)", cos_sim);
    }

    #[test]
    fn test_fast_round_scale() {
        let fp4_max_inv = 1.0 / 6.0f32;
        let s = fast_round_scale(12.0, fp4_max_inv);
        assert!(s > 0.0 && (s - 2.0).abs() < 1e-5, "fast_round_scale(12) = {}, expected 2.0", s);
        let s2 = fast_round_scale(3.0, fp4_max_inv);
        assert!(s2 > 0.0 && (s2 - 0.5).abs() < 1e-5, "fast_round_scale(3) = {}, expected 0.5", s2);
        let s3 = fast_round_scale(6.0, fp4_max_inv);
        assert!(s3 > 0.0 && (s3 - 1.0).abs() < 1e-5, "fast_round_scale(6) = {}, expected 1.0", s3);
        let s4 = fast_round_scale(0.57, fp4_max_inv);
        assert!(s4 > 0.0 && (s4 - 0.125).abs() < 1e-5, "fast_round_scale(0.57) = {}, expected 0.125", s4);
        let ratio = 0.57f32 / s4;
        assert!(ratio <= 6.0, "0.57/scale = {} should be <= 6.0", ratio);
    }

    #[test]
    fn test_hadamard_preserves_norm() {
        let dim = 16usize;
        let mut data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.7).cos()).collect();
        let orig_sq_norm: f32 = data.iter().map(|v| v * v).sum();
        hadamard_transform(&mut data, dim);
        let new_sq_norm: f32 = data.iter().map(|v| v * v).sum();
        assert!(
            (orig_sq_norm - new_sq_norm).abs() / orig_sq_norm < 1e-4,
            "norm not preserved: {} vs {}", orig_sq_norm, new_sq_norm
        );
    }
}
