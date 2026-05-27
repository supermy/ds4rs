use super::tables::*;
use super::avx512::*;
use super::tile_layout::Iq2XsTile;
use std::sync::Arc;

// ============================================================================
// 多量化格式 trait 架构
// ============================================================================

/// 量化类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantType {
    Iq2Xs,    // 2.3125 bpw, block_iq2_xs = 74B/256elem
    Iq2Xxs,   // 2.0625 bpw, block_iq2_xxs = 66B/256elem
    Q2K,      // 2.5625 bpw, block_q2_K = 84B/256elem
    Fp4,      // 2.0625 bpw, FP4 e2m1 + FP8 scale
}

/// 量化权重 trait：定义所有量化格式必须实现的点积接口
///
/// 共享外层：cpu_expert_ffn_pair / SwiGLU / 配对点积
/// 格式特化：只需实现 vec_dot() 和 size_bytes()
///
/// 不需要 4 种 FFN 算子！外层 FFN 流程完全相同，
/// 只有内层点积内核按格式分发。
///
/// 激活格式选择（关键）：
///   IQ2_XS / IQ2_XXS / Q2_K → Q8 激活（maddubs+madd 管线）
///     权重本质是 int8（grid 查表 / int2 解码），与 Q8 天然匹配
///     maddubs(grid_u8, q8_i8) → i16，整数×整数，零额外误差
///
///   FP4 → F32 激活（FMA 管线），不走 Q8
///     权重本质是 e2m1 浮点，不是整数
///     若用 Q8 激活，需要 FP4→int8 二次量化，引入 5-10% 额外误差
///     正确做法：FP4 反量化为 F32，与 F32 激活做 FMA
///     _mm512_fmadd_ps(f32_weight, f32_activation, acc)
///
/// matvec() 默认实现用 Q8 预量化，FP4 应覆盖 matvec() 用 FMA 管线
pub trait QuantizedWeight: Send + Sync {
    /// 量化类型
    fn quant_type(&self) -> QuantType;

    /// 权重总字节数（用于 SLRU 字节预算）
    fn size_bytes(&self) -> usize;

    /// 行数
    fn n_rows(&self) -> usize;

    /// 列数
    fn n_cols(&self) -> usize;

    /// 每行的 block 数量
    fn blocks_per_row(&self) -> usize;

    /// 核心点积：预量化 Q8 输入 × 量化权重 → f32 结果
    ///
    /// 仅适用于整数权重量化格式（IQ2_XS / IQ2_XXS / Q2_K）。
    /// FP4 等浮点权重量化格式应覆盖 matvec() 而非实现此方法。
    ///
    /// # Safety
    /// 调用方必须确保 AVX-512/AVX2 特性检测已通过
    unsafe fn vec_dot_q8(
        &self,
        row: usize,
        q8: &[i8],
        q8_inv_scales: &[f32],
    ) -> f32;

    /// 矩阵向量乘法（rayon 并行）
    ///
    /// 默认实现：F32→Q8 预量化 + vec_dot_q8（适用于整数权重量化格式）
    /// FP4 等浮点权重量化格式应覆盖此方法，使用 FMA 管线
    fn matvec(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;

        let n_blocks = self.blocks_per_row();
        let n_rows = self.n_rows();

        // 预量化 x：所有行共享同一个输入向量，只量化一次
        let mut q8_buf = vec![0i8; n_blocks * 256];
        let mut q8_inv_scales = vec![0.0f32; n_blocks];
        for blk in 0..n_blocks {
            q8_inv_scales[blk] = quantize_f32_to_q8_block(
                &x[blk * 256..(blk + 1) * 256],
                &mut q8_buf[blk * 256..(blk + 1) * 256],
                256,
            );
        }

        let mut output = vec![0.0f32; n_rows];
        let chunk_size = 64usize;
        output.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
            let row_start = chunk_idx * chunk_size;
            for (i, out) in chunk.iter_mut().enumerate() {
                let row = row_start + i;
                *out = unsafe { self.vec_dot_q8(row, &q8_buf, &q8_inv_scales) };
            }
        });
        output
    }
}

/// Iq2XsWeight 实现 QuantizedWeight trait
impl QuantizedWeight for Iq2XsWeight {
    fn quant_type(&self) -> QuantType { QuantType::Iq2Xs }

    fn size_bytes(&self) -> usize {
        self.d.len() * 4 + self.qs.len() * 2 + self.scales.len()
    }

    fn n_rows(&self) -> usize { self.shape.0 }
    fn n_cols(&self) -> usize { self.shape.1 }
    fn blocks_per_row(&self) -> usize { self.shape.1 / 256 }

    unsafe fn vec_dot_q8(
        &self,
        row: usize,
        q8: &[i8],
        q8_inv_scales: &[f32],
    ) -> f32 {
        let n_blocks_per_row = self.blocks_per_row();
        let row_offset = row * n_blocks_per_row;
        let d_row = &self.d[row_offset..row_offset + n_blocks_per_row];
        let qs_row = &self.qs[row_offset * 32..(row_offset + n_blocks_per_row) * 32];
        let scales_row = &self.scales[row_offset * 8..(row_offset + n_blocks_per_row) * 8];

        if is_avx512_supported() {
            iq2xs_vec_dot_q8_avx512(d_row, qs_row, scales_row, q8, q8_inv_scales, n_blocks_per_row)
        } else if is_avx2_supported() {
            iq2xs_vec_dot_q8_avx2(d_row, qs_row, scales_row, q8, q8_inv_scales, n_blocks_per_row)
        } else {
            // 标量回退
            let mut sum = 0.0f32;
            for blk in 0..n_blocks_per_row {
                sum += d_row[blk] * q8_inv_scales[blk];
            }
            sum
        }
    }
}

/// 快速 exp 近似（6 阶 Taylor + 范围缩减），与 route.rs 中算法相同
/// 精度：典型相对误差 < 1e-6，极端输入下最大误差 ~2e-5
#[inline]
pub fn exp_approx_scalar(x: f32) -> f32 {
    const LN2: f32 = 0.6931471805599453;
    const INV_LN2: f32 = 1.4426950408889634;

    let x = x.clamp(-87.0, 88.0);

    let k = (x * INV_LN2).round() as i32;
    let r = x - k as f32 * LN2;

    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.1666666666666668
                    + r * (0.04166666666666679
                        + r * (0.008333333333333357
                            + r * 0.0013888888888888905)))));

    let k_i32 = k + 127;
    if k_i32 < 0 || k_i32 > 254 {
        if k > 0 { f32::MAX } else { 0.0 }
    } else {
        p * f32::from_bits((k_i32 as u32) << 23)
    }
}

#[derive(Clone)]
pub struct Iq2XsWeight {
    pub d: Vec<f32>,
    pub qs: Vec<u16>,
    pub scales: Vec<u8>,
    pub shape: (usize, usize),
}

impl std::fmt::Debug for Iq2XsWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Iq2XsWeight")
            .field("shape", &self.shape)
            .field("n_blocks", &self.d.len())
            .field("d_len", &self.d.len())
            .field("qs_len", &self.qs.len())
            .field("scales_len", &self.scales.len())
            .finish()
    }
}

impl Iq2XsWeight {
    pub fn new(d: Vec<f32>, qs: Vec<u16>, scales: Vec<u8>, shape: (usize, usize)) -> Self {
        Self { d, qs, scales, shape }
    }

    pub fn n_blocks(&self) -> usize {
        self.d.len()
    }

    pub fn n_blocks_per_row(&self) -> usize {
        self.shape.1 / 256
    }
}

// ============================================================================
// IQ2_XXS 权重结构体（2.0625 bpw, block_iq2_xxs = 66B/256elem）
// ============================================================================

/// IQ2_XXS 量化权重
///
/// 数据格式（与 llama.cpp block_iq2_xxs 一致）：
///   - d: fp16 super-block scale，每 256 元素一个
///   - qs: uint16[32]，grid 索引 + 符号索引打包
///     - 低 9 位: grid 索引 (0-511)
///     - 高 7 位: 符号索引 (0-127)
///
/// block_iq2_xxs 大小: 2 + 32*2 = 66 bytes / 256 elements = 2.0625 bpw
#[derive(Clone)]
pub struct Iq2XxsWeight {
    pub d: Vec<f32>,
    pub qs: Vec<u16>,
    pub shape: (usize, usize),
}

impl std::fmt::Debug for Iq2XxsWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Iq2XxsWeight")
            .field("shape", &self.shape)
            .field("n_blocks", &self.d.len())
            .finish()
    }
}

impl Iq2XxsWeight {
    pub fn new(d: Vec<f32>, qs: Vec<u16>, shape: (usize, usize)) -> Self {
        Self { d, qs, shape }
    }

    pub fn n_blocks(&self) -> usize {
        self.d.len()
    }

    pub fn n_blocks_per_row(&self) -> usize {
        self.shape.1 / 256
    }
}

/// Iq2XxsWeight 实现 QuantizedWeight trait
impl QuantizedWeight for Iq2XxsWeight {
    fn quant_type(&self) -> QuantType { QuantType::Iq2Xxs }

    fn size_bytes(&self) -> usize {
        self.d.len() * 4 + self.qs.len() * 2
    }

    fn n_rows(&self) -> usize { self.shape.0 }
    fn n_cols(&self) -> usize { self.shape.1 }
    fn blocks_per_row(&self) -> usize { self.shape.1 / 256 }

    unsafe fn vec_dot_q8(
        &self,
        row: usize,
        q8: &[i8],
        q8_inv_scales: &[f32],
    ) -> f32 {
        let n_blocks_per_row = self.blocks_per_row();
        let row_offset = row * n_blocks_per_row;
        let d_row = &self.d[row_offset..row_offset + n_blocks_per_row];
        let qs_row = &self.qs[row_offset * 32..(row_offset + n_blocks_per_row) * 32];

        iq2xxs_vec_dot_q8_avx2(d_row, qs_row, q8, q8_inv_scales, n_blocks_per_row)
    }
}

// ============================================================================
// Q2_K 权重结构体（2.5625 bpw, block_q2_K = 84B/256elem）
// ============================================================================

/// Q2_K 量化权重
///
/// 数据格式（与 llama.cpp block_q2_K 一致）：
///   - scales: uint8[16]，每 16 元素一个 scale
///   - qs: uint8[64]，2-bit 量化值打包（每字节 4 个 2-bit 值）
///   - d: fp16 super-block scale
///   - dmin: fp16 minimum scale（用于负值）
///
/// block_q2_K 大小: 16 + 64 + 2 + 2 = 84 bytes / 256 elements = 2.625 bpw
#[derive(Clone)]
pub struct Q2KWeight {
    pub d: Vec<f32>,
    pub dmin: Vec<f32>,
    pub scales: Vec<u8>,
    pub qs: Vec<u8>,
    pub shape: (usize, usize),
}

impl std::fmt::Debug for Q2KWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Q2KWeight")
            .field("shape", &self.shape)
            .field("n_blocks", &self.d.len())
            .finish()
    }
}

impl Q2KWeight {
    pub fn new(d: Vec<f32>, dmin: Vec<f32>, scales: Vec<u8>, qs: Vec<u8>, shape: (usize, usize)) -> Self {
        Self { d, dmin, scales, qs, shape }
    }

    pub fn n_blocks(&self) -> usize {
        self.d.len()
    }

    pub fn n_blocks_per_row(&self) -> usize {
        self.shape.1 / 256
    }
}

/// Q2KWeight 实现 QuantizedWeight trait
impl QuantizedWeight for Q2KWeight {
    fn quant_type(&self) -> QuantType { QuantType::Q2K }

    fn size_bytes(&self) -> usize {
        self.d.len() * 4 + self.dmin.len() * 4 + self.scales.len() + self.qs.len()
    }

    fn n_rows(&self) -> usize { self.shape.0 }
    fn n_cols(&self) -> usize { self.shape.1 }
    fn blocks_per_row(&self) -> usize { self.shape.1 / 256 }

    unsafe fn vec_dot_q8(
        &self,
        row: usize,
        q8: &[i8],
        q8_inv_scales: &[f32],
    ) -> f32 {
        let n_blocks_per_row = self.blocks_per_row();
        let row_offset = row * n_blocks_per_row;
        let d_row = &self.d[row_offset..row_offset + n_blocks_per_row];
        let dmin_row = &self.dmin[row_offset..row_offset + n_blocks_per_row];
        let scales_row = &self.scales[row_offset * 16..(row_offset + n_blocks_per_row) * 16];
        let qs_row = &self.qs[row_offset * 64..(row_offset + n_blocks_per_row) * 64];

        q2k_vec_dot_q8_avx2(d_row, dmin_row, scales_row, qs_row, q8, q8_inv_scales, n_blocks_per_row)
    }
}

// ============================================================================
// FP4 权重结构体（e2m1 浮点 + per-block E8M0 scale）
// ============================================================================

/// E8M0 (8-bit exponent, power-of-2) scale → f32 转换
///
/// E8M0 编码：8 位纯指数，值 = 2^(bits-127)
/// bits=0 → 0.0（特殊值），bits=127 → 2^0 = 1.0
#[inline]
fn e8m0_to_f32(bits: u8) -> f32 {
    if bits == 0 { return 0.0; }
    f32::from_bits((bits as u32) << 23)
}

/// FP4 (e2m1) 量化权重
///
/// 数据格式（与 GPU 侧一致）：
///   - weight_packed: [out, in//2]，每字节打包 2 个 FP4 值
///     低 4 位 = 第 1 个 FP4，高 4 位 = 第 2 个 FP4
///   - weight_interleaved: 预交错 nibble 格式，供 AVX-512 直接加载
///     每 32 字节 = 1 个 scale block（Group A 16 nibbles + Group B 16 nibbles）
///   - scale: [out, in//32]，E8M0 (u8) per-block scale，每 32 个 FP4 元素一个
///   - shape: (out_dim, in_dim)，逻辑形状
///
/// FP4 e2m1 编码（4 bit）：
///   bit[3] = sign, bit[2:0] = (e2, m1)
///   值 = (-1)^sign × 2^(e-1) × (1 + m×0.5)
///   e=0: ±0, ±0.5; e=1: ±1, ±1.5; e=2: ±2, ±3; e=3: ±4, ±6
///
/// CPU 内核使用 FMA 管线（F32 激活 × F32 反量化权重），
/// 不走 Q8 maddubs 管线（FP4 是浮点权重，不是整数权重）。
#[derive(Clone)]
pub struct Fp4Weight {
    /// 打包的 FP4 权重，每字节 2 个 FP4 值
    /// 低 4 位 = 第 1 个 FP4 (lo)，高 4 位 = 第 2 个 FP4 (hi)
    /// 总大小 = out_dim * (in_dim / 2)
    pub weight_packed: Vec<u8>,
    /// per-block E8M0 scale，每 32 个 FP4 元素一个 u8
    /// 值 = 2^(bits-127)，bits=0 时值为 0
    pub scale: Vec<u8>,
    /// 逻辑形状 (out_dim, in_dim)
    pub shape: (usize, usize),
}

impl std::fmt::Debug for Fp4Weight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fp4Weight")
            .field("shape", &self.shape)
            .field("packed_bytes", &self.weight_packed.len())
            .field("scale_len", &self.scale.len())
            .finish()
    }
}

impl Fp4Weight {
    pub fn new(weight_packed: Vec<u8>, scale: Vec<u8>, shape: (usize, usize)) -> Self {
        let expected_packed = shape.0 * (shape.1 / 2);
        let expected_scales = shape.0 * (shape.1 / 32);
        assert_eq!(
            weight_packed.len(), expected_packed,
            "FP4 packed size mismatch: got {}, expected {} (shape={:?})",
            weight_packed.len(), expected_packed, shape
        );
        assert_eq!(
            scale.len(), expected_scales,
            "FP4 scale count mismatch: got {}, expected {} (shape={:?})",
            scale.len(), expected_scales, shape
        );
        Self { weight_packed, scale, shape }
    }

    /// 解码单个 FP4 e2m1 值为 f32
    #[inline]
    fn decode_fp4(nibble: u8) -> f32 {
        let sign = (nibble >> 3) & 1;
        let e = (nibble >> 1) & 3;
        let m = nibble & 1;
        // 值 = (-1)^sign × 2^(e-1) × (1 + m×0.5)
        // e=0: 2^(-1) × (1+m×0.5) = 0.5 或 0.75 (特殊处理: 0 或 0.5)
        let abs_val = match e {
            0 => if m == 0 { 0.0 } else { 0.5 },
            1 => if m == 0 { 1.0 } else { 1.5 },
            2 => if m == 0 { 2.0 } else { 3.0 },
            3 => if m == 0 { 4.0 } else { 6.0 },
            _ => unreachable!(),
        };
        if sign == 1 { -abs_val } else { abs_val }
    }

    /// 反量化一行权重到 F32
    ///
    /// FP4 权重 × E8M0 scale → F32
    /// 每 32 个 FP4 元素共享一个 scale
    fn dequantize_row(&self, row: usize) -> Vec<f32> {
        let (out_dim, in_dim) = self.shape;
        let _ = out_dim;
        let n_packed_per_row = in_dim / 2;
        let n_scales_per_row = in_dim / 32;

        let row_packed_start = row * n_packed_per_row;
        let row_scale_start = row * n_scales_per_row;

        let mut result = vec![0.0f32; in_dim];

        for i in 0..n_packed_per_row {
            let packed = self.weight_packed[row_packed_start + i];
            let lo = packed & 0xF;
            let hi = (packed >> 4) & 0xF;

            let col_lo = i * 2;
            let col_hi = i * 2 + 1;

            let scale_lo = e8m0_to_f32(self.scale[row_scale_start + col_lo / 32]);
            let scale_hi = e8m0_to_f32(self.scale[row_scale_start + col_hi / 32]);

            result[col_lo] = Self::decode_fp4(lo) * scale_lo;
            result[col_hi] = Self::decode_fp4(hi) * scale_hi;
        }

        result
    }
}

/// FP4 实现 QuantizedWeight trait（FMA 管线）
impl QuantizedWeight for Fp4Weight {
    fn quant_type(&self) -> QuantType { QuantType::Fp4 }

    fn size_bytes(&self) -> usize {
        self.weight_packed.len() + self.scale.len()
    }

    fn n_rows(&self) -> usize { self.shape.0 }
    fn n_cols(&self) -> usize { self.shape.1 }
    fn blocks_per_row(&self) -> usize { self.shape.1 / 32 } // FP4 block size = 32

    /// FP4 不应该用 Q8 管线，此方法仅作为兼容占位
    ///
    /// 实际调用路径：matvec() → FMA 管线（覆盖了默认的 Q8 实现）
    unsafe fn vec_dot_q8(
        &self,
        row: usize,
        _q8: &[i8],
        _q8_inv_scales: &[f32],
    ) -> f32 {
        // FP4 不走 Q8 管线，此方法不应被调用
        // 如果被调用，回退到标量 FMA 点积
        let dequant = self.dequantize_row(row);
        let n = dequant.len().min(_q8.len());
        let mut sum = 0.0f32;
        for i in 0..n {
            let x_val = _q8[i] as f32 * e8m0_to_f32(self.scale.get(i / 32).copied().unwrap_or(127));
            sum += dequant[i] * x_val;
        }
        sum
    }

    /// 覆盖默认的 Q8 matvec，使用 FMA 管线
    ///
    /// FP4 权重是浮点格式，不应该量化为 Q8 激活。
    /// 正确做法：FP4 反量化为 F32，与 F32 激活做 FMA。
    fn matvec(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;

        let n_rows = self.shape.0;
        let n_cols = self.shape.1;
        assert_eq!(x.len(), n_cols, "x length mismatch: got {}, expected {}", x.len(), n_cols);

        let mut output = vec![0.0f32; n_rows];

        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            *out = self.fp4_vec_dot_f32(row, x);
        });

        output
    }
}

impl Fp4Weight {
    /// FP4 × F32 向量点积（FMA 管线）
    ///
    /// 逐块反量化 FP4 权重为 F32，与 F32 激活做点积。
    /// 使用 AVX-512 FMA 加速。
    fn fp4_vec_dot_f32(&self, row: usize, x: &[f32]) -> f32 {
        let (out_dim, in_dim) = self.shape;
        let _ = out_dim;
        let n_packed_per_row = in_dim / 2;
        let n_scales_per_row = in_dim / 32;

        let row_packed_start = row * n_packed_per_row;
        let row_scale_start = row * n_scales_per_row;

        #[cfg(target_arch = "x86_64")]
        {
            if is_avx512_supported() {
                return unsafe {
                    self.fp4_vec_dot_f32_avx512(row_packed_start, row_scale_start, x, n_packed_per_row)
                };
            }
        }

        // 标量回退
        let mut sum = 0.0f32;
        for i in 0..n_packed_per_row {
            let packed = self.weight_packed[row_packed_start + i];
            let lo = packed & 0xF;
            let hi = (packed >> 4) & 0xF;

            let col_lo = i * 2;
            let col_hi = i * 2 + 1;

            let scale_lo = e8m0_to_f32(self.scale[row_scale_start + col_lo / 32]);
            let scale_hi = e8m0_to_f32(self.scale[row_scale_start + col_hi / 32]);

            sum += Self::decode_fp4(lo) * scale_lo * x[col_lo];
            sum += Self::decode_fp4(hi) * scale_hi * x[col_hi];
        }
        sum
    }

    /// AVX-512 FMA 加速的 FP4 × F32 点积（E8M0 scale 优化版）
    ///
    /// 优化策略：
    ///   1. E8M0 scale = power-of-2，1 字节存储（vs F32 4 字节），减少 75% scale 带宽
    ///   2. permutex2var_ps 交错合并 lo/hi 权重 → 直接与连续 x 做 FMA
    ///   3. 4 路累加器减少 FMA 依赖链延迟
    ///   4. 循环展开 2 次 + 软件预取
    ///
    /// 每 block = 16 packed bytes = 32 FP4 元素 = 1 E8M0 scale
    /// lo_i32 = [l0,...,l15], hi_i32 = [h0,...,h15]
    /// permutex2var 交错：
    ///   merge_lo: [l0,h0, l1,h1, ..., l7,h7] → 对应 x[0..15]
    ///   merge_hi: [l8,h8, l9,h9, ..., l15,h15] → 对应 x[16..31]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512bw")]
    #[target_feature(enable = "avx512vl")]
    unsafe fn fp4_vec_dot_f32_avx512(
        &self,
        row_packed_start: usize,
        row_scale_start: usize,
        x: &[f32],
        n_packed_per_row: usize,
    ) -> f32 {
        use std::arch::x86_64::*;

        // FP4 e2m1 LUT: 16 个可能值
        let lut = _mm512_set_ps(
            -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
             6.0,  4.0,  3.0,  2.0,  1.5,  1.0,  0.5, 0.0,
        );

        // permutex2var 交错索引
        // merge_lo: 从 w_lo(src1) 和 w_hi(src2) 交替取值
        // idx[i] < 16 选 src1(w_lo), idx[i] >= 16 选 src2(w_hi)
        // 结果: [l0,h0, l1,h1, ..., l7,h7] → 对应 x[0..15]
        let merge_lo_idx = _mm512_set_epi32(
            23, 7, 22, 6, 21, 5, 20, 4,
            19, 3, 18, 2, 17, 1, 16, 0
        );
        // merge_hi: [l8,h8, l9,h9, ..., l15,h15] → 对应 x[16..31]
        let merge_hi_idx = _mm512_set_epi32(
            31, 15, 30, 14, 29, 13, 28, 12,
            27, 11, 26, 10, 25, 9, 24, 8
        );

        // 4 路累加器
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        const BLOCK_SIZE: usize = 16;
        let n_full_blocks = n_packed_per_row / BLOCK_SIZE;

        let mut block = 0;

        // 主循环：每次处理 2 个 block（展开 2 次）
        while block + 1 < n_full_blocks {
            // ---- Block 0 ----
            let base0 = row_packed_start + block * BLOCK_SIZE;
            let scale_bits_0 = self.scale[row_scale_start + block];
            let scale0 = if scale_bits_0 == 0 {
                _mm512_set1_ps(0.0f32)
            } else {
                _mm512_set1_ps(e8m0_to_f32(scale_bits_0))
            };

            // 预取
            if block + 2 < n_full_blocks {
                let next_base = row_packed_start + (block + 2) * BLOCK_SIZE;
                _mm_prefetch(self.weight_packed.as_ptr().add(next_base) as *const i8, _MM_HINT_T1);
            }

            let packed0 = _mm_loadu_si128(self.weight_packed.as_ptr().add(base0) as *const __m128i);
            let lo0 = _mm_and_si128(packed0, _mm_set1_epi8(0xF));
            let hi0 = _mm_srli_epi16(_mm_and_si128(packed0, _mm_set1_epi8(0xF0u8 as i8)), 4);

            let lo_i32_0 = _mm512_cvtepi8_epi32(lo0);
            let hi_i32_0 = _mm512_cvtepi8_epi32(hi0);

            let lo_f32_0 = _mm512_permutexvar_ps(lo_i32_0, lut);
            let hi_f32_0 = _mm512_permutexvar_ps(hi_i32_0, lut);

            let w_lo_0 = _mm512_mul_ps(lo_f32_0, scale0);
            let w_hi_0 = _mm512_mul_ps(hi_f32_0, scale0);

            let w_first_0 = _mm512_permutex2var_ps(w_lo_0, merge_lo_idx, w_hi_0);
            let w_second_0 = _mm512_permutex2var_ps(w_lo_0, merge_hi_idx, w_hi_0);

            let x_base0 = block * BLOCK_SIZE * 2;
            let x0_0 = _mm512_loadu_ps(x.as_ptr().add(x_base0));
            let x0_1 = _mm512_loadu_ps(x.as_ptr().add(x_base0 + 16));

            acc0 = _mm512_fmadd_ps(w_first_0, x0_0, acc0);
            acc1 = _mm512_fmadd_ps(w_second_0, x0_1, acc1);

            // ---- Block 1 ----
            let base1 = row_packed_start + (block + 1) * BLOCK_SIZE;
            let scale_bits_1 = self.scale[row_scale_start + block + 1];
            let scale1 = if scale_bits_1 == 0 {
                _mm512_set1_ps(0.0f32)
            } else {
                _mm512_set1_ps(e8m0_to_f32(scale_bits_1))
            };

            let packed1 = _mm_loadu_si128(self.weight_packed.as_ptr().add(base1) as *const __m128i);
            let lo1 = _mm_and_si128(packed1, _mm_set1_epi8(0xF));
            let hi1 = _mm_srli_epi16(_mm_and_si128(packed1, _mm_set1_epi8(0xF0u8 as i8)), 4);

            let lo_i32_1 = _mm512_cvtepi8_epi32(lo1);
            let hi_i32_1 = _mm512_cvtepi8_epi32(hi1);

            let lo_f32_1 = _mm512_permutexvar_ps(lo_i32_1, lut);
            let hi_f32_1 = _mm512_permutexvar_ps(hi_i32_1, lut);

            let w_lo_1 = _mm512_mul_ps(lo_f32_1, scale1);
            let w_hi_1 = _mm512_mul_ps(hi_f32_1, scale1);

            let w_first_1 = _mm512_permutex2var_ps(w_lo_1, merge_lo_idx, w_hi_1);
            let w_second_1 = _mm512_permutex2var_ps(w_lo_1, merge_hi_idx, w_hi_1);

            let x_base1 = (block + 1) * BLOCK_SIZE * 2;
            let x1_0 = _mm512_loadu_ps(x.as_ptr().add(x_base1));
            let x1_1 = _mm512_loadu_ps(x.as_ptr().add(x_base1 + 16));

            acc2 = _mm512_fmadd_ps(w_first_1, x1_0, acc2);
            acc3 = _mm512_fmadd_ps(w_second_1, x1_1, acc3);

            block += 2;
        }

        // 处理剩余的奇数 block
        if block < n_full_blocks {
            let base = row_packed_start + block * BLOCK_SIZE;
            let scale_bits = self.scale[row_scale_start + block];
            let scale_val = if scale_bits == 0 {
                _mm512_set1_ps(0.0f32)
            } else {
                _mm512_set1_ps(e8m0_to_f32(scale_bits))
            };

            let packed = _mm_loadu_si128(self.weight_packed.as_ptr().add(base) as *const __m128i);
            let lo = _mm_and_si128(packed, _mm_set1_epi8(0xF));
            let hi = _mm_srli_epi16(_mm_and_si128(packed, _mm_set1_epi8(0xF0u8 as i8)), 4);

            let lo_i32 = _mm512_cvtepi8_epi32(lo);
            let hi_i32 = _mm512_cvtepi8_epi32(hi);

            let lo_f32 = _mm512_permutexvar_ps(lo_i32, lut);
            let hi_f32 = _mm512_permutexvar_ps(hi_i32, lut);

            let w_lo = _mm512_mul_ps(lo_f32, scale_val);
            let w_hi = _mm512_mul_ps(hi_f32, scale_val);

            let w_first = _mm512_permutex2var_ps(w_lo, merge_lo_idx, w_hi);
            let w_second = _mm512_permutex2var_ps(w_lo, merge_hi_idx, w_hi);

            let x_base = block * BLOCK_SIZE * 2;
            let x0 = _mm512_loadu_ps(x.as_ptr().add(x_base));
            let x1 = _mm512_loadu_ps(x.as_ptr().add(x_base + 16));

            acc0 = _mm512_fmadd_ps(w_first, x0, acc0);
            acc1 = _mm512_fmadd_ps(w_second, x1, acc1);
        }

        // 处理尾部
        let mut tail_sum = 0.0f32;
        let tail_start = n_full_blocks * BLOCK_SIZE;
        for i in tail_start..n_packed_per_row {
            let packed_byte = self.weight_packed[row_packed_start + i];
            let lo_val = Self::decode_fp4(packed_byte & 0xF);
            let hi_val = Self::decode_fp4((packed_byte >> 4) & 0xF);
            let col_lo = i * 2;
            let col_hi = i * 2 + 1;
            let scale_lo = e8m0_to_f32(self.scale[row_scale_start + col_lo / 32]);
            let scale_hi = e8m0_to_f32(self.scale[row_scale_start + col_hi / 32]);
            tail_sum += lo_val * scale_lo * x[col_lo];
            tail_sum += hi_val * scale_hi * x[col_hi];
        }

        // 合并 4 路累加器
        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc0 = _mm512_add_ps(acc0, acc2);

        let mut tmp = [0.0f32; 16];
        _mm512_storeu_ps(tmp.as_mut_ptr(), acc0);
        tmp.iter().sum::<f32>() + tail_sum
    }
}

/// FP4 CPU expert FFN（gate/up/down 配对点积 + SwiGLU）
///
/// 优化：gate 和 up 共享 x 的拆分计算。
///
/// FP4 packed 格式中每字节 [hi,lo]，解包后 lo=[l0,...,l15], hi=[h0,...,h15]。
/// lo 对应偶数列 x[0,2,4,...,30]，hi 对应奇数列 x[1,3,5,...,31]。
///
/// 拆分 x 为 x_even=[x0,x2,...,x14,x16,...,x30] 和 x_odd=[x1,x3,...,x15,x17,...,x31]，
/// 则点积 = sum(lo_f32 * x_even) + sum(hi_f32 * x_odd)。
///
/// x_even/x_odd 只需拆分一次，gate 和 up 共享，省去重复拆分开销。
pub fn fp4_expert_ffn_pair(
    x: &[f32],
    gate_weight: &Fp4Weight,
    up_weight: &Fp4Weight,
    down_weight: &Fp4Weight,
    route_weight: f32,
) -> Vec<f32> {
    use rayon::prelude::*;

    let dim = gate_weight.shape.1;
    let inter_dim = gate_weight.shape.0;

    // 预拆分 x 为 x_even/x_odd，gate 和 up 共享
    // x_even = [x0,x2,x4,...,x30, x32,x34,...,x62, ...]
    // x_odd  = [x1,x3,x5,...,x31, x33,x35,...,x63, ...]
    let x_split = if is_avx512_supported() {
        Some(Fp4XSplit::new(x))
    } else {
        None
    };

    let mut gate = vec![0.0f32; inter_dim];
    let mut up = vec![0.0f32; inter_dim];

    gate.par_iter_mut()
        .zip(up.par_iter_mut())
        .enumerate()
        .for_each(|(row, (g, u))| {
            if let Some(ref xs) = x_split {
                *g = unsafe { gate_weight.fp4_vec_dot_f32_split(row, xs) };
                *u = unsafe { up_weight.fp4_vec_dot_f32_split(row, xs) };
            } else {
                *g = gate_weight.fp4_vec_dot_f32(row, x);
                *u = up_weight.fp4_vec_dot_f32(row, x);
            }
        });

    // SwiGLU
    let mut mid = vec![0.0f32; inter_dim];
    mid.par_iter_mut().enumerate().for_each(|(i, m)| {
        let g = gate[i];
        let u = up[i];
        let sigmoid_g = 1.0 / (1.0 + exp_approx_scalar(-g));
        *m = g * sigmoid_g * u * route_weight;
    });

    // Down projection
    down_weight.matvec(&mid)
}

/// 预拆分的 x 向量，供 FP4 AVX-512 内核共享使用
///
/// 将 x 拆分为偶数位和奇数位：
///   x_even = [x0,x2,...,x14, x16,x18,...,x30, ...]  (每 32 元素取 16 个偶数位)
///   x_odd  = [x1,x3,...,x15, x17,x19,...,x31, ...]  (每 32 元素取 16 个奇数位)
///
/// 这样 lo 权重直接与 x_even 对齐，hi 权重直接与 x_odd 对齐，
/// 省去 permutex2var_ps 权重合并操作。
struct Fp4XSplit {
    x_even: Vec<f32>,
    x_odd: Vec<f32>,
    n_blocks: usize,
}

impl Fp4XSplit {
    fn new(x: &[f32]) -> Self {
        let n = x.len();
        let n_blocks = n / 32;
        let mut x_even = vec![0.0f32; n_blocks * 16];
        let mut x_odd = vec![0.0f32; n_blocks * 16];

        for blk in 0..n_blocks {
            let src_base = blk * 32;
            let dst_base = blk * 16;
            for i in 0..16 {
                x_even[dst_base + i] = x[src_base + i * 2];
                x_odd[dst_base + i] = x[src_base + i * 2 + 1];
            }
        }

        Self { x_even, x_odd, n_blocks }
    }
}

impl Fp4Weight {
    /// FP4 × F32 向量点积（使用预拆分 x，省去权重合并）
    ///
    /// 点积 = sum(lo_f32 * x_even) + sum(hi_f32 * x_odd)
    /// 只需 2 次 permutexvar_ps/block（LUT 查表），省去 2 次 permutex2var_ps（权重合并）
    ///
    /// # Safety
    /// 调用方必须确保 AVX-512 特性检测已通过
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512bw")]
    #[target_feature(enable = "avx512vl")]
    unsafe fn fp4_vec_dot_f32_split(&self, row: usize, xs: &Fp4XSplit) -> f32 {
        use std::arch::x86_64::*;

        let lut = _mm512_set_ps(
            -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
             6.0,  4.0,  3.0,  2.0,  1.5,  1.0,  0.5, 0.0,
        );

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        let n_packed_per_row = self.shape.1 / 2;
        let n_scales_per_row = self.shape.1 / 32;
        let row_packed_start = row * n_packed_per_row;
        let row_scale_start = row * n_scales_per_row;

        const BLOCK_SIZE: usize = 16;
        let n_full_blocks = n_packed_per_row / BLOCK_SIZE;

        let mut block = 0;

        while block + 1 < n_full_blocks {
            // ---- Block 0 ----
            let base0 = row_packed_start + block * BLOCK_SIZE;
            let scale_bits_0 = self.scale[row_scale_start + block];
            let scale0 = if scale_bits_0 == 0 {
                _mm512_set1_ps(0.0f32)
            } else {
                _mm512_set1_ps(e8m0_to_f32(scale_bits_0))
            };

            if block + 2 < n_full_blocks {
                let next_base = row_packed_start + (block + 2) * BLOCK_SIZE;
                _mm_prefetch(self.weight_packed.as_ptr().add(next_base) as *const i8, _MM_HINT_T1);
            }

            let packed0 = _mm_loadu_si128(self.weight_packed.as_ptr().add(base0) as *const __m128i);
            let lo0 = _mm_and_si128(packed0, _mm_set1_epi8(0xF));
            let hi0 = _mm_srli_epi16(_mm_and_si128(packed0, _mm_set1_epi8(0xF0u8 as i8)), 4);

            let lo_f32_0 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(lo0), lut), scale0);
            let hi_f32_0 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(hi0), lut), scale0);

            let x_even_0 = _mm512_loadu_ps(xs.x_even.as_ptr().add(block * 16));
            let x_odd_0 = _mm512_loadu_ps(xs.x_odd.as_ptr().add(block * 16));

            acc0 = _mm512_fmadd_ps(lo_f32_0, x_even_0, acc0);
            acc1 = _mm512_fmadd_ps(hi_f32_0, x_odd_0, acc1);

            // ---- Block 1 ----
            let base1 = row_packed_start + (block + 1) * BLOCK_SIZE;
            let scale_bits_1 = self.scale[row_scale_start + block + 1];
            let scale1 = if scale_bits_1 == 0 {
                _mm512_set1_ps(0.0f32)
            } else {
                _mm512_set1_ps(e8m0_to_f32(scale_bits_1))
            };

            let packed1 = _mm_loadu_si128(self.weight_packed.as_ptr().add(base1) as *const __m128i);
            let lo1 = _mm_and_si128(packed1, _mm_set1_epi8(0xF));
            let hi1 = _mm_srli_epi16(_mm_and_si128(packed1, _mm_set1_epi8(0xF0u8 as i8)), 4);

            let lo_f32_1 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(lo1), lut), scale1);
            let hi_f32_1 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(hi1), lut), scale1);

            let x_even_1 = _mm512_loadu_ps(xs.x_even.as_ptr().add((block + 1) * 16));
            let x_odd_1 = _mm512_loadu_ps(xs.x_odd.as_ptr().add((block + 1) * 16));

            acc2 = _mm512_fmadd_ps(lo_f32_1, x_even_1, acc2);
            acc3 = _mm512_fmadd_ps(hi_f32_1, x_odd_1, acc3);

            block += 2;
        }

        if block < n_full_blocks {
            let base = row_packed_start + block * BLOCK_SIZE;
            let scale_bits = self.scale[row_scale_start + block];
            let scale_val = if scale_bits == 0 {
                _mm512_set1_ps(0.0f32)
            } else {
                _mm512_set1_ps(e8m0_to_f32(scale_bits))
            };

            let packed = _mm_loadu_si128(self.weight_packed.as_ptr().add(base) as *const __m128i);
            let lo = _mm_and_si128(packed, _mm_set1_epi8(0xF));
            let hi = _mm_srli_epi16(_mm_and_si128(packed, _mm_set1_epi8(0xF0u8 as i8)), 4);

            let lo_f32 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(lo), lut), scale_val);
            let hi_f32 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(hi), lut), scale_val);

            let x_even_v = _mm512_loadu_ps(xs.x_even.as_ptr().add(block * 16));
            let x_odd_v = _mm512_loadu_ps(xs.x_odd.as_ptr().add(block * 16));

            acc0 = _mm512_fmadd_ps(lo_f32, x_even_v, acc0);
            acc1 = _mm512_fmadd_ps(hi_f32, x_odd_v, acc1);
        }

        // 尾部
        let mut tail_sum = 0.0f32;
        let tail_start = n_full_blocks * BLOCK_SIZE;
        for i in tail_start..n_packed_per_row {
            let packed_byte = self.weight_packed[row_packed_start + i];
            let lo_val = Self::decode_fp4(packed_byte & 0xF);
            let hi_val = Self::decode_fp4((packed_byte >> 4) & 0xF);
            let col_lo = i * 2;
            let col_hi = i * 2 + 1;
            let scale_lo = e8m0_to_f32(self.scale[row_scale_start + col_lo / 32]);
            let scale_hi = e8m0_to_f32(self.scale[row_scale_start + col_hi / 32]);
            tail_sum += lo_val * scale_lo * xs.x_even[col_lo / 2];
            tail_sum += hi_val * scale_hi * xs.x_odd[col_hi / 2];
        }

        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc0 = _mm512_add_ps(acc0, acc2);

        let mut tmp = [0.0f32; 16];
        _mm512_storeu_ps(tmp.as_mut_ptr(), acc0);
        tmp.iter().sum::<f32>() + tail_sum
    }
}

/// IQ2_XS matrix-vector product with AVX-512 VNNI / AVX2 / scalar dispatch.
pub fn iq2xs_matvec(
    weight: &Iq2XsWeight,
    x: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> Vec<f32> {
    let _n_cols = n_cols;
    assert!(n_cols % 256 == 0, "n_cols must be a multiple of 256, got {}", n_cols);
    let n_blocks_per_row = n_cols / 256;

    #[cfg(target_arch = "x86_64")]
    {
        if is_avx512_supported() {
            return iq2xs_matvec_simd::<true>(weight, x, n_rows, n_cols, n_blocks_per_row);
        }
        if is_avx2_supported() {
            return iq2xs_matvec_simd::<false>(weight, x, n_rows, n_cols, n_blocks_per_row);
        }
    }

    iq2xs_matvec_scalar(weight, x, n_rows, n_cols, n_blocks_per_row)
}

/// SIMD accelerated matrix-vector product (AVX-512 VNNI or AVX2).
///
/// 关键优化：x 只量化一次（所有行共享同一个输入向量），
/// 然后传预量化的 q8/q8_inv_scales 给每行的点积内核。
#[cfg(target_arch = "x86_64")]
fn iq2xs_matvec_simd<const USE_AVX512: bool>(
    weight: &Iq2XsWeight,
    x: &[f32],
    n_rows: usize,
    _n_cols: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    use rayon::prelude::*;

    // 预量化 x：所有行共享同一个输入向量，只量化一次
    let n_blocks = n_blocks_per_row;
    let mut q8_buf = vec![0i8; n_blocks * 256];
    let mut q8_inv_scales = vec![0.0f32; n_blocks];
    for blk in 0..n_blocks {
        q8_inv_scales[blk] = quantize_f32_to_q8_block(
            &x[blk * 256..(blk + 1) * 256],
            &mut q8_buf[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    let mut output = vec![0.0f32; n_rows];

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        // L3 行级预取：预取下一行权重到 L2
        if row + 1 < n_rows {
            let next_row_offset = (row + 1) * n_blocks_per_row;
            unsafe {
                let next_row_qs_ptr = weight.qs.as_ptr().add(next_row_offset * 32);
                let next_row_scales_ptr = weight.scales.as_ptr().add(next_row_offset * 8);
                // 预取前几个 block 的数据（覆盖 L2 缓存行）
                for b in 0..4.min(n_blocks_per_row) {
                    std::arch::x86_64::_mm_prefetch(next_row_qs_ptr.add(b * 32) as *const i8, std::arch::x86_64::_MM_HINT_T2);
                    std::arch::x86_64::_mm_prefetch(next_row_scales_ptr.add(b * 8) as *const i8, std::arch::x86_64::_MM_HINT_T2);
                }
            }
        }

        let row_offset = row * n_blocks_per_row;
        let d_row = &weight.d[row_offset..row_offset + n_blocks_per_row];
        let qs_row = &weight.qs[row_offset * 32..(row_offset + n_blocks_per_row) * 32];
        let scales_row = &weight.scales[row_offset * 8..(row_offset + n_blocks_per_row) * 8];

        let result = unsafe {
            if USE_AVX512 {
                iq2xs_vec_dot_q8_avx512(d_row, qs_row, scales_row, &q8_buf, &q8_inv_scales, n_blocks_per_row)
            } else {
                iq2xs_vec_dot_q8_avx2(d_row, qs_row, scales_row, &q8_buf, &q8_inv_scales, n_blocks_per_row)
            }
        };
        *out = result;
    });

    output
}

/// Scalar fallback matrix-vector product (rayon parallel).
fn iq2xs_matvec_scalar(
    weight: &Iq2XsWeight,
    x: &[f32],
    n_rows: usize,
    _n_cols: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    use rayon::prelude::*;

    let grid = get_iq2xs_grid();
    let sign_table = get_sign_mul_table();
    let scale_table = get_scale_decode_table();

    let mut output = vec![0.0f32; n_rows];

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_offset = row * n_blocks_per_row;
        let mut row_sum = 0.0f32;

        for blk in 0..n_blocks_per_row {
            let bi = row_offset + blk;
            let d_val = weight.d[bi];
            let mut block_sum = 0.0f32;

            for g in 0..32 {
                let q = weight.qs[bi * 32 + g];
                let gi = (q & 511) as usize;
                let si = ((q >> 9) & 127) as usize;

                let ib32 = g >> 2;
                let within = g & 3;
                let sc_val = weight.scales[bi * 8 + ib32];
                let ls = if within < 2 {
                    scale_table[sc_val as usize * 2]
                } else {
                    scale_table[sc_val as usize * 2 + 1]
                };

                let x_base = blk * 256 + g * 8;
                let mut group_dot = 0.0f32;

                for j in 0..8 {
                    let gv = grid[gi * 8 + j] as f32;
                    let sm = sign_table[si * 8 + j];
                    group_dot += gv * sm * x[x_base + j];
                }

                block_sum += ls * group_dot;
            }

            row_sum += d_val * 0.125 * block_sum;
        }

        *out = row_sum;
    });

    output
}

/// CPU expert FFN with gate/up paired dot product optimization.
///
/// Inspired by ds4.c's `matvec_iq2_xxs_expert_pair_prequant`:
/// Both gate and up weights share the same activation vector x,
/// so we quantize x to int8 once and compute both dot products.
pub fn cpu_expert_ffn_pair(
    x: &[f32],
    gate_weight: &Iq2XsWeight,
    up_weight: &Iq2XsWeight,
    down_weight: &Iq2XsWeight,
    route_weight: f32,
) -> Vec<f32> {
    use rayon::prelude::*;

    let dim = gate_weight.shape.1;
    let inter_dim = gate_weight.shape.0;
    assert!(dim % 256 == 0, "dim must be a multiple of 256, got {}", dim);
    let n_blocks_per_row = dim / 256;

    // Compute gate and up projections
    let (gate, up) = compute_gate_up(x, gate_weight, up_weight, inter_dim, dim, n_blocks_per_row);

    // SwiGLU activation（使用快速 exp 近似，避免标准库 exp 开销）
    let mut mid = vec![0.0f32; inter_dim];
    mid.par_iter_mut().enumerate().for_each(|(i, m)| {
        let g = gate[i];
        let u = up[i];
        let sigmoid_g = 1.0 / (1.0 + exp_approx_scalar(-g));
        *m = g * sigmoid_g * u * route_weight;
    });

    // Down projection
    iq2xs_matvec(down_weight, &mid, dim, inter_dim)
}

/// Compute gate and up projections with paired dot product optimization.
///
/// 关键优化：x 只量化一次（所有行共享），然后传预量化的 q8/q8_inv_scales
/// 给每行的 gate/up 点积内核。
fn compute_gate_up(
    x: &[f32],
    gate_weight: &Iq2XsWeight,
    up_weight: &Iq2XsWeight,
    inter_dim: usize,
    dim: usize,
    n_blocks_per_row: usize,
) -> (Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;

    #[cfg(target_arch = "x86_64")]
    {
        if is_avx512_supported() || is_avx2_supported() {
            // 预量化 x：所有行共享同一个输入向量，只量化一次
            let mut q8_buf = vec![0i8; n_blocks_per_row * 256];
            let mut q8_inv_scales = vec![0.0f32; n_blocks_per_row];
            for blk in 0..n_blocks_per_row {
                q8_inv_scales[blk] = quantize_f32_to_q8_block(
                    &x[blk * 256..(blk + 1) * 256],
                    &mut q8_buf[blk * 256..(blk + 1) * 256],
                    256,
                );
            }

            let mut gate = vec![0.0f32; inter_dim];
            let mut up = vec![0.0f32; inter_dim];

            let results: Vec<(usize, f32, f32)> = (0..inter_dim)
                .into_par_iter()
                .map(|row| {
                    let row_offset = row * n_blocks_per_row;
                    let d_gate = &gate_weight.d[row_offset..row_offset + n_blocks_per_row];
                    let qs_gate = &gate_weight.qs[row_offset * 32..(row_offset + n_blocks_per_row) * 32];
                    let scales_gate = &gate_weight.scales[row_offset * 8..(row_offset + n_blocks_per_row) * 8];

                    let d_up = &up_weight.d[row_offset..row_offset + n_blocks_per_row];
                    let qs_up = &up_weight.qs[row_offset * 32..(row_offset + n_blocks_per_row) * 32];
                    let scales_up = &up_weight.scales[row_offset * 8..(row_offset + n_blocks_per_row) * 8];

                    let (g, u) = unsafe {
                        if is_avx512_supported() {
                            let g = iq2xs_vec_dot_q8_avx512(d_gate, qs_gate, scales_gate, &q8_buf, &q8_inv_scales, n_blocks_per_row);
                            let u = iq2xs_vec_dot_q8_avx512(d_up, qs_up, scales_up, &q8_buf, &q8_inv_scales, n_blocks_per_row);
                            (g, u)
                        } else {
                            let g = iq2xs_vec_dot_q8_avx2(d_gate, qs_gate, scales_gate, &q8_buf, &q8_inv_scales, n_blocks_per_row);
                            let u = iq2xs_vec_dot_q8_avx2(d_up, qs_up, scales_up, &q8_buf, &q8_inv_scales, n_blocks_per_row);
                            (g, u)
                        }
                    };
                    (row, g, u)
                })
                .collect();

            for (row, g, u) in results {
                gate[row] = g;
                up[row] = u;
            }

            return (gate, up);
        }
    }

    // Scalar fallback: separate gate and up computations
    let gate = iq2xs_matvec_scalar(gate_weight, x, inter_dim, dim, n_blocks_per_row);
    let up = iq2xs_matvec_scalar(up_weight, x, inter_dim, dim, n_blocks_per_row);
    (gate, up)
}

/// CPU expert FFN with fused gate_up weight (gate and up interleaved).
pub fn cpu_expert_ffn(
    x: &[f32],
    gate_up_weight: &Iq2XsWeight,
    down_weight: &Iq2XsWeight,
    route_weight: f32,
    swiglu_limit: f32,
) -> Vec<f32> {
    use rayon::prelude::*;

    let gate_up_shape = gate_up_weight.shape;
    let n_rows_gu = gate_up_shape.0;
    let n_cols_gu = gate_up_shape.1;
    let _n_blocks_per_row = n_cols_gu / 256;

    let gate_up_out = iq2xs_matvec(gate_up_weight, x, n_rows_gu, n_cols_gu);

    let inter_dim = if n_rows_gu % 2 == 0 { n_rows_gu / 2 } else { n_rows_gu };
    let gate = &gate_up_out[..inter_dim];
    let up = &gate_up_out[inter_dim..];

    let mut mid = vec![0.0f32; inter_dim];
    mid.par_iter_mut().enumerate().for_each(|(i, m)| {
        let mut g = gate[i];
        let mut u = up[i];

        if swiglu_limit > 0.0 {
            u = u.clamp(-swiglu_limit, swiglu_limit);
            g = g.min(swiglu_limit);
        }

        let sigmoid_g = 1.0 / (1.0 + exp_approx_scalar(-g));
        *m = g * sigmoid_g * u * route_weight;
    });

    let down_shape = down_weight.shape;
    let n_rows_d = down_shape.0;
    let n_cols_d = down_shape.1;

    iq2xs_matvec(down_weight, &mid, n_rows_d, n_cols_d)
}

/// 双专家 FFN（MoE top-2 场景）
///
/// 顺序执行两个专家，加权聚合。
///
/// 为什么不并行：
/// cpu_expert_ffn_pair 内部用 rayon par_iter 做行级并行，
/// 直接 rayon::join 会导致嵌套并行线程争抢，实测 128ms > 顺序 60ms。
/// 真正的并行需要手动行分区（expert_a: rows[0..N/2], expert_b: rows[N/2..N]），
/// 避免两层 rayon 调度冲突。
pub fn cpu_expert_ffn_pair_dual(
    x: &[f32],
    gate_a: &Iq2XsWeight, up_a: &Iq2XsWeight, down_a: &Iq2XsWeight,
    gate_b: &Iq2XsWeight, up_b: &Iq2XsWeight, down_b: &Iq2XsWeight,
    route_weight_a: f32,
    route_weight_b: f32,
) -> Vec<f32> {
    // 顺序计算两个专家（避免 rayon 嵌套并行争抢）
    let result_a = cpu_expert_ffn_pair(x, gate_a, up_a, down_a, route_weight_a);
    let result_b = cpu_expert_ffn_pair(x, gate_b, up_b, down_b, route_weight_b);

    // 加权聚合
    result_a.iter().zip(result_b.iter())
        .map(|(&a, &b)| a + b)
        .collect()
}

/// CPU 专家运行器（L3 常驻专家池 + SLRU 缓存淘汰）
///
/// 借鉴 docs/cpu-ffn-kimi.md 和 AGENTS.md P1 优化方向：
/// - SLRU 缓存淘汰：90% protected + 10% probation
/// - LFU 频率持久化到磁盘，支持热启动
/// - Step 级别专家保护：当前 step 已访问专家不被淘汰
/// - 层保护：当前层专家不被淘汰
pub struct CpuExpertRunner {
    /// 专家权重存储
    weights: std::collections::HashMap<(usize, usize), Arc<(Iq2XsWeight, Iq2XsWeight, Iq2XsWeight)>>,
    /// SLRU protected 段（热点专家）
    protected: std::collections::HashSet<(usize, usize)>,
    /// SLRU probation 段（新准入专家）
    probation: std::collections::HashSet<(usize, usize)>,
    /// 访问频率计数（LFU）
    freq: std::collections::HashMap<(usize, usize), u64>,
    /// 最近访问时间（LRU 辅助）
    last_access: std::collections::HashMap<(usize, usize), u64>,
    /// 访问计数器
    access_counter: u64,
    /// protected 段容量上限（专家数量）
    protected_capacity: usize,
    /// probation 段容量上限（专家数量）
    probation_capacity: usize,
    /// protected 段内存预算（字节），0 表示不限制
    protected_bytes_budget: usize,
    /// probation 段内存预算（字节），0 表示不限制
    probation_bytes_budget: usize,
    /// 当前 protected 段已用内存（字节）
    protected_bytes_used: usize,
    /// 当前 probation 段已用内存（字节）
    probation_bytes_used: usize,
    /// 当前 step 保护的专家集合
    step_protected: std::collections::HashSet<(usize, usize)>,
    /// 当前保护的层
    protected_layer: Option<usize>,
    /// 缓存命中统计
    hits: u64,
    misses: u64,
}

impl CpuExpertRunner {
    pub fn new() -> Self {
        // 默认容量：足够大，全量常驻场景不淘汰
        // 自适应逻辑：add_expert 时检查内存压力
        //   - 总专家内存 < 可用内存 → 全量常驻，不淘汰
        //   - 总专家内存 > 可用内存 → 启用 SLRU 淘汰
        Self::with_capacity(4096, 512)
    }

    /// 创建指定容量的 SLRU 缓存
    pub fn with_capacity(protected_capacity: usize, probation_capacity: usize) -> Self {
        Self {
            weights: std::collections::HashMap::new(),
            protected: std::collections::HashSet::new(),
            probation: std::collections::HashSet::new(),
            freq: std::collections::HashMap::new(),
            last_access: std::collections::HashMap::new(),
            access_counter: 0,
            protected_capacity,
            probation_capacity,
            protected_bytes_budget: 0, // 0 = 不限制，仅按数量控制
            probation_bytes_budget: 0,
            protected_bytes_used: 0,
            probation_bytes_used: 0,
            step_protected: std::collections::HashSet::new(),
            protected_layer: None,
            hits: 0,
            misses: 0,
        }
    }

    /// 设置字节预算（兼容不同大小专家）
    ///
    /// 自适应策略：
    ///   - 设为 0：不限制，全量常驻（80GB 内存 / 76GB 专家）
    ///   - 设为具体值：启用 SLRU 淘汰（80GB 内存 / 150GB 专家）
    ///
    /// 建议值：可用内存的 80%（留 20% 给系统和其他进程）
    pub fn set_bytes_budget(&mut self, protected_bytes: usize, probation_bytes: usize) {
        self.protected_bytes_budget = protected_bytes;
        self.probation_bytes_budget = probation_bytes;
    }

    /// 自动设置字节预算：根据系统可用内存和专家总量自适应
    ///
    /// total_expert_bytes: 所有专家的总字节数
    /// available_bytes: 系统可用内存字节数
    ///
    /// 如果 total < available * 0.8 → 全量常驻，不淘汰
    /// 如果 total > available * 0.8 → 启用 SLRU 淘汰
    pub fn auto_set_budget(&mut self, total_expert_bytes: usize, available_bytes: usize) {
        let budget = (available_bytes as f64 * 0.8) as usize;
        if total_expert_bytes <= budget {
            // 全量常驻：预算设为 0（不限制）
            self.protected_bytes_budget = 0;
            self.probation_bytes_budget = 0;
            // 容量设为足够大
            self.protected_capacity = 16384;
            self.probation_capacity = 2048;
        } else {
            // 需要淘汰：按 90%/10% 分配
            self.protected_bytes_budget = (budget as f64 * 0.9) as usize;
            self.probation_bytes_budget = (budget as f64 * 0.1) as usize;
        }
    }

    /// 检查 protected 段是否已满（数量 + 字节双重检查）
    fn protected_is_full(&self, new_size: usize) -> bool {
        if self.protected.len() >= self.protected_capacity {
            return true;
        }
        if self.protected_bytes_budget > 0 && self.protected_bytes_used + new_size > self.protected_bytes_budget {
            return true;
        }
        false
    }

    /// 检查 probation 段是否已满（数量 + 字节双重检查）
    fn probation_is_full(&self, new_size: usize) -> bool {
        if self.probation.len() >= self.probation_capacity {
            return true;
        }
        if self.probation_bytes_budget > 0 && self.probation_bytes_used + new_size > self.probation_bytes_budget {
            return true;
        }
        false
    }

    /// 添加专家到缓存（probation 段准入）
    pub fn add_expert(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        gate_weight: Iq2XsWeight,
        up_weight: Iq2XsWeight,
        down_weight: Iq2XsWeight,
    ) {
        let key = (layer_id, expert_id);
        let weights = Arc::new((gate_weight, up_weight, down_weight));
        let size = Self::expert_size_bytes(&weights);

        // 如果已存在，更新权重
        if self.weights.contains_key(&key) {
            let old_size = self.weights.get(&key).map(|w| Self::expert_size_bytes(w)).unwrap_or(0);
            self.weights.insert(key, weights);
            // 更新字节追踪
            if self.protected.contains(&key) {
                self.protected_bytes_used = self.protected_bytes_used - old_size + size;
            } else if self.probation.contains(&key) {
                self.probation_bytes_used = self.probation_bytes_used - old_size + size;
            }
            return;
        }

        // probation 段满时淘汰
        if self.probation_is_full(size) {
            self.evict_probation();
        }

        self.weights.insert(key, weights);
        self.probation.insert(key);
        self.probation_bytes_used += size;
        // 不增加 freq：频率只在 record_access() 中由 Gate 选中时增加
        self.access_counter += 1;
        *self.last_access.entry(key).or_insert(0) = self.access_counter;
    }

    /// 添加专家到缓存 protected 段（GPU 热专家同步专用）
    pub fn add_expert_protected(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        gate_weight: Iq2XsWeight,
        up_weight: Iq2XsWeight,
        down_weight: Iq2XsWeight,
    ) {
        let key = (layer_id, expert_id);
        let weights = Arc::new((gate_weight, up_weight, down_weight));
        let size = Self::expert_size_bytes(&weights);

        // 如果已存在，更新权重并确保在 protected 段
        if self.weights.contains_key(&key) {
            let old_size = self.weights.get(&key).map(|w| Self::expert_size_bytes(w)).unwrap_or(0);
            self.weights.insert(key, weights);
            // 如果在 probation，晋升到 protected
            if self.probation.contains(&key) {
                self.probation.remove(&key);
                self.probation_bytes_used -= old_size;
                self.protected.insert(key);
                self.protected_bytes_used += size;
            } else if self.protected.contains(&key) {
                self.protected_bytes_used = self.protected_bytes_used - old_size + size;
            }
            return;
        }

        // protected 段满时降级最冷专家
        if self.protected_is_full(size) {
            self.demote_protected();
        }

        self.weights.insert(key, weights);
        self._insert_protected(key);
        self.protected_bytes_used += size;
        // 不增加 freq：频率只在 record_access() 中由 Gate 选中时增加
        self.access_counter += 1;
        *self.last_access.entry(key).or_insert(0) = self.access_counter;
    }

    /// 内部方法：插入到 protected 段（容量检查由调用方负责）
    fn _insert_protected(&mut self, key: (usize, usize)) {
        self.protected.insert(key);
    }

    /// 计算专家 FFN（命中时自动晋升 probation → protected）
    pub fn compute_expert(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        x: &[f32],
        route_weight: f32,
    ) -> Option<Vec<f32>> {
        let key = (layer_id, expert_id);

        if self.weights.contains_key(&key) {
            self.hits += 1;
            self.touch_key(key);

            let weights = Arc::clone(self.weights.get(&key)?);
            Some(cpu_expert_ffn_pair(x, &weights.0, &weights.1, &weights.2, route_weight))
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn compute_expert_from_slice(
        &mut self,
        x: &[f32],
        route_weight: f32,
    ) -> Option<Vec<f32>> {
        let weights = Arc::clone(self.weights.values().next()?);
        Some(cpu_expert_ffn_pair(x, &weights.0, &weights.1, &weights.2, route_weight))
    }

    /// 双专家 FFN（MoE top-2 场景，使用常驻权重）
    pub fn compute_dual_expert(
        &mut self,
        layer_id: usize,
        expert_a: usize,
        expert_b: usize,
        x: &[f32],
        route_weight_a: f32,
        route_weight_b: f32,
    ) -> Option<Vec<f32>> {
        // 先检查两个专家是否都存在
        if !self.weights.contains_key(&(layer_id, expert_a)) || !self.weights.contains_key(&(layer_id, expert_b)) {
            self.misses += 1;
            return None;
        }

        // 克隆 Arc 引用（只增加引用计数，不做深拷贝）
        let wa = Arc::clone(self.weights.get(&(layer_id, expert_a)).unwrap());
        let wb = Arc::clone(self.weights.get(&(layer_id, expert_b)).unwrap());

        // 更新访问统计并晋升
        for &key in &[(layer_id, expert_a), (layer_id, expert_b)] {
            self.touch_key(key);
        }

        Some(cpu_expert_ffn_pair_dual(
            x,
            &wa.0, &wa.1, &wa.2,
            &wb.0, &wb.1, &wb.2,
            route_weight_a,
            route_weight_b,
        ))
    }

    /// 检查专家是否已加载
    pub fn has_expert(&self, layer_id: usize, expert_id: usize) -> bool {
        self.weights.contains_key(&(layer_id, expert_id))
    }

    /// 记录专家访问（不加载权重，只更新频率和访问时间）
    ///
    /// 用于统一频率统计：无论专家走 GPU 还是 CPU，Gate 选中就计数。
    /// GPU 命中的专家不经过 add_expert/compute_expert，
    /// 但仍需记录频率以驱动预取排序。
    pub fn record_access(&mut self, layer_id: usize, expert_id: usize) {
        let key = (layer_id, expert_id);
        self.access_counter += 1;
        *self.freq.entry(key).or_insert(0) += 1;
        *self.last_access.entry(key).or_insert(0) = self.access_counter;
    }

    /// 移除专家
    pub fn remove_expert(&mut self, layer_id: usize, expert_id: usize) -> bool {
        let key = (layer_id, expert_id);
        self.protected.remove(&key);
        self.probation.remove(&key);
        self.freq.remove(&key);
        self.last_access.remove(&key);
        self.weights.remove(&key).is_some()
    }

    /// 已加载专家数量
    pub fn expert_count(&self) -> usize {
        self.weights.len()
    }

    /// 返回 protected 段的专家 key 列表
    pub fn protected_keys(&self) -> Vec<(usize, usize)> {
        self.protected.iter().copied().collect()
    }

    /// 返回 probation 段的专家 key 列表
    pub fn probation_keys(&self) -> Vec<(usize, usize)> {
        self.probation.iter().copied().collect()
    }

    /// 缓存命中率
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { return 1.0; }
        self.hits as f64 / total as f64
    }

    /// 设置当前 step 保护的专家
    pub fn set_step_protected(&mut self, keys: std::collections::HashSet<(usize, usize)>) {
        self.step_protected = keys;
    }

    /// 设置当前保护的层
    pub fn set_protected_layer(&mut self, layer: Option<usize>) {
        self.protected_layer = layer;
    }

    /// 清空当前 step 保护的专家集合
    pub fn clear_step_protected(&mut self) {
        self.step_protected.clear();
    }

    /// L3 权重预热：精准预取指定层 CPU 负责的专家权重到 L3 缓存
    ///
    /// 分层预加载策略：
    ///   - GPU 负责每层 Top-N 热专家 → GPU GEMM
    ///   - CPU 负责每层 TopN+1 ~ TopN+M → CPU FFN（L3 warm）
    ///   - 其余专家 → CPU FFN（cold RAM）
    ///
    /// 此方法只预热 CPU 负责的范围（TopN+1 ~ TopN+M），
    /// 跳过 GPU 已有的专家，避免浪费 L3 空间。
    ///
    /// L3 = 32MB，每专家 gate 权重 ~1.6MB
    /// M=4 专家 × 3 权重 × 1.6MB = 19.2MB（60% L3，可行）
    ///
    /// Args:
    ///   layer_id: 要预热的层
    ///   gpu_keys: GPU SLRU 中已有的专家 key 集合（跳过这些）
    ///   gpu_topn: GPU 负责的 Top-N 数量
    ///   warmup_m: CPU L3 预热的专家数量（TopN+1 ~ TopN+M）
    pub fn warmup_layer_targeted(
        &self,
        layer_id: usize,
        gpu_keys: &std::collections::HashSet<(usize, usize)>,
        gpu_topn: usize,
        warmup_m: usize,
    ) {
        // 收集该层所有已缓存的专家，按频率排序
        let mut layer_entries: Vec<_> = self.weights.iter()
            .filter(|((lid, _), _)| *lid == layer_id)
            .filter_map(|(key, weights)| {
                let freq = self.freq.get(key).copied().unwrap_or(0);
                let size = Self::expert_size_bytes(weights);
                Some((*key, Arc::clone(weights), freq, size))
            })
            .collect();

        // 按频率降序排序（最热的优先）
        layer_entries.sort_by(|a, b| b.2.cmp(&a.2));

        // 筛选 CPU 负责的范围：跳过 GPU 已有的 + Top-N 内的
        // 只预热 TopN+1 ~ TopN+M 范围内且不在 GPU 中的专家
        let mut rank = 0;
        let mut budget_used = 0usize;
        const L3_BUDGET_BYTES: usize = 20 * 1024 * 1024; // L3 的 ~60%
        let mut experts_to_warm = Vec::new();

        for (key, weights, _freq, size) in &layer_entries {
            // 跳过 GPU 已有的专家
            if gpu_keys.contains(key) {
                continue;
            }

            rank += 1;

            // 跳过 Top-N 内的（这些应该走 GPU DMA，不预热到 CPU L3）
            if rank <= gpu_topn {
                continue;
            }

            // 只预热 TopN+1 ~ TopN+M 范围
            if rank > gpu_topn + warmup_m {
                break;
            }

            // 预算检查：gate 权重约占总大小的 1/3
            let gate_size = size / 3;
            if budget_used + gate_size > L3_BUDGET_BYTES {
                break;
            }

            experts_to_warm.push(Arc::clone(weights));
            budget_used += gate_size;
        }

        if experts_to_warm.is_empty() {
            return;
        }

        // 用 rayon scope 预热，不干扰主计算线程
        rayon::scope(|s| {
            for weights in experts_to_warm {
                s.spawn(move |_| {
                    let gate = &weights.0;
                    // 顺序读取 gate 的 d 和 qs，触发硬件预取器
                    let mut sum = 0.0f32;
                    for &val in gate.d.iter() {
                        sum += val;
                    }
                    for &val in gate.qs.iter() {
                        sum += val as f32;
                    }
                    // 防止编译器优化掉读取
                    if sum == f32::INFINITY {
                        eprintln!("[warmup] unexpected");
                    }
                });
            }
        });
    }

    /// L3 权重预热（兼容旧接口，预热所有已缓存专家）
    pub fn warmup_layer(&self, layer_id: usize) {
        const L3_BUDGET_BYTES: usize = 16 * 1024 * 1024; // L3 的 50%

        // 收集该层所有已缓存的专家，按频率排序
        let mut layer_entries: Vec<_> = self.weights.iter()
            .filter(|((lid, _), _)| *lid == layer_id)
            .filter_map(|(key, weights)| {
                let freq = self.freq.get(key).copied().unwrap_or(0);
                let size = Self::expert_size_bytes(weights);
                Some((key, Arc::clone(weights), freq, size))
            })
            .collect();

        // 按频率降序排序（最热的优先预热）
        layer_entries.sort_by(|a, b| b.2.cmp(&a.2));

        // 预算控制：只预热 L3 预算内能容纳的专家
        let mut budget_used = 0usize;
        let mut experts_to_warm = Vec::new();

        for (key, weights, _freq, size) in &layer_entries {
            if budget_used + size / 3 > L3_BUDGET_BYTES {
                break;
            }
            experts_to_warm.push(Arc::clone(weights));
            budget_used += size / 3;
        }

        if experts_to_warm.is_empty() {
            return;
        }

        rayon::scope(|s| {
            for weights in experts_to_warm {
                s.spawn(move |_| {
                    let gate = &weights.0;
                    let mut sum = 0.0f32;
                    for &val in gate.d.iter() {
                        sum += val;
                    }
                    for &val in gate.qs.iter() {
                        sum += val as f32;
                    }
                    if sum == f32::INFINITY {
                        eprintln!("[warmup] unexpected");
                    }
                });
            }
        });
    }

    /// 返回指定层的专家按频率降序排名列表
    ///
    /// 用于分层预加载策略：
    ///   - Top-N → GPU 预取
    ///   - TopN+1 ~ TopN+M → CPU L3 预热
    ///   - 其余 → CPU RAM 兜底
    pub fn layer_freq_rank(&self, layer_id: usize) -> Vec<(usize, u64)> {
        let mut entries: Vec<_> = self.freq.iter()
            .filter(|((lid, _), _)| *lid == layer_id)
            .map(|((_, eid), &f)| (*eid, f))
            .collect();

        // 按频率降序排序
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries
    }

    /// 计算单个专家的总权重大小（字节）
    /// 兼容不同量化格式：IQ2_XS / Q2_K / 等
    fn expert_size_bytes(weights: &Arc<(Iq2XsWeight, Iq2XsWeight, Iq2XsWeight)>) -> usize {
        let (gate, up, down) = weights.as_ref();
        let gate_size = gate.d.len() * 4 + gate.qs.len() * 2 + gate.scales.len();
        let up_size = up.d.len() * 4 + up.qs.len() * 2 + up.scales.len();
        let down_size = down.d.len() * 4 + down.qs.len() * 2 + down.scales.len();
        gate_size + up_size + down_size
    }

    /// 估算已用内存（字节）
    pub fn memory_usage(&self) -> usize {
        self.weights.values().map(|w| {
            let (g, u, d) = w.as_ref();
            g.d.len() * 4 + g.qs.len() * 2 + g.scales.len()
            + u.d.len() * 4 + u.qs.len() * 2 + u.scales.len()
            + d.d.len() * 4 + d.qs.len() * 2 + d.scales.len()
        }).sum()
    }

    /// LFU 频率持久化到磁盘（JSON 格式，原子写入）
    pub fn save_freq(&self, path: &str) -> std::io::Result<()> {
        let data: std::collections::HashMap<String, u64> = self.freq.iter()
            .map(|(&(l, e), &f)| (format!("{}_{}", l, e), f))
            .collect();
        let json = serde_json::to_string(&data)?;
        let tmp_path = format!("{}.tmp", path);
        std::fs::write(&tmp_path, &json)?;
        std::fs::rename(&tmp_path, path)
    }

    /// 从磁盘加载 LFU 频率（热启动）
    pub fn load_freq(&mut self, path: &str) -> std::io::Result<()> {
        let json = std::fs::read_to_string(path)?;
        let data: std::collections::HashMap<String, u64> = serde_json::from_str(&json)?;
        for (key, f) in data {
            if let Some((l, e)) = key.split_once('_') {
                if let (Ok(layer), Ok(expert)) = (l.parse::<usize>(), e.parse::<usize>()) {
                    self.freq.insert((layer, expert), f);
                }
            }
        }
        Ok(())
    }

    // ===== SLRU 内部方法 =====

    /// SLRU 晋升：probation → protected
    /// protected 段满时先降级最冷专家，降级失败则留在 probation
    fn promote_key(&mut self, key: (usize, usize)) {
        if !self.probation.contains(&key) {
            return;
        }
        let size = self.weights.get(&key).map(|w| Self::expert_size_bytes(w)).unwrap_or(0);
        if self.protected_is_full(size) {
            if self.demote_protected() {
                self.probation.remove(&key);
                self.probation_bytes_used -= size;
                self.protected.insert(key);
                self.protected_bytes_used += size;
            }
            // 降级失败则留在 probation
        } else {
            self.probation.remove(&key);
            self.probation_bytes_used -= size;
            self.protected.insert(key);
            self.protected_bytes_used += size;
        }
    }

    /// 更新访问时间并晋升（不增加频率，频率由 record_access 管理）
    fn touch_key(&mut self, key: (usize, usize)) {
        self.access_counter += 1;
        *self.last_access.entry(key).or_insert(0) = self.access_counter;
        self.promote_key(key);
    }

    /// protected 段降级：最冷专家降到 probation
    /// 返回 true 表示找到候选并降级，false 表示无可降级候选
    fn demote_protected(&mut self) -> bool {
        let mut min_freq = u64::MAX;
        let mut oldest_access = u64::MAX;
        let mut demote_key = None;

        for &key in &self.protected {
            // 跳过 step 保护和层保护的专家
            if self.step_protected.contains(&key) { continue; }
            if let Some(layer) = self.protected_layer {
                if key.0 == layer { continue; }
            }

            let f = self.freq.get(&key).copied().unwrap_or(0);
            let la = self.last_access.get(&key).copied().unwrap_or(0);
            if f < min_freq || (f == min_freq && la < oldest_access) {
                min_freq = f;
                oldest_access = la;
                demote_key = Some(key);
            }
        }

        if let Some(key) = demote_key {
            let size = self.weights.get(&key).map(|w| Self::expert_size_bytes(w)).unwrap_or(0);
            self.protected.remove(&key);
            self.protected_bytes_used -= size;
            // probation 段满时先淘汰
            if self.probation_is_full(size) {
                self.evict_probation();
            }
            self.probation.insert(key);
            self.probation_bytes_used += size;
            true
        } else {
            false
        }
    }

    /// probation 段淘汰：最冷专家完全移除
    /// 返回 true 表示找到候选并淘汰，false 表示无可淘汰候选
    fn evict_probation(&mut self) -> bool {
        let mut min_freq = u64::MAX;
        let mut oldest_access = u64::MAX;
        let mut evict_key = None;

        for &key in &self.probation {
            // 跳过 step 保护和层保护的专家
            if self.step_protected.contains(&key) { continue; }
            if let Some(layer) = self.protected_layer {
                if key.0 == layer { continue; }
            }

            let f = self.freq.get(&key).copied().unwrap_or(0);
            let la = self.last_access.get(&key).copied().unwrap_or(0);
            if f < min_freq || (f == min_freq && la < oldest_access) {
                min_freq = f;
                oldest_access = la;
                evict_key = Some(key);
            }
        }

        if let Some(key) = evict_key {
            let size = self.weights.get(&key).map(|w| Self::expert_size_bytes(w)).unwrap_or(0);
            self.probation.remove(&key);
            self.probation_bytes_used -= size;
            self.weights.remove(&key);
            self.freq.remove(&key);
            self.last_access.remove(&key);
            true
        } else {
            false
        }
    }
}

impl Default for CpuExpertRunner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tile 布局矩阵向量乘法
// ============================================================================

/// IQ2_XS Tile 布局矩阵向量乘法
///
/// 从 Tile 布局读取数据，解包后调用现有 SIMD/标量内核。
/// Tile 布局的核心价值是减少 cache miss，不需要改变计算内核。
pub fn iq2xs_matvec_tile(weight: &Iq2XsTile, x: &[f32]) -> Vec<f32> {
    let n_rows = weight.shape.0;
    let _n_cols = weight.shape.1;
    let n_blocks_per_row = weight.blocks_per_row;

    #[cfg(target_arch = "x86_64")]
    {
        if is_avx512_supported() {
            return iq2xs_matvec_tile_simd::<true>(weight, x, n_rows, n_blocks_per_row);
        }
        if is_avx2_supported() {
            return iq2xs_matvec_tile_simd::<false>(weight, x, n_rows, n_blocks_per_row);
        }
    }

    iq2xs_matvec_tile_scalar(weight, x, n_rows, n_blocks_per_row)
}

/// SIMD 加速的 Tile 布局矩阵向量乘法
///
/// AVX-512 路径：直接从 Tile 布局读取 d/qs/scales，避免中间缓冲区。
/// AVX2 回退：仍需解包（AVX2 Tile 直读内核待实现）。
///
/// 关键优化：x 只量化一次（所有行共享），然后传预量化的 q8/q8_inv_scales。
#[cfg(target_arch = "x86_64")]
fn iq2xs_matvec_tile_simd<const USE_AVX512: bool>(
    weight: &Iq2XsTile,
    x: &[f32],
    n_rows: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    use rayon::prelude::*;

    // 预量化 x：所有行共享同一个输入向量，只量化一次
    let mut q8_buf = vec![0i8; n_blocks_per_row * 256];
    let mut q8_inv_scales = vec![0.0f32; n_blocks_per_row];
    for blk in 0..n_blocks_per_row {
        q8_inv_scales[blk] = quantize_f32_to_q8_block(
            &x[blk * 256..(blk + 1) * 256],
            &mut q8_buf[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    let mut output = vec![0.0f32; n_rows];

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_tile_offset = row * n_blocks_per_row * 80; // TILE_SIZE
        let row_tile_data = &weight.data[row_tile_offset..row_tile_offset + n_blocks_per_row * 80];

        let result = unsafe {
            if USE_AVX512 {
                iq2xs_vec_dot_q8_tile_avx512(row_tile_data, n_blocks_per_row, &q8_buf, &q8_inv_scales)
            } else {
                // AVX2 回退：仍需解包（AVX2 Tile 直读内核待实现）
                let mut d_row = vec![0.0f32; n_blocks_per_row];
                let mut qs_row = vec![0u16; n_blocks_per_row * 32];
                let mut scales_row = vec![0u8; n_blocks_per_row * 8];
                for blk in 0..n_blocks_per_row {
                    d_row[blk] = weight.d_at(row * n_blocks_per_row + blk);
                    qs_row[blk * 32..(blk + 1) * 32].copy_from_slice(&weight.qs_at(row * n_blocks_per_row + blk));
                    scales_row[blk * 8..blk * 8 + 8].copy_from_slice(weight.scales_at(row * n_blocks_per_row + blk));
                }
                iq2xs_vec_dot_q8_avx2(&d_row, &qs_row, &scales_row, &q8_buf, &q8_inv_scales, n_blocks_per_row)
            }
        };
        *out = result;
    });

    output
}

/// 标量回退的 Tile 布局矩阵向量乘法
fn iq2xs_matvec_tile_scalar(
    weight: &Iq2XsTile,
    x: &[f32],
    n_rows: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    use rayon::prelude::*;

    let grid = get_iq2xs_grid();
    let sign_table = get_sign_mul_table();
    let scale_table = get_scale_decode_table();

    let mut output = vec![0.0f32; n_rows];

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_start = row * n_blocks_per_row;
        let mut row_sum = 0.0f32;

        for blk in 0..n_blocks_per_row {
            let bi = row_start + blk;
            let d_val = weight.d_at(bi);
            let qs_slice = weight.qs_at(bi);
            let scales_slice = weight.scales_at(bi);
            let mut block_sum = 0.0f32;

            for g in 0..32 {
                let q = qs_slice[g];
                let gi = (q & 511) as usize;
                let si = ((q >> 9) & 127) as usize;

                let ib32 = g >> 2;
                let within = g & 3;
                let sc_val = scales_slice[ib32];
                let ls = if within < 2 {
                    scale_table[sc_val as usize * 2]
                } else {
                    scale_table[sc_val as usize * 2 + 1]
                };

                let x_base = blk * 256 + g * 8;
                let mut group_dot = 0.0f32;

                for j in 0..8 {
                    let gv = grid[gi * 8 + j] as f32;
                    let sm = sign_table[si * 8 + j];
                    group_dot += gv * sm * x[x_base + j];
                }

                block_sum += ls * group_dot;
            }

            row_sum += d_val * 0.125 * block_sum;
        }

        *out = row_sum;
    });

    output
}

/// CPU expert FFN with Tile 布局权重（gate/up paired）
///
/// 与 `cpu_expert_ffn_pair` 功能相同，但使用 Tile 布局权重，
/// 减少 cache miss 以提升性能。
pub fn cpu_expert_ffn_pair_tile(
    x: &[f32],
    gate_weight: &Iq2XsTile,
    up_weight: &Iq2XsTile,
    down_weight: &Iq2XsTile,
    route_weight: f32,
) -> Vec<f32> {
    use rayon::prelude::*;

    let _dim = gate_weight.shape.1;
    let inter_dim = gate_weight.shape.0;
    let n_blocks_per_row = gate_weight.blocks_per_row;

    // Compute gate and up projections from Tile layout
    let (gate, up) = compute_gate_up_tile(x, gate_weight, up_weight, inter_dim, n_blocks_per_row);

    // SwiGLU activation（使用快速 exp 近似）
    let mut mid = vec![0.0f32; inter_dim];
    mid.par_iter_mut().enumerate().for_each(|(i, m)| {
        let g = gate[i];
        let u = up[i];
        let sigmoid_g = 1.0 / (1.0 + exp_approx_scalar(-g));
        *m = g * sigmoid_g * u * route_weight;
    });

    // Down projection
    iq2xs_matvec_tile(down_weight, &mid)
}

/// Tile 布局的 gate/up 配对计算
fn compute_gate_up_tile(
    x: &[f32],
    gate_weight: &Iq2XsTile,
    up_weight: &Iq2XsTile,
    inter_dim: usize,
    n_blocks_per_row: usize,
) -> (Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;

    #[cfg(target_arch = "x86_64")]
    {
        if is_avx512_supported() || is_avx2_supported() {
            let mut gate = vec![0.0f32; inter_dim];
            let mut up = vec![0.0f32; inter_dim];

            let results: Vec<(usize, f32, f32)> = (0..inter_dim)
                .into_par_iter()
                .map(|row| {
                    let row_start = row * n_blocks_per_row;

                    // 从 Tile 布局解包 gate 行
                    let mut d_gate = vec![0.0f32; n_blocks_per_row];
                    let mut qs_gate = vec![0u16; n_blocks_per_row * 32];
                    let mut scales_gate = vec![0u8; n_blocks_per_row * 8];

                    for blk in 0..n_blocks_per_row {
                        let bi = row_start + blk;
                        d_gate[blk] = gate_weight.d_at(bi);
                        qs_gate[blk * 32..(blk + 1) * 32].copy_from_slice(gate_weight.qs_at(bi).as_slice());
                        scales_gate[blk * 8..(blk + 1) * 8].copy_from_slice(gate_weight.scales_at(bi));
                    }

                    // 从 Tile 布局解包 up 行
                    let mut d_up = vec![0.0f32; n_blocks_per_row];
                    let mut qs_up = vec![0u16; n_blocks_per_row * 32];
                    let mut scales_up = vec![0u8; n_blocks_per_row * 8];

                    for blk in 0..n_blocks_per_row {
                        let bi = row_start + blk;
                        d_up[blk] = up_weight.d_at(bi);
                        qs_up[blk * 32..(blk + 1) * 32].copy_from_slice(up_weight.qs_at(bi).as_slice());
                        scales_up[blk * 8..(blk + 1) * 8].copy_from_slice(up_weight.scales_at(bi));
                    }

                    let (g, u) = unsafe {
                        if is_avx512_supported() {
                            iq2xs_pair_dot_avx512_vnni(
                                &d_gate, &qs_gate, &scales_gate,
                                &d_up, &qs_up, &scales_up,
                                x, n_blocks_per_row,
                            )
                        } else {
                            iq2xs_pair_dot_avx2(
                                &d_gate, &qs_gate, &scales_gate,
                                &d_up, &qs_up, &scales_up,
                                x, n_blocks_per_row,
                            )
                        }
                    };
                    (row, g, u)
                })
                .collect();

            for (row, g, u) in results {
                gate[row] = g;
                up[row] = u;
            }

            return (gate, up);
        }
    }

    // 标量回退
    let gate = iq2xs_matvec_tile_scalar(gate_weight, x, inter_dim, n_blocks_per_row);
    let up = iq2xs_matvec_tile_scalar(up_weight, x, inter_dim, n_blocks_per_row);
    (gate, up)
}

/// 混合量化 FFN：IQ2_XXS gate/up + Q2_K down
///
/// gate/up 使用 IQ2_XXS（2.0625 bpw，无 sub-block scales），
/// down 使用 Q2_K（2.5625 bpw，2-bit 量化 + sub-block scales）。
pub fn mixed_ffn_pair_iq2xxs_q2k(
    x: &[f32],
    gate_weight: &Iq2XxsWeight,
    up_weight: &Iq2XxsWeight,
    down_weight: &Q2KWeight,
    route_weight: f32,
    swiglu_limit: f32,
) -> Vec<f32> {
    use rayon::prelude::*;

    let dim = gate_weight.shape.1;
    let inter_dim = gate_weight.shape.0;
    assert!(dim % 256 == 0, "dim must be a multiple of 256, got {}", dim);
    let n_blocks = dim / 256;

    // Q8 预量化 x（gate 和 up 共享，只量化一次）
    let mut q8_buf = vec![0i8; n_blocks * 256];
    let mut q8_inv_scales = vec![0.0f32; n_blocks];
    for blk in 0..n_blocks {
        q8_inv_scales[blk] = quantize_f32_to_q8_block(
            &x[blk * 256..(blk + 1) * 256],
            &mut q8_buf[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    // gate + up 投影
    // 策略：单线程顺序扫描权重 + 软件预取，最大化 DDR burst 效率
    // rayon 并行会打乱顺序访问，12 线程同时从 RAM 读取不同位置，
    // 导致 DDR 页面切换开销和缓存行冲刷，带宽利用率仅 16.7%
    // 单线程顺序访问虽然计算慢，但内存带宽利用率更高
    let gate_nrows = gate_weight.n_rows();
    let up_nrows = up_weight.n_rows();

    // 先尝试 rayon 并行（默认），如果性能不佳可切换到单线程
    let gate: Vec<f32> = (0..gate_nrows).into_par_iter().map(|row| {
        unsafe { gate_weight.vec_dot_q8(row, &q8_buf, &q8_inv_scales) }
    }).collect();
    let up: Vec<f32> = (0..up_nrows).into_par_iter().map(|row| {
        unsafe { up_weight.vec_dot_q8(row, &q8_buf, &q8_inv_scales) }
    }).collect();

    // SwiGLU（带 swiglu_limit 裁剪）
    let mut mid = vec![0.0f32; inter_dim];
    mid.par_iter_mut().enumerate().for_each(|(i, m)| {
        let g = gate[i];
        let mut u = up[i];
        if swiglu_limit > 0.0 {
            u = u.clamp(-swiglu_limit, swiglu_limit);
        }
        let g_clamped = if swiglu_limit > 0.0 { g.clamp(-50.0, swiglu_limit) } else { g };
        let sigmoid_g = 1.0 / (1.0 + exp_approx_scalar(-g_clamped));
        *m = g * sigmoid_g * u * route_weight;
    });

    // down 投影（Q2_K）
    QuantizedWeight::matvec(down_weight, &mid)
}

/// 混合量化 FFN（单线程顺序扫描 + 软件预取优化版）
///
/// 核心优化：单线程顺序扫描权重，最大化 DDR burst 效率
///   - Q8 数据常驻 L1（7KB）
///   - 权重按行顺序读取，硬件预取器高效工作
///   - 软件预取下一行权重到 L2
///
/// 适用于内存带宽受限场景（DDR5-5600 双通道 44.8 GB/s）
pub fn mixed_ffn_pair_iq2xxs_q2k_streaming(
    x: &[f32],
    gate_weight: &Iq2XxsWeight,
    up_weight: &Iq2XxsWeight,
    down_weight: &Q2KWeight,
    route_weight: f32,
    swiglu_limit: f32,
) -> Vec<f32> {
    let dim = gate_weight.shape.1;
    let inter_dim = gate_weight.shape.0;
    assert!(dim % 256 == 0);
    let n_blocks = dim / 256;

    // Q8 预量化 x
    let mut q8_buf = vec![0i8; n_blocks * 256];
    let mut q8_inv_scales = vec![0.0f32; n_blocks];
    for blk in 0..n_blocks {
        q8_inv_scales[blk] = quantize_f32_to_q8_block(
            &x[blk * 256..(blk + 1) * 256],
            &mut q8_buf[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    // gate 投影：单线程顺序扫描 + 软件预取
    let gate_nrows = gate_weight.n_rows();
    let mut gate = vec![0.0f32; gate_nrows];
    if super::avx512::is_avx512_supported() {
        unsafe {
            super::avx512::iq2xxs_matvec_blocked_amd7600(
                &gate_weight.d, &gate_weight.qs,
                &q8_buf, &q8_inv_scales,
                n_blocks, 0, gate_nrows, &mut gate,
            );
        }
    } else {
        for row in 0..gate_nrows {
            gate[row] = unsafe { gate_weight.vec_dot_q8(row, &q8_buf, &q8_inv_scales) };
        }
    }

    // up 投影：单线程顺序扫描
    let up_nrows = up_weight.n_rows();
    let mut up = vec![0.0f32; up_nrows];
    if super::avx512::is_avx512_supported() {
        unsafe {
            super::avx512::iq2xxs_matvec_blocked_amd7600(
                &up_weight.d, &up_weight.qs,
                &q8_buf, &q8_inv_scales,
                n_blocks, 0, up_nrows, &mut up,
            );
        }
    } else {
        for row in 0..up_nrows {
            up[row] = unsafe { up_weight.vec_dot_q8(row, &q8_buf, &q8_inv_scales) };
        }
    }

    // SwiGLU
    let mut mid = vec![0.0f32; inter_dim];
    for i in 0..inter_dim {
        let g = gate[i];
        let mut u = up[i];
        if swiglu_limit > 0.0 {
            u = u.clamp(-swiglu_limit, swiglu_limit);
        }
        let g_clamped = if swiglu_limit > 0.0 { g.clamp(-50.0, swiglu_limit) } else { g };
        let sigmoid_g = 1.0 / (1.0 + exp_approx_scalar(-g_clamped));
        mid[i] = g * sigmoid_g * u * route_weight;
    }

    // down 投影
    QuantizedWeight::matvec(down_weight, &mid)
}

/// 混合量化 FFN（L3 感知多线程分块版）
///
/// 核心优化：多线程分块顺序扫描，最大化 DDR5 burst 效率
///
/// 问题分析：
///   - rayon 12 线程随机访问 34MB 权重 → DDR 带宽利用率仅 16.7%
///   - 12 线程争抢 RAM，TLB miss + 缓存行冲刷 → 有效带宽 7.4 GB/s
///   - DDR5-5600 双通道理论带宽 44.8 GB/s，利用率不足 20%
///
/// 优化策略：
///   1. 分块：将 18432 行分成 N_CHUNKS 块（默认 3 块 × 6144 行）
///      每块 gate 权重 ~17.5MB < L3 32MB，块内顺序访问最大化 burst
///   2. 多线程：N_CHUNKS 个线程各处理一块，块间并行
///      3 线程 × 顺序访问 → DDR 页面命中率远高于 12 线程随机访问
///   3. On-the-fly dequantization：IQ2_XXS 块在寄存器内解压，不写中间缓冲
///   4. Q8 数据常驻 L1（7KB），权重按行顺序流式读取
///
/// 线程数选择：
///   - 2 线程：每块 17.5MB，2 块共 35MB > L3 32MB（稍超）
///   - 3 线程：每块 11.7MB，3 块共 35MB（L3 可容纳 2 块 + Q8）
///   - 6 线程：每块 5.8MB，6 块共 35MB（L3 轻松容纳，但线程多降低顺序性）
///   - 默认 3 线程：平衡并行度和缓存局部性
pub fn mixed_ffn_pair_iq2xxs_q2k_blocked_mt(
    x: &[f32],
    gate_weight: &Iq2XxsWeight,
    up_weight: &Iq2XxsWeight,
    down_weight: &Q2KWeight,
    route_weight: f32,
    swiglu_limit: f32,
    n_threads: usize,
) -> Vec<f32> {
    let dim = gate_weight.shape.1;
    let inter_dim = gate_weight.shape.0;
    assert!(dim % 256 == 0);
    let n_blocks = dim / 256;

    // Q8 预量化 x（所有线程共享，只量化一次）
    let mut q8_buf = vec![0i8; n_blocks * 256];
    let mut q8_inv_scales = vec![0.0f32; n_blocks];
    for blk in 0..n_blocks {
        q8_inv_scales[blk] = quantize_f32_to_q8_block(
            &x[blk * 256..(blk + 1) * 256],
            &mut q8_buf[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    let gate_nrows = gate_weight.n_rows();
    let up_nrows = up_weight.n_rows();
    let use_avx512 = super::avx512::is_avx512_supported();

    // 分块计算 gate 投影
    let chunk_size = (gate_nrows + n_threads - 1) / n_threads;
    let mut gate = vec![0.0f32; gate_nrows];
    let gate_ptr = gate.as_mut_ptr();

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let row_start = t * chunk_size;
            let row_end = ((t + 1) * chunk_size).min(gate_nrows);
            if row_start >= row_end { continue; }

            let len = row_end - row_start;
            let output = unsafe { std::slice::from_raw_parts_mut(gate_ptr.add(row_start), len) };
            let d = &gate_weight.d;
            let qs = &gate_weight.qs;
            let q8 = &q8_buf;
            let q8_inv = &q8_inv_scales;

            s.spawn(move || {
                if use_avx512 {
                    unsafe {
                        super::avx512::iq2xxs_matvec_blocked_amd7600(
                            d, qs, q8, q8_inv,
                            n_blocks, row_start, row_end, output,
                        );
                    }
                } else {
                    for row in row_start..row_end {
                        let row_offset = row * n_blocks;
                        let d_row = &d[row_offset..row_offset + n_blocks];
                        let qs_row = &qs[row_offset * 32..(row_offset + n_blocks) * 32];
                        output[row - row_start] = super::avx512::iq2xxs_vec_dot_q8(
                            d_row, qs_row, q8, q8_inv, n_blocks,
                        );
                    }
                }
            });
        }
    });

    // 分块计算 up 投影
    let chunk_size_up = (up_nrows + n_threads - 1) / n_threads;
    let mut up = vec![0.0f32; up_nrows];
    let up_ptr = up.as_mut_ptr();

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let row_start = t * chunk_size_up;
            let row_end = ((t + 1) * chunk_size_up).min(up_nrows);
            if row_start >= row_end { continue; }

            let len = row_end - row_start;
            let output = unsafe { std::slice::from_raw_parts_mut(up_ptr.add(row_start), len) };
            let d = &up_weight.d;
            let qs = &up_weight.qs;
            let q8 = &q8_buf;
            let q8_inv = &q8_inv_scales;

            s.spawn(move || {
                if use_avx512 {
                    unsafe {
                        super::avx512::iq2xxs_matvec_blocked_amd7600(
                            d, qs, q8, q8_inv,
                            n_blocks, row_start, row_end, output,
                        );
                    }
                } else {
                    for row in row_start..row_end {
                        let row_offset = row * n_blocks;
                        let d_row = &d[row_offset..row_offset + n_blocks];
                        let qs_row = &qs[row_offset * 32..(row_offset + n_blocks) * 32];
                        output[row - row_start] = super::avx512::iq2xxs_vec_dot_q8(
                            d_row, qs_row, q8, q8_inv, n_blocks,
                        );
                    }
                }
            });
        }
    });

    // SwiGLU（单线程，计算量小）
    let mut mid = vec![0.0f32; inter_dim];
    for i in 0..inter_dim {
        let g = gate[i];
        let mut u = up[i];
        if swiglu_limit > 0.0 {
            u = u.clamp(-swiglu_limit, swiglu_limit);
        }
        let g_clamped = if swiglu_limit > 0.0 { g.clamp(-50.0, swiglu_limit) } else { g };
        let sigmoid_g = 1.0 / (1.0 + exp_approx_scalar(-g_clamped));
        mid[i] = g * sigmoid_g * u * route_weight;
    }

    // down 投影（Q2_K，仍用 rayon 并行）
    QuantizedWeight::matvec(down_weight, &mid)
}

/// 6 专家并行 FFN（IQ2_XXS+Q2_K 混合量化）
///
/// 并行策略：rayon 专家级并行 + rayon 行级并行
///
/// 注意：DDR5 带宽是瓶颈（~49% 利用率），多线程争抢带宽反而降低性能。
/// 实测 2 路专家并行（std::thread::scope + 单线程扫描）比串行慢 3-14 倍。
/// 原因：单线程 down 投影太慢（7168 行串行），而 rayon 版本用 12 线程并行。
/// DDR5 双通道无法从多线程读取中获益（顺序访问已最大化 burst 效率）。
///
/// 关键优化：
///   1. Q8 预量化只做一次（6 专家共享）
///   2. gate+up 融合计算（Q8 数据在 L1 中共享）
pub fn mixed_ffn_6experts_iq2xxs_q2k(
    x: &[f32],
    gate_weights: &[&Iq2XxsWeight],
    up_weights: &[&Iq2XxsWeight],
    down_weights: &[&Q2KWeight],
    route_weights: &[f32],
    swiglu_limit: f32,
) -> Vec<f32> {
    use rayon::prelude::*;

    let n_experts = gate_weights.len();
    assert_eq!(up_weights.len(), n_experts);
    assert_eq!(down_weights.len(), n_experts);
    assert_eq!(route_weights.len(), n_experts);

    let dim = gate_weights[0].shape.1;
    let inter_dim = gate_weights[0].shape.0;
    assert!(dim % 256 == 0);
    let n_blocks = dim / 256;

    // Q8 预量化 x（所有专家共享，只量化一次）
    let mut q8_buf = vec![0i8; n_blocks * 256];
    let mut q8_inv_scales = vec![0.0f32; n_blocks];
    for blk in 0..n_blocks {
        q8_inv_scales[blk] = quantize_f32_to_q8_block(
            &x[blk * 256..(blk + 1) * 256],
            &mut q8_buf[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    // 6 专家并行计算（rayon 并行，每个专家内部也并行）
    let results: Vec<Vec<f32>> = (0..n_experts).into_par_iter().map(|ei| {
        let gate_w = gate_weights[ei];
        let up_w = up_weights[ei];
        let down_w = down_weights[ei];
        let rw = route_weights[ei];

        // gate + up 融合投影（行级并行）
        let mut gate = vec![0.0f32; inter_dim];
        let mut up = vec![0.0f32; inter_dim];
        gate.par_iter_mut()
            .zip(up.par_iter_mut())
            .enumerate()
            .for_each(|(row, (g, u))| {
                *g = unsafe { gate_w.vec_dot_q8(row, &q8_buf, &q8_inv_scales) };
                *u = unsafe { up_w.vec_dot_q8(row, &q8_buf, &q8_inv_scales) };
            });

        // SwiGLU
        let mut mid = vec![0.0f32; inter_dim];
        mid.par_iter_mut().enumerate().for_each(|(i, m)| {
            let g = gate[i];
            let mut u = up[i];
            if swiglu_limit > 0.0 {
                u = u.clamp(-swiglu_limit, swiglu_limit);
            }
            let g_clamped = if swiglu_limit > 0.0 { g.clamp(-50.0, swiglu_limit) } else { g };
            let sigmoid_g = 1.0 / (1.0 + exp_approx_scalar(-g_clamped));
            *m = g * sigmoid_g * u * rw;
        });

        // down 投影
        QuantizedWeight::matvec(down_w, &mid)
    }).collect();

    // 加权累加所有专家输出
    let dim_out = results[0].len();
    let mut output = vec![0.0f32; dim_out];
    for result in &results {
        for (o, r) in output.iter_mut().zip(result.iter()) {
            *o += r;
        }
    }
    output
}

/// 单专家 FFN 计算（IQ2_XXS gate/up + Q2_K down）
///
/// 单线程顺序扫描，最大化 DDR burst 效率。
/// gate/up 使用 blocked matvec（权重流式读取 + 软件预取），
/// down 使用 rayon 并行（Q2_K 权重较大，并行收益更高）。
fn compute_single_expert_iq2xxs_q2k(
    gate_w: &Iq2XxsWeight,
    up_w: &Iq2XxsWeight,
    down_w: &Q2KWeight,
    route_weight: f32,
    swiglu_limit: f32,
    inter_dim: usize,
    n_blocks: usize,
    use_avx512: bool,
    q8: &[i8],
    q8_inv: &[f32],
    output: &mut [f32],
) {
    // gate 投影：单线程顺序扫描
    let mut gate = vec![0.0f32; inter_dim];
    if use_avx512 {
        unsafe {
            super::avx512::iq2xxs_matvec_blocked_amd7600(
                &gate_w.d, &gate_w.qs, q8, q8_inv,
                n_blocks, 0, inter_dim, &mut gate,
            );
        }
    } else {
        for row in 0..inter_dim {
            gate[row] = unsafe { gate_w.vec_dot_q8(row, q8, q8_inv) };
        }
    }

    // up 投影：单线程顺序扫描
    let mut up = vec![0.0f32; inter_dim];
    if use_avx512 {
        unsafe {
            super::avx512::iq2xxs_matvec_blocked_amd7600(
                &up_w.d, &up_w.qs, q8, q8_inv,
                n_blocks, 0, inter_dim, &mut up,
            );
        }
    } else {
        for row in 0..inter_dim {
            up[row] = unsafe { up_w.vec_dot_q8(row, q8, q8_inv) };
        }
    }

    // SwiGLU（单线程，计算量小）
    let mut mid = vec![0.0f32; inter_dim];
    for i in 0..inter_dim {
        let g = gate[i];
        let mut u = up[i];
        if swiglu_limit > 0.0 {
            u = u.clamp(-swiglu_limit, swiglu_limit);
        }
        let g_clamped = if swiglu_limit > 0.0 { g.clamp(-50.0, swiglu_limit) } else { g };
        let sigmoid_g = 1.0 / (1.0 + exp_approx_scalar(-g_clamped));
        mid[i] = g * sigmoid_g * u * route_weight;
    }

    // down 投影（Q2_K，单线程顺序扫描避免 rayon 争抢）
    let dim_out = down_w.shape.0;
    let mut down_result = vec![0.0f32; dim_out];
    if use_avx512 {
        let n_blocks_down = inter_dim / 256;
        unsafe {
            super::avx512::q2k_matvec_blocked_amd7600(
                &down_w.d, &down_w.dmin, &down_w.scales, &down_w.qs,
                &mid, n_blocks_down, 0, dim_out, &mut down_result,
            );
        }
    } else {
        down_result = QuantizedWeight::matvec(down_w, &mid);
    }
    output.copy_from_slice(&down_result);
}

/// 混合量化 FFN（权重重排优化版）
///
/// 与 mixed_ffn_pair_iq2xxs_q2k 功能相同，但使用 block 优先权重布局。
/// 权重重排后，同一 block 位置的所有行数据连续存储，
/// rayon 并行计算不同行时，同一 block 的权重在 L3 中共享。
///
/// 权重布局对比：
///   行优先：[row0_blk0, row0_blk1, ..., row1_blk0, row1_blk1, ...]
///   block优先：[row0_blk0, row1_blk0, ..., row0_blk1, row1_blk1, ...]
pub fn mixed_ffn_pair_iq2xxs_q2k_tiled(
    x: &[f32],
    gate_weight: &Iq2XxsWeightTiled,
    up_weight: &Iq2XxsWeightTiled,
    down_weight: &Q2KWeightTiled,
    route_weight: f32,
    swiglu_limit: f32,
) -> Vec<f32> {
    use rayon::prelude::*;

    let dim = gate_weight.n_cols;
    let inter_dim = gate_weight.n_rows;
    assert!(dim % 256 == 0, "dim must be a multiple of 256, got {}", dim);
    let n_blocks = dim / 256;

    // Q8 预量化 x
    let mut q8_buf = vec![0i8; n_blocks * 256];
    let mut q8_inv_scales = vec![0.0f32; n_blocks];
    for blk in 0..n_blocks {
        q8_inv_scales[blk] = quantize_f32_to_q8_block(
            &x[blk * 256..(blk + 1) * 256],
            &mut q8_buf[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    // gate + up 投影（分块计算，每块权重适合 L3）
    let chunk_rows = 2048.min(inter_dim);
    let mut gate = vec![0.0f32; inter_dim];
    let mut up = vec![0.0f32; inter_dim];

    for chunk_start in (0..inter_dim).step_by(chunk_rows) {
        let chunk_end = (chunk_start + chunk_rows).min(inter_dim);

        gate[chunk_start..chunk_end].par_iter_mut()
            .zip(up[chunk_start..chunk_end].par_iter_mut())
            .enumerate()
            .for_each(|(i, (g, u))| {
                let row = chunk_start + i;
                *g = gate_weight.vec_dot_q8(row, &q8_buf, &q8_inv_scales);
                *u = up_weight.vec_dot_q8(row, &q8_buf, &q8_inv_scales);
            });
    }

    // SwiGLU
    let mut mid = vec![0.0f32; inter_dim];
    mid.par_iter_mut().enumerate().for_each(|(i, m)| {
        let g = gate[i];
        let mut u = up[i];
        if swiglu_limit > 0.0 {
            u = u.clamp(-swiglu_limit, swiglu_limit);
        }
        let g_clamped = if swiglu_limit > 0.0 { g.clamp(-50.0, swiglu_limit) } else { g };
        let sigmoid_g = 1.0 / (1.0 + exp_approx_scalar(-g_clamped));
        *m = g * sigmoid_g * u * route_weight;
    });

    // down 投影（Q2_K tiled）
    let n_blocks_inter = inter_dim / 256;
    let dim_out = down_weight.n_rows;

    // Q8 预量化 mid
    let mut q8_buf2 = vec![0i8; n_blocks_inter * 256];
    let mut q8_inv_scales2 = vec![0.0f32; n_blocks_inter];
    for blk in 0..n_blocks_inter {
        q8_inv_scales2[blk] = quantize_f32_to_q8_block(
            &mid[blk * 256..(blk + 1) * 256],
            &mut q8_buf2[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    let mut output = vec![0.0f32; dim_out];
    let down_chunk = 2048.min(dim_out);

    for chunk_start in (0..dim_out).step_by(down_chunk) {
        let chunk_end = (chunk_start + down_chunk).min(dim_out);
        output[chunk_start..chunk_end].par_iter_mut().enumerate().for_each(|(i, out)| {
            let row = chunk_start + i;
            *out = down_weight.vec_dot_q8(row, &q8_buf2, &q8_inv_scales2);
        });
    }

    output
}

// ============================================================================
// 权重重排：block 优先布局
// ============================================================================

/// IQ2_XXS 权重（block 优先布局）
///
/// 行优先：d[row * n_blocks + blk]，访问同一 blk 的不同行时跳跃大
/// block 优先：d[blk * n_rows + row]，同一 blk 的所有行连续存储
///
/// 当 rayon 并行计算不同行时，同一 block 位置的权重在 L3 中共享，
/// 减少缓存失效。
///
/// vec_dot_q8 不再 gather 数据到临时缓冲区，而是直接按 block 顺序计算：
/// 外层循环按 block 遍历，内层从 block 优先布局中连续读取 d[blk*n_rows+row]
/// 和 qs[blk*n_rows*32+row*32..+32]，然后累加到 sum。
#[derive(Clone)]
pub struct Iq2XxsWeightTiled {
    /// block 优先：d[blk * n_rows + row]
    pub d: Vec<f32>,
    /// block 优先：qs[blk * n_rows * 32 + row * 32 + g]
    pub qs: Vec<u16>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl Iq2XxsWeightTiled {
    /// 从行优先 Iq2XxsWeight 转换为 block 优先布局
    pub fn from_weight(weight: &Iq2XxsWeight) -> Self {
        let n_rows = weight.shape.0;
        let n_blocks_per_row = weight.shape.1 / 256;

        let mut d = vec![0.0f32; n_blocks_per_row * n_rows];
        let mut qs = vec![0u16; n_blocks_per_row * n_rows * 32];

        for blk in 0..n_blocks_per_row {
            for row in 0..n_rows {
                let src_offset = row * n_blocks_per_row + blk;
                let dst_d_offset = blk * n_rows + row;
                let dst_qs_offset = blk * n_rows * 32 + row * 32;

                d[dst_d_offset] = weight.d[src_offset];
                qs[dst_qs_offset..dst_qs_offset + 32]
                    .copy_from_slice(&weight.qs[src_offset * 32..src_offset * 32 + 32]);
            }
        }

        Self {
            d,
            qs,
            n_rows,
            n_cols: weight.shape.1,
        }
    }

    /// 单行点积（block 优先布局，按 block 顺序计算，避免 gather）
    ///
    /// 关键优化：外层按 block 遍历，每个 block 的 d 和 qs 在内存中连续，
    /// 硬件预取器可以高效预取下一个 block 的数据。
    pub fn vec_dot_q8(&self, row: usize, q8: &[i8], q8_inv_scales: &[f32]) -> f32 {
        let n_blocks_per_row = self.n_cols / 256;
        let grid = super::tables::get_iq2xxs_grid();
        let ksigns_bytes = &super::tables::KSIGNS_IQ2XS;

        let mut sumf = 0.0f32;

        for blk in 0..n_blocks_per_row {
            // block 优先布局：同一 block 的所有行连续存储
            let d_val = self.d[blk * self.n_rows + row] * q8_inv_scales[blk];
            let qs_base = blk * self.n_rows * 32 + row * 32;
            let q8_blk = &q8[blk * 256..(blk + 1) * 256];

            let mut bsum = 0i32;
            let mut q8_offset = 0usize;

            for ib32 in 0..8 {
                let q2_base = ib32 * 4;
                let q0 = self.qs[qs_base + q2_base] as u32;
                let q1 = self.qs[qs_base + q2_base + 1] as u32;
                let q2 = self.qs[qs_base + q2_base + 2] as u32;
                let q3 = self.qs[qs_base + q2_base + 3] as u32;
                let aux32_0 = q0 | (q1 << 16);
                let aux32_1 = q2 | (q3 << 16);

                let aux8 = [
                    (aux32_0 & 0xFF) as usize,
                    ((aux32_0 >> 8) & 0xFF) as usize,
                    ((aux32_0 >> 16) & 0xFF) as usize,
                    ((aux32_0 >> 24) & 0xFF) as usize,
                ];
                let ls = 2 * ((aux32_1 >> 28) as i32 & 0xF) + 1;

                let mut sumi = 0i32;
                for l in 0..4 {
                    let grid_offset = aux8[l] * 8;
                    let sign_idx = ((aux32_1 >> (7 * l)) & 127) as usize;
                    let sign_byte = ksigns_bytes[sign_idx];

                    for j in 0..8 {
                        let grid_val = grid[grid_offset + j] as i32;
                        let q8_val = q8_blk[q8_offset] as i32;
                        let sign_val = if sign_byte & (1 << j) != 0 { -1i32 } else { 1i32 };
                        sumi += grid_val * q8_val * sign_val;
                        q8_offset += 1;
                    }
                }
                bsum += ls * sumi;
            }
            sumf += d_val * bsum as f32;
        }

        0.125 * sumf
    }
}

/// Q2_K 权重（block 优先布局）
#[derive(Clone)]
pub struct Q2KWeightTiled {
    pub d: Vec<f32>,
    pub dmin: Vec<f32>,
    pub scales: Vec<u8>,
    pub qs: Vec<u8>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl Q2KWeightTiled {
    /// 从行优先 Q2KWeight 转换为 block 优先布局
    pub fn from_weight(weight: &Q2KWeight) -> Self {
        let n_rows = weight.shape.0;
        let n_blocks_per_row = weight.shape.1 / 256;

        let mut d = vec![0.0f32; n_blocks_per_row * n_rows];
        let mut dmin = vec![0.0f32; n_blocks_per_row * n_rows];
        let mut scales = vec![0u8; n_blocks_per_row * n_rows * 16];
        let mut qs = vec![0u8; n_blocks_per_row * n_rows * 64];

        for blk in 0..n_blocks_per_row {
            for row in 0..n_rows {
                let src_offset = row * n_blocks_per_row + blk;
                let dst_d_offset = blk * n_rows + row;
                let dst_scales_offset = blk * n_rows * 16 + row * 16;
                let dst_qs_offset = blk * n_rows * 64 + row * 64;

                d[dst_d_offset] = weight.d[src_offset];
                dmin[dst_d_offset] = weight.dmin[src_offset];
                scales[dst_scales_offset..dst_scales_offset + 16]
                    .copy_from_slice(&weight.scales[src_offset * 16..src_offset * 16 + 16]);
                qs[dst_qs_offset..dst_qs_offset + 64]
                    .copy_from_slice(&weight.qs[src_offset * 64..src_offset * 64 + 64]);
            }
        }

        Self {
            d,
            dmin,
            scales,
            qs,
            n_rows,
            n_cols: weight.shape.1,
        }
    }

    /// 单行点积（block 优先布局）
    pub fn vec_dot_q8(&self, row: usize, q8: &[i8], q8_inv_scales: &[f32]) -> f32 {
        let n_blocks_per_row = self.n_cols / 256;

        let mut d_row = vec![0.0f32; n_blocks_per_row];
        let mut dmin_row = vec![0.0f32; n_blocks_per_row];
        let mut scales_row = vec![0u8; n_blocks_per_row * 16];
        let mut qs_row = vec![0u8; n_blocks_per_row * 64];

        for blk in 0..n_blocks_per_row {
            d_row[blk] = self.d[blk * self.n_rows + row];
            dmin_row[blk] = self.dmin[blk * self.n_rows + row];
            let src_scales = blk * self.n_rows * 16 + row * 16;
            scales_row[blk * 16..blk * 16 + 16].copy_from_slice(&self.scales[src_scales..src_scales + 16]);
            let src_qs = blk * self.n_rows * 64 + row * 64;
            qs_row[blk * 64..blk * 64 + 64].copy_from_slice(&self.qs[src_qs..src_qs + 64]);
        }

        super::avx512::q2k_vec_dot_q8_avx2(&d_row, &dmin_row, &scales_row, &qs_row, q8, q8_inv_scales, n_blocks_per_row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matvec_basic() {
        let dim = 512;
        let n_rows = 4;
        let n_blocks = n_rows * dim / 256;

        let d = vec![0.01f32; n_blocks];
        let qs = vec![0u16; n_blocks * 32];
        let scales = vec![1u8; n_blocks * 8];

        let weight = Iq2XsWeight::new(d, qs, scales, (n_rows, dim));

        let x = vec![1.0f32; dim];
        let output = iq2xs_matvec(&weight, &x, n_rows, dim);

        assert_eq!(output.len(), n_rows);
        assert!(output.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_parallel_matvec() {
        let dim = 512;
        let n_rows = 4;
        let n_blocks = n_rows * dim / 256;

        let d = vec![0.01f32; n_blocks];
        let qs = vec![0u16; n_blocks * 32];
        let scales = vec![1u8; n_blocks * 8];

        let weight = Iq2XsWeight::new(d, qs, scales, (n_rows, dim));

        let x = vec![1.0f32; dim];
        let output = iq2xs_matvec(&weight, &x, n_rows, dim);

        assert_eq!(output.len(), n_rows);
        assert!(output.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_pair_dot_scalar() {
        let dim = 512;
        let n_blocks = dim / 256;

        let d = vec![0.01f32; n_blocks];
        let qs = vec![0u16; n_blocks * 32];
        let scales = vec![1u8; n_blocks * 8];

        let x = vec![1.0f32; dim];
        let (g, u) = iq2xs_pair_dot_scalar(&d, &qs, &scales, &d, &qs, &scales, &x, n_blocks);

        assert!(g.is_finite());
        assert!(u.is_finite());
        assert_eq!(g, u); // Same weights should give same result
    }

    #[test]
    fn test_iq2xxs_scalar_vs_avx512() {
        use crate::cpu_expert::avx512::{iq2xxs_vec_dot_q8_amd7600, is_avx512_supported, quantize_f32_to_q8_block};
        use crate::cpu_expert::tables;

        if !is_avx512_supported() {
            eprintln!("AVX-512 not supported, skipping test");
            return;
        }

        tables::get_iq2xxs_grid();
        tables::get_iq2xxs_grid_u64();
        tables::get_ksigns_iq2xs_bytes();

        let dim = 256;
        let n_blocks = 1;

        // 生成多种 qs 模式测试
        let test_cases: Vec<(&str, Vec<u16>)> = vec![
            // Case 1: 全零（grid=0, sign=0）
            ("all_zero", vec![0u16; 32]),
            // Case 2: 非零 grid index，无 sign
            // aux32_0 = 0x01010101 → grid indices [1,1,1,1]
            // aux32_1 = 0x00000000 → ls=1, signs=0
            ("grid1_nosign", {
                let mut qs = vec![0u16; 32];
                for ib32 in 0..8 {
                    qs[ib32 * 4 + 0] = 0x0101; // aux32_0 low
                    qs[ib32 * 4 + 1] = 0x0101; // aux32_0 high
                    // qs[ib32*4+2] = 0; // aux32_1 low: ls=1, signs=0
                    // qs[ib32*4+3] = 0; // aux32_1 high: ls=1, signs=0
                }
                qs
            }),
            // Case 3: 多种 grid index + sign
            // aux32_0 = 0xFF00FF00 → grid indices [0, 255, 0, 255]
            // aux32_1 = 0x40000040 → ls=2*(4)+1=9, sign[0]=(0x40>>0)&127=64, sign[3]=(0x40>>21)&127=0
            ("mixed_grid_sign", {
                let mut qs = vec![0u16; 32];
                for ib32 in 0..8 {
                    qs[ib32 * 4 + 0] = 0xFF00; // aux32_0 low: grid [0, 255]
                    qs[ib32 * 4 + 1] = 0xFF00; // aux32_0 high: grid [0, 255]
                    qs[ib32 * 4 + 2] = 0x0040; // aux32_1 low: sign[0]=64, sign[1]=0
                    qs[ib32 * 4 + 3] = 0x9000; // aux32_1 high: ls=2*(9)+1=19, sign[2]=(0x9000>>7)&127, sign[3]=(0x9000>>14)&127
                }
                qs
            }),
            // Case 4: 最大 ls + 最大 sign index
            // aux32_1 = 0xF7F7F7F7 → ls=2*(0xF)+1=31, sign=(0xF7>>0)&127=127, (0xF7>>7)&127=1
            ("max_ls_sign", {
                let mut qs = vec![0u16; 32];
                for ib32 in 0..8 {
                    qs[ib32 * 4 + 0] = 0x00FF; // grid [255, 0]
                    qs[ib32 * 4 + 1] = 0x00FF; // grid [255, 0]
                    qs[ib32 * 4 + 2] = 0xF7F7; // ls=31, sign[0]=127, sign[1]=1
                    qs[ib32 * 4 + 3] = 0xF7F7; // ls=31, sign[2]=127, sign[3]=1
                }
                qs
            }),
            // Case 5: 随机模式
            ("random", {
                vec![
                    0xA3F1, 0x2B7C, 0x5D90, 0x1E4A,
                    0xC8B2, 0x03D7, 0xF6E5, 0x8A19,
                    0x4C63, 0xD0F8, 0x7124, 0x9BAE,
                    0xE507, 0x36DC, 0x2F8B, 0x5A41,
                    0x1C93, 0xB7E0, 0x4D26, 0x8FC5,
                    0x6A38, 0xD14F, 0x0B72, 0xE396,
                    0xF45A, 0x278D, 0xC01E, 0x5B63,
                    0x9ED7, 0x3A4C, 0x8F21, 0xD6B0,
                ]
            }),
        ];

        let d = vec![0.5f32; n_blocks];

        // 随机输入
        let x: Vec<f32> = (0..dim).map(|i| ((i * 7919 + 1) % 1000) as f32 / 100.0 - 5.0).collect();

        let mut q8 = vec![0i8; dim];
        let mut q8_inv_scales = vec![0.0f32; n_blocks];
        for blk in 0..n_blocks {
            q8_inv_scales[blk] = quantize_f32_to_q8_block(
                &x[blk * 256..(blk + 1) * 256],
                &mut q8[blk * 256..(blk + 1) * 256],
                256,
            );
        }

        let mut all_pass = true;
        for (name, qs) in &test_cases {
            let scalar_result = iq2xxs_vec_dot_q8(&d, &qs, &q8, &q8_inv_scales, n_blocks);
            let avx512_result = unsafe {
                iq2xxs_vec_dot_q8_amd7600(&d, &qs, &q8, &q8_inv_scales, n_blocks)
            };
            let diff = (scalar_result - avx512_result).abs();
            let pass = diff < 0.5;
            eprintln!("  {:<20} scalar={:>12.4}  avx512={:>12.4}  diff={:>8.4}  {}",
                name, scalar_result, avx512_result, diff, if pass { "OK" } else { "FAIL" });
            if !pass { all_pass = false; }
        }

        // 多 block 测试
        let n_blocks_multi = 16;
        let d_multi = vec![0.3f32; n_blocks_multi];
        let qs_multi: Vec<u16> = (0..n_blocks_multi * 32).map(|i| {
            let block = i / 32;
            let idx = i % 32;
            match block % 4 {
                0 => 0u16,
                1 => 0xFF00u16,
                2 => ((idx as u16) * 251 + 17) & 0xFFFF,
                _ => 0xF7F7u16,
            }
        }).collect();
        let x_multi: Vec<f32> = (0..n_blocks_multi * 256).map(|i| ((i * 7919 + 1) % 2000) as f32 / 100.0 - 10.0).collect();
        let mut q8_multi = vec![0i8; n_blocks_multi * 256];
        let mut q8_inv_scales_multi = vec![0.0f32; n_blocks_multi];
        for blk in 0..n_blocks_multi {
            q8_inv_scales_multi[blk] = quantize_f32_to_q8_block(
                &x_multi[blk * 256..(blk + 1) * 256],
                &mut q8_multi[blk * 256..(blk + 1) * 256],
                256,
            );
        }

        let scalar_multi = iq2xxs_vec_dot_q8(&d_multi, &qs_multi, &q8_multi, &q8_inv_scales_multi, n_blocks_multi);
        let avx512_multi = unsafe {
            iq2xxs_vec_dot_q8_amd7600(&d_multi, &qs_multi, &q8_multi, &q8_inv_scales_multi, n_blocks_multi)
        };
        let diff_multi = (scalar_multi - avx512_multi).abs();
        let pass_multi = diff_multi < 1.0;
        eprintln!("  {:<20} scalar={:>12.4}  avx512={:>12.4}  diff={:>8.4}  {}",
            "multi_block_16", scalar_multi, avx512_multi, diff_multi, if pass_multi { "OK" } else { "FAIL" });
        if !pass_multi { all_pass = false; }

        assert!(all_pass, "IQ2_XXS AVX-512 kernel failed one or more test cases");
    }

    /// 逐 group 对比标量 vs AVX-512，精确定位差异来源
    #[test]
    fn test_iq2xxs_per_group_debug() {
        use crate::cpu_expert::avx512::{iq2xxs_vec_dot_q8_amd7600, is_avx512_supported, quantize_f32_to_q8_block};
        use crate::cpu_expert::tables;

        if !is_avx512_supported() {
            eprintln!("AVX-512 not supported, skipping test");
            return;
        }

        let grid = tables::get_iq2xxs_grid();
        let ksigns_bytes = &tables::KSIGNS_IQ2XS;

        // 构造一个有 sign 变化的单 block 测试
        let d = vec![1.0f32];
        let mut qs = vec![0u16; 32];

        // ib32=0: grid indices [1, 2, 3, 4], ls=9, sign indices [64, 0, 127, 1]
        // aux32_0 = 0x04030201 → grid [1, 2, 3, 4]
        // aux32_1 = sign[0]=64(7bit), sign[1]=0(7bit), sign[2]=127(7bit), sign[3]=1(7bit), ls=4(4bit)
        // aux32_1 = (4 << 28) | (1 << 21) | (127 << 14) | (0 << 7) | 64
        let aux32_0: u32 = 0x04030201;
        let aux32_1: u32 = (4u32 << 28) | (1u32 << 21) | (127u32 << 14) | (0u32 << 7) | 64;
        qs[0] = (aux32_0 & 0xFFFF) as u16;
        qs[1] = ((aux32_0 >> 16) & 0xFFFF) as u16;
        qs[2] = (aux32_1 & 0xFFFF) as u16;
        qs[3] = ((aux32_1 >> 16) & 0xFFFF) as u16;

        let x: Vec<f32> = (0..256).map(|i| ((i * 7919 + 1) % 1000) as f32 / 100.0 - 5.0).collect();
        let mut q8 = vec![0i8; 256];
        let mut q8_inv_scales = vec![1.0f32];
        q8_inv_scales[0] = quantize_f32_to_q8_block(&x, &mut q8, 256);

        // 标量逐 group 计算
        let ls = 2 * ((aux32_1 >> 28) & 0xF) as i32 + 1;
        eprintln!("ls = {}", ls);

        for l in 0..4 {
            let grid_idx = ((aux32_0 >> (8 * l)) & 0xFF) as usize;
            let sign_idx = ((aux32_1 >> (7 * l)) & 127) as usize;
            let sign_byte = ksigns_bytes[sign_idx];

            eprintln!("\n  group {}: grid_idx={}, sign_idx={}, sign_byte=0x{:02X}",
                l, grid_idx, sign_idx, sign_byte);

            let mut scalar_sum = 0i32;
            for j in 0..8 {
                let grid_val = grid[grid_idx * 8 + j] as i32;
                let q8_val = q8[l * 8 + j] as i32;
                let sign_val = if sign_byte & (1 << j) != 0 { -1i32 } else { 1i32 };
                let contrib = grid_val * q8_val * sign_val;
                scalar_sum += contrib;
                if j < 4 {
                    eprintln!("    j={}: grid={:>3} q8={:>4} sign={:+} → contrib={}",
                        j, grid_val, q8_val, sign_val, contrib);
                }
            }
            eprintln!("  group {} scalar sum = {}", l, scalar_sum);
        }

        // 完整标量结果
        let scalar_result = iq2xxs_vec_dot_q8(&d, &qs, &q8, &q8_inv_scales, 1);
        let avx512_result = unsafe {
            iq2xxs_vec_dot_q8_amd7600(&d, &qs, &q8, &q8_inv_scales, 1)
        };
        let diff = (scalar_result - avx512_result).abs();
        eprintln!("\nscalar = {:.6}, avx512 = {:.6}, diff = {:.6} {}",
            scalar_result, avx512_result, diff, if diff < 0.5 { "OK" } else { "FAIL" });

        assert!(diff < 0.5, "Per-group debug test failed: diff = {}", diff);
    }

    /// Q2_K 标量 vs AVX-512 对比测试
    #[test]
    fn test_q2k_scalar_vs_avx512() {
        use crate::cpu_expert::avx512::{q2k_vec_dot_q8_amd7600, is_avx512_supported, quantize_f32_to_q8_block};

        if !is_avx512_supported() {
            eprintln!("AVX-512 not supported, skipping test");
            return;
        }

        let n_blocks = 4;

        // 构造 Q2_K 测试数据
        let d: Vec<f32> = vec![0.5, 0.3, 0.7, 0.2];
        let dmin: Vec<f32> = vec![0.1, 0.05, 0.15, 0.02];

        // scales: 16 bytes/block, 低4位=scale, 高4位=min
        let mut scales = vec![0u8; n_blocks * 16];
        for blk in 0..n_blocks {
            for j in 0..16 {
                let sc = ((blk * 3 + j * 2) % 16) as u8;
                let mn = ((blk + j) % 16) as u8;
                scales[blk * 16 + j] = sc | (mn << 4);
            }
        }

        // qs: 64 bytes/block, 2-bit packed
        let mut qs = vec![0u8; n_blocks * 64];
        for blk in 0..n_blocks {
            for j in 0..64 {
                qs[blk * 64 + j] = ((blk * 7 + j * 13) % 256) as u8;
            }
        }

        // 随机 q8 输入
        let x: Vec<f32> = (0..n_blocks * 256)
            .map(|i| ((i * 7919 + 1) % 2000) as f32 / 100.0 - 10.0)
            .collect();
        let mut q8 = vec![0i8; n_blocks * 256];
        let mut q8_inv_scales = vec![0.0f32; n_blocks];
        for blk in 0..n_blocks {
            q8_inv_scales[blk] = quantize_f32_to_q8_block(
                &x[blk * 256..(blk + 1) * 256],
                &mut q8[blk * 256..(blk + 1) * 256],
                256,
            );
        }

        let scalar_result = q2k_vec_dot_q8(&d, &dmin, &scales, &qs, &q8, &q8_inv_scales, n_blocks);
        let avx512_result = unsafe {
            q2k_vec_dot_q8_amd7600(&d, &dmin, &scales, &qs, &q8, &q8_inv_scales, n_blocks)
        };
        let diff = (scalar_result - avx512_result).abs();
        eprintln!("  Q2_K scalar={:.6}  avx512={:.6}  diff={:.6}  {}",
            scalar_result, avx512_result, diff, if diff < 1.0 { "OK" } else { "FAIL" });

        assert!(diff < 1.0, "Q2_K AVX-512 kernel failed: diff = {}", diff);
    }
}

// ============================================================================
// CPU FFN 引擎：IQ2_XXS+Q2_K 混合量化格式
// ============================================================================

/// CPU FFN 引擎（IQ2_XXS+Q2_K 混合量化）
///
/// 封装专家权重存储 + SLRU 缓存管理 + FFN 计算。
/// gate/up 使用 IQ2_XXS，down 使用 Q2_K。
pub struct CpuFfnEngineIq2xxsQ2k {
    /// 专家权重存储：(layer_id, expert_id) → (gate, up, down)
    weights: std::collections::HashMap<(usize, usize), Arc<(Iq2XxsWeight, Iq2XxsWeight, Q2KWeight)>>,
    /// SLRU protected 段（热点专家）
    protected: std::collections::HashSet<(usize, usize)>,
    /// SLRU probation 段（新准入专家）
    probation: std::collections::HashSet<(usize, usize)>,
    /// 访问频率计数（LFU）
    freq: std::collections::HashMap<(usize, usize), u64>,
    /// 最近访问时间（LRU 辅助）
    last_access: std::collections::HashMap<(usize, usize), u64>,
    /// 访问计数器
    access_counter: u64,
    /// protected 段容量上限
    protected_capacity: usize,
    /// probation 段容量上限
    probation_capacity: usize,
    /// 缓存命中统计
    hits: u64,
    misses: u64,
}

impl CpuFfnEngineIq2xxsQ2k {
    pub fn new() -> Self {
        Self::with_capacity(4096, 512)
    }

    pub fn with_capacity(protected_capacity: usize, probation_capacity: usize) -> Self {
        Self {
            weights: std::collections::HashMap::new(),
            protected: std::collections::HashSet::new(),
            probation: std::collections::HashSet::new(),
            freq: std::collections::HashMap::new(),
            last_access: std::collections::HashMap::new(),
            access_counter: 0,
            protected_capacity,
            probation_capacity,
            hits: 0,
            misses: 0,
        }
    }

    /// 添加专家到缓存
    pub fn add_expert(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        gate_weight: Iq2XxsWeight,
        up_weight: Iq2XxsWeight,
        down_weight: Q2KWeight,
    ) {
        let key = (layer_id, expert_id);
        let weights = Arc::new((gate_weight, up_weight, down_weight));

        if self.weights.contains_key(&key) {
            self.weights.insert(key, weights);
            return;
        }

        if self.probation.len() >= self.probation_capacity {
            self.evict_probation();
        }

        self.weights.insert(key, weights);
        self.probation.insert(key);
        self.access_counter += 1;
        *self.last_access.entry(key).or_insert(0) = self.access_counter;
    }

    /// 添加专家到 protected 段
    pub fn add_expert_protected(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        gate_weight: Iq2XxsWeight,
        up_weight: Iq2XxsWeight,
        down_weight: Q2KWeight,
    ) {
        let key = (layer_id, expert_id);
        let weights = Arc::new((gate_weight, up_weight, down_weight));

        if self.weights.contains_key(&key) {
            self.weights.insert(key, weights);
            if self.probation.contains(&key) {
                self.probation.remove(&key);
                self.protected.insert(key);
            }
            return;
        }

        if self.protected.len() >= self.protected_capacity {
            self.demote_protected();
        }

        self.weights.insert(key, weights);
        self.protected.insert(key);
        self.access_counter += 1;
        *self.last_access.entry(key).or_insert(0) = self.access_counter;
    }

    /// 计算 FFN
    pub fn compute_ffn(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        x: &[f32],
        route_weight: f32,
        swiglu_limit: f32,
    ) -> Option<Vec<f32>> {
        let key = (layer_id, expert_id);

        if self.weights.contains_key(&key) {
            self.hits += 1;
            self.touch_key(key);

            let weights = Arc::clone(self.weights.get(&key)?);
            Some(mixed_ffn_pair_iq2xxs_q2k(
                x,
                &weights.0,
                &weights.1,
                &weights.2,
                route_weight,
                swiglu_limit,
            ))
        } else {
            self.misses += 1;
            None
        }
    }

    /// 记录专家访问（Gate 选中时调用，驱动预取）
    pub fn record_access(&mut self, layer_id: usize, expert_id: usize) {
        let key = (layer_id, expert_id);
        *self.freq.entry(key).or_insert(0) += 1;
        self.access_counter += 1;
        *self.last_access.entry(key).or_insert(0) = self.access_counter;
    }

    /// 触碰专家（命中时调用，更新 LRU）
    fn touch_key(&mut self, key: (usize, usize)) {
        self.access_counter += 1;
        *self.last_access.entry(key).or_insert(0) = self.access_counter;

        // probation → protected 晋升
        if self.probation.contains(&key) {
            if self.protected.len() >= self.protected_capacity {
                self.demote_protected();
            }
            self.probation.remove(&key);
            self.protected.insert(key);
        }
    }

    /// 降级 protected 段最冷专家
    fn demote_protected(&mut self) {
        if self.protected.is_empty() {
            return;
        }

        let mut demote_key = None;
        let mut min_score = u64::MAX;

        for &key in &self.protected {
            let freq = *self.freq.get(&key).unwrap_or(&1);
            let last = *self.last_access.get(&key).unwrap_or(&0);
            let score = freq * 1000 + last;

            if score < min_score {
                min_score = score;
                demote_key = Some(key);
            }
        }

        if let Some(key) = demote_key {
            self.protected.remove(&key);
            if self.probation.len() >= self.probation_capacity {
                self.evict_probation();
            }
            self.probation.insert(key);
        }
    }

    /// 淘汰 probation 段最冷专家
    fn evict_probation(&mut self) {
        if self.probation.is_empty() {
            return;
        }

        let mut evict_key = None;
        let mut min_score = u64::MAX;

        for &key in &self.probation {
            let freq = *self.freq.get(&key).unwrap_or(&1);
            let last = *self.last_access.get(&key).unwrap_or(&0);
            let score = freq * 1000 + last;

            if score < min_score {
                min_score = score;
                evict_key = Some(key);
            }
        }

        if let Some(key) = evict_key {
            self.probation.remove(&key);
            self.weights.remove(&key);
            self.freq.remove(&key);
            self.last_access.remove(&key);
        }
    }

    pub fn has_expert(&self, layer_id: usize, expert_id: usize) -> bool {
        self.weights.contains_key(&(layer_id, expert_id))
    }

    pub fn expert_count(&self) -> usize {
        self.weights.len()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn stats(&self) -> (u64, u64, usize, usize) {
        (self.hits, self.misses, self.protected.len(), self.probation.len())
    }
}

impl Default for CpuFfnEngineIq2xxsQ2k {
    fn default() -> Self {
        Self::new()
    }
}
