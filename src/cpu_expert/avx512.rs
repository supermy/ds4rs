/// AVX-512 BW / AVX2 优化 IQ2_XS 点积内核
///
/// 借鉴 docs/cpu-ffn-kimi.md 优化策略：
/// - 512-bit maddubs+madd 全程（替代 256-bit 内核 + 512-bit 累加）
/// - AVX-512 BW: _mm512_maddubs_epi16 + _mm512_madd_epi16
/// - grid_u64 预计算缓存（4KB，L1 常驻）
/// - scale shuffle 表编译期常量
/// - _mm_prefetch 预取下一块权重数据
///
/// 为什么不用 dpbusd (VNNI)：
/// IQ2_XS 的 per-group scale 在 maddubs 和 madd 之间应用：
///   maddubs(grid_u8, q8s_i8) → i16  (grid × q8 点积)
///   madd(dot_i16, scale_i16) → i32  (乘以 scale 并累加)
/// dpbusd 会跳过中间的 scale 乘法，无法正确计算。
/// 唯一可行方案是预解压到 INT8（3.5× 膨胀），但 L3 容量不足。
///
/// AMD Ryzen 5 7600 拓扑：
/// - 单 CCD 6 核，32MB L3 统一
/// - Zen 4: 512-bit 指令拆分为 2×256-bit uops（仍有指令密度优势）
/// - L1=32KB/核, L2=1MB/核, 缓存行=64B

// ============================================================================
// 符号计算共享常量（AVX-512 / AVX2 / Tile 路径共用）
// ============================================================================

/// bit_helper 查找表：用于计算奇偶校验位（llama.cpp parity trick）
const K_BIT_HELPER: [u8; 32] = [
    0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80,
    0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80,
    0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
];

/// 符号 shuffle 表 1：将 full_sign_bits 低 128 位广播到各 group
const BLOCK_SIGN_SHUFFLE_1: [u8; 32] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
];

/// 符号 shuffle 表 2：将 full_sign_bits 高 128 位广播到各 group
const BLOCK_SIGN_SHUFFLE_2: [u8; 32] = [
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
    0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c,
    0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
];

/// 位选择掩码：用于从 shuffled sign byte 中提取特定位
const BIT_SELECTOR_MASK: [u8; 32] = [
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
];

pub fn is_avx512_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

pub fn is_avx2_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// F32 激活量化为 int8，返回逆缩放因子
#[inline]
pub fn quantize_f32_to_q8_block(x: &[f32], q8: &mut [i8], n: usize) -> f32 {
    let mut amax = 0.0f32;
    for i in 0..n {
        amax = amax.max(x[i].abs());
    }
    let scale = if amax > 1e-6 { 127.0 / amax } else { 0.0 };
    let inv_scale = if amax > 1e-6 { amax / 127.0 } else { 0.0 };

    for i in 0..n {
        q8[i] = (x[i] * scale).round().clamp(-128.0, 127.0) as i8;
    }

    inv_scale
}

// ============================================================================
// AVX-512 BW 路径: 512-bit maddubs+madd 全程
// ============================================================================

/// AVX-512 IQ2_XS × Q8 向量点积（内部，使用预量化激活）
///
/// 512-bit maddubs+madd 全程，处理 16 groups/迭代。
/// 符号计算保持 256-bit（与 llama.cpp 一致），算术合并到 512-bit。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn iq2xs_vec_dot_q8_avx512(
    d: &[f32],
    qs: &[u16],
    scales: &[u8],
    q8: &[i8],
    q8_inv_scales: &[f32],
    n_blocks: usize,
) -> f32 {
    use std::arch::x86_64::*;

    let grid_u64 = super::tables::get_iq2xs_grid_u64();
    let grid_u64_ptr = grid_u64.as_ptr();
    let k_scale_shuffle = &super::tables::K_SCALE_SHUFFLE;

    let m4 = _mm_set1_epi8(0xf);
    let m1 = _mm_set1_epi8(1);
    let m511 = _mm256_set1_epi16(511);
    let mone = _mm256_set1_epi8(1);

    // 符号计算常量（模块级共享）
    let bit_helper = _mm256_loadu_si256(K_BIT_HELPER.as_ptr() as *const __m256i);
    let sign_shuffle_1 = _mm256_loadu_si256(BLOCK_SIGN_SHUFFLE_1.as_ptr() as *const __m256i);
    let sign_shuffle_2 = _mm256_loadu_si256(BLOCK_SIGN_SHUFFLE_2.as_ptr() as *const __m256i);
    let bit_selector_mask = _mm256_loadu_si256(BIT_SELECTOR_MASK.as_ptr() as *const __m256i);

    let mut accumf = _mm512_setzero_ps();

    for blk in 0..n_blocks {
        let d_val = d[blk] * q8_inv_scales[blk];
        let qs_ptr = qs.as_ptr().add(blk * 32);
        let q8_ptr = q8.as_ptr().add(blk * 256);

        // 预取下一块权重数据到 L2
        if blk + 1 < n_blocks {
            _mm_prefetch(qs.as_ptr().add((blk + 1) * 32) as *const i8, _MM_HINT_T1);
            _mm_prefetch(q8.as_ptr().add((blk + 1) * 256) as *const i8, _MM_HINT_T1);
        }

        // 解码 scales: 8 bytes → 16 i8 values
        let mut aux64: u64 = 0;
        for b in 0..8 {
            aux64 |= (scales[blk * 8 + b] as u64) << (b * 8);
        }
        let mut stmp = _mm_set1_epi64x(aux64 as i64);
        stmp = _mm_unpacklo_epi8(
            _mm_and_si128(stmp, m4),
            _mm_and_si128(_mm_srli_epi16(stmp, 4), m4),
        );
        let scales_vec = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        let mut sumi_512 = _mm512_setzero_si512();

        // 每 16 groups 为一轮 (step_by(4), 2 轮覆盖 32 groups)
        for ib32 in (0..8).step_by(4) {
            // 加载 16 u16 qs 值
            let q2_data = _mm256_loadu_si256(qs_ptr.add(ib32 * 4) as *const __m256i);
            let aux_gindex = _mm256_and_si256(q2_data, m511);
            let mut gidx = [0u16; 16];
            _mm256_storeu_si256(gidx.as_mut_ptr() as *mut __m256i, aux_gindex);

            // 符号计算 (256-bit, 与 llama.cpp 一致)
            let partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            let partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            let partial_sign_bits_for_counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);
            let odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            let full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);

            // 加载 4 × 32 q8 值 = 128 q8 值 (16 groups × 8 q8/group)
            let q8_1 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32) as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32 + 32) as *const __m256i);
            let q8_3 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32 + 64) as *const __m256i);
            let q8_4 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32 + 96) as *const __m256i);

            // 查找 16 grid_u64 值 (标量查找，4KB 表 L1 命中)
            let q2_1 = _mm256_set_epi64x(
                *grid_u64_ptr.add(gidx[3] as usize) as i64,
                *grid_u64_ptr.add(gidx[2] as usize) as i64,
                *grid_u64_ptr.add(gidx[1] as usize) as i64,
                *grid_u64_ptr.add(gidx[0] as usize) as i64,
            );
            let q2_2 = _mm256_set_epi64x(
                *grid_u64_ptr.add(gidx[7] as usize) as i64,
                *grid_u64_ptr.add(gidx[6] as usize) as i64,
                *grid_u64_ptr.add(gidx[5] as usize) as i64,
                *grid_u64_ptr.add(gidx[4] as usize) as i64,
            );
            let q2_3 = _mm256_set_epi64x(
                *grid_u64_ptr.add(gidx[11] as usize) as i64,
                *grid_u64_ptr.add(gidx[10] as usize) as i64,
                *grid_u64_ptr.add(gidx[9] as usize) as i64,
                *grid_u64_ptr.add(gidx[8] as usize) as i64,
            );
            let q2_4 = _mm256_set_epi64x(
                *grid_u64_ptr.add(gidx[15] as usize) as i64,
                *grid_u64_ptr.add(gidx[14] as usize) as i64,
                *grid_u64_ptr.add(gidx[13] as usize) as i64,
                *grid_u64_ptr.add(gidx[12] as usize) as i64,
            );

            // 符号应用 (256-bit)
            let full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            let full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            let full_signs_1 = _mm256_insertf128_si256(
                _mm256_castsi128_si256(full_signs_l), full_signs_l, 1);
            let full_signs_2 = _mm256_insertf128_si256(
                _mm256_castsi128_si256(full_signs_h), full_signs_h, 1);

            let mut signs;
            signs = _mm256_shuffle_epi8(full_signs_1, sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            let q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_1, sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            let q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            let q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            let q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));

            // ===== 512-bit maddubs + madd =====
            // 合并 2 × 256-bit → 512-bit（inserti64x4 是 rename 操作，零延迟）
            let grid_512_a = _mm512_inserti64x4(
                _mm512_castsi256_si512(q2_1), q2_2, 1);
            let grid_512_b = _mm512_inserti64x4(
                _mm512_castsi256_si512(q2_3), q2_4, 1);
            let q8s_512_a = _mm512_inserti64x4(
                _mm512_castsi256_si512(q8s_1), q8s_2, 1);
            let q8s_512_b = _mm512_inserti64x4(
                _mm512_castsi256_si512(q8s_3), q8s_4, 1);

            // 512-bit maddubs: 64 u8*i8 → 32 i16
            let dot_512_a = _mm512_maddubs_epi16(grid_512_a, q8s_512_a);
            let dot_512_b = _mm512_maddubs_epi16(grid_512_b, q8s_512_b);

            // 加载 scales 并合并为 512-bit
            let sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales_vec,
                _mm_loadu_si128(k_scale_shuffle.as_ptr().add((ib32 + 0) * 16) as *const __m128i)));
            let sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales_vec,
                _mm_loadu_si128(k_scale_shuffle.as_ptr().add((ib32 + 1) * 16) as *const __m128i)));
            let sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales_vec,
                _mm_loadu_si128(k_scale_shuffle.as_ptr().add((ib32 + 2) * 16) as *const __m128i)));
            let sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales_vec,
                _mm_loadu_si128(k_scale_shuffle.as_ptr().add((ib32 + 3) * 16) as *const __m128i)));

            let sc_512_a = _mm512_inserti64x4(
                _mm512_castsi256_si512(sc1), sc2, 1);
            let sc_512_b = _mm512_inserti64x4(
                _mm512_castsi256_si512(sc3), sc4, 1);

            // 512-bit madd: 32 i16*i16 → 16 i32
            let prod_512_a = _mm512_madd_epi16(dot_512_a, sc_512_a);
            let prod_512_b = _mm512_madd_epi16(dot_512_b, sc_512_b);

            // 累加
            sumi_512 = _mm512_add_epi32(sumi_512, prod_512_a);
            sumi_512 = _mm512_add_epi32(sumi_512, prod_512_b);
        }

        // 最终: sumi_512 → f32, 乘以 d_val
        accumf = _mm512_fmadd_ps(
            _mm512_set1_ps(d_val),
            _mm512_cvtepi32_ps(sumi_512),
            accumf,
        );
    }

    0.125 * _mm512_reduce_add_ps(accumf)
}

// ============================================================================
// IQ2_XXS AVX-512 / AVX2 内核
// ============================================================================

/// IQ2_XXS 点积内核（标量，匹配 llama.cpp ggml_vec_dot_iq2_xxs_q8_K_generic）
///
/// block_iq2_xxs 布局（66 bytes / 256 elements）：
///   - d: fp16 super-block scale
///   - qs[32]: uint16，编码 grid index + sign index + scale
///
/// 每 32 元素一个子块（ib32），读 4 个 uint16（8 bytes = 2 uint32）：
///   - aux8[0..3] = 4 个 grid index（8-bit, 0-255），索引到 iq2xxs_grid（256 entries）
///   - aux32[1] 包含 sign indices（7-bit × 4 = 28 bits）+ scale（4 bits）
///   - scale = 2*(aux32[1] >> 28) + 1
///   - sign = ksigns_iq2xs[(aux32[1] >> 7*l) & 127]
///
/// vec_dot 公式：result = d * 0.125 * Σ(ls * Σ(grid[j] * q8[j] * sign[j]))
pub fn iq2xxs_vec_dot_q8(
    d: &[f32],
    qs: &[u16],
    q8: &[i8],
    q8_inv_scales: &[f32],
    n_blocks: usize,
) -> f32 {
    let grid = super::tables::get_iq2xxs_grid(); // IQ2_XXS grid (256 entries × 8 i8)
    // 直接使用 KSIGNS_IQ2XS 字节表（与 AVX-512 内核一致），
    // 不用 get_sign_mul_table()（依赖 init_tables()，默认全 1.0）
    let ksigns_bytes = &super::tables::KSIGNS_IQ2XS;

    let mut sumf = 0.0f32;

    for blk in 0..n_blocks {
        let d_val = d[blk] * q8_inv_scales[blk];
        let qs_blk = &qs[blk * 32..(blk + 1) * 32];
        let q8_blk = &q8[blk * 256..(blk + 1) * 256];

        let mut bsum = 0i32;
        let mut q8_offset = 0usize;

        for ib32 in 0..8 {
            // 每 32 元素一个子块，读 4 个 uint16
            let q2_base = ib32 * 4;
            let qs_slice = &qs_blk[q2_base..q2_base + 4];

            // 将 4 个 uint16 复制为 2 个 uint32（小端序）
            let aux32_0 = (qs_slice[0] as u32) | ((qs_slice[1] as u32) << 16);
            let aux32_1 = (qs_slice[2] as u32) | ((qs_slice[3] as u32) << 16);

            // aux8[0..3] = aux32_0 的 4 个字节 = 4 个 grid index
            let aux8: [u8; 4] = [
                (aux32_0 & 0xFF) as u8,
                ((aux32_0 >> 8) & 0xFF) as u8,
                ((aux32_0 >> 16) & 0xFF) as u8,
                ((aux32_0 >> 24) & 0xFF) as u8,
            ];

            // scale: 4 bits from top of aux32_1
            let ls = 2 * ((aux32_1 >> 28) & 0xF) as i32 + 1;

            let mut sumi = 0i32;
            for l in 0..4 {
                // grid lookup: iq2xxs_grid[aux8[l]] → 8 个 int8 值
                let grid_idx = aux8[l] as usize;
                let grid_offset = grid_idx * 8;

                // sign lookup: ksigns_iq2xs[(aux32_1 >> 7*l) & 127]
                let sign_idx = ((aux32_1 >> (7 * l)) & 127) as usize;

                for j in 0..8 {
                    let grid_val = grid[grid_offset + j] as i32;
                    let q8_val = q8_blk[q8_offset] as i32;
                    // sign: KSIGNS_IQ2XS[sign_idx] bit j = 1 → -1, bit j = 0 → +1
                    let sign_val = if ksigns_bytes[sign_idx] & (1 << j) != 0 { -1i32 } else { 1i32 };
                    sumi += grid_val * q8_val * sign_val;
                    q8_offset += 1;
                }
            }
            bsum += sumi * ls;
        }

        sumf += d_val * bsum as f32;
    }

    0.125 * sumf
}

/// IQ2_XXS AVX2 点积内核（回退，使用标量实现）
pub fn iq2xxs_vec_dot_q8_avx2(
    d: &[f32],
    qs: &[u16],
    q8: &[i8],
    q8_inv_scales: &[f32],
    n_blocks: usize,
) -> f32 {
    if is_avx512_supported() {
        unsafe { iq2xxs_vec_dot_q8_amd7600(d, qs, q8, q8_inv_scales, n_blocks) }
    } else {
        iq2xxs_vec_dot_q8(d, qs, q8, q8_inv_scales, n_blocks)
    }
}

/// AVX-512 IQ2_XXS × Q8 向量点积（256-bit 双 ib32 融合管线）
///
/// 优化策略（对标 llama.cpp AVX2 实现 quants_x86.c:2536-2576）：
/// - 一次处理 2 个 ib32（64 个 q8 值），2 次 256-bit maddubs
/// - grid 值全部在 [8, 43] 范围（unsigned byte），直接传给 maddubs
/// - sign 只需应用到 q8：sign_epi8(q8, signs64) → maddubs(grid, q8s)
/// - ls 融入 madd：madd(dot, set1(2*ls+1))
/// - 预计算符号掩码表（1KB L1 常驻）
///
/// 与 llama.cpp AVX2 的关键区别：
/// - llama.cpp 用 keven_signs_q2xs (u64*) 直接查表
/// - 我们用 KSIGNS_IQ2XS_MASKS (i8[8]) 查表 + _mm256_set_epi64x 打包
/// - 数学等价：signs64[idx] = KSIGNS_IQ2XS_MASKS[idx] 的 u64 重解释
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn iq2xxs_vec_dot_q8_amd7600(
    d: &[f32],
    qs: &[u16],
    q8: &[i8],
    q8_inv_scales: &[f32],
    n_blocks: usize,
) -> f32 {
    use std::arch::x86_64::*;

    let grid_u64 = super::tables::get_iq2xxs_grid_u64();
    let grid_u64_ptr = grid_u64.as_ptr();
    let sign_u64 = &super::tables::KSIGNS_IQ2XS_U64;

    let mut sumf = 0.0f32;

    for blk in 0..n_blocks {
        let d_val = d[blk] * q8_inv_scales[blk];
        let qs_ptr = qs.as_ptr().add(blk * 32);
        let q8_ptr = q8.as_ptr().add(blk * 256);

        if blk + 1 < n_blocks {
            _mm_prefetch(qs.as_ptr().add((blk + 1) * 32) as *const i8, _MM_HINT_T1);
        }

        let mut block_sum: i32 = 0;

        // 一次处理 2 个 ib32（与 llama.cpp 一致）
        // 优化：用 read_unaligned 一次读 16 字节（8 个 u16 = 4 个 u32），
        // 替代逐个 u16 读取和拼接，减少标量解码开销约 50%
        for ib32 in (0..8).step_by(2) {
            // 一次读 8 个 u16（16 字节）= 4 个 u32
            // aux32[0] = qs[0..1] 的 grid indices（4 个 u8）
            // aux32[1] = qs[2..3] 的 sign indices + ls
            // aux32[2] = qs[4..5] 的 grid indices（4 个 u8）
            // aux32[3] = qs[6..7] 的 sign indices + ls
            let aux32: [u32; 4] = std::ptr::read_unaligned(qs_ptr.add(ib32 * 4) as *const [u32; 4]);
            let aux8 = &aux32 as *const [u32; 4] as *const u8;

            let ls1 = 2 * ((aux32[1] >> 28) as i32 & 0xF) + 1;
            let ls2 = 2 * ((aux32[3] >> 28) as i32 & 0xF) + 1;

            // 2 个 grid_256（每 ib32 一个，4 groups 打包为 __m256i）
            let grid_1 = _mm256_set_epi64x(
                *grid_u64_ptr.add(*aux8.add(3) as usize) as i64,
                *grid_u64_ptr.add(*aux8.add(2) as usize) as i64,
                *grid_u64_ptr.add(*aux8.add(1) as usize) as i64,
                *grid_u64_ptr.add(*aux8.add(0) as usize) as i64,
            );
            let grid_2 = _mm256_set_epi64x(
                *grid_u64_ptr.add(*aux8.add(11) as usize) as i64,
                *grid_u64_ptr.add(*aux8.add(10) as usize) as i64,
                *grid_u64_ptr.add(*aux8.add(9) as usize) as i64,
                *grid_u64_ptr.add(*aux8.add(8) as usize) as i64,
            );

            // 2 个 q8_256（每 ib32 32 个 q8 值）
            let q8_1 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32) as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8_ptr.add((ib32 + 1) * 32) as *const __m256i);

            // 2 个 sign_256（预计算符号掩码表查表，1KB L1 常驻）
            let sign_1 = _mm256_set_epi64x(
                sign_u64[((aux32[1] >> 21) & 127) as usize] as i64,
                sign_u64[((aux32[1] >> 14) & 127) as usize] as i64,
                sign_u64[((aux32[1] >> 7) & 127) as usize] as i64,
                sign_u64[((aux32[1] >> 0) & 127) as usize] as i64,
            );
            let sign_2 = _mm256_set_epi64x(
                sign_u64[((aux32[3] >> 21) & 127) as usize] as i64,
                sign_u64[((aux32[3] >> 14) & 127) as usize] as i64,
                sign_u64[((aux32[3] >> 7) & 127) as usize] as i64,
                sign_u64[((aux32[3] >> 0) & 127) as usize] as i64,
            );

            // sign 只应用到 q8（grid 是 unsigned byte，不需要符号吸收）
            let q8s_1 = _mm256_sign_epi8(q8_1, sign_1);
            let q8s_2 = _mm256_sign_epi8(q8_2, sign_2);

            // 256-bit maddubs: grid(unsigned) × q8s(signed) → 16 i16
            let dot_1 = _mm256_maddubs_epi16(grid_1, q8s_1);
            let dot_2 = _mm256_maddubs_epi16(grid_2, q8s_2);

            // ls 融入 madd
            let p1 = _mm256_madd_epi16(dot_1, _mm256_set1_epi16(ls1 as i16));
            let p2 = _mm256_madd_epi16(dot_2, _mm256_set1_epi16(ls2 as i16));

            // 累加 2 个 ib32 的结果
            let sum_256 = _mm256_add_epi32(p1, p2);

            // hsum 8 i32 → 1 i32
            let hi = _mm256_extracti128_si256(sum_256, 1);
            let lo = _mm256_castsi256_si128(sum_256);
            let sum128 = _mm_add_epi32(lo, hi);
            let s1 = _mm_shuffle_epi32(sum128, 0xB1);
            let s2 = _mm_add_epi32(sum128, s1);
            let s3 = _mm_shuffle_epi32(s2, 0x4E);
            let s4 = _mm_add_epi32(s2, s3);
            block_sum += _mm_cvtsi128_si32(s4);
        }

        sumf += d_val * block_sum as f32;
    }

    0.125 * sumf
}

// ============================================================================
// Q2_K AVX-512 / AVX2 内核
// ============================================================================

/// Q2_K 点积内核（标量，匹配 llama.cpp ggml_vec_dot_q2_K_q8_K）
///
/// 分块 matvec：IQ2_XXS 权重流式读取 + Q8 数据常驻 L1
///
/// 核心优化：
///   1. Q8 数据（7168 bytes）常驻 L1 缓存（< 32KB）
///   2. 权重流式读取：按行顺序访问 d 和 qs，最大化 DDR burst 效率
///   3. on-the-fly dequantization：在寄存器内解压 grid + sign + ls，立即点积
///   4. 软件预取：提前预取下一行权重到 L2
///   5. 预计算符号掩码表：KSIGNS_IQ2XS_MASKS（1KB L1 常驻），避免逐 bit 构造
///
/// 分块策略：将 18432 行分成 3 块 × 6144 行
///   每块 gate 权重 17.5MB，gate+up = 23.4MB < 30MB L3
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn iq2xxs_matvec_blocked_amd7600(
    d: &[f32],
    qs: &[u16],
    q8: &[i8],
    q8_inv_scales: &[f32],
    n_blocks_per_row: usize,
    row_start: usize,
    row_end: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    let grid_u64 = super::tables::get_iq2xxs_grid_u64();
    let grid_u64_ptr = grid_u64.as_ptr();
    let sign_u64 = &super::tables::KSIGNS_IQ2XS_U64;

    for row in row_start..row_end {
        // 预取下一行的权重到 L2
        if row + 1 < row_end {
            let next_qs_ptr = qs.as_ptr().add((row + 1) * n_blocks_per_row * 32);
            _mm_prefetch(next_qs_ptr as *const i8, _MM_HINT_T1);
            let next_d_ptr = d.as_ptr().add((row + 1) * n_blocks_per_row);
            _mm_prefetch(next_d_ptr as *const i8, _MM_HINT_T1);
        }

        let d_row = &d[row * n_blocks_per_row..(row + 1) * n_blocks_per_row];
        let qs_row = &qs[row * n_blocks_per_row * 32..(row + 1) * n_blocks_per_row * 32];

        let mut sumf = 0.0f32;

        for blk in 0..n_blocks_per_row {
            let d_val = d_row[blk] * q8_inv_scales[blk];
            let qs_ptr = qs_row.as_ptr().add(blk * 32);
            let q8_ptr = q8.as_ptr().add(blk * 256);

            if blk + 1 < n_blocks_per_row {
                _mm_prefetch(qs_row.as_ptr().add((blk + 1) * 32) as *const i8, _MM_HINT_T0);
            }

            let mut block_sum: i32 = 0;

            // 双 ib32 融合 + memcpy 批量读取（与 vec_dot 内核一致）
            for ib32 in (0..8).step_by(2) {
                // 一次读 8 个 u16（16 字节）= 4 个 u32
                let aux32: [u32; 4] = std::ptr::read_unaligned(qs_ptr.add(ib32 * 4) as *const [u32; 4]);
                let aux8 = &aux32 as *const [u32; 4] as *const u8;

                let ls1 = 2 * ((aux32[1] >> 28) as i32 & 0xF) + 1;
                let ls2 = 2 * ((aux32[3] >> 28) as i32 & 0xF) + 1;

                let grid_1 = _mm256_set_epi64x(
                    *grid_u64_ptr.add(*aux8.add(3) as usize) as i64,
                    *grid_u64_ptr.add(*aux8.add(2) as usize) as i64,
                    *grid_u64_ptr.add(*aux8.add(1) as usize) as i64,
                    *grid_u64_ptr.add(*aux8.add(0) as usize) as i64,
                );
                let grid_2 = _mm256_set_epi64x(
                    *grid_u64_ptr.add(*aux8.add(11) as usize) as i64,
                    *grid_u64_ptr.add(*aux8.add(10) as usize) as i64,
                    *grid_u64_ptr.add(*aux8.add(9) as usize) as i64,
                    *grid_u64_ptr.add(*aux8.add(8) as usize) as i64,
                );

                let q8_1 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32) as *const __m256i);
                let q8_2 = _mm256_loadu_si256(q8_ptr.add((ib32 + 1) * 32) as *const __m256i);

                let sign_1 = _mm256_set_epi64x(
                    sign_u64[((aux32[1] >> 21) & 127) as usize] as i64,
                    sign_u64[((aux32[1] >> 14) & 127) as usize] as i64,
                    sign_u64[((aux32[1] >> 7) & 127) as usize] as i64,
                    sign_u64[((aux32[1] >> 0) & 127) as usize] as i64,
                );
                let sign_2 = _mm256_set_epi64x(
                    sign_u64[((aux32[3] >> 21) & 127) as usize] as i64,
                    sign_u64[((aux32[3] >> 14) & 127) as usize] as i64,
                    sign_u64[((aux32[3] >> 7) & 127) as usize] as i64,
                    sign_u64[((aux32[3] >> 0) & 127) as usize] as i64,
                );

                // sign 只应用到 q8（grid 是 unsigned byte）
                let q8s_1 = _mm256_sign_epi8(q8_1, sign_1);
                let q8s_2 = _mm256_sign_epi8(q8_2, sign_2);

                let dot_1 = _mm256_maddubs_epi16(grid_1, q8s_1);
                let dot_2 = _mm256_maddubs_epi16(grid_2, q8s_2);

                let p1 = _mm256_madd_epi16(dot_1, _mm256_set1_epi16(ls1 as i16));
                let p2 = _mm256_madd_epi16(dot_2, _mm256_set1_epi16(ls2 as i16));

                let sum_256 = _mm256_add_epi32(p1, p2);

                let hi = _mm256_extracti128_si256(sum_256, 1);
                let lo = _mm256_castsi256_si128(sum_256);
                let sum128 = _mm_add_epi32(lo, hi);
                let s1 = _mm_shuffle_epi32(sum128, 0xB1);
                let s2 = _mm_add_epi32(sum128, s1);
                let s3 = _mm_shuffle_epi32(s2, 0x4E);
                let s4 = _mm_add_epi32(s2, s3);
                block_sum += _mm_cvtsi128_si32(s4);
            }

            sumf += d_val * block_sum as f32;
        }

        output[row - row_start] = 0.125 * sumf;
    }
}

/// block_q2_K 布局（84 bytes / 256 elements）：
///   - scales[16]: 4-bit packed，低 4 位 = scale，高 4 位 = min
///   - qs[64]: 2-bit 量化值打包（每字节 4 个 2-bit 值，0-3 无偏移）
///   - d: fp16 super-block scale
///   - dmin: fp16 super-block minimum scale
///
/// vec_dot 公式：result = q8_d * (q2_d * isum - q2_dmin * summs)
///   isum  = Σ (scales[j] & 0xF) * dot(q2_sub_block, q8_sub_block)
///   summs = Σ bsums[j] * (scales[j] >> 4)
///
/// qs 索引：每个 q2 字节被 4 个子块共享（通过 shift 0/2/4/6）
///   k=0: q2[0..31],  shift=0,2,4,6 → 8 个子块 × 16 元素 = 128
///   k=1: q2[32..63], shift=0,2,4,6 → 8 个子块 × 16 元素 = 128
pub fn q2k_vec_dot_q8(
    d: &[f32],
    dmin: &[f32],
    scales: &[u8],
    qs: &[u8],
    q8: &[i8],
    q8_inv_scales: &[f32],
    n_blocks: usize,
) -> f32 {
    let mut sumf = 0.0f32;

    for blk in 0..n_blocks {
        let d_val = d[blk];
        let dmin_val = dmin[blk];
        let sc = &scales[blk * 16..(blk + 1) * 16];
        let qs_blk = &qs[blk * 64..(blk + 1) * 64];
        let q8_blk = &q8[blk * 256..(blk + 1) * 256];

        // min 贡献：summs = Σ bsums[j] * (sc[j] >> 4)
        // bsums[j] = Σ q8[j*16..(j+1)*16]
        let mut summs = 0i32;
        for j in 0..16 {
            let mut bsum = 0i32;
            for l in 0..16 {
                bsum += q8_blk[j * 16 + l] as i32;
            }
            summs += bsum * ((sc[j] >> 4) as i32);
        }

        // scale 贡献：isum = Σ (sc[is] & 0xF) * dot(q2_sub, q8_sub)
        // 遍历 2 个 128 元素组（k=0,1），每组 4 个子组（j=0..3），
        // 每个子组 2 个子块（各 16 元素），共 16 个子块
        let mut isum = 0i32;
        let mut is_idx = 0usize; // scales 索引
        let mut q8_offset = 0usize; // q8 偏移

        for k in 0..2 {
            let q2_base = k * 32; // k=0: qs[0..31], k=1: qs[32..63]
            let mut shift = 0u32;

            for _j in 0..4 {
                // 第一个子块：q2[0..15] >> shift
                let sc_val = (sc[is_idx] & 0xF) as i32;
                is_idx += 1;
                let mut isuml = 0i32;
                for l in 0..16 {
                    let q2_val = ((qs_blk[q2_base + l] >> shift) & 3) as i32;
                    isuml += q2_val * q8_blk[q8_offset + l] as i32;
                }
                isum += sc_val * isuml;
                q8_offset += 16;

                // 第二个子块：q2[16..31] >> shift
                let sc_val = (sc[is_idx] & 0xF) as i32;
                is_idx += 1;
                let mut isuml = 0i32;
                for l in 0..16 {
                    let q2_val = ((qs_blk[q2_base + 16 + l] >> shift) & 3) as i32;
                    isuml += q2_val * q8_blk[q8_offset + l] as i32;
                }
                isum += sc_val * isuml;
                q8_offset += 16;

                shift += 2;
            }
        }

        // result = q8_d * (q2_d * isum - q2_dmin * summs)
        sumf += q8_inv_scales[blk] * (d_val * isum as f32 - dmin_val * summs as f32);
    }

    sumf
}

/// Q2_K AVX2 点积内核（AVX-512 优先，回退标量）
pub fn q2k_vec_dot_q8_avx2(
    d: &[f32],
    dmin: &[f32],
    scales: &[u8],
    qs: &[u8],
    q8: &[i8],
    q8_inv_scales: &[f32],
    n_blocks: usize,
) -> f32 {
    if is_avx512_supported() {
        unsafe { q2k_vec_dot_q8_amd7600(d, dmin, scales, qs, q8, q8_inv_scales, n_blocks) }
    } else {
        q2k_vec_dot_q8(d, dmin, scales, qs, q8, q8_inv_scales, n_blocks)
    }
}

/// AVX-512 Q2_K × Q8 向量点积（优化版）
///
/// 优化策略：
/// - summs: 256-bit 批量处理（一次 2 个子块），8 次替代 16 次
/// - isum: scale 融入 madd + 累加后 1 次全局 hsum（替代 8 次 hsum）
/// - _mm_prefetch 预取下一块权重数据
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn q2k_vec_dot_q8_amd7600(
    d: &[f32],
    dmin: &[f32],
    scales: &[u8],
    qs: &[u8],
    q8: &[i8],
    q8_inv_scales: &[f32],
    n_blocks: usize,
) -> f32 {
    use std::arch::x86_64::*;

    let m3 = _mm_set1_epi8(3);
    let mut sumf = 0.0f32;

    for blk in 0..n_blocks {
        let d_val = d[blk];
        let dmin_val = dmin[blk];
        let sc_ptr = scales.as_ptr().add(blk * 16);
        let qs_ptr = qs.as_ptr().add(blk * 64);
        let q8_ptr = q8.as_ptr().add(blk * 256);

        // 预取下一块权重数据到 L2
        if blk + 1 < n_blocks {
            _mm_prefetch(scales.as_ptr().add((blk + 1) * 16) as *const i8, _MM_HINT_T1);
            _mm_prefetch(qs.as_ptr().add((blk + 1) * 64) as *const i8, _MM_HINT_T1);
        }

        // ===== summs: min 贡献 =====
        // 256-bit 批量处理：一次处理 2 个子块（32 个 q8 值）
        // maddubs(1, q8_256) → 16 i16 → madd(1) → 8 i32
        // 低 4 个 i32 = 子块 0 的和，高 4 个 = 子块 1 的和
        let mut summs = 0i32;
        for j in (0..16).step_by(2) {
            // 加载 2×16 q8 值
            let q8_lo = _mm_loadu_si128(q8_ptr.add(j * 16) as *const __m128i);
            let q8_hi = _mm_loadu_si128(q8_ptr.add((j + 1) * 16) as *const __m128i);
            let q8_256 = _mm256_inserti128_si256(
                _mm256_castsi128_si256(q8_lo), q8_hi, 1);

            // maddubs(1, q8) + madd(1) = sum of 32 q8 values
            let dot16 = _mm256_maddubs_epi16(_mm256_set1_epi8(1), q8_256);
            let sum32 = _mm256_madd_epi16(dot16, _mm256_set1_epi16(1));

            // 分别提取低半和高半的和
            let lo = _mm256_castsi256_si128(sum32);
            let hi = _mm256_extractf128_si256(sum32, 1);
            let s1 = _mm_shuffle_epi32(lo, 0xB1);
            let s2 = _mm_add_epi32(lo, s1);
            let s3 = _mm_shuffle_epi32(s2, 0x4E);
            let s4 = _mm_add_epi32(s2, s3);
            let bsum0 = _mm_cvtsi128_si32(s4);
            let s5 = _mm_shuffle_epi32(hi, 0xB1);
            let s6 = _mm_add_epi32(hi, s5);
            let s7 = _mm_shuffle_epi32(s6, 0x4E);
            let s8 = _mm_add_epi32(s6, s7);
            let bsum1 = _mm_cvtsi128_si32(s8);

            summs += bsum0 * ((*sc_ptr.add(j) >> 4) as i32);
            summs += bsum1 * ((*sc_ptr.add(j + 1) >> 4) as i32);
        }

        // ===== isum: scale 贡献 =====
        // 优化：累加 madd 结果，最后只做 1 次全局 hsum
        let mut isum = 0i32;
        let mut is_idx = 0usize;
        let mut q8_offset = 0usize;

        for k in 0..2 {
            let q2_base = k * 32;
            let mut shift = 0u32;

            for _j in 0..4 {
                let sc_val1 = (*sc_ptr.add(is_idx) & 0xF) as i16;
                is_idx += 1;
                let sc_val2 = (*sc_ptr.add(is_idx) & 0xF) as i16;
                is_idx += 1;

                // 加载 2×16 qs 字节
                let qs_lo = _mm_loadu_si128(qs_ptr.add(q2_base) as *const __m128i);
                let qs_hi = _mm_loadu_si128(qs_ptr.add(q2_base + 16) as *const __m128i);

                // 解包 2-bit 值: (byte >> shift) & 3
                let shift_vec = _mm_cvtsi32_si128(shift as i32);
                let q2_lo = _mm_and_si128(_mm_srl_epi16(qs_lo, shift_vec), m3);
                let q2_hi = _mm_and_si128(_mm_srl_epi16(qs_hi, shift_vec), m3);

                // 加载 2×16 q8 值
                let q8_1 = _mm_loadu_si128(q8_ptr.add(q8_offset) as *const __m128i);
                q8_offset += 16;
                let q8_2 = _mm_loadu_si128(q8_ptr.add(q8_offset) as *const __m128i);
                q8_offset += 16;

                // 256-bit maddubs: 合并 2 个 128-bit 子块
                let q2_256 = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(q2_lo), q2_hi, 1);
                let q8_256 = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(q8_1), q8_2, 1);

                // 256-bit maddubs: 32 u8×i8 → 16 i16
                let dot_256 = _mm256_maddubs_epi16(q2_256, q8_256);

                // scale 融入 madd
                let sc_256 = _mm256_set_epi16(
                    sc_val2, sc_val2, sc_val2, sc_val2, sc_val2, sc_val2, sc_val2, sc_val2,
                    sc_val1, sc_val1, sc_val1, sc_val1, sc_val1, sc_val1, sc_val1, sc_val1,
                );
                let sum_256 = _mm256_madd_epi16(dot_256, sc_256);

                // 累加到标量（避免每轮 hsum）
                let lo = _mm256_castsi256_si128(sum_256);
                let hi = _mm256_extractf128_si256(sum_256, 1);
                let s1 = _mm_shuffle_epi32(lo, 0xB1);
                let s2 = _mm_add_epi32(lo, s1);
                let s3 = _mm_shuffle_epi32(s2, 0x4E);
                let s4 = _mm_add_epi32(s2, s3);
                isum += _mm_cvtsi128_si32(s4);
                let s5 = _mm_shuffle_epi32(hi, 0xB1);
                let s6 = _mm_add_epi32(hi, s5);
                let s7 = _mm_shuffle_epi32(s6, 0x4E);
                let s8 = _mm_add_epi32(s6, s7);
                isum += _mm_cvtsi128_si32(s8);

                shift += 2;
            }
        }

        sumf += q8_inv_scales[blk] * (d_val * isum as f32 - dmin_val * summs as f32);
    }

    sumf
}

/// Q2_K 单线程顺序扫描矩阵向量乘法
///
/// 与 iq2xxs_matvec_blocked_amd7600 类似的优化策略：
/// 单线程顺序扫描权重，最大化 DDR burst 效率。
/// 用于 6 专家并行场景，避免 rayon 线程池争抢。
///
/// 内联 AVX-512 指令，复用 q2k_vec_dot_q8_amd7600 的优化逻辑
#[cfg(target_arch = "x86_64")]
pub unsafe fn q2k_matvec_blocked_amd7600(
    d: &[f32],
    dmin: &[f32],
    scales: &[u8],
    qs: &[u8],
    x: &[f32],
    n_blocks_per_row: usize,
    row_start: usize,
    row_end: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    // Q8 预量化 x（所有行共享）
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

    let m3 = _mm_set1_epi8(3);

    for row in row_start..row_end {
        // 预取下一行权重到 L2
        if row + 1 < row_end {
            let next_qs_ptr = qs.as_ptr().add((row + 1) * n_blocks_per_row * 64);
            _mm_prefetch(next_qs_ptr as *const i8, _MM_HINT_T1);
            let next_scales_ptr = scales.as_ptr().add((row + 1) * n_blocks_per_row * 16);
            _mm_prefetch(next_scales_ptr as *const i8, _MM_HINT_T1);
        }

        let row_offset = row * n_blocks_per_row;
        let d_row = &d[row_offset..row_offset + n_blocks_per_row];
        let dmin_row = &dmin[row_offset..row_offset + n_blocks_per_row];
        let scales_row = &scales[row_offset * 16..(row_offset + n_blocks_per_row) * 16];
        let qs_row = &qs[row_offset * 64..(row_offset + n_blocks_per_row) * 64];

        let mut sumf = 0.0f32;

        for blk in 0..n_blocks_per_row {
            let d_val = d_row[blk];
            let dmin_val = dmin_row[blk];
            let sc_ptr = scales_row.as_ptr().add(blk * 16);
            let qs_ptr = qs_row.as_ptr().add(blk * 64);
            let q8_ptr = q8_buf.as_ptr().add(blk * 256);

            // 预取下一个 block 的 qs 和 scales
            if blk + 1 < n_blocks_per_row {
                _mm_prefetch(qs_row.as_ptr().add((blk + 1) * 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(scales_row.as_ptr().add((blk + 1) * 16) as *const i8, _MM_HINT_T0);
            }

            // ===== summs: min 贡献 =====
            let mut summs = 0i32;
            for j in (0..16).step_by(2) {
                let q8_lo = _mm_loadu_si128(q8_ptr.add(j * 16) as *const __m128i);
                let q8_hi = _mm_loadu_si128(q8_ptr.add((j + 1) * 16) as *const __m128i);
                let q8_256 = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(q8_lo), q8_hi, 1);

                let dot16 = _mm256_maddubs_epi16(_mm256_set1_epi8(1), q8_256);
                let sum32 = _mm256_madd_epi16(dot16, _mm256_set1_epi16(1));

                let lo = _mm256_castsi256_si128(sum32);
                let hi = _mm256_extractf128_si256(sum32, 1);
                let s1 = _mm_shuffle_epi32(lo, 0xB1);
                let s2 = _mm_add_epi32(lo, s1);
                let s3 = _mm_shuffle_epi32(s2, 0x4E);
                let s4 = _mm_add_epi32(s2, s3);
                let bsum0 = _mm_cvtsi128_si32(s4);
                let s5 = _mm_shuffle_epi32(hi, 0xB1);
                let s6 = _mm_add_epi32(hi, s5);
                let s7 = _mm_shuffle_epi32(s6, 0x4E);
                let s8 = _mm_add_epi32(s6, s7);
                let bsum1 = _mm_cvtsi128_si32(s8);

                summs += bsum0 * ((*sc_ptr.add(j) >> 4) as i32);
                summs += bsum1 * ((*sc_ptr.add(j + 1) >> 4) as i32);
            }

            // ===== isum: scale 贡献 =====
            let mut isum = 0i32;
            let mut is_idx = 0usize;
            let mut q8_offset = 0usize;

            for k in 0..2 {
                let q2_base = k * 32;
                let mut shift = 0u32;

                for _j in 0..4 {
                    let sc_val1 = (*sc_ptr.add(is_idx) & 0xF) as i16;
                    is_idx += 1;
                    let sc_val2 = (*sc_ptr.add(is_idx) & 0xF) as i16;
                    is_idx += 1;

                    let qs_lo = _mm_loadu_si128(qs_ptr.add(q2_base) as *const __m128i);
                    let qs_hi = _mm_loadu_si128(qs_ptr.add(q2_base + 16) as *const __m128i);

                    let shift_vec = _mm_cvtsi32_si128(shift as i32);
                    let q2_lo = _mm_and_si128(_mm_srl_epi16(qs_lo, shift_vec), m3);
                    let q2_hi = _mm_and_si128(_mm_srl_epi16(qs_hi, shift_vec), m3);

                    let q8_1 = _mm_loadu_si128(q8_ptr.add(q8_offset) as *const __m128i);
                    q8_offset += 16;
                    let q8_2 = _mm_loadu_si128(q8_ptr.add(q8_offset) as *const __m128i);
                    q8_offset += 16;

                    let q2_256 = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(q2_lo), q2_hi, 1);
                    let q8_256 = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(q8_1), q8_2, 1);

                    let dot_256 = _mm256_maddubs_epi16(q2_256, q8_256);

                    let sc_256 = _mm256_set_epi16(
                        sc_val2, sc_val2, sc_val2, sc_val2, sc_val2, sc_val2, sc_val2, sc_val2,
                        sc_val1, sc_val1, sc_val1, sc_val1, sc_val1, sc_val1, sc_val1, sc_val1,
                    );
                    let sum_256 = _mm256_madd_epi16(dot_256, sc_256);

                    let lo = _mm256_castsi256_si128(sum_256);
                    let hi = _mm256_extractf128_si256(sum_256, 1);
                    let s1 = _mm_shuffle_epi32(lo, 0xB1);
                    let s2 = _mm_add_epi32(lo, s1);
                    let s3 = _mm_shuffle_epi32(s2, 0x4E);
                    let s4 = _mm_add_epi32(s2, s3);
                    isum += _mm_cvtsi128_si32(s4);
                    let s5 = _mm_shuffle_epi32(hi, 0xB1);
                    let s6 = _mm_add_epi32(hi, s5);
                    let s7 = _mm_shuffle_epi32(s6, 0x4E);
                    let s8 = _mm_add_epi32(s6, s7);
                    isum += _mm_cvtsi128_si32(s8);

                    shift += 2;
                }
            }

            sumf += q8_inv_scales[blk] * (d_val * isum as f32 - dmin_val * summs as f32);
        }

        output[row - row_start] = sumf;
    }
}

/// AVX-512 IQ2_XS × F32 向量点积（单行）
///
/// 量化 F32→Q8 后调用 512-bit 内核。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn iq2xs_vec_dot_avx512_vnni(
    d: &[f32],
    qs: &[u16],
    scales: &[u8],
    x: &[f32],
    n_blocks: usize,
) -> f32 {
    let mut q8_buf = vec![0i8; n_blocks * 256];
    let mut q8_inv_scales = vec![0.0f32; n_blocks];
    for blk in 0..n_blocks {
        q8_inv_scales[blk] = quantize_f32_to_q8_block(
            &x[blk * 256..(blk + 1) * 256],
            &mut q8_buf[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    iq2xs_vec_dot_q8_avx512(d, qs, scales, &q8_buf, &q8_inv_scales, n_blocks)
}

/// AVX-512 配对 gate/up 点积
///
/// 共享 Q8 量化：x 量化一次，gate 和 up 各查一次权重表。
/// 512-bit maddubs+madd 全程。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn iq2xs_pair_dot_avx512_vnni(
    d_gate: &[f32],
    qs_gate: &[u16],
    scales_gate: &[u8],
    d_up: &[f32],
    qs_up: &[u16],
    scales_up: &[u8],
    x: &[f32],
    n_blocks: usize,
) -> (f32, f32) {
    // 共享 Q8 量化：一次量化，两次使用
    let mut q8_buf = vec![0i8; n_blocks * 256];
    let mut q8_inv_scales = vec![0.0f32; n_blocks];
    for blk in 0..n_blocks {
        q8_inv_scales[blk] = quantize_f32_to_q8_block(
            &x[blk * 256..(blk + 1) * 256],
            &mut q8_buf[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    let gate = iq2xs_vec_dot_q8_avx512(d_gate, qs_gate, scales_gate, &q8_buf, &q8_inv_scales, n_blocks);
    let up = iq2xs_vec_dot_q8_avx512(d_up, qs_up, scales_up, &q8_buf, &q8_inv_scales, n_blocks);

    (gate, up)
}

// ============================================================================
// AVX2 路径
// ============================================================================

/// AVX2 IQ2_XS × F32 向量点积（单行）
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn iq2xs_vec_dot_avx2(
    d: &[f32],
    qs: &[u16],
    scales: &[u8],
    x: &[f32],
    n_blocks: usize,
) -> f32 {
    let mut q8_buf = vec![0i8; n_blocks * 256];
    let mut q8_inv_scales = vec![0.0f32; n_blocks];
    for blk in 0..n_blocks {
        q8_inv_scales[blk] = quantize_f32_to_q8_block(
            &x[blk * 256..(blk + 1) * 256],
            &mut q8_buf[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    iq2xs_vec_dot_q8_avx2(d, qs, scales, &q8_buf, &q8_inv_scales, n_blocks)
}

/// AVX2 IQ2_XS × Q8 向量点积（内部，使用预量化激活）
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn iq2xs_vec_dot_q8_avx2(
    d: &[f32],
    qs: &[u16],
    scales: &[u8],
    q8: &[i8],
    q8_inv_scales: &[f32],
    n_blocks: usize,
) -> f32 {
    use std::arch::x86_64::*;

    let grid_u64 = super::tables::get_iq2xs_grid_u64();
    let grid_u64_ptr = grid_u64.as_ptr();
    let k_scale_shuffle = &super::tables::K_SCALE_SHUFFLE;

    let mone = _mm256_set1_epi8(1);
    let m511 = _mm256_set1_epi16(511);
    let m4 = _mm_set1_epi8(0xf);
    let m1 = _mm_set1_epi8(1);

    // 符号计算常量（模块级共享）
    let sign_shuffle_1 = _mm256_loadu_si256(BLOCK_SIGN_SHUFFLE_1.as_ptr() as *const __m256i);
    let sign_shuffle_2 = _mm256_loadu_si256(BLOCK_SIGN_SHUFFLE_2.as_ptr() as *const __m256i);
    let bit_selector_mask = _mm256_loadu_si256(BIT_SELECTOR_MASK.as_ptr() as *const __m256i);
    let bit_helper = _mm256_loadu_si256(K_BIT_HELPER.as_ptr() as *const __m256i);

    let mut accumf = _mm256_setzero_ps();

    for blk in 0..n_blocks {
        let d_val = d[blk] * q8_inv_scales[blk];
        let qs_ptr = qs.as_ptr().add(blk * 32);
        let q8_ptr = q8.as_ptr().add(blk * 256);

        if blk + 1 < n_blocks {
            _mm_prefetch(qs.as_ptr().add((blk + 1) * 32) as *const i8, _MM_HINT_T1);
            _mm_prefetch(q8.as_ptr().add((blk + 1) * 256) as *const i8, _MM_HINT_T1);
        }

        let mut aux64: u64 = 0;
        for b in 0..8 {
            aux64 |= (scales[blk * 8 + b] as u64) << (b * 8);
        }
        let mut stmp = _mm_set1_epi64x(aux64 as i64);
        stmp = _mm_unpacklo_epi8(
            _mm_and_si128(stmp, m4),
            _mm_and_si128(_mm_srli_epi16(stmp, 4), m4),
        );
        let scales_vec = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        let mut sumi1 = _mm256_setzero_si256();
        let mut sumi2 = _mm256_setzero_si256();

        for ib32 in (0..8).step_by(4) {
            let q2_data = _mm256_loadu_si256(qs_ptr.add(ib32 * 4) as *const __m256i);
            let aux_gindex = _mm256_and_si256(q2_data, m511);
            let mut gidx = [0u16; 16];
            _mm256_storeu_si256(gidx.as_mut_ptr() as *mut __m256i, aux_gindex);

            let partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            let partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            let partial_sign_bits_for_counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);
            let odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            let full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);

            let q8_1 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32) as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32 + 32) as *const __m256i);
            let q8_3 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32 + 64) as *const __m256i);
            let q8_4 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32 + 96) as *const __m256i);

            let q2_1 = _mm256_set_epi64x(
                *grid_u64_ptr.add(gidx[3] as usize) as i64,
                *grid_u64_ptr.add(gidx[2] as usize) as i64,
                *grid_u64_ptr.add(gidx[1] as usize) as i64,
                *grid_u64_ptr.add(gidx[0] as usize) as i64,
            );
            let q2_2 = _mm256_set_epi64x(
                *grid_u64_ptr.add(gidx[7] as usize) as i64,
                *grid_u64_ptr.add(gidx[6] as usize) as i64,
                *grid_u64_ptr.add(gidx[5] as usize) as i64,
                *grid_u64_ptr.add(gidx[4] as usize) as i64,
            );
            let q2_3 = _mm256_set_epi64x(
                *grid_u64_ptr.add(gidx[11] as usize) as i64,
                *grid_u64_ptr.add(gidx[10] as usize) as i64,
                *grid_u64_ptr.add(gidx[9] as usize) as i64,
                *grid_u64_ptr.add(gidx[8] as usize) as i64,
            );
            let q2_4 = _mm256_set_epi64x(
                *grid_u64_ptr.add(gidx[15] as usize) as i64,
                *grid_u64_ptr.add(gidx[14] as usize) as i64,
                *grid_u64_ptr.add(gidx[13] as usize) as i64,
                *grid_u64_ptr.add(gidx[12] as usize) as i64,
            );

            let full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            let full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            let full_signs_1 = _mm256_insertf128_si256(
                _mm256_castsi128_si256(full_signs_l), full_signs_l, 1);
            let full_signs_2 = _mm256_insertf128_si256(
                _mm256_castsi128_si256(full_signs_h), full_signs_h, 1);

            let mut signs;
            signs = _mm256_shuffle_epi8(full_signs_1, sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            let q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_1, sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            let q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            let q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            let q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));

            let dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            let dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            let dot3 = _mm256_maddubs_epi16(q2_3, q8s_3);
            let dot4 = _mm256_maddubs_epi16(q2_4, q8s_4);

            let sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales_vec,
                _mm_loadu_si128(k_scale_shuffle.as_ptr().add((ib32 + 0) * 16) as *const __m128i)));
            let sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales_vec,
                _mm_loadu_si128(k_scale_shuffle.as_ptr().add((ib32 + 1) * 16) as *const __m128i)));
            let sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales_vec,
                _mm_loadu_si128(k_scale_shuffle.as_ptr().add((ib32 + 2) * 16) as *const __m128i)));
            let sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales_vec,
                _mm_loadu_si128(k_scale_shuffle.as_ptr().add((ib32 + 3) * 16) as *const __m128i)));

            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot3, sc3));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot4, sc4));
        }

        accumf = _mm256_fmadd_ps(
            _mm256_set1_ps(d_val),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
            accumf,
        );
    }

    0.125 * hsum_float_8(accumf)
}

/// AVX2 配对 gate/up 点积
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn iq2xs_pair_dot_avx2(
    d_gate: &[f32],
    qs_gate: &[u16],
    scales_gate: &[u8],
    d_up: &[f32],
    qs_up: &[u16],
    scales_up: &[u8],
    x: &[f32],
    n_blocks: usize,
) -> (f32, f32) {
    let mut q8_buf = vec![0i8; n_blocks * 256];
    let mut q8_inv_scales = vec![0.0f32; n_blocks];
    for blk in 0..n_blocks {
        q8_inv_scales[blk] = quantize_f32_to_q8_block(
            &x[blk * 256..(blk + 1) * 256],
            &mut q8_buf[blk * 256..(blk + 1) * 256],
            256,
        );
    }

    let gate = iq2xs_vec_dot_q8_avx2(d_gate, qs_gate, scales_gate, &q8_buf, &q8_inv_scales, n_blocks);
    let up = iq2xs_vec_dot_q8_avx2(d_up, qs_up, scales_up, &q8_buf, &q8_inv_scales, n_blocks);

    (gate, up)
}

// ============================================================================
// 标量回退
// ============================================================================

pub fn iq2xs_vec_dot_scalar(
    d: &[f32], qs: &[u16], scales: &[u8], x: &[f32], n_blocks: usize,
) -> f32 {
    let grid = super::tables::get_iq2xs_grid();
    let sign_table = super::tables::get_sign_mul_table();
    let scale_table = super::tables::get_scale_decode_table();

    let mut row_sum = 0.0f32;
    for blk in 0..n_blocks {
        let d_val = d[blk];
        let mut block_sum = 0.0f32;
        for g in 0..32 {
            let q = qs[blk * 32 + g];
            let gi = (q & 511) as usize;
            let si = ((q >> 9) & 127) as usize;
            let ib32 = g >> 2;
            let within = g & 3;
            let sc_val = scales[blk * 8 + ib32];
            let ls = if within < 2 { scale_table[sc_val as usize * 2] } else { scale_table[sc_val as usize * 2 + 1] };
            let x_base = blk * 256 + g * 8;
            let mut group_dot = 0.0f32;
            for j in 0..8 { group_dot += grid[gi * 8 + j] as f32 * sign_table[si * 8 + j] * x[x_base + j]; }
            block_sum += ls * group_dot;
        }
        row_sum += d_val * 0.125 * block_sum;
    }
    row_sum
}

pub fn iq2xs_pair_dot_scalar(
    d_gate: &[f32], qs_gate: &[u16], scales_gate: &[u8],
    d_up: &[f32], qs_up: &[u16], scales_up: &[u8],
    x: &[f32], n_blocks: usize,
) -> (f32, f32) {
    let grid = super::tables::get_iq2xs_grid();
    let sign_table = super::tables::get_sign_mul_table();
    let scale_table = super::tables::get_scale_decode_table();

    let mut gate_sum = 0.0f32;
    let mut up_sum = 0.0f32;

    for blk in 0..n_blocks {
        let x_base = blk * 256;

        let d_g = d_gate[blk];
        let mut blk_gate = 0.0f32;
        for g in 0..32 {
            let q = qs_gate[blk * 32 + g];
            let gi = (q & 511) as usize;
            let si = ((q >> 9) & 127) as usize;
            let ib32 = g >> 2;
            let within = g & 3;
            let sc_val = scales_gate[blk * 8 + ib32];
            let ls = if within < 2 { scale_table[sc_val as usize * 2] } else { scale_table[sc_val as usize * 2 + 1] };
            let mut dot = 0.0f32;
            for j in 0..8 { dot += grid[gi * 8 + j] as f32 * sign_table[si * 8 + j] * x[x_base + g * 8 + j]; }
            blk_gate += ls * dot;
        }
        gate_sum += d_g * 0.125 * blk_gate;

        let d_u = d_up[blk];
        let mut blk_up = 0.0f32;
        for g in 0..32 {
            let q = qs_up[blk * 32 + g];
            let gi = (q & 511) as usize;
            let si = ((q >> 9) & 127) as usize;
            let ib32 = g >> 2;
            let within = g & 3;
            let sc_val = scales_up[blk * 8 + ib32];
            let ls = if within < 2 { scale_table[sc_val as usize * 2] } else { scale_table[sc_val as usize * 2 + 1] };
            let mut dot = 0.0f32;
            for j in 0..8 { dot += grid[gi * 8 + j] as f32 * sign_table[si * 8 + j] * x[x_base + g * 8 + j]; }
            blk_up += ls * dot;
        }
        up_sum += d_u * 0.125 * blk_up;
    }

    (gate_sum, up_sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_float_8(x: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(x, 1);
    let lo = _mm256_castps256_ps128(x);
    let mut sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    sum128 = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(shuf, sum128);
    sum128 = _mm_add_ss(sum128, shuf);
    _mm_cvtss_f32(sum128)
}

// ============================================================================
// AVX-512 Tile 直读路径: 直接从 80B Tile 块读取 d/qs/scales
// ============================================================================

/// AVX-512 IQ2_XS × Q8 向量点积（直接从 Tile 布局读取）
///
/// 与 iq2xs_vec_dot_q8_avx512 功能相同，但直接从 80B Tile 块读取 d/qs/scales，
/// 避免解包到中间缓冲区。Tile 布局确保 d+qs+scales 在同一缓存行附近，
/// 减少 cache miss。
///
/// 参数：
/// - tile_data: Tile 布局的字节数组
/// - n_blocks: block 数量
/// - q8: 预量化的 int8 激活
/// - q8_inv_scales: 预量化的逆缩放因子
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn iq2xs_vec_dot_q8_tile_avx512(
    tile_data: &[u8],
    n_blocks: usize,
    q8: &[i8],
    q8_inv_scales: &[f32],
) -> f32 {
    use std::arch::x86_64::*;

    let grid_u64 = super::tables::get_iq2xs_grid_u64();
    let grid_u64_ptr = grid_u64.as_ptr();
    let k_scale_shuffle = &super::tables::K_SCALE_SHUFFLE;

    let m4 = _mm_set1_epi8(0xf);
    let m1 = _mm_set1_epi8(1);
    let m511 = _mm256_set1_epi16(511);
    let mone = _mm256_set1_epi8(1);

    // 符号计算常量（模块级共享）
    let bit_helper = _mm256_loadu_si256(K_BIT_HELPER.as_ptr() as *const __m256i);
    let sign_shuffle_1 = _mm256_loadu_si256(BLOCK_SIGN_SHUFFLE_1.as_ptr() as *const __m256i);
    let sign_shuffle_2 = _mm256_loadu_si256(BLOCK_SIGN_SHUFFLE_2.as_ptr() as *const __m256i);
    let bit_selector_mask = _mm256_loadu_si256(BIT_SELECTOR_MASK.as_ptr() as *const __m256i);

    let mut accumf = _mm512_setzero_ps();

    for blk in 0..n_blocks {
        let tile_offset = blk * 80; // TILE_SIZE = 80

        // 直接从 Tile 读取 d (4 字节 f32)
        let d_bytes: [u8; 4] = [
            tile_data[tile_offset], tile_data[tile_offset + 1],
            tile_data[tile_offset + 2], tile_data[tile_offset + 3],
        ];
        let d_val = f32::from_le_bytes(d_bytes) * q8_inv_scales[blk];

        // 直接从 Tile 读取 qs (32 个 u16, 偏移 4)
        let qs_ptr = tile_data.as_ptr().add(tile_offset + 4);
        let q8_ptr = q8.as_ptr().add(blk * 256);

        // 预取下一块 Tile 数据到 L2（80 字节跨 2 个缓存行，需预取两行）
        if blk + 1 < n_blocks {
            _mm_prefetch(tile_data.as_ptr().add((blk + 1) * 80) as *const i8, _MM_HINT_T1);
            _mm_prefetch(tile_data.as_ptr().add((blk + 1) * 80 + 64) as *const i8, _MM_HINT_T1);
        }

        // 直接从 Tile 读取 scales (8 个 u8, 偏移 68)
        let scales_ptr = tile_data.as_ptr().add(tile_offset + 68);
        let mut aux64: u64 = 0;
        for b in 0..8 {
            aux64 |= (*scales_ptr.add(b) as u64) << (b * 8);
        }
        let mut stmp = _mm_set1_epi64x(aux64 as i64);
        stmp = _mm_unpacklo_epi8(
            _mm_and_si128(stmp, m4),
            _mm_and_si128(_mm_srli_epi16(stmp, 4), m4),
        );
        let scales_vec = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        let mut sumi_512 = _mm512_setzero_si512();

        for ib32 in (0..8).step_by(4) {
            // 加载 16 个 u16 qs 值（直接从 Tile 偏移 4 处读取）
            // 注意：qs_ptr 是 *const u8，每个 u16 占 2 字节，所以偏移 = ib32 * 4 * sizeof(u16) = ib32 * 8
            let q2_data = _mm256_loadu_si256(qs_ptr.add(ib32 * 8) as *const __m256i);
            let aux_gindex = _mm256_and_si256(q2_data, m511);
            let mut gidx = [0u16; 16];
            _mm256_storeu_si256(gidx.as_mut_ptr() as *mut __m256i, aux_gindex);

            // 符号计算
            let partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            let partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            let partial_sign_bits_for_counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);
            let odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            let full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);

            let q8_1 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32) as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32 + 32) as *const __m256i);
            let q8_3 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32 + 64) as *const __m256i);
            let q8_4 = _mm256_loadu_si256(q8_ptr.add(ib32 * 32 + 96) as *const __m256i);

            let q2_1 = _mm256_set_epi64x(
                *grid_u64_ptr.add(gidx[3] as usize) as i64,
                *grid_u64_ptr.add(gidx[2] as usize) as i64,
                *grid_u64_ptr.add(gidx[1] as usize) as i64,
                *grid_u64_ptr.add(gidx[0] as usize) as i64,
            );
            let q2_2 = _mm256_set_epi64x(
                *grid_u64_ptr.add(gidx[7] as usize) as i64,
                *grid_u64_ptr.add(gidx[6] as usize) as i64,
                *grid_u64_ptr.add(gidx[5] as usize) as i64,
                *grid_u64_ptr.add(gidx[4] as usize) as i64,
            );
            let q2_3 = _mm256_set_epi64x(
                *grid_u64_ptr.add(gidx[11] as usize) as i64,
                *grid_u64_ptr.add(gidx[10] as usize) as i64,
                *grid_u64_ptr.add(gidx[9] as usize) as i64,
                *grid_u64_ptr.add(gidx[8] as usize) as i64,
            );
            let q2_4 = _mm256_set_epi64x(
                *grid_u64_ptr.add(gidx[15] as usize) as i64,
                *grid_u64_ptr.add(gidx[14] as usize) as i64,
                *grid_u64_ptr.add(gidx[13] as usize) as i64,
                *grid_u64_ptr.add(gidx[12] as usize) as i64,
            );

            let full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            let full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            let full_signs_1 = _mm256_insertf128_si256(
                _mm256_castsi128_si256(full_signs_l), full_signs_l, 1);
            let full_signs_2 = _mm256_insertf128_si256(
                _mm256_castsi128_si256(full_signs_h), full_signs_h, 1);

            let mut signs;
            signs = _mm256_shuffle_epi8(full_signs_1, sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            let q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_1, sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            let q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            let q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            let q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));

            // 512-bit maddubs + madd
            let grid_512_a = _mm512_inserti64x4(_mm512_castsi256_si512(q2_1), q2_2, 1);
            let grid_512_b = _mm512_inserti64x4(_mm512_castsi256_si512(q2_3), q2_4, 1);
            let q8s_512_a = _mm512_inserti64x4(_mm512_castsi256_si512(q8s_1), q8s_2, 1);
            let q8s_512_b = _mm512_inserti64x4(_mm512_castsi256_si512(q8s_3), q8s_4, 1);

            let dot_512_a = _mm512_maddubs_epi16(grid_512_a, q8s_512_a);
            let dot_512_b = _mm512_maddubs_epi16(grid_512_b, q8s_512_b);

            let sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales_vec,
                _mm_loadu_si128(k_scale_shuffle.as_ptr().add((ib32 + 0) * 16) as *const __m128i)));
            let sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales_vec,
                _mm_loadu_si128(k_scale_shuffle.as_ptr().add((ib32 + 1) * 16) as *const __m128i)));
            let sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales_vec,
                _mm_loadu_si128(k_scale_shuffle.as_ptr().add((ib32 + 2) * 16) as *const __m128i)));
            let sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales_vec,
                _mm_loadu_si128(k_scale_shuffle.as_ptr().add((ib32 + 3) * 16) as *const __m128i)));

            let sc_512_a = _mm512_inserti64x4(_mm512_castsi256_si512(sc1), sc2, 1);
            let sc_512_b = _mm512_inserti64x4(_mm512_castsi256_si512(sc3), sc4, 1);

            let prod_512_a = _mm512_madd_epi16(dot_512_a, sc_512_a);
            let prod_512_b = _mm512_madd_epi16(dot_512_b, sc_512_b);

            sumi_512 = _mm512_add_epi32(sumi_512, prod_512_a);
            sumi_512 = _mm512_add_epi32(sumi_512, prod_512_b);
        }

        accumf = _mm512_fmadd_ps(
            _mm512_set1_ps(d_val),
            _mm512_cvtepi32_ps(sumi_512),
            accumf,
        );
    }

    0.125 * _mm512_reduce_add_ps(accumf)
}

// ============================================================
// FP4 (e2m1 + E8M0) amd7600 专用内核
// ============================================================

/// E8M0 scale 转 f32（内联版本，避免函数调用开销）
#[inline(always)]
fn e8m0_to_f32_inline(bits: u8) -> f32 {
    if bits == 0 {
        0.0f32
    } else {
        f32::from_bits((bits as u32) << 23)
    }
}

/// FP4 单线程顺序扫描矩阵向量乘法（amd7600 专用，x_split 版）
///
/// 优化策略：
///   1. x_split 预拆分：将 x 拆分为 x_even/x_odd，lo 权重直接与 x_even 对齐
///   2. 512-bit FMA + permutexvar_ps LUT 查表（16 个 FP4 值 → f32）
///   3. 4 路累加器减少 FMA 依赖链延迟
///   4. 循环展开 2 次 + 软件预取
///   5. 单线程顺序扫描权重，最大化 DDR burst 效率
#[cfg(target_arch = "x86_64")]
pub unsafe fn fp4_matvec_blocked_amd7600(
    weight_packed: &[u8],
    scale: &[u8],
    x_even: &[f32],
    x_odd: &[f32],
    n_packed_per_row: usize,
    n_scales_per_row: usize,
    row_start: usize,
    row_end: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    let lut = _mm512_set_ps(
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
         6.0,  4.0,  3.0,  2.0,  1.5,  1.0,  0.5, 0.0,
    );

    const BLOCK_SIZE: usize = 16;

    for row in row_start..row_end {
        if row + 1 < row_end {
            let next_packed = weight_packed.as_ptr().add((row + 1) * n_packed_per_row);
            _mm_prefetch(next_packed as *const i8, _MM_HINT_T1);
        }

        let row_packed_start = row * n_packed_per_row;
        let row_scale_start = row * n_scales_per_row;
        let n_full_blocks = n_packed_per_row / BLOCK_SIZE;

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        let mut block = 0usize;

        while block + 1 < n_full_blocks {
            let base0 = row_packed_start + block * BLOCK_SIZE;
            let scale_bits_0 = scale[row_scale_start + block];
            let scale0 = if scale_bits_0 == 0 {
                _mm512_set1_ps(0.0f32)
            } else {
                _mm512_set1_ps(e8m0_to_f32_inline(scale_bits_0))
            };

            if block + 2 < n_full_blocks {
                let next_base = row_packed_start + (block + 2) * BLOCK_SIZE;
                _mm_prefetch(weight_packed.as_ptr().add(next_base) as *const i8, _MM_HINT_T1);
            }

            let packed0 = _mm_loadu_si128(weight_packed.as_ptr().add(base0) as *const __m128i);
            let lo0 = _mm_and_si128(packed0, _mm_set1_epi8(0xF));
            let hi0 = _mm_srli_epi16(_mm_and_si128(packed0, _mm_set1_epi8(0xF0u8 as i8)), 4);

            let lo_f32_0 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(lo0), lut), scale0);
            let hi_f32_0 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(hi0), lut), scale0);

            let x_even_0 = _mm512_loadu_ps(x_even.as_ptr().add(block * 16));
            let x_odd_0 = _mm512_loadu_ps(x_odd.as_ptr().add(block * 16));

            acc0 = _mm512_fmadd_ps(lo_f32_0, x_even_0, acc0);
            acc1 = _mm512_fmadd_ps(hi_f32_0, x_odd_0, acc1);

            let base1 = row_packed_start + (block + 1) * BLOCK_SIZE;
            let scale_bits_1 = scale[row_scale_start + block + 1];
            let scale1 = if scale_bits_1 == 0 {
                _mm512_set1_ps(0.0f32)
            } else {
                _mm512_set1_ps(e8m0_to_f32_inline(scale_bits_1))
            };

            let packed1 = _mm_loadu_si128(weight_packed.as_ptr().add(base1) as *const __m128i);
            let lo1 = _mm_and_si128(packed1, _mm_set1_epi8(0xF));
            let hi1 = _mm_srli_epi16(_mm_and_si128(packed1, _mm_set1_epi8(0xF0u8 as i8)), 4);

            let lo_f32_1 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(lo1), lut), scale1);
            let hi_f32_1 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(hi1), lut), scale1);

            let x_even_1 = _mm512_loadu_ps(x_even.as_ptr().add((block + 1) * 16));
            let x_odd_1 = _mm512_loadu_ps(x_odd.as_ptr().add((block + 1) * 16));

            acc2 = _mm512_fmadd_ps(lo_f32_1, x_even_1, acc2);
            acc3 = _mm512_fmadd_ps(hi_f32_1, x_odd_1, acc3);

            block += 2;
        }

        if block < n_full_blocks {
            let base = row_packed_start + block * BLOCK_SIZE;
            let scale_bits = scale[row_scale_start + block];
            let scale_val = if scale_bits == 0 {
                _mm512_set1_ps(0.0f32)
            } else {
                _mm512_set1_ps(e8m0_to_f32_inline(scale_bits))
            };

            let packed = _mm_loadu_si128(weight_packed.as_ptr().add(base) as *const __m128i);
            let lo = _mm_and_si128(packed, _mm_set1_epi8(0xF));
            let hi = _mm_srli_epi16(_mm_and_si128(packed, _mm_set1_epi8(0xF0u8 as i8)), 4);

            let lo_f32 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(lo), lut), scale_val);
            let hi_f32 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(hi), lut), scale_val);

            let x_even_v = _mm512_loadu_ps(x_even.as_ptr().add(block * 16));
            let x_odd_v = _mm512_loadu_ps(x_odd.as_ptr().add(block * 16));

            acc0 = _mm512_fmadd_ps(lo_f32, x_even_v, acc0);
            acc1 = _mm512_fmadd_ps(hi_f32, x_odd_v, acc1);
        }

        // 尾部
        let mut tail_sum = 0.0f32;
        let tail_start = n_full_blocks * BLOCK_SIZE;
        for i in tail_start..n_packed_per_row {
            let packed_byte = weight_packed[row_packed_start + i];
            let lo_val = crate::cpu_expert::kernel::Fp4Weight::decode_fp4(packed_byte & 0xF);
            let hi_val = crate::cpu_expert::kernel::Fp4Weight::decode_fp4((packed_byte >> 4) & 0xF);
            let scale_lo = e8m0_to_f32_inline(scale[row_scale_start + (i * 2) / 32]);
            let scale_hi = e8m0_to_f32_inline(scale[row_scale_start + (i * 2 + 1) / 32]);
            tail_sum += lo_val * scale_lo * x_even[i];
            tail_sum += hi_val * scale_hi * x_odd[i];
        }

        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc0 = _mm512_add_ps(acc0, acc2);

        output[row - row_start] = _mm512_reduce_add_ps(acc0) + tail_sum;
    }
}

/// FP4 单线程顺序扫描矩阵向量乘法（amd7600 专用，非 x_split 版）
///
/// 用于 down 投影（mid 不需要 x_split 预拆分）
/// 使用 permutex2var_ps 交错合并 lo/hi 权重
#[cfg(target_arch = "x86_64")]
pub unsafe fn fp4_matvec_blocked_amd7600_direct(
    weight_packed: &[u8],
    scale: &[u8],
    x: &[f32],
    n_packed_per_row: usize,
    n_scales_per_row: usize,
    row_start: usize,
    row_end: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    let lut = _mm512_set_ps(
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
         6.0,  4.0,  3.0,  2.0,  1.5,  1.0,  0.5, 0.0,
    );

    let merge_lo_idx = _mm512_set_epi32(
        23, 7, 22, 6, 21, 5, 20, 4,
        19, 3, 18, 2, 17, 1, 16, 0
    );
    let merge_hi_idx = _mm512_set_epi32(
        31, 15, 30, 14, 29, 13, 28, 12,
        27, 11, 26, 10, 25, 9, 24, 8
    );

    const BLOCK_SIZE: usize = 16;

    for row in row_start..row_end {
        if row + 1 < row_end {
            let next_packed = weight_packed.as_ptr().add((row + 1) * n_packed_per_row);
            _mm_prefetch(next_packed as *const i8, _MM_HINT_T1);
        }

        let row_packed_start = row * n_packed_per_row;
        let row_scale_start = row * n_scales_per_row;
        let n_full_blocks = n_packed_per_row / BLOCK_SIZE;

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        let mut block = 0usize;

        while block + 1 < n_full_blocks {
            let base0 = row_packed_start + block * BLOCK_SIZE;
            let scale_bits_0 = scale[row_scale_start + block];
            let scale0 = if scale_bits_0 == 0 {
                _mm512_set1_ps(0.0f32)
            } else {
                _mm512_set1_ps(e8m0_to_f32_inline(scale_bits_0))
            };

            if block + 2 < n_full_blocks {
                let next_base = row_packed_start + (block + 2) * BLOCK_SIZE;
                _mm_prefetch(weight_packed.as_ptr().add(next_base) as *const i8, _MM_HINT_T1);
            }

            let packed0 = _mm_loadu_si128(weight_packed.as_ptr().add(base0) as *const __m128i);
            let lo0 = _mm_and_si128(packed0, _mm_set1_epi8(0xF));
            let hi0 = _mm_srli_epi16(_mm_and_si128(packed0, _mm_set1_epi8(0xF0u8 as i8)), 4);

            let lo_f32_0 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(lo0), lut), scale0);
            let hi_f32_0 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(hi0), lut), scale0);

            let w_first_0 = _mm512_permutex2var_ps(lo_f32_0, merge_lo_idx, hi_f32_0);
            let w_second_0 = _mm512_permutex2var_ps(lo_f32_0, merge_hi_idx, hi_f32_0);

            let x_base0 = block * BLOCK_SIZE * 2;
            let x0_0 = _mm512_loadu_ps(x.as_ptr().add(x_base0));
            let x0_1 = _mm512_loadu_ps(x.as_ptr().add(x_base0 + 16));

            acc0 = _mm512_fmadd_ps(w_first_0, x0_0, acc0);
            acc1 = _mm512_fmadd_ps(w_second_0, x0_1, acc1);

            let base1 = row_packed_start + (block + 1) * BLOCK_SIZE;
            let scale_bits_1 = scale[row_scale_start + block + 1];
            let scale1 = if scale_bits_1 == 0 {
                _mm512_set1_ps(0.0f32)
            } else {
                _mm512_set1_ps(e8m0_to_f32_inline(scale_bits_1))
            };

            let packed1 = _mm_loadu_si128(weight_packed.as_ptr().add(base1) as *const __m128i);
            let lo1 = _mm_and_si128(packed1, _mm_set1_epi8(0xF));
            let hi1 = _mm_srli_epi16(_mm_and_si128(packed1, _mm_set1_epi8(0xF0u8 as i8)), 4);

            let lo_f32_1 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(lo1), lut), scale1);
            let hi_f32_1 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(hi1), lut), scale1);

            let w_first_1 = _mm512_permutex2var_ps(lo_f32_1, merge_lo_idx, hi_f32_1);
            let w_second_1 = _mm512_permutex2var_ps(lo_f32_1, merge_hi_idx, hi_f32_1);

            let x_base1 = (block + 1) * BLOCK_SIZE * 2;
            let x1_0 = _mm512_loadu_ps(x.as_ptr().add(x_base1));
            let x1_1 = _mm512_loadu_ps(x.as_ptr().add(x_base1 + 16));

            acc2 = _mm512_fmadd_ps(w_first_1, x1_0, acc2);
            acc3 = _mm512_fmadd_ps(w_second_1, x1_1, acc3);

            block += 2;
        }

        if block < n_full_blocks {
            let base = row_packed_start + block * BLOCK_SIZE;
            let scale_bits = scale[row_scale_start + block];
            let scale_val = if scale_bits == 0 {
                _mm512_set1_ps(0.0f32)
            } else {
                _mm512_set1_ps(e8m0_to_f32_inline(scale_bits))
            };

            let packed = _mm_loadu_si128(weight_packed.as_ptr().add(base) as *const __m128i);
            let lo = _mm_and_si128(packed, _mm_set1_epi8(0xF));
            let hi = _mm_srli_epi16(_mm_and_si128(packed, _mm_set1_epi8(0xF0u8 as i8)), 4);

            let lo_f32 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(lo), lut), scale_val);
            let hi_f32 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepi8_epi32(hi), lut), scale_val);

            let w_first = _mm512_permutex2var_ps(lo_f32, merge_lo_idx, hi_f32);
            let w_second = _mm512_permutex2var_ps(lo_f32, merge_hi_idx, hi_f32);

            let x_base = block * BLOCK_SIZE * 2;
            let x0 = _mm512_loadu_ps(x.as_ptr().add(x_base));
            let x1 = _mm512_loadu_ps(x.as_ptr().add(x_base + 16));

            acc0 = _mm512_fmadd_ps(w_first, x0, acc0);
            acc1 = _mm512_fmadd_ps(w_second, x1, acc1);
        }

        // 尾部
        let mut tail_sum = 0.0f32;
        let tail_start = n_full_blocks * BLOCK_SIZE;
        for i in tail_start..n_packed_per_row {
            let packed_byte = weight_packed[row_packed_start + i];
            let lo_val = crate::cpu_expert::kernel::Fp4Weight::decode_fp4(packed_byte & 0xF);
            let hi_val = crate::cpu_expert::kernel::Fp4Weight::decode_fp4((packed_byte >> 4) & 0xF);
            let col_lo = i * 2;
            let col_hi = i * 2 + 1;
            let scale_lo = e8m0_to_f32_inline(scale[row_scale_start + col_lo / 32]);
            let scale_hi = e8m0_to_f32_inline(scale[row_scale_start + col_hi / 32]);
            tail_sum += lo_val * scale_lo * x[col_lo];
            tail_sum += hi_val * scale_hi * x[col_hi];
        }

        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc0 = _mm512_add_ps(acc0, acc2);

        output[row - row_start] = _mm512_reduce_add_ps(acc0) + tail_sum;
    }
}

// ============================================================
// FP4+VNNI 内核（VPDPBUSD 替代 FMA 管线）
// ============================================================

/// FP4 e2m1 → int8 幅度 LUT（值 × 2 以保持整数）
/// nibble 0-7: {0, 1, 2, 3, 4, 6, 8, 12}（正数幅度）
/// nibble 8-15: 同上（符号由 sign LUT 单独处理）
///
/// 参考 llama.cpp kvalues_mxfp4: {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}
/// 区别：我们分离幅度和符号，因为 VPDPBUSD 要求第一操作数为 uint8
const KVALUES_MXFP4_ABS: [u8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12];

/// FP4 符号 LUT：nibble 0-7 为正（+1），nibble 8-15 为负（-1）
/// sign_epi8(a, b): b>0 → a, b<0 → -a, b==0 → 0
/// 因此正数用 +1（保持），负数用 -1（取反）
const KSIGN_MXFP4: [i8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1];

/// 将 x_even + x_odd 量化为 Q8_0 格式（32 元素/块，FP4 权重对齐布局）
///
/// 输出布局: q8[0:15] = Q8(x_even), q8[16:31] = Q8(x_odd)
/// 与 FP4 权重布局对齐: lo nibbles → q8[0:15], hi nibbles → q8[16:31]
///
/// 返回 (q8_buf, q8_scales)，q8_scales[i] = 第 i 块的逆缩放因子
pub fn quantize_f32_to_q8_fp4_split(
    x_even: &[f32],
    x_odd: &[f32],
    n_blocks: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut q8 = vec![0i8; n_blocks * 32];
    let mut scales = vec![0.0f32; n_blocks];

    for blk in 0..n_blocks {
        let mut amax = 0.0f32;
        for i in 0..16 {
            amax = amax.max(x_even[blk * 16 + i].abs());
            amax = amax.max(x_odd[blk * 16 + i].abs());
        }
        let inv_scale = if amax > 1e-6 { 127.0 / amax } else { 0.0 };
        scales[blk] = if amax > 1e-6 { amax / 127.0 } else { 0.0 };

        for i in 0..16 {
            q8[blk * 32 + i] = (x_even[blk * 16 + i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            q8[blk * 32 + 16 + i] = (x_odd[blk * 16 + i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    (q8, scales)
}

/// 将 x 量化为 Q8_0 格式（32 元素/块），同时重排为 FP4 权重对齐的顺序
///
/// 输出布局: q8[0:15] = Q8(x[0,2,4,...,30]), q8[16:31] = Q8(x[1,3,5,...,31])
/// 用于 down 投影（mid 不需要预拆分，量化时直接重排）
pub fn quantize_f32_to_q8_fp4_direct(
    x: &[f32],
    n_blocks: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut q8 = vec![0i8; n_blocks * 32];
    let mut scales = vec![0.0f32; n_blocks];

    for blk in 0..n_blocks {
        let base = blk * 32;
        let mut amax = 0.0f32;
        for i in 0..32 {
            amax = amax.max(x[base + i].abs());
        }
        let inv_scale = if amax > 1e-6 { 127.0 / amax } else { 0.0 };
        scales[blk] = if amax > 1e-6 { amax / 127.0 } else { 0.0 };

        for i in 0..16 {
            q8[blk * 32 + i] = (x[base + i * 2] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            q8[blk * 32 + 16 + i] = (x[base + i * 2 + 1] * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    (q8, scales)
}

/// FP4+VNNI 单线程顺序扫描矩阵向量乘法（amd7600 专用，x_split 版）
///
/// 优化策略：
///   1. FP4 nibble → uint8 幅度 + sign 分离（shuffle_epi8 LUT）
///   2. sign_epi8(q8, sign_mask): 将权重符号应用到 Q8 激活
///   3. VPDPBUSD(uint8_magnitude, int8_signed_activation) → int32
///   4. cvtepi32_ps + fmadd_ps: SIMD 浮点累加（避免逐块 hsum）
///   5. 2× 循环展开 + 双累加器
///
/// Q8 量化由调用方预计算（FFN 函数级别），避免 per-chunk 重复量化
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni")]
pub unsafe fn fp4_matvec_vnni_amd7600(
    weight_packed: &[u8],
    scale: &[u8],
    q8_buf: &[i8],
    q8_scales: &[f32],
    n_packed_per_row: usize,
    n_scales_per_row: usize,
    row_start: usize,
    row_end: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    let values128 = _mm_loadu_si128(KVALUES_MXFP4_ABS.as_ptr() as *const __m128i);
    let signs128 = _mm_loadu_si128(KSIGN_MXFP4.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0xF);

    const BLOCK_SIZE: usize = 16; // 16 packed bytes = 32 FP4 elements
    let n_blocks = n_packed_per_row / BLOCK_SIZE;

    for row in row_start..row_end {
        if row + 1 < row_end {
            let next_packed = weight_packed.as_ptr().add((row + 1) * n_packed_per_row);
            _mm_prefetch(next_packed as *const i8, _MM_HINT_T1);
        }

        let row_packed_start = row * n_packed_per_row;
        let row_scale_start = row * n_scales_per_row;

        // 双累加器 + SIMD 浮点累加（避免逐块 hsum）
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        // 2× 循环展开
        let n_full = n_blocks & !1;
        for block in (0..n_full).step_by(2) {
            // --- Block 0 ---
            let base0 = row_packed_start + block * BLOCK_SIZE;
            let sc0 = _mm256_set1_ps(e8m0_to_f32_inline(scale[row_scale_start + block]) * q8_scales[block] * 0.5);

            let packed0 = _mm_loadu_si128(weight_packed.as_ptr().add(base0) as *const __m128i);
            let lo0 = _mm_and_si128(packed0, m4b);
            let hi0 = _mm_and_si128(_mm_srli_epi16(packed0, 4), m4b);
            let lo_mag0 = _mm_shuffle_epi8(values128, lo0);
            let hi_mag0 = _mm_shuffle_epi8(values128, hi0);
            let lo_sign0 = _mm_shuffle_epi8(signs128, lo0);
            let hi_sign0 = _mm_shuffle_epi8(signs128, hi0);
            let w0 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_mag0), hi_mag0, 1);
            let sgn0 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_sign0), hi_sign0, 1);
            let q8_0 = _mm256_loadu_si256(q8_buf.as_ptr().add(block * 32) as *const __m256i);
            let q8s_0 = _mm256_sign_epi8(q8_0, sgn0);
            let dot0 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w0, q8s_0);
            acc0 = _mm256_fmadd_ps(sc0, _mm256_cvtepi32_ps(dot0), acc0);

            // --- Block 1 ---
            let base1 = row_packed_start + (block + 1) * BLOCK_SIZE;
            let sc1 = _mm256_set1_ps(e8m0_to_f32_inline(scale[row_scale_start + block + 1]) * q8_scales[block + 1] * 0.5);

            let packed1 = _mm_loadu_si128(weight_packed.as_ptr().add(base1) as *const __m128i);
            let lo1 = _mm_and_si128(packed1, m4b);
            let hi1 = _mm_and_si128(_mm_srli_epi16(packed1, 4), m4b);
            let lo_mag1 = _mm_shuffle_epi8(values128, lo1);
            let hi_mag1 = _mm_shuffle_epi8(values128, hi1);
            let lo_sign1 = _mm_shuffle_epi8(signs128, lo1);
            let hi_sign1 = _mm_shuffle_epi8(signs128, hi1);
            let w1 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_mag1), hi_mag1, 1);
            let sgn1 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_sign1), hi_sign1, 1);
            let q8_1 = _mm256_loadu_si256(q8_buf.as_ptr().add((block + 1) * 32) as *const __m256i);
            let q8s_1 = _mm256_sign_epi8(q8_1, sgn1);
            let dot1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w1, q8s_1);
            acc1 = _mm256_fmadd_ps(sc1, _mm256_cvtepi32_ps(dot1), acc1);
        }

        // 处理奇数块
        if n_blocks % 2 != 0 {
            let block = n_blocks - 1;
            let base = row_packed_start + block * BLOCK_SIZE;
            let sc = _mm256_set1_ps(e8m0_to_f32_inline(scale[row_scale_start + block]) * q8_scales[block] * 0.5);
            let packed = _mm_loadu_si128(weight_packed.as_ptr().add(base) as *const __m128i);
            let lo = _mm_and_si128(packed, m4b);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), m4b);
            let lo_mag = _mm_shuffle_epi8(values128, lo);
            let hi_mag = _mm_shuffle_epi8(values128, hi);
            let lo_sign = _mm_shuffle_epi8(signs128, lo);
            let hi_sign = _mm_shuffle_epi8(signs128, hi);
            let w = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_mag), hi_mag, 1);
            let sgn = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_sign), hi_sign, 1);
            let q8 = _mm256_loadu_si256(q8_buf.as_ptr().add(block * 32) as *const __m256i);
            let q8s = _mm256_sign_epi8(q8, sgn);
            let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, q8s);
            acc0 = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(dot), acc0);
        }

        // 最终 hsum
        let sum256 = _mm256_add_ps(acc0, acc1);
        output[row - row_start] = hsum_ps256(sum256);
    }
}

/// __m256 8×f32 水平求和
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_ps256(x: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extractf128_ps(x, 1);
    let lo128 = _mm256_castps256_ps128(x);
    let sum128 = _mm_add_ps(hi128, lo128);
    let s1 = _mm_movehl_ps(sum128, sum128);
    let s2 = _mm_add_ps(sum128, s1);
    let s3 = _mm_shuffle_ps(s2, s2, 0x01);
    let s4 = _mm_add_ss(s2, s3);
    _mm_cvtss_f32(s4)
}

/// FP4+VNNI 单线程顺序扫描矩阵向量乘法（amd7600 专用，direct 版）
///
/// 用于 down 投影：x 不预拆分，Q8 量化时直接重排为 FP4 权重对齐顺序
/// q8[0:15] = Q8(x[0,2,...,30]), q8[16:31] = Q8(x[1,3,...,31])
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni")]
pub unsafe fn fp4_matvec_vnni_amd7600_direct(
    weight_packed: &[u8],
    scale: &[u8],
    q8_buf: &[i8],
    q8_scales: &[f32],
    n_packed_per_row: usize,
    n_scales_per_row: usize,
    row_start: usize,
    row_end: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    let values128 = _mm_loadu_si128(KVALUES_MXFP4_ABS.as_ptr() as *const __m128i);
    let signs128 = _mm_loadu_si128(KSIGN_MXFP4.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0xF);

    const BLOCK_SIZE: usize = 16;
    let n_blocks = n_packed_per_row / BLOCK_SIZE;

    for row in row_start..row_end {
        if row + 1 < row_end {
            let next_packed = weight_packed.as_ptr().add((row + 1) * n_packed_per_row);
            _mm_prefetch(next_packed as *const i8, _MM_HINT_T1);
        }

        let row_packed_start = row * n_packed_per_row;
        let row_scale_start = row * n_scales_per_row;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let n_full = n_blocks & !1;
        for block in (0..n_full).step_by(2) {
            let base0 = row_packed_start + block * BLOCK_SIZE;
            let sc0 = _mm256_set1_ps(e8m0_to_f32_inline(scale[row_scale_start + block]) * q8_scales[block] * 0.5);

            let packed0 = _mm_loadu_si128(weight_packed.as_ptr().add(base0) as *const __m128i);
            let lo0 = _mm_and_si128(packed0, m4b);
            let hi0 = _mm_and_si128(_mm_srli_epi16(packed0, 4), m4b);
            let lo_mag0 = _mm_shuffle_epi8(values128, lo0);
            let hi_mag0 = _mm_shuffle_epi8(values128, hi0);
            let lo_sign0 = _mm_shuffle_epi8(signs128, lo0);
            let hi_sign0 = _mm_shuffle_epi8(signs128, hi0);
            let w0 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_mag0), hi_mag0, 1);
            let sgn0 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_sign0), hi_sign0, 1);
            let q8_0 = _mm256_loadu_si256(q8_buf.as_ptr().add(block * 32) as *const __m256i);
            let q8s_0 = _mm256_sign_epi8(q8_0, sgn0);
            let dot0 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w0, q8s_0);
            acc0 = _mm256_fmadd_ps(sc0, _mm256_cvtepi32_ps(dot0), acc0);

            let base1 = row_packed_start + (block + 1) * BLOCK_SIZE;
            let sc1 = _mm256_set1_ps(e8m0_to_f32_inline(scale[row_scale_start + block + 1]) * q8_scales[block + 1] * 0.5);

            let packed1 = _mm_loadu_si128(weight_packed.as_ptr().add(base1) as *const __m128i);
            let lo1 = _mm_and_si128(packed1, m4b);
            let hi1 = _mm_and_si128(_mm_srli_epi16(packed1, 4), m4b);
            let lo_mag1 = _mm_shuffle_epi8(values128, lo1);
            let hi_mag1 = _mm_shuffle_epi8(values128, hi1);
            let lo_sign1 = _mm_shuffle_epi8(signs128, lo1);
            let hi_sign1 = _mm_shuffle_epi8(signs128, hi1);
            let w1 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_mag1), hi_mag1, 1);
            let sgn1 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_sign1), hi_sign1, 1);
            let q8_1 = _mm256_loadu_si256(q8_buf.as_ptr().add((block + 1) * 32) as *const __m256i);
            let q8s_1 = _mm256_sign_epi8(q8_1, sgn1);
            let dot1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w1, q8s_1);
            acc1 = _mm256_fmadd_ps(sc1, _mm256_cvtepi32_ps(dot1), acc1);
        }

        if n_blocks % 2 != 0 {
            let block = n_blocks - 1;
            let base = row_packed_start + block * BLOCK_SIZE;
            let sc = _mm256_set1_ps(e8m0_to_f32_inline(scale[row_scale_start + block]) * q8_scales[block] * 0.5);
            let packed = _mm_loadu_si128(weight_packed.as_ptr().add(base) as *const __m128i);
            let lo = _mm_and_si128(packed, m4b);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), m4b);
            let lo_mag = _mm_shuffle_epi8(values128, lo);
            let hi_mag = _mm_shuffle_epi8(values128, hi);
            let lo_sign = _mm_shuffle_epi8(signs128, lo);
            let hi_sign = _mm_shuffle_epi8(signs128, hi);
            let w = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_mag), hi_mag, 1);
            let sgn = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_sign), hi_sign, 1);
            let q8 = _mm256_loadu_si256(q8_buf.as_ptr().add(block * 32) as *const __m256i);
            let q8s = _mm256_sign_epi8(q8, sgn);
            let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, q8s);
            acc0 = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(dot), acc0);
        }

        let sum256 = _mm256_add_ps(acc0, acc1);
        output[row - row_start] = hsum_ps256(sum256);
    }
}
