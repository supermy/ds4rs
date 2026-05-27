/// MoE 路由逻辑 SIMD 优化
///
/// 实现流程：
///   1. 对 GEMM 输出的原始分数应用 score_func（AVX-512/AVX2 向量化）
///   2. 加偏置 bias
///   3. TopK 选择（部分排序，只需前 topk 个）
///   4. 归一化权重 + 乘 route_scale
///   5. rayon 并行处理多个 token
///
/// 性能目标：单 token 路由 < 10μs（n_experts=256, topk=6）
///
/// 优化要点：
///   - AVX-512 向量化 score_func 计算，16 个 f32 并行处理
///   - AVX2 回退路径，8 个 f32 并行处理
///   - TopK 使用选择排序而非全排序，避免 O(n log n) 开销
///   - exp 近似：多项式逼近（6 阶 Taylor），精度 < 1e-6
///   - rayon 并行处理多 token，单 token 路径约 3 次小量堆分配

use rayon::prelude::*;
use std::cell::RefCell;

// ============================================================================
// 线程局部路由 buffer（消除每 token 的堆分配）
// ============================================================================

/// 线程局部路由 buffer，消除每 token 的堆分配
///
/// 预分配 n_experts 大小的 buffer，路由时复用。
/// 典型 n_experts=256，每 buffer 约 1KB，3 个 buffer 共 3KB。
struct RouteBuffer {
    /// score_func 输出缓冲
    activated: Vec<f32>,
    /// 原始激活值备份
    original: Vec<f32>,
    /// TopK 选择排序临时缓冲
    topk_temp: Vec<f32>,
    /// TopK 权重输出
    topk_weights: Vec<f32>,
    /// TopK 索引输出
    topk_indices: Vec<i32>,
}

impl RouteBuffer {
    fn new(n_experts: usize, topk: usize) -> Self {
        Self {
            activated: vec![0.0f32; n_experts],
            original: vec![0.0f32; n_experts],
            topk_temp: vec![0.0f32; n_experts],
            topk_weights: vec![0.0f32; topk],
            topk_indices: vec![0i32; topk],
        }
    }

    fn resize(&mut self, n_experts: usize, topk: usize) {
        if self.activated.len() != n_experts {
            self.activated.resize(n_experts, 0.0);
            self.original.resize(n_experts, 0.0);
            self.topk_temp.resize(n_experts, 0.0);
        }
        if self.topk_weights.len() != topk {
            self.topk_weights.resize(topk, 0.0);
            self.topk_indices.resize(topk, 0);
        }
    }
}

thread_local! {
    static ROUTE_BUF: RefCell<RouteBuffer> = RefCell::new(RouteBuffer::new(256, 8));
}

// ============================================================================
// 数据结构
// ============================================================================

/// 激活函数类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoreFunc {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}

impl ScoreFunc {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "softmax" => ScoreFunc::Softmax,
            "sigmoid" => ScoreFunc::Sigmoid,
            _ => ScoreFunc::SqrtSoftplus,
        }
    }
}

/// 路由输入
pub struct RouteInput<'a> {
    /// GEMM 输出的原始分数，shape [num_tokens, n_experts]
    pub scores: &'a [f32],
    /// 偏置，shape [n_experts]
    pub bias: Option<&'a [f32]>,
    /// 专家数量
    pub n_experts: usize,
    /// TopK 值
    pub topk: usize,
    /// 激活函数
    pub score_func: ScoreFunc,
    /// 路由缩放因子
    pub route_scale: f32,
}

/// 路由输出
pub struct RouteOutput {
    /// 权重，shape [num_tokens, topk]
    pub weights: Vec<f32>,
    /// 索引，shape [num_tokens, topk]
    pub indices: Vec<i32>,
}

// ============================================================================
// exp 近似：6 阶 Taylor 多项式
// ============================================================================

/// exp(x) 多项式近似，适用于 x ∈ [-88, 88]
///
/// 使用范围缩减 + 多项式逼近：
///   exp(x) = 2^k * exp(r)，其中 k = round(x/ln2)，r = x - k*ln2
///   exp(r) 用 6 阶多项式逼近
///
/// 精度：典型相对误差 < 1e-6，极端输入下最大误差 ~2e-5
#[inline]
fn exp_approx_scalar(mut x: f32) -> f32 {
    const LN2: f32 = 0.6931471805599453;
    const INV_LN2: f32 = 1.4426950408889634;

    if x < -87.0 {
        return 0.0;
    }
    if x > 88.0 {
        // 钳制到 88.0 后计算 exp(88)，与 SIMD 路径行为一致
        // exp(88) ≈ 1.65e38，远小于 f32::MAX ≈ 3.4e38
        x = 88.0;
    }

    // 范围缩减
    let k = (x * INV_LN2).round() as i32;
    let r = x - k as f32 * LN2;

    // 6 阶多项式逼近 exp(r)，r ∈ [-0.5*ln2, 0.5*ln2]
    // 系数为 Taylor 级数系数 (1/n!)
    // 使用 Horner 法则计算，避免中间变量
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.1666666666666668
                    + r * (0.04166666666666679
                        + r * (0.008333333333333357
                            + r * 0.0013888888888888905)))));

    // 2^k * p，用位操作实现快速 2^k
    let k_i32 = k + 127;
    if k_i32 < 0 || k_i32 > 254 {
        return if k > 0 { f32::MAX } else { 0.0 };
    }
    let scale = f32::from_bits((k_i32 as u32) << 23);
    p * scale
}

// ============================================================================
// 标量路径（回退 + 公共逻辑）
// ============================================================================

/// 对单个 token 的分数应用 score_func
#[inline]
fn apply_score_func_scalar(scores: &[f32], out: &mut [f32], n: usize, func: ScoreFunc) {
    match func {
        ScoreFunc::Sigmoid => {
            for i in 0..n {
                out[i] = 1.0 / (1.0 + exp_approx_scalar(-scores[i]));
            }
        }
        ScoreFunc::SqrtSoftplus => {
            for i in 0..n {
                let s = scores[i];
                out[i] = if s > 20.0 {
                    s.sqrt()
                } else {
                    (1.0 + exp_approx_scalar(s)).ln().sqrt()
                };
            }
        }
        ScoreFunc::Softmax => {
            // softmax 先拷贝原始值，后面单独处理
            out[..n].copy_from_slice(&scores[..n]);
        }
    }
}

/// Softmax 归一化（标量路径）
#[inline]
fn softmax_normalize_scalar(row: &mut [f32], n: usize) {
    let max = row[..n].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row[..n].iter_mut() {
        *v = exp_approx_scalar(*v - max);
        sum += *v;
    }
    if sum > 0.0 {
        for v in row[..n].iter_mut() {
            *v /= sum;
        }
    }
}

/// TopK 选择排序：只找前 topk 个最大值的索引
///
/// 对于 topk=6 这种小值，选择排序 O(n*topk) 远优于全排序 O(n log n)
/// n=256, topk=6 时：1536 次比较 vs 2048 次（log2(256)*256）
#[inline]
fn topk_select(activated: &[f32], n: usize, topk: usize) -> (Vec<f32>, Vec<i32>) {
    assert!(n > 0 && topk <= n, "topk_select: topk={} > n={}", topk, n);
    let mut weights = vec![0.0f32; topk];
    let mut indices = vec![0i32; topk];

    // 选择排序：每轮找最大值，找到后标记为 -inf
    let mut temp = activated[..n].to_vec();
    for k in 0..topk {
        let mut max_idx = 0;
        let mut max_val = temp[0];
        for i in 1..n {
            if temp[i] > max_val {
                max_val = temp[i];
                max_idx = i;
            }
        }
        weights[k] = max_val;
        indices[k] = max_idx as i32;
        temp[max_idx] = f32::NEG_INFINITY;
    }

    (weights, indices)
}

/// 原地 TopK 选择排序，使用预分配 buffer
#[inline]
fn topk_select_inplace(activated: &[f32], temp: &mut [f32], weights: &mut [f32], indices: &mut [i32], n: usize, topk: usize) {
    assert!(n > 0 && topk <= n);
    temp[..n].copy_from_slice(&activated[..n]);
    for k in 0..topk {
        let mut max_idx = 0;
        let mut max_val = temp[0];
        for i in 1..n {
            if temp[i] > max_val {
                max_val = temp[i];
                max_idx = i;
            }
        }
        weights[k] = max_val;
        indices[k] = max_idx as i32;
        temp[max_idx] = f32::NEG_INFINITY;
    }
}

/// 单 token 路由处理（标量路径）
fn route_single_token_scalar(
    scores_row: &[f32],
    bias: Option<&[f32]>,
    n_experts: usize,
    topk: usize,
    score_func: ScoreFunc,
    route_scale: f32,
    buf: &mut RouteBuffer,
) -> (Vec<f32>, Vec<i32>) {
    buf.resize(n_experts, topk);

    // 1. 应用 score_func → buf.activated
    apply_score_func_scalar(scores_row, &mut buf.activated, n_experts, score_func);

    // 2. Softmax 归一化
    if score_func == ScoreFunc::Softmax {
        softmax_normalize_scalar(&mut buf.activated, n_experts);
    }

    // 3. 备份原始激活值
    buf.original[..n_experts].copy_from_slice(&buf.activated[..n_experts]);

    // 4. 加偏置
    if let Some(bias) = bias {
        for j in 0..n_experts {
            buf.activated[j] += bias[j];
        }
    }

    // 5. TopK
    topk_select_inplace(&buf.activated, &mut buf.topk_temp, &mut buf.topk_weights, &mut buf.topk_indices, n_experts, topk);

    // 6. 用原始激活值替换权重
    for k in 0..topk {
        buf.topk_weights[k] = buf.original[buf.topk_indices[k] as usize];
    }

    // 7. 归一化
    if score_func != ScoreFunc::Softmax {
        let sum: f32 = buf.topk_weights[..topk].iter().sum();
        if sum > 0.0 {
            for w in buf.topk_weights[..topk].iter_mut() {
                *w /= sum;
            }
        }
    }

    // 8. 乘 route_scale
    for w in buf.topk_weights[..topk].iter_mut() {
        *w *= route_scale;
    }

    (buf.topk_weights[..topk].to_vec(), buf.topk_indices[..topk].to_vec())
}

// ============================================================================
// AVX-512 向量化路径
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod avx512_route {
    use super::*;

    /// AVX-512 exp 近似：16 个 f32 并行计算
    ///
    /// 范围缩减 + 多项式逼近，与标量版本相同算法
    /// 精度：典型相对误差 < 1e-6，极端输入下最大误差 ~2e-5
    #[target_feature(enable = "avx512f")]
    unsafe fn exp_approx_avx512(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;

        // 范围钳制：防止极端输入导致溢出或 NaN
        let x = _mm512_max_ps(x, _mm512_set1_ps(-87.0));
        let x = _mm512_min_ps(x, _mm512_set1_ps(88.0));

        const LN2: f32 = 0.6931471805599453;
        const INV_LN2: f32 = 1.4426950408889634;

        let ln2_vec = _mm512_set1_ps(LN2);
        let inv_ln2_vec = _mm512_set1_ps(INV_LN2);

        // k = round(x * 1/ln2)
        let k_f32 = _mm512_roundscale_ps(
            _mm512_mul_ps(x, inv_ln2_vec),
            0, // round to nearest
        );
        let k_i32 = _mm512_cvtps_epi32(k_f32);

        // r = x - k * ln2
        let r = _mm512_fnmadd_ps(k_f32, ln2_vec, x);

        // 6 阶多项式逼近 exp(r)
        let c0 = _mm512_set1_ps(1.0);
        let c1 = _mm512_set1_ps(1.0);
        let c2 = _mm512_set1_ps(0.5);
        let c3 = _mm512_set1_ps(0.1666666666666668);
        let c4 = _mm512_set1_ps(0.04166666666666679);
        let c5 = _mm512_set1_ps(0.008333333333333357);
        let c6 = _mm512_set1_ps(0.0013888888888888905);

        // Horner 法则：p = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*c6)))))
        let p = _mm512_fmadd_ps(
            r,
            _mm512_fmadd_ps(
                r,
                _mm512_fmadd_ps(
                    r,
                    _mm512_fmadd_ps(
                        r,
                        _mm512_fmadd_ps(
                            r,
                            _mm512_fmadd_ps(r, c6, c5),
                            c4,
                        ),
                        c3,
                    ),
                    c2,
                ),
                c1,
            ),
            c0,
        );

        // 2^k：将 k+127 移位到指数位
        let k_shifted = _mm512_add_epi32(k_i32, _mm512_set1_epi32(127));
        let scale = _mm512_castsi512_ps(_mm512_slli_epi32(k_shifted, 23));

        _mm512_mul_ps(p, scale)
    }

    /// AVX-512 Sigmoid：1 / (1 + exp(-x))
    #[target_feature(enable = "avx512f")]
    unsafe fn sigmoid_avx512(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;
        let neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
        let exp_neg = exp_approx_avx512(neg_x);
        _mm512_div_ps(
            _mm512_set1_ps(1.0),
            _mm512_add_ps(_mm512_set1_ps(1.0), exp_neg),
        )
    }

    /// AVX-512 SqrtSoftplus：sqrt(ln(1 + exp(x)))
    /// x > 20 时退化为 sqrt(x)
    #[target_feature(enable = "avx512f")]
    unsafe fn sqrt_softplus_avx512(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;
        let threshold = _mm512_set1_ps(20.0);
        // _MM_CMP_GT_OQ = 14 (x86 comparison predicate for >)
        let mask = _mm512_cmp_ps_mask(x, threshold, 14);

        // 大值路径：sqrt(x)
        let sqrt_x = _mm512_sqrt_ps(x);

        // 小值路径：sqrt(ln(1 + exp(x)))
        // AVX-512 没有 _mm512_log_ps（需要 SVML），标量回退 ln
        let exp_x = exp_approx_avx512(x);
        let one_plus_exp = _mm512_add_ps(_mm512_set1_ps(1.0), exp_x);
        // 标量 ln 回退（用 storeu 而非 transmute，避免对齐问题）
        let mut ope_arr = [0.0f32; 16];
        _mm512_storeu_ps(ope_arr.as_mut_ptr(), one_plus_exp);
        let mut ln_arr = [0.0f32; 16];
        for i in 0..16 {
            ln_arr[i] = ope_arr[i].ln();
        }
        let ln_vec = _mm512_loadu_ps(ln_arr.as_ptr());
        let sqrt_ln = _mm512_sqrt_ps(ln_vec);

        // 根据掩码选择
        _mm512_mask_blend_ps(mask, sqrt_ln, sqrt_x)
    }

    /// AVX-512 Softmax 归一化
    #[target_feature(enable = "avx512f")]
    unsafe fn softmax_normalize_avx512(row: &mut [f32], n: usize) {
        use std::arch::x86_64::*;

        // 求 max
        let mut max_val = f32::NEG_INFINITY;
        let mut i = 0;
        while i + 16 <= n {
            let v = _mm512_loadu_ps(row.as_ptr().add(i));
            let m = _mm512_reduce_max_ps(v);
            if m > max_val {
                max_val = m;
            }
            i += 16;
        }
        while i < n {
            if row[i] > max_val {
                max_val = row[i];
            }
            i += 1;
        }

        // exp(x - max) 并求和
        let max_vec = _mm512_set1_ps(max_val);
        let mut sum_vec = _mm512_setzero_ps();
        i = 0;
        while i + 16 <= n {
            let v = _mm512_loadu_ps(row.as_ptr().add(i));
            let shifted = _mm512_sub_ps(v, max_vec);
            let exp_v = exp_approx_avx512(shifted);
            _mm512_storeu_ps(row.as_mut_ptr().add(i), exp_v);
            sum_vec = _mm512_add_ps(sum_vec, exp_v);
            i += 16;
        }
        let mut sum = _mm512_reduce_add_ps(sum_vec);
        while i < n {
            let exp_v = exp_approx_scalar(row[i] - max_val);
            row[i] = exp_v;
            sum += exp_v;
            i += 1;
        }

        // 除以 sum
        if sum > 0.0 {
            let inv_sum = _mm512_set1_ps(1.0 / sum);
            i = 0;
            while i + 16 <= n {
                let v = _mm512_loadu_ps(row.as_ptr().add(i));
                let normalized = _mm512_mul_ps(v, inv_sum);
                _mm512_storeu_ps(row.as_mut_ptr().add(i), normalized);
                i += 16;
            }
            while i < n {
                row[i] /= sum;
                i += 1;
            }
        }
    }

    /// AVX-512 单 token 路由处理
    #[target_feature(enable = "avx512f")]
    unsafe fn route_single_token_avx512(
        scores_row: &[f32],
        bias: Option<&[f32]>,
        n_experts: usize,
        topk: usize,
        score_func: ScoreFunc,
        route_scale: f32,
        buf: &mut RouteBuffer,
    ) -> (Vec<f32>, Vec<i32>) {
        use std::arch::x86_64::*;

        buf.resize(n_experts, topk);

        // 1. 应用 score_func（AVX-512 向量化）
        match score_func {
            ScoreFunc::Sigmoid => {
                let mut i = 0;
                while i + 16 <= n_experts {
                    let v = _mm512_loadu_ps(scores_row.as_ptr().add(i));
                    let sig = sigmoid_avx512(v);
                    _mm512_storeu_ps(buf.activated.as_mut_ptr().add(i), sig);
                    i += 16;
                }
                // 尾部标量处理
                while i < n_experts {
                    buf.activated[i] = 1.0 / (1.0 + exp_approx_scalar(-scores_row[i]));
                    i += 1;
                }
            }
            ScoreFunc::SqrtSoftplus => {
                let mut i = 0;
                while i + 16 <= n_experts {
                    let v = _mm512_loadu_ps(scores_row.as_ptr().add(i));
                    let sp = sqrt_softplus_avx512(v);
                    _mm512_storeu_ps(buf.activated.as_mut_ptr().add(i), sp);
                    i += 16;
                }
                while i < n_experts {
                    let s = scores_row[i];
                    buf.activated[i] = if s > 20.0 {
                        s.sqrt()
                    } else {
                        (1.0 + exp_approx_scalar(s)).ln().sqrt()
                    };
                    i += 1;
                }
            }
            ScoreFunc::Softmax => {
                buf.activated[..n_experts].copy_from_slice(&scores_row[..n_experts]);
            }
        }

        // 2. Softmax 归一化
        if score_func == ScoreFunc::Softmax {
            softmax_normalize_avx512(&mut buf.activated, n_experts);
        }

        // 3. 备份原始激活值
        buf.original[..n_experts].copy_from_slice(&buf.activated[..n_experts]);

        // 4. 加偏置（AVX-512 向量化）
        if let Some(bias) = bias {
            let mut i = 0;
            while i + 16 <= n_experts {
                let v = _mm512_loadu_ps(buf.activated.as_ptr().add(i));
                let b = _mm512_loadu_ps(bias.as_ptr().add(i));
                let result = _mm512_add_ps(v, b);
                _mm512_storeu_ps(buf.activated.as_mut_ptr().add(i), result);
                i += 16;
            }
            while i < n_experts {
                buf.activated[i] += bias[i];
                i += 1;
            }
        }

        // 5. TopK 选择排序
        topk_select_inplace(&buf.activated, &mut buf.topk_temp, &mut buf.topk_weights, &mut buf.topk_indices, n_experts, topk);

        // 6. 用原始激活值替换权重
        for k in 0..topk {
            buf.topk_weights[k] = buf.original[buf.topk_indices[k] as usize];
        }

        // 7. 非 Softmax 时归一化
        if score_func != ScoreFunc::Softmax {
            let sum: f32 = buf.topk_weights[..topk].iter().sum();
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for w in buf.topk_weights[..topk].iter_mut() {
                    *w *= inv_sum;
                }
            }
        }

        // 8. 乘 route_scale（topk 通常很小，标量即可）
        for w in buf.topk_weights[..topk].iter_mut() {
            *w *= route_scale;
        }

        (buf.topk_weights[..topk].to_vec(), buf.topk_indices[..topk].to_vec())
    }

    /// AVX-512 路由主函数
    pub fn route_scores_avx512(input: &RouteInput) -> RouteOutput {
        let num_tokens = input.scores.len() / input.n_experts;
        let n_experts = input.n_experts;
        let topk = input.topk;
        let score_func = input.score_func;
        let route_scale = input.route_scale;
        let bias = input.bias;

        let results: Vec<(Vec<f32>, Vec<i32>)> = (0..num_tokens)
            .into_par_iter()
            .map(|t| {
                let row_start = t * n_experts;
                let scores_row = &input.scores[row_start..row_start + n_experts];
                ROUTE_BUF.with(|buf| {
                    let mut buf = buf.borrow_mut();
                    // 安全性：在运行时已检测 AVX-512 支持
                    unsafe {
                        route_single_token_avx512(
                            scores_row, bias, n_experts, topk, score_func, route_scale, &mut buf,
                        )
                    }
                })
            })
            .collect();

        let mut all_weights = Vec::with_capacity(num_tokens * topk);
        let mut all_indices = Vec::with_capacity(num_tokens * topk);
        for (w, idx) in results {
            all_weights.extend_from_slice(&w);
            all_indices.extend_from_slice(&idx);
        }

        RouteOutput {
            weights: all_weights,
            indices: all_indices,
        }
    }
}

// ============================================================================
// AVX2 向量化路径
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod avx2_route {
    use super::*;

    /// AVX2 exp 近似：8 个 f32 并行计算
    #[target_feature(enable = "avx2")]
    unsafe fn exp_approx_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;

        // 范围钳制：防止极端输入导致溢出或 NaN
        let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
        let x = _mm256_min_ps(x, _mm256_set1_ps(88.0));

        const LN2: f32 = 0.6931471805599453;
        const INV_LN2: f32 = 1.4426950408889634;

        let ln2_vec = _mm256_set1_ps(LN2);
        let inv_ln2_vec = _mm256_set1_ps(INV_LN2);

        // k = round(x * 1/ln2)
        let k_f32 = _mm256_round_ps(
            _mm256_mul_ps(x, inv_ln2_vec),
            0, // round to nearest
        );
        let k_i32 = _mm256_cvtps_epi32(k_f32);

        // r = x - k * ln2
        let r = _mm256_fnmadd_ps(k_f32, ln2_vec, x);

        // 6 阶多项式逼近
        let c0 = _mm256_set1_ps(1.0);
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(0.1666666666666668);
        let c4 = _mm256_set1_ps(0.04166666666666679);
        let c5 = _mm256_set1_ps(0.008333333333333357);
        let c6 = _mm256_set1_ps(0.0013888888888888905);

        let p = _mm256_fmadd_ps(
            r,
            _mm256_fmadd_ps(
                r,
                _mm256_fmadd_ps(
                    r,
                    _mm256_fmadd_ps(
                        r,
                        _mm256_fmadd_ps(
                            r,
                            _mm256_fmadd_ps(r, c6, c5),
                            c4,
                        ),
                        c3,
                    ),
                    c2,
                ),
                c1,
            ),
            c0,
        );

        // 2^k
        let k_shifted = _mm256_add_epi32(k_i32, _mm256_set1_epi32(127));
        let scale = _mm256_castsi256_ps(_mm256_slli_epi32(k_shifted, 23));

        _mm256_mul_ps(p, scale)
    }

    /// AVX2 水平求和：8 个 f32 → 1 个 f32
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hsum_float_8_avx2(x: std::arch::x86_64::__m256) -> f32 {
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

    /// AVX2 Sigmoid
    #[target_feature(enable = "avx2")]
    unsafe fn sigmoid_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;
        let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        let exp_neg = exp_approx_avx2(neg_x);
        _mm256_div_ps(
            _mm256_set1_ps(1.0),
            _mm256_add_ps(_mm256_set1_ps(1.0), exp_neg),
        )
    }

    /// AVX2 SqrtSoftplus
    #[target_feature(enable = "avx2")]
    unsafe fn sqrt_softplus_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;

        // AVX2 没有掩码混合，用比较 + blend 替代
        let threshold = _mm256_set1_ps(20.0);

        // 大值路径：sqrt(x)
        let sqrt_x = _mm256_sqrt_ps(x);

        // 小值路径：sqrt(ln(1 + exp(x)))
        let exp_x = exp_approx_avx2(x);
        let one_plus_exp = _mm256_add_ps(_mm256_set1_ps(1.0), exp_x);
        // AVX2 没有 _mm256_log_ps，标量回退
        let mut ope_arr = [0.0f32; 8];
        _mm256_storeu_ps(ope_arr.as_mut_ptr(), one_plus_exp);
        let mut ln_result = [0.0f32; 8];
        for i in 0..8 {
            ln_result[i] = ope_arr[i].ln();
        }
        let ln_vec = _mm256_loadu_ps(ln_result.as_ptr());
        let sqrt_ln = _mm256_sqrt_ps(ln_vec);

        // 比较：x > 20 时选 sqrt_x，否则选 sqrt_ln
        // _MM_CMP_GT_OQ = 14
        let mask = _mm256_cmp_ps(x, threshold, 14);
        _mm256_blendv_ps(sqrt_ln, sqrt_x, mask)
    }

    /// AVX2 Softmax 归一化
    #[target_feature(enable = "avx2")]
    unsafe fn softmax_normalize_avx2(row: &mut [f32], n: usize) {
        use std::arch::x86_64::*;

        // 求 max
        let mut max_val = f32::NEG_INFINITY;
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(row.as_ptr().add(i));
            // 水平求最大值
            let mut arr = [0.0f32; 8];
            _mm256_storeu_ps(arr.as_mut_ptr(), v);
            for j in 0..8 {
                if arr[j] > max_val {
                    max_val = arr[j];
                }
            }
            i += 8;
        }
        while i < n {
            if row[i] > max_val {
                max_val = row[i];
            }
            i += 1;
        }

        // exp(x - max) 并求和
        let max_vec = _mm256_set1_ps(max_val);
        let mut sum_vec = _mm256_setzero_ps();
        i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(row.as_ptr().add(i));
            let shifted = _mm256_sub_ps(v, max_vec);
            let exp_v = exp_approx_avx2(shifted);
            _mm256_storeu_ps(row.as_mut_ptr().add(i), exp_v);
            sum_vec = _mm256_add_ps(sum_vec, exp_v);
            i += 8;
        }
        let mut sum = hsum_float_8_avx2(sum_vec);
        while i < n {
            let exp_v = exp_approx_scalar(row[i] - max_val);
            row[i] = exp_v;
            sum += exp_v;
            i += 1;
        }

        // 除以 sum
        if sum > 0.0 {
            let inv_sum = _mm256_set1_ps(1.0 / sum);
            i = 0;
            while i + 8 <= n {
                let v = _mm256_loadu_ps(row.as_ptr().add(i));
                let normalized = _mm256_mul_ps(v, inv_sum);
                _mm256_storeu_ps(row.as_mut_ptr().add(i), normalized);
                i += 8;
            }
            while i < n {
                row[i] /= sum;
                i += 1;
            }
        }
    }

    /// AVX2 单 token 路由处理
    #[target_feature(enable = "avx2")]
    unsafe fn route_single_token_avx2(
        scores_row: &[f32],
        bias: Option<&[f32]>,
        n_experts: usize,
        topk: usize,
        score_func: ScoreFunc,
        route_scale: f32,
        buf: &mut RouteBuffer,
    ) -> (Vec<f32>, Vec<i32>) {
        use std::arch::x86_64::*;

        buf.resize(n_experts, topk);

        // 1. 应用 score_func（AVX2 向量化）
        match score_func {
            ScoreFunc::Sigmoid => {
                let mut i = 0;
                while i + 8 <= n_experts {
                    let v = _mm256_loadu_ps(scores_row.as_ptr().add(i));
                    let sig = sigmoid_avx2(v);
                    _mm256_storeu_ps(buf.activated.as_mut_ptr().add(i), sig);
                    i += 8;
                }
                while i < n_experts {
                    buf.activated[i] = 1.0 / (1.0 + exp_approx_scalar(-scores_row[i]));
                    i += 1;
                }
            }
            ScoreFunc::SqrtSoftplus => {
                let mut i = 0;
                while i + 8 <= n_experts {
                    let v = _mm256_loadu_ps(scores_row.as_ptr().add(i));
                    let sp = sqrt_softplus_avx2(v);
                    _mm256_storeu_ps(buf.activated.as_mut_ptr().add(i), sp);
                    i += 8;
                }
                while i < n_experts {
                    let s = scores_row[i];
                    buf.activated[i] = if s > 20.0 {
                        s.sqrt()
                    } else {
                        (1.0 + exp_approx_scalar(s)).ln().sqrt()
                    };
                    i += 1;
                }
            }
            ScoreFunc::Softmax => {
                buf.activated[..n_experts].copy_from_slice(&scores_row[..n_experts]);
            }
        }

        // 2. Softmax 归一化
        if score_func == ScoreFunc::Softmax {
            softmax_normalize_avx2(&mut buf.activated, n_experts);
        }

        // 3. 备份原始激活值
        buf.original[..n_experts].copy_from_slice(&buf.activated[..n_experts]);

        // 4. 加偏置（AVX2 向量化）
        if let Some(bias) = bias {
            let mut i = 0;
            while i + 8 <= n_experts {
                let v = _mm256_loadu_ps(buf.activated.as_ptr().add(i));
                let b = _mm256_loadu_ps(bias.as_ptr().add(i));
                let result = _mm256_add_ps(v, b);
                _mm256_storeu_ps(buf.activated.as_mut_ptr().add(i), result);
                i += 8;
            }
            while i < n_experts {
                buf.activated[i] += bias[i];
                i += 1;
            }
        }

        // 5. TopK 选择排序
        topk_select_inplace(&buf.activated, &mut buf.topk_temp, &mut buf.topk_weights, &mut buf.topk_indices, n_experts, topk);

        // 6. 用原始激活值替换权重
        for k in 0..topk {
            buf.topk_weights[k] = buf.original[buf.topk_indices[k] as usize];
        }

        // 7. 非 Softmax 时归一化
        if score_func != ScoreFunc::Softmax {
            let sum: f32 = buf.topk_weights[..topk].iter().sum();
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for w in buf.topk_weights[..topk].iter_mut() {
                    *w *= inv_sum;
                }
            }
        }

        // 8. 乘 route_scale
        for w in buf.topk_weights[..topk].iter_mut() {
            *w *= route_scale;
        }

        (buf.topk_weights[..topk].to_vec(), buf.topk_indices[..topk].to_vec())
    }

    /// AVX2 路由主函数
    pub fn route_scores_avx2(input: &RouteInput) -> RouteOutput {
        let num_tokens = input.scores.len() / input.n_experts;
        let n_experts = input.n_experts;
        let topk = input.topk;
        let score_func = input.score_func;
        let route_scale = input.route_scale;
        let bias = input.bias;

        let results: Vec<(Vec<f32>, Vec<i32>)> = (0..num_tokens)
            .into_par_iter()
            .map(|t| {
                let row_start = t * n_experts;
                let scores_row = &input.scores[row_start..row_start + n_experts];
                ROUTE_BUF.with(|buf| {
                    let mut buf = buf.borrow_mut();
                    unsafe {
                        route_single_token_avx2(
                            scores_row, bias, n_experts, topk, score_func, route_scale, &mut buf,
                        )
                    }
                })
            })
            .collect();

        let mut all_weights = Vec::with_capacity(num_tokens * topk);
        let mut all_indices = Vec::with_capacity(num_tokens * topk);
        for (w, idx) in results {
            all_weights.extend_from_slice(&w);
            all_indices.extend_from_slice(&idx);
        }

        RouteOutput {
            weights: all_weights,
            indices: all_indices,
        }
    }
}

// ============================================================================
// 标量回退路径
// ============================================================================

fn route_scores_scalar(input: &RouteInput) -> RouteOutput {
    let num_tokens = input.scores.len() / input.n_experts;
    let n_experts = input.n_experts;
    let topk = input.topk;
    let score_func = input.score_func;
    let route_scale = input.route_scale;
    let bias = input.bias;

    let results: Vec<(Vec<f32>, Vec<i32>)> = (0..num_tokens)
        .into_par_iter()
        .map(|t| {
            let row_start = t * n_experts;
            let scores_row = &input.scores[row_start..row_start + n_experts];
            ROUTE_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                route_single_token_scalar(
                    scores_row, bias, n_experts, topk, score_func, route_scale, &mut buf,
                )
            })
        })
        .collect();

    let mut all_weights = Vec::with_capacity(num_tokens * topk);
    let mut all_indices = Vec::with_capacity(num_tokens * topk);
    for (w, idx) in results {
        all_weights.extend_from_slice(&w);
        all_indices.extend_from_slice(&idx);
    }

    RouteOutput {
        weights: all_weights,
        indices: all_indices,
    }
}

// ============================================================================
// 公共接口：SIMD 分发
// ============================================================================

/// SIMD 优化的路由计算
///
/// 自动检测 CPU 特性，选择最优路径：
///   1. AVX-512F：16 路 f32 并行
///   2. AVX2：8 路 f32 并行
///   3. 标量回退
///
/// 使用 rayon 并行处理多个 token
pub fn route_scores_simd(input: &RouteInput) -> RouteOutput {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx512::is_avx512_supported() {
            return avx512_route::route_scores_avx512(input);
        }
        if super::avx512::is_avx2_supported() {
            return avx2_route::route_scores_avx2(input);
        }
    }

    route_scores_scalar(input)
}

// ============================================================================
// PyO3 绑定辅助
// ============================================================================

/// PyO3 调用入口：接收 numpy 数组，返回 (weights, indices)
pub fn route_simd_impl(
    scores: &[f32],
    bias: Option<&[f32]>,
    n_experts: usize,
    topk: usize,
    score_func: &str,
    route_scale: f32,
) -> (Vec<f32>, Vec<i32>) {
    assert!(n_experts > 0, "n_experts must be positive");
    assert!(topk > 0 && topk <= n_experts, "topk must be in [1, n_experts]");
    assert!(scores.len() % n_experts == 0, "scores length must be multiple of n_experts");
    if let Some(b) = bias { assert_eq!(b.len(), n_experts, "bias length must equal n_experts"); }
    assert!(route_scale.is_finite(), "route_scale must be finite");
    let input = RouteInput {
        scores,
        bias,
        n_experts,
        topk,
        score_func: ScoreFunc::from_str(score_func),
        route_scale,
    };
    let output = route_scores_simd(&input);
    (output.weights, output.indices)
}
