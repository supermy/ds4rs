//! IQ2_XS 量化 C FFI 测试程序
//!
//! 从 safetensors 读取专家权重，调用 C 语言 IQ2_XS 量化/反量化，
//! 显示进度、量化精度（MSE）、压缩比和速度。

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use std::fs;
use std::path::Path;
use std::time::Instant;

// ============================================================================
// FFI 绑定
// ============================================================================

#[link(name = "iq2_xs_c")]
extern "C" {
    fn iq2xs_init_ffi();
    fn iq2xs_free_ffi();
    fn quantize_iq2_xs_ffi(
        src: *const f32,
        dst: *mut u8,
        nrow: i64,
        n_per_row: i64,
    ) -> usize;
    fn dequantize_iq2_xs_ffi(blocks: *const u8, dst: *mut f32, n: i64);
    fn iq2_xs_block_size_ffi() -> usize;
    fn iq2_xs_qk_k_ffi() -> i32;
}

// ============================================================================
// FP4 解码（与 Python 脚本对齐）
// ============================================================================

/// FP4 E2M1 查找表
const FP4_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// 将 int8 打包的 FP4 解码为 float32
fn decode_fp4_to_f32(packed: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(packed.len() * 2);
    for &byte in packed {
        let lo = (byte & 0x0F) as usize;
        let hi = ((byte >> 4) & 0x0F) as usize;
        out.push(FP4_TABLE[lo]);
        out.push(FP4_TABLE[hi]);
    }
    out
}

// ============================================================================
// safetensors 读取
// ============================================================================

fn read_shard_metadata(path: &Path) -> Result<Vec<(String, usize, usize, usize)>> {
    let data = fs::read(path).with_context(|| format!("读取文件失败: {:?}", path))?;
    let st = SafeTensors::deserialize(&data)?;

    let mut entries = Vec::new();
    for name in st.names() {
        let tensor = st.tensor(name)?;
        let dtype = tensor.dtype().to_string();
        let shape: Vec<usize> = tensor.shape().to_vec();
        let n_elements: usize = shape.iter().product();
        let byte_len = tensor.data().len();
        entries.push((name.to_string(), n_elements, byte_len, shape.len()));
    }
    Ok(entries)
}

/// 从 safetensors 读取指定 key 的 FP4 权重并解码为 float32
fn read_fp4_weight(path: &Path, key: &str) -> Result<Vec<f32>> {
    let data = fs::read(path).with_context(|| format!("读取文件失败: {:?}", path))?;
    let st = SafeTensors::deserialize(&data)?;
    let tensor = st.tensor(key)?;
    let raw = tensor.data();
    // FP4 权重以 int8 打包存储
    let packed: &[u8] = bytemuck::cast_slice(raw);
    Ok(decode_fp4_to_f32(packed))
}

// ============================================================================
// 量化测试
// ============================================================================

struct QuantResult {
    key: String,
    n_elements: usize,
    original_bytes: usize,
    quantized_bytes: usize,
    mse: f64,
    quant_ms: u128,
    dequant_ms: u128,
}

fn quantize_one_weight(
    key: &str,
    f32_data: &[f32],
    qk_k: usize,
    block_size: usize,
) -> Result<QuantResult> {
    let n = f32_data.len();
    if n == 0 {
        anyhow::bail!("空权重: {}", key);
    }

    // 对齐到 QK_K 的倍数
    let n_padded = ((n + qk_k - 1) / qk_k) * qk_k;
    let n_blocks = n_padded / qk_k;
    let nrow = 1i64;
    let n_per_row = n_padded as i64;

    // 准备输入（补零对齐）
    let mut src = vec![0.0f32; n_padded];
    src[..n].copy_from_slice(f32_data);

    // 分配输出
    let dst_size = n_blocks * block_size;
    let mut dst = vec![0u8; dst_size];

    // 量化
    let t0 = Instant::now();
    let written = unsafe {
        quantize_iq2_xs_ffi(src.as_ptr(), dst.as_mut_ptr(), nrow, n_per_row)
    };
    let quant_ms = t0.elapsed().as_millis();

    assert_eq!(written, dst_size, "量化输出大小不匹配");

    // 反量化
    let mut dequant = vec![0.0f32; n_padded];
    let t1 = Instant::now();
    unsafe {
        dequantize_iq2_xs_ffi(dst.as_ptr(), dequant.as_mut_ptr(), n_per_row);
    }
    let dequant_ms = t1.elapsed().as_millis();

    // 计算 MSE
    let mut sum_sq = 0.0f64;
    for i in 0..n {
        let diff = f32_data[i] as f64 - dequant[i] as f64;
        sum_sq += diff * diff;
    }
    let mse = sum_sq / n as f64;

    Ok(QuantResult {
        key: key.to_string(),
        n_elements: n,
        original_bytes: n * 4, // float32
        quantized_bytes: written,
        mse,
        quant_ms,
        dequant_ms,
    })
}

// ============================================================================
// 内存使用
// ============================================================================

fn get_ram_usage_gb() -> f64 {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let kb: f64 = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            return kb / 1024.0 / 1024.0;
        }
    }
    0.0
}

// ============================================================================
// 主函数
// ============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let shard_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/models/model-00002-of-00046.safetensors");
    let limit: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0); // 0 = 全部

    println!("======================================================================");
    println!("IQ2_XS 量化 C FFI 测试");
    println!("======================================================================");
    println!("输入: {}", shard_path);

    // 初始化 IQ2_XS 运行时
    println!("[初始化] 构建 kmap/kneighbors...");
    let t_init = Instant::now();
    unsafe { iq2xs_init_ffi() };
    println!("[初始化] 完成 ({:.1}s)", t_init.elapsed().as_secs_f64());

    let qk_k = unsafe { iq2_xs_qk_k_ffi() } as usize;
    let block_size = unsafe { iq2_xs_block_size_ffi() };
    println!("[常量] QK_K={}, block_size={}, bpw={:.4}", qk_k, block_size,
             block_size as f64 * 8.0 / qk_k as f64);

    // 读取 shard 元数据
    let path = Path::new(shard_path);
    let data = fs::read(path).with_context(|| format!("读取文件失败: {:?}", path))?;
    let st = SafeTensors::deserialize(&data)?;

    // 筛选专家权重
    let expert_keys: Vec<String> = st
        .names()
        .iter()
        .filter(|n| n.contains(".ffn.experts.") && n.contains(".weight"))
        .map(|s| s.to_string())
        .collect();

    let total = if limit > 0 && limit < expert_keys.len() {
        limit
    } else {
        expert_keys.len()
    };
    println!("[扫描] 找到 {} 个专家权重，测试 {} 个", expert_keys.len(), total);

    // 逐个处理
    let mut results = Vec::new();
    let start_all = Instant::now();

    for (idx, key) in expert_keys.iter().enumerate() {
        if idx >= total {
            break;
        }

        // 读取 FP4 权重并解码为 float32
        let tensor = st.tensor(key)?;
        let raw = tensor.data();
        let packed: &[u8] = bytemuck::cast_slice(raw);
        let f32_data = decode_fp4_to_f32(packed);

        // 量化测试
        let result = quantize_one_weight(key, &f32_data, qk_k, block_size)?;
        results.push(result);

        // 进度显示
        let elapsed = start_all.elapsed().as_secs_f64();
        let avg = elapsed / (idx + 1) as f64;
        let remaining = avg * (total - idx - 1) as f64;
        let r = results.last().unwrap();
        let ratio = r.original_bytes as f64 / r.quantized_bytes as f64;
        print!(
            "\r  [{}/{}] {} MSE={:.6} 压缩={:.2}x 量化={}ms 反量化={}ms RAM={:.1}GB 剩余~{:.0}s   ",
            idx + 1,
            total,
            key.split('.').last().unwrap_or(""),
            r.mse,
            ratio,
            r.quant_ms,
            r.dequant_ms,
            get_ram_usage_gb(),
            remaining,
        );
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!();

    // 汇总统计
    let total_time = start_all.elapsed().as_secs_f64();
    let avg_mse: f64 = results.iter().map(|r| r.mse).sum::<f64>() / results.len() as f64;
    let max_mse = results.iter().map(|r| r.mse).fold(f64::MIN, f64::max);
    let min_mse = results.iter().map(|r| r.mse).fold(f64::MAX, f64::min);
    let total_orig: usize = results.iter().map(|r| r.original_bytes).sum();
    let total_quant: usize = results.iter().map(|r| r.quantized_bytes).sum();
    let total_quant_ms: u128 = results.iter().map(|r| r.quant_ms).sum();
    let total_dequant_ms: u128 = results.iter().map(|r| r.dequant_ms).sum();
    let total_elements: usize = results.iter().map(|r| r.n_elements).sum();

    println!();
    println!("======================================================================");
    println!("测试结果汇总");
    println!("======================================================================");
    println!("  专家数量:       {}", results.len());
    println!("  总元素数:       {:.3}M", total_elements as f64 / 1e6);
    println!("  总耗时:         {:.1}s", total_time);
    println!("  量化总耗时:     {:.1}s", total_quant_ms as f64 / 1000.0);
    println!("  反量化总耗时:   {:.1}s", total_dequant_ms as f64 / 1000.0);
    println!("  量化速度:       {:.1}M 元素/s", total_elements as f64 / 1e6 / (total_quant_ms as f64 / 1000.0).max(0.001));
    println!("  反量化速度:     {:.1}M 元素/s", total_elements as f64 / 1e6 / (total_dequant_ms as f64 / 1000.0).max(0.001));
    println!("  平均 MSE:       {:.6}", avg_mse);
    println!("  最大 MSE:       {:.6}", max_mse);
    println!("  最小 MSE:       {:.6}", min_mse);
    println!("  压缩比:         {:.2}x (float32→IQ2_XS)", total_orig as f64 / total_quant as f64);
    println!("  原始大小:       {:.2}MB", total_orig as f64 / 1024.0 / 1024.0);
    println!("  量化后大小:     {:.2}MB", total_quant as f64 / 1024.0 / 1024.0);
    println!("  bpw:            {:.4} bits/weight", total_quant as f64 * 8.0 / total_elements as f64);
    println!("  RAM 峰值:       {:.1}GB", get_ram_usage_gb());

    // 释放
    unsafe { iq2xs_free_ffi() };

    Ok(())
}
