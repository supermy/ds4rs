//! IQ2_XS 权重的 NHWC/Tile 内存布局优化
//!
//! # 动机
//!
//! 当前 `Iq2XsWeight` 的 d/qs/scales 三个数组分离存储，访问一个 block 需要三次
//! 内存访问，可能产生三次 cache miss。d 和 scales 很小（4+8=12 字节），但与
//! qs（64 字节）不在同一缓存行，导致 cache thrashing。
//!
//! # Tile 布局
//!
//! 将每个 block 的 d + qs + scales 打包为一个 80 字节的连续块（4 字节填充对齐）：
//!
//! ```text
//! Tile 布局（80 字节/block，4 字节填充对齐）:
//! ┌────────┬──────────────────────────────────────────────────────┬──────────────┬──────┐
//! │ d(4B)  │                    qs(64B)                           │  scales(8B)  │pad(4B)│
//! └────────┴──────────────────────────────────────────────────────┴──────────────┴──────┘
//! ```
//!
//! # 缓存行对齐分析
//!
//! - 80 字节/block，5 个 block = 400 字节 = 6.25 个缓存行
//! - 4 个 block = 320 字节 = 正好 5 个缓存行
//! - 对比 76 字节（无填充）：4 block = 304 字节，跨 5 个缓存行但不对齐
//!
//! 选择 80 字节填充方案：
//! - 4 个 block 恰好对齐到 5 个缓存行（320B = 5×64B）
//! - 预取更高效：硬件预取器偏好规则步长
//! - 4 字节填充开销：每 block 5%，换取消除跨行不对齐的额外 cache miss
//!
//! # 缓存行为改善估算
//!
//! 分离布局（每行 56 blocks，14336×4096 权重）：
//! - 访问一个 block：3 次 cache miss（d, qs, scales 各一次）
//! - d 和 scales 共 12B，与 qs 的 64B 在不同缓存行
//! - 行间跳跃时 d/scales 几乎必定 miss（空间局部性差）
//! - 每行 56 blocks × 3 = 168 次 L1 miss
//!
//! Tile 布局：
//! - 访问一个 block：1 次 cache miss（d+qs+scales 连续）
//! - 80B 跨 2 个缓存行，但硬件预取器能识别 80B 步长
//! - d(4B) 和 scales(8B) 与 qs(64B) 在同一 Tile，预取 qs 时自动带入
//! - 每行 56 blocks × 1 = 56 次 L1 miss
//!
//! L1 命中率改善：约 3× cache miss 减少

use super::kernel::Iq2XsWeight;

/// Tile 大小：d(4B) + qs(64B) + scales(8B) + padding(4B) = 80 字节
pub const TILE_SIZE: usize = 80;

/// d 字段在 Tile 中的偏移
const D_OFFSET: usize = 0;
/// qs 字段在 Tile 中的偏移
const QS_OFFSET: usize = 4;
/// scales 字段在 Tile 中的偏移
const SCALES_OFFSET: usize = 68;
/// 填充字段在 Tile 中的偏移
const _PAD_OFFSET: usize = 76;

/// IQ2_XS 权重的 NHWC/Tile 内存布局
///
/// 将每个 block 的 d + qs + scales 打包为连续的 80 字节 Tile，
/// 减少单 block 访问的 cache miss 次数（从 3 次降到 1 次）。
#[derive(Clone, Debug)]
pub struct Iq2XsTile {
    /// 打包后的数据：每 block 80 字节 (d_f32:4 + qs:64 + scales:8 + pad:4)
    /// 布局：[block_0_d, block_0_qs_0..31, block_0_scales_0..7, block_0_pad,
    ///        block_1_d, block_1_qs_0..31, ...]
    pub data: Vec<u8>,
    /// (n_rows, n_cols)
    pub shape: (usize, usize),
    /// 每行的 block 数
    pub blocks_per_row: usize,
}

impl Iq2XsTile {
    /// 从 Iq2XsWeight 转换为 Tile 布局
    pub fn from_weight(weight: &Iq2XsWeight) -> Self {
        Self::from_separate(&weight.d, &weight.qs, &weight.scales, weight.shape)
    }

    /// 从分离的 d/qs/scales 数组构建 Tile 布局
    pub fn from_separate(d: &[f32], qs: &[u16], scales: &[u8], shape: (usize, usize)) -> Self {
        let n_blocks = d.len();
        assert_eq!(qs.len(), n_blocks * 32, "qs length mismatch: expected {}, got {}", n_blocks * 32, qs.len());
        assert_eq!(scales.len(), n_blocks * 8, "scales length mismatch: expected {}, got {}", n_blocks * 8, scales.len());
        let blocks_per_row = shape.1 / 256;

        let mut data = vec![0u8; n_blocks * TILE_SIZE];

        for blk in 0..n_blocks {
            let tile_offset = blk * TILE_SIZE;

            // 写入 d (4 字节, f32 little-endian)
            let d_bytes = d[blk].to_le_bytes();
            data[tile_offset + D_OFFSET..tile_offset + D_OFFSET + 4].copy_from_slice(&d_bytes);

            // 写入 qs (32 个 u16 = 64 字节, little-endian)
            let qs_base = tile_offset + QS_OFFSET;
            for g in 0..32 {
                let q_bytes = qs[blk * 32 + g].to_le_bytes();
                data[qs_base + g * 2..qs_base + g * 2 + 2].copy_from_slice(&q_bytes);
            }

            // 写入 scales (8 个 u8 = 8 字节)
            let scales_base = tile_offset + SCALES_OFFSET;
            data[scales_base..scales_base + 8].copy_from_slice(&scales[blk * 8..blk * 8 + 8]);

            // padding 4 字节已经是 0（vec! 初始化）
        }

        Self {
            data,
            shape,
            blocks_per_row,
        }
    }

    /// 获取指定 block 的 d 值
    #[inline]
    pub fn d_at(&self, block_idx: usize) -> f32 {
        let offset = block_idx * TILE_SIZE + D_OFFSET;
        let bytes: [u8; 4] = self.data[offset..offset + 4].try_into().unwrap();
        f32::from_le_bytes(bytes)
    }

    /// 获取指定 block 的 qs 数组（32 个 u16）
    ///
    /// 注意：由于 u16 的对齐要求，使用 `ptr::read_unaligned` 逐个读取，
    /// 确保在任意对齐的 Tile 数据上安全访问。
    #[inline]
    pub fn qs_at(&self, block_idx: usize) -> [u16; 32] {
        let offset = block_idx * TILE_SIZE + QS_OFFSET;
        let mut result = [0u16; 32];
        unsafe {
            let ptr = self.data.as_ptr().add(offset) as *const u16;
            for i in 0..32 {
                result[i] = std::ptr::read_unaligned(ptr.add(i));
            }
        }
        result
    }

    /// 获取指定 block 的 scales 切片（8 个 u8）
    #[inline]
    pub fn scales_at(&self, block_idx: usize) -> &[u8] {
        let offset = block_idx * TILE_SIZE + SCALES_OFFSET;
        &self.data[offset..offset + 8]
    }

    /// 获取指定行的所有 block 起始偏移（字节）
    #[inline]
    pub fn row_offset(&self, row: usize) -> usize {
        row * self.blocks_per_row * TILE_SIZE
    }

    /// 获取指定行的 Tile 数据切片
    #[inline]
    pub fn row_tiles(&self, row: usize) -> &[u8] {
        let start = self.row_offset(row);
        let len = self.blocks_per_row * TILE_SIZE;
        &self.data[start..start + len]
    }

    /// 总 block 数
    #[inline]
    pub fn n_blocks(&self) -> usize {
        self.data.len() / TILE_SIZE
    }

    /// 内存占用（字节）
    pub fn memory_usage(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_roundtrip() {
        let dim = 512;
        let n_rows = 4;
        let n_blocks = n_rows * dim / 256;

        let d: Vec<f32> = (0..n_blocks).map(|i| i as f32 * 0.01).collect();
        let qs: Vec<u16> = (0..n_blocks * 32).map(|i| (i % 1024) as u16).collect();
        let scales: Vec<u8> = (0..n_blocks * 8).map(|i| (i % 256) as u8).collect();

        let weight = Iq2XsWeight::new(d.clone(), qs.clone(), scales.clone(), (n_rows, dim));
        let tile = Iq2XsTile::from_weight(&weight);

        assert_eq!(tile.shape, (n_rows, dim));
        assert_eq!(tile.blocks_per_row, dim / 256);
        assert_eq!(tile.n_blocks(), n_blocks);
        assert_eq!(tile.data.len(), n_blocks * TILE_SIZE);

        // 验证每个 block 的数据正确
        for blk in 0..n_blocks {
            assert!((tile.d_at(blk) - d[blk]).abs() < 1e-6, "d mismatch at block {}", blk);

            let qs_slice = tile.qs_at(blk);
            for g in 0..32 {
                assert_eq!(qs_slice[g], qs[blk * 32 + g], "qs mismatch at block {} group {}", blk, g);
            }

            let scales_slice = tile.scales_at(blk);
            for s in 0..8 {
                assert_eq!(scales_slice[s], scales[blk * 8 + s], "scales mismatch at block {} scale {}", blk, s);
            }
        }
    }

    #[test]
    fn test_from_separate() {
        let dim = 256;
        let n_blocks = 2;

        let d = vec![1.0f32, 2.0f32];
        let qs = vec![100u16; 64];
        let scales = vec![42u8; 16];

        let tile = Iq2XsTile::from_separate(&d, &qs, &scales, (2, dim));

        assert_eq!(tile.n_blocks(), 2);
        assert!((tile.d_at(0) - 1.0f32).abs() < 1e-6);
        assert!((tile.d_at(1) - 2.0f32).abs() < 1e-6);
        assert_eq!(tile.qs_at(0)[0], 100);
        assert_eq!(tile.scales_at(0)[0], 42);
    }

    #[test]
    fn test_row_offset() {
        let dim = 512;
        let n_blocks_per_row = dim / 256; // 2

        let d = vec![0.0f32; 6];
        let qs = vec![0u16; 192];
        let scales = vec![0u8; 48];

        let tile = Iq2XsTile::from_separate(&d, &qs, &scales, (3, dim));

        assert_eq!(tile.row_offset(0), 0);
        assert_eq!(tile.row_offset(1), n_blocks_per_row * TILE_SIZE);
        assert_eq!(tile.row_offset(2), 2 * n_blocks_per_row * TILE_SIZE);
    }

    #[test]
    fn test_cache_line_alignment() {
        // 4 个 Tile = 320 字节 = 5 个缓存行（64B），完美对齐
        assert_eq!(4 * TILE_SIZE, 320);
        assert_eq!(4 * TILE_SIZE % 64, 0);
    }
}
