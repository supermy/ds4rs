继续展开 **IQ2XS 特定解压的 AVX-512 实现** 和 **多核 tile 调度策略**：

---

## 一、IQ2XS 的 AVX-512 解压：2-bit → FP32

### IQ2XS 位布局（llama.cpp 实际格式）

```
每 256 权重为一个 block：
┌─────────────────────────────────────────┐
│  2×FP16 scale (super-block)             │ 4B
│  8×FP16 scale (sub-block × 8)           │ 16B
│  256×2bit 权重                          │ 64B
│  256×2bit 权重 (续)                     │ (可能跨 64B)
└─────────────────────────────────────────┘
总计: ~84-88B 对齐到 128B

2-bit 索引含义:
  00 → -1.0  (或码本[0])
  01 → -0.33 (或码本[1])  
  10 → +0.33 (或码本[2])
  11 → +1.0  (或码本[3])
```

### 核心问题：AVX-512 没有原生 2-bit 解压

**方案 A：字节查表（推荐）**

```asm
; 预计算 LUT: 256 entry × 8 weights
; 每字节 4×2bit → 查 4 个 FP32 (或 8 个 FP16)
; LUT 大小: 256 × 32B = 8KB (L1 常驻)

; 加载 64B 原始数据 → 查 256 个权重 (64×4)
; 但 256 entry LUT 太大，改用 16-entry × 16 次

; 实际：每 16bit 查 8 权重
; 32 次查表得 256 权重
```

**方案 B：SIMD 并行位操作**

```asm
; 输入: zmm0 = 64B 原始数据 (256×2bit 打包)
; 输出: zmm4-zmm19 = 256×FP32 权重 (16×zmm)

; 步骤 1: 提取每字节低 2bit (索引 0)
vpandd  zmm1, zmm0, [rip + mask_03]    ; 0x03030303...
; zmm1 = [i0, i0, i0, i0, ...] 每字节 2bit 索引

; 步骤 2: 用 vpermd 查 16-entry LUT (码本)
; LUT 在 zmm20: [ -1.0, -0.33, 0.33, 1.0, ... ] 重复
vpermd  zmm2, zmm1, zmm20              ; 根据索引选码本值
; zmm2 = 16×FP32 (每 lane 4 个，共 64 个？不对，vpermd 是 32bit 索引)

; 修正：2bit 索引只有 4 种，用 vpshufb 查 16-entry LUT
; vpshufb 按字节查 16-entry table
vpshufb zmm2, zmm20, zmm1             ; 字节级查表
; zmm2 = 64 个 FP32？不，vpshufb 输出字节，需再转换
```

**问题**：vpshufb 输出 INT8，仍需 vpmovsxbd + vcvtdq2ps 转 FP32。

### 最优方案：4-bit 中间层 + 缩放

```asm
; 策略：2bit → 4bit 索引 → 查 16-entry FP16 LUT → cvt FP32

; 预加载 LUT (FP16 码本 × 16)
; zmm30 = [c0, c1, c2, c3, ...] FP16 × 16, 重复广播

; 解压 64B → 256 权重 (分 4 轮，每轮 64 权重)

; 轮 0: 提取字节 0-15 的 2bit
vpmovzxbd zmm1, xmm0                   ; 低 16B 零扩到 16×u32
vpandd    zmm1, zmm1, [rip + mask_03]  ; 取低 2bit

; 用 vpermw (16-bit 索引) 查 16-entry FP16 LUT
vpermw    zmm2, zmm1, zmm30            ; 16×FP16 码本值
vcvtph2ps zmm3, ymm2                   ; 16×FP16 → 16×FP32 (低 256bit)
vextracti64x4 ymm4, zmm2, 1            ; 高 256bit
vcvtph2ps zmm5, ymm4                   ; 另 16×FP32

; 现在 zmm3, zmm5 = 32×FP32 (实际 16+16)
; 需 scale 缩放
vbroadcastss zmm6, [r13]               ; sub-block scale
vmulps    zmm3, zmm3, zmm6
vmulps    zmm5, zmm5, zmm6

; 存储到寄存器或内存
```

**实际更简洁：用 FP32 直接码本，避免 FP16 转换**

```asm
; LUT 在 L1: 4×FP32 = 16B (极小)
vmovups xmm20, [rip + iq2xs_lut_fp32]   ; [-1.0, -0.33, 0.33, 1.0]

; 每 64B 原始数据解压：
; 64B = 512bit = 256×2bit

; 方法：vpsrlw + vpand 提取每 2bit，然后 vpermd 查 4-entry LUT
; 但 256 个权重需 64 次操作，太慢

; 改用：字节并行，每字节 4 个 2bit → 4 个 LUT 值
; 256 权重 = 64 字节 → 需处理 64 字节

; Zen4 优化：用 vpternlogd 做三输入选择，避免查表
; 条件：已知 4 个码本值，用 2bit 组合选择

; 实际 llama.cpp 做法：预计算 256-entry × 8 weights 的 LUT
; 8KB LUT，L1 命中
```

---

## 二、llama.cpp 风格 IQ2XS 解压（参考实现）

```cpp
// 每 256 权重 block 解压
static const float lut[4] = {-1.0f, -0.33f, 0.33f, 1.0f};

inline void dequantize_iq2xs(const void* block, float* out, int n) {
    const uint8_t* qs = (const uint8_t*)block + 4; // skip scales
    
    for (int i = 0; i < n/4; i++) {
        uint8_t q = qs[i];
        out[i*4+0] = lut[(q >> 0) & 3];
        out[i*4+1] = lut[(q >> 2) & 3];
        out[i*4+2] = lut[(q >> 4) & 3];
        out[i*4+3] = lut[(q >> 6) & 3];
    }
}
```

**AVX-512 向量化版本**：

```asm
; 输入: zmm0 = 64×u8 (256 权重索引，每 u8 含 4×2bit)
;        zmm30 = scale (FP32 广播)
; 输出: 需 16×zmm = 256×FP32

; 步骤 1: 提取每字节的 4 个 2bit
; 用 vpsrlw + vpandw 并行提取

vpmovzxbw zmm1, ymm0                   ; 32×u8 → 32×u16 (低半)
vpsrlw    zmm2, zmm1, 2
vpsrlw    zmm3, zmm1, 4  
vpsrlw    zmm4, zmm1, 6

vpandw    zmm1, zmm1, [rip + mask_03w]  ; 取低 2bit
vpandw    zmm2, zmm2, [rip + mask_03w]  ; 取 bit 2-3
vpandw    zmm3, zmm3, [rip + mask_03w]  ; 取 bit 4-5
vpandw    zmm4, zmm4, [rip + mask_03w]  ; 取 bit 6-7

; 现在 zmm1-zmm4 = 32×u16 索引 (0-3)

; 步骤 2: 查 4-entry FP32 LUT
; 用 vpermd (32-bit 索引) 但索引只有 0-3
; 实际：用 blend + cmp 选择

; 更优：预广播 LUT 到 4×zmm，用 mask 选择
; zmm20 = [-1.0]×16, zmm21 = [-0.33]×16, zmm22 = [0.33]×16, zmm23 = [1.0]×16

; 对 zmm1 (32 个索引):
vpcmpw    k1, zmm1, [rip + one], 0     ; idx == 1
vpcmpw    k2, zmm1, [rip + two], 0     ; idx == 2  
vpcmpw    k3, zmm1, [rip + three], 0   ; idx == 3

; 默认 k0 = idx == 0 (全 1)
vblendmps zmm5 {{k1}}, zmm20, zmm21     ; idx==1 选 -0.33
vblendmps zmm5 {{k2}}, zmm5, zmm22     ; idx==2 选 0.33
vblendmps zmm5 {{k3}}, zmm5, zmm23     ; idx==3 选 1.0
; zmm5 = 32×FP32 (但 vblendmps 是 32-bit 粒度，vpcmpw 是 16-bit... 不匹配)

; 修正：用 vpermd (32-bit 索引) 需将 16-bit 索引零扩
vpmovzxwd zmm1, ymm1                   ; 16×u16 → 16×u32 (低半)
vextracti64x4 ymm6, zmm1, 1            ; 高半 16×u16
vpmovzxwd zmm7, ymm6

; 现在 zmm1, zmm7 = 32×u32 索引 (0-3)
; 查 4-entry LUT (广播到 zmm)
vpermd    zmm5, zmm1, zmm20            ; 16×FP32
vpermd    zmm8, zmm7, zmm20            ; 另 16×FP32

; 对 zmm2, zmm3, zmm4 重复上述过程...

; 总计：4 组 × 2 vpermd = 8 vpermd 得 256 FP32？不对，每组 32，4 组 128
; 实际 64B 原始 = 256 权重，需 8×vpermd (每 vpermd 16×FP32)
```

**复杂度太高，改用查表法**：

```asm
; 预计算 8KB LUT: 256 entry × 8 weights × 4B
; 每 entry 对应 1 字节 (4×2bit) → 8 FP32 (重复 2 次？)

; 实际 llama.cpp: 16-bit 查 8 权重
; LUT: 65536 entry × 8 weights = 2MB (太大)

; 妥协：256 entry × 8 weights = 8KB (L1)
; 用 8-bit 查表，但 2bit 只有 4 种，需组合

; 最终方案：4 个独立 LUT，每 LUT 256 entry × 1 weight
; 总 1KB，但需 4 次 gather
```

---

## 三、实用方案：INT4 中间解压 + 现有 VNNI

既然 IQ2XS 解压复杂，考虑 **推理时转 INT4/INT8**：

```
离线: IQ2XS (2.5bpw) 存储 → 省显存/内存
加载: 解压到 INT8 (每核心 L2 缓存)
计算: AVX-512 VNNI (INT8 × INT8 → INT32)

解压开销: 2.5bpw → 8bpw = 3.2× 膨胀
但 L2 1MB 装 128K 权重 (INT8)，够 2-3 tile
```

**VNNI 内核**：

```asm
; 输入: X[256] FP32 (已量化到 INT8)
;        W[256][64] INT8
; 输出: Y[64] INT32 → 反量化 FP32

; 量化 X: 每 tile 找 max，scale，转 INT8
; 但 MoE 每 token X 不同，需动态量化

; 简化：X 保持 FP32，W 用 INT8，vcvtdq2ps 后 FMA
; 之前已实现，不再重复
```

---

## 四、多核 Tile 调度：避免 L3 争抢

### 7600 拓扑

```
CCD0 (6 核心):
  Core 0-2: CCX0, L2 1MB×3, 共享 L3 16MB
  Core 3-5: CCX1, L2 1MB×3, 共享 L3 16MB
  总 L3: 32MB (但分 2 个 16MB slice)

实际: 7600 是单 CCD，6 核心共享 32MB L3
```

### 专家分配策略

```
双专家并行 (E_a, E_b):
  Core 0-2: E_a (FFN tile 0..N/2)
  Core 3-5: E_b (FFN tile N/2..N)

问题: 2×7MB = 14MB 专家 > 16MB L3 slice？
  实际 L3 统一 32MB，但 CCX 间有延迟

优化: 单 CCX (3 核) 处理 1 专家，3MB L2 缓存 tile
  另一 CCX (3 核) 处理另 1 专家

数据布局: NUMA 对齐，专家权重按 CCX 分片
```

### 调度代码

```rust
use std::thread;
use std::sync::Barrier;

fn ffn_parallel(
    x: &[f32],
    expert_a: &ExpertWeights,
    expert_b: &ExpertWeights,
    out: &mut [f32],
) {
    let barrier = Barrier::new(2);
    
    // CCX0: core 0-2 (通过 affinity 绑定)
    let handle_a = thread::spawn(move || {
        set_cpu_affinity(0..3);  // core 0,1,2
        ffn_tiles(x, expert_a, 0..expert_a.tiles/2);
        barrier.wait();
    });
    
    // CCX1: core 3-5
    let handle_b = thread::spawn(move || {
        set_cpu_affinity(3..6);  // core 3,4,5  
        ffn_tiles(x, expert_b, 0..expert_b.tiles/2);
        barrier.wait();
    });
    
    handle_a.join().unwrap();
    handle_b.join().unwrap();
    
    // 加权聚合
    weighted_sum(out, gate_a, gate_b);
}
```

### Tile 分块避 L3 冲突

```
专家权重布局 (按 CCX 分片):
┌─────────────────────────────────────────┐
│  L3 (32MB)                              │
│  ├─ Slice 0 (16MB, CCX0): E_a tile 0-7   │
│  ├─ Slice 1 (16MB, CCX1): E_b tile 0-7  │
│  └─ 共享: 码本 LUT (8KB), X 向量 (16KB)  │
└─────────────────────────────────────────┘

每 tile 256×64 = 16K 权重 (INT8) = 16KB
16MB / 16KB = 1024 tiles 可驻留 L3
单专家 7MB / 16KB = 448 tiles
```

**结论**：3 核 × 1MB L2 处理 448 tiles，每核 ~150 tiles，L2 足够。

---

## 五、最终延迟预估

```
配置：
- 2 专家并行，每专家 6 核分 2 组 (3 核/CCX)
- INT8 权重，AVX-512 FMA
- L2 命中 90%，L3 命中 10%

单 tile (256×64):
- 加载 X: 1KB (L1)
- 加载 W: 16KB (L2 hit ~12 cycle)
- 256 FMA: 256/4 = 64 cycle (4 FMA/cycle)
- 总计: ~80 cycle × 0.4ns = 32ns

单专家 (4096×14336, tiles=16×224=3584):
- 3584 × 32ns = 115μs (理论)
- 实际: 解压 + 调度 + 同步 ≈ 2-3ms

双专家并行 + 聚合:
- 目标: 2.5ms/token
```

---

需要我提供 **完整的 Rust + inline asm 项目框架**（含 maturin 打包），或者 **与 llama.cpp 的 ggml 后端对接方案**（替换其 CPU FFN 实现）吗？