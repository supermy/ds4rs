/**
 * IQ2_XS CUDA GEMM 内核实现 — 从 llama.cpp 抽取
 *
 * 来源文件:
 *   ggml/src/ggml-cuda/convert.cu   — dequantize_block_iq2_xs (反量化 kernel)
 *   ggml/src/ggml-cuda/vecdotq.cuh  — vec_dot_iq2_xs_q8_1 (MMVQ 点积)
 *   ggml/src/ggml-cuda/mmq.cuh      — load_tiles_iq2_xs (MMQ 数据加载)
 *   ggml/src/ggml-cuda/common.cuh   — DP4A 辅助函数
 *   ggml/src/ggml-cuda/mmq.cu       — MMQ 分发
 *   ggml/src/ggml-cuda/mmvq.cu      — MMVQ 分发
 *   ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu — MMQ 模板实例化
 *
 * 实现函数:
 *   1. unpack_ksigns           — 符号索引展开
 *   2. ggml_cuda_dp4a          — DP4A 4 元素字节级点积
 *   3. get_int_b2 / get_int_b4 — 内存读取辅助
 *   4. dequantize_block_iq2_xs — CUDA 反量化 kernel
 *   5. vec_dot_iq2_xs_q8_1     — CUDA 向量点积 (MMVQ 用)
 *   6. load_tiles_iq2_xs       — CUDA MMQ 数据加载
 *   7. iq2xs_init_cuda         — 常量内存初始化
 *
 * 自包含: 不依赖 llama.cpp 的其他头文件
 */
#include "iq2_xs.cuh"

// ============================================================================
// 第一部分: Device 常量内存定义
// 将查找表放入常量内存加速访问 (每个 SM 有独立常量缓存, 延迟极低)
// ============================================================================

__constant__ uint64_t c_iq2xs_grid[512];
__constant__ uint8_t  c_ksigns_iq2xs[128];
__constant__ uint8_t  c_kmask_iq2xs[8];

// ============================================================================
// 第二部分: 辅助函数实现
// ============================================================================

// unpack_ksigns: 将 7-bit 符号索引展开为 32-bit 符号掩码
// 来源: ggml/src/ggml-cuda/vecdotq.cuh:97
//
// 算法:
//   1. v 是 7-bit 值 (0-127), 第 8 位由 popcnt(v) 的奇偶性决定
//   2. p = popcnt(v) & 1 — 如果 v 中 1 的个数为奇数, 则 p=1
//   3. s = v ^ (p << 7) — 修正第 8 位, 使 popcnt(s) 为偶数
//   4. s * 0x01010101 — 将 8-bit 值广播到 32-bit (每个字节相同)
//   5. 广播后可用 0x08040201 / 0x80402010 作为选择器提取符号位
//
// 为什么需要偶数 popcnt:
//   IQ2_XS 的符号编码只使用 7 bit, 但 8 个元素需要 8 bit 符号
//   约定: 第 8 个符号 = popcnt(前 7 个符号) 的奇偶性
//   这保证了负数个数始终为偶数, 是 IQ 量化的核心约束
static __device__ __forceinline__ uint32_t unpack_ksigns(uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101;
}

// ggml_cuda_dp4a: 4 元素字节级点积 (DP4A 指令加速)
// 来源: ggml/src/ggml-cuda/common.cuh
//
// 计算: c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
// 其中 a, b 被解释为 4 个 int8 的打包值
//
// SM 6.1+ (Pascal 及以上) 使用 dp4a 指令, 单周期完成
// 更旧架构使用标量展开 (实际不会用于 IQ2_XS, 因为需要 Pascal+)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(c) : "r"(a), "r"(b), "r"(c));
    return c;
}
#else
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
    const int8_t * a8 = (const int8_t *)&a;
    const int8_t * b8 = (const int8_t *)&b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
}
#endif

// get_int_b2: 从 uint16_t 数组读取 2 个连续值, 拼接为 32-bit int
// 来源: ggml/src/ggml-cuda/vecdotq.cuh:18
//
// 假设至少 2 字节对齐
// x16[2*i32 + 0] 放入低 16-bit, x16[2*i32 + 1] 放入高 16-bit
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

// get_int_b4: 从 int 数组读取 1 个 32-bit 值 (4 字节对齐)
// 来源: ggml/src/ggml-cuda/vecdotq.cuh:27
static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32];
}

// ============================================================================
// 第三部分: CUDA 反量化 Kernel
// 来源: ggml/src/ggml-cuda/convert.cu:314
// ============================================================================

// dequantize_block_iq2_xs: 将 IQ2_XS block 解码为 float/half
//
// 线程组织: 每个线程块处理一个 256 元素 block, 使用 32 个线程
//   tid/8 = il (0-3): 8 元素组索引 (对应 qs 中的 4 个 uint16_t)
//   tid%8 = ib (0-7): 32 元素 sub-block 索引
//
// 反量化公式:
//   y[j] = d * (0.5 + scale_4bit) * 0.25 * grid[j] * (signs & mask[j] ? -1 : 1)
//
// 其中:
//   d = block.d (FP16 全局缩放)
//   scale_4bit = (scales[ib] >> 4*(il/2)) & 0xf (4-bit 子块缩放)
//   grid = iq2xs_grid[qs[4*ib + il] & 511] (物理网格查找)
//   signs = ksigns_iq2xs[qs[4*ib + il] >> 9] (符号查找)
template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7

    dst_t * y = yy + i*QK_K + 32*ib + 8*il;

    const uint16_t * q2 = x[i].qs + 4*ib;

    // 从物理网格查找量化值
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));

    // 计算 scale: d * (0.5 + scale_4bit) * 0.25
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;

    // 符号查找
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];

    // 反量化: grid[j] 为正整数 (8, 25, 43), signs 控制符号
    for (int j = 0; j < 8; ++j) {
        y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    }
}

// 反量化入口函数
// k 必须是 QK_K (256) 的倍数
static void dequantize_row_iq2_xs_cuda(
    const void * vx, float * y, int64_t k, cudaStream_t stream)
{
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<float><<<nb, 32, 0, stream>>>(vx, y);
}

// ============================================================================
// 第四部分: CUDA vec_dot 函数 (MMVQ 用)
// 来源: ggml/src/ggml-cuda/vecdotq.cuh:1020
// ============================================================================

// vec_dot_iq2_xs_q8_1: IQ2_XS 与 Q8_1 的向量点积 (DP4A 加速)
//
// 核心算法:
//   1. 从 qs 读取 4 个 uint16_t, 每个 uint16_t 编码:
//      - 低 9-bit: grid 索引 (0-511)
//      - 高 7-bit: 符号索引
//   2. 从 iq2xs_grid 查找 8 个 int8 量化值 (uint2 = 8 字节)
//   3. 用 XOR+SUB 技巧应用符号:
//      signs0 = __vcmpne4(signs & 0x08040201, 0) — 提取低 4 字节符号, 生成 0x00000000 或 0xFFFFFFFF
//      grid_l = __vsub4(grid_pos.x ^ signs0, signs0) — 条件取反
//      当 signs0 = 0x00000000: grid_l = grid_pos.x (正)
//      当 signs0 = 0xFFFFFFFF: grid_l = ~grid_pos.x + 1 = -grid_pos.x (负)
//   4. DP4A 计算 grid × q8 点积, 分为前 4 组 (ls0) 和后 4 组 (ls1)
//   5. 最终结果: d * (sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2) / 4
//      等价于: d * 0.125 * (sumi0*(2*ls0+1) + sumi1*(2*ls1+1))
//
// 参数:
//   vbq:   指向 block_iq2_xs 数组的指针
//   bq8_1: 指向 block_q8_1 数组的指针 (Q8_1 量化值)
//   kbx:   block 索引偏移
//   iqs:   量化值起始索引 (0 或 QK_K/4)
static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

    // 读取 4 个 qs 值 (2 个 int2 = 4 个 uint16_t)
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;

    // 读取 scales: 低 4-bit 和高 4-bit 各编码一个子块缩放
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        // 从网格查找量化值 (uint2 = 8 字节 = 8 个 int8)
        const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l0/2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);

        // 低 4 字节: 应用符号 (XOR + SUB 技巧实现条件取反)
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

        // Q8_1 值
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        // 高 4 字节: 应用符号
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        // DP4A 点积: 分为前 4 组 (ls0) 和后 4 组 (ls1)
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }

    // 最终结果: d * (sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2) / 4
    // 等价于: d * 0.125 * (sumi0*(2*ls0+1) + sumi1*(2*ls1+1))
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// ============================================================================
// 第五部分: CUDA MMQ 数据加载
// 来源: ggml/src/ggml-cuda/mmq.cuh:2803
// ============================================================================

// load_tiles_iq2_xs: 为 MMQ (MulMatQ) 加载 IQ2_XS 数据到 shared memory
//
// 将 IQ2_XS block 解码为:
//   x_qs: int8 量化值 (带符号), 用于 DP4A 点积
//   x_df: float scale 值, 用于最终缩放
//
// 支持 DP4A 和 MMA 双路径 (通过编译时宏切换 shared memory 布局):
//   DP4A 路径: x_qs 按 (2*MMQ_TILE_NE_K + 1) stride 存储
//   MMA  路径: x_qs 按 MMQ_MMA_TILE_X_K_Q3_K stride 存储
//
// 线程组织:
//   threads_per_row = (MMQ_ITER_K / (4 * QR2_XS)) / 2 = 8
//   nrows = warp_size / threads_per_row = 4
//   每个线程处理 kqsx 列的 QR2_XS=4 个元素
//
// 参数:
//   x:       指向 block_iq2_xs 数组的 char 指针
//   x_tile:  shared memory 指针 (int 数组)
//   kbx0:    block 索引偏移
//   i_max:   最大行索引 (用于边界检查)
//   stride:  行 stride (block 数)
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_iq2_xs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    // nwarps 和 warp_size 在 llama.cpp 中通过 mmq_get_nwarps_device() 获取
    // NVIDIA GPU: nwarps=8, warp_size=32
    // AMD GPU (GFX9/GFX8): nwarps=4, warp_size=64
    constexpr int nwarps = 8;
    constexpr int warp_size = 32;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    // MMA 路径: x_qs 和 x_df 连续存储
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    // DP4A 路径: 使用 MMQ_DP4A_TXS_Q8_0_16 布局
    // qs = mmq_y*MMQ_TILE_NE_K*2 + mmq_y
    // dm = mmq_y*MMQ_TILE_NE_K*4/QI8_0 + mmq_y/(QI8_0/4)
    constexpr int txs_qs = mmq_y*MMQ_TILE_NE_K*2 + mmq_y;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs_qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    // 每行线程数: (256 / (4*4)) / 2 = 8
    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XS)) / 2;
    // 每个 warp 处理的行数: 32 / 8 = 4
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_xs * bxi = (const block_iq2_xs *) x + kbx0 + i*stride;

        // 读取 4 个 qs 值 (2 个 int2 = 4 个 uint16_t)
        const int2 q2_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint16_t * q2 = (const uint16_t *) &q2_packed;

        // 解码每个 qs 值 (QR2_XS = 4 个)
    #pragma unroll
        for (int l = 0; l < QR2_XS; ++l) {
            // 从网格查找量化值
            const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l] >> 9);

            // 应用符号 (XOR + SUB 技巧)
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            // 存入 shared memory (根据路径选择布局)
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        // 计算 scale 并存入 shared memory
        // scale = ((ls * d) + d/2) / 4 = d * (ls + 0.5) / 4 = d * (0.5 + ls) * 0.25
        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K                   + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K                   + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// ============================================================================
// 第六部分: MMQ 类型特征 (IQ2_XS 特化)
// 来源: ggml/src/ggml-cuda/mmq.cuh:3391
// ============================================================================

// IQ2_XS 的 MMQ 类型特征:
//   vdr          = VDR_IQ2_XS_Q8_1_MMQ = 2
//   load_tiles   = load_tiles_iq2_xs
//   vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma (IQ2_XS 使用 Q8_0_16 布局, 与 Q3_K 相同)
//   vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a
//
// 注: vec_dot_mma 和 vec_dot_dp4a 是通用函数, 定义在 mmq.cuh 中
//     IQ2_XS 复用 Q8_0_16 布局的版本 (与 Q3_K, NVFP4 相同的 MMA tile 大小)
//     这些函数的完整实现需要 mmq.cuh 中的其他辅助代码, 此处仅记录接口

// ============================================================================
// 第七部分: MMVQ 分发信息
// 来源: ggml/src/ggml-cuda/mmvq.cu
// ============================================================================

// IQ2_XS 在 MMVQ 中的配置:
//   vec_dot 函数:  vec_dot_iq2_xs_q8_1
//   VDR:           VDR_IQ2_XS_Q8_1_MMVQ = 2
//   ncols_dst 限制:
//     SM80+:  5 (Turing/Ampere/Ada/Hopper)
//     SM75:   4 (Turing)
//     SM70:   4 (Volta)
//     SM60:   4 (Pascal)
//   batch_size 限制:
//     SM80+:  5
//     SM75-:  4
//
// 在 llama.cpp 中通过 mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS> 分发

// ============================================================================
// 第八部分: MMQ 模板实例化
// 来源: ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu
// ============================================================================

// 在 llama.cpp 中, MMQ 通过 DECL_MMQ_CASE(GGML_TYPE_IQ2_XS) 实例化
// 这会为所有 mmq_x, mmq_y 组合生成 mul_mat_q 的特化版本
// 实际使用时需要包含完整的 mmq.cuh 框架

// ============================================================================
// 第九部分: 性能阈值
// ============================================================================

// IQ2_XS 的 MMQ 在 RDNA3.5+ GPU 上总是启用
// 在其他 GPU 上, 仅当 ne11 <= 128 时启用 MMQ, 否则回退到 MMVQ
// 参考: ggml/src/ggml-cuda/mmq.cu 中的 mmq_need_quant_condition

// ============================================================================
// 第十部分: 初始化函数
// ============================================================================

// iq2xs_init_cuda: 将查找表拷贝到 GPU 常量内存
// 必须在程序启动时调用一次 (在任意 CUDA 操作之前)
//
// 常量内存优势:
//   - 每个 SM 有独立的常量缓存 (8KB/SM)
//   - 广播读取时延迟极低 (1 个周期 vs 全局内存 400+ 周期)
//   - 适合查找表等只读数据
static void iq2xs_init_cuda() {
    cudaMemcpyToSymbol(c_iq2xs_grid,    iq2xs_grid,    sizeof(iq2xs_grid));
    cudaMemcpyToSymbol(c_ksigns_iq2xs,  ksigns_iq2xs,  sizeof(ksigns_iq2xs));
    cudaMemcpyToSymbol(c_kmask_iq2xs,   kmask_iq2xs,   sizeof(kmask_iq2xs));
}
