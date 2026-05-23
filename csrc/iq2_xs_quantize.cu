/**
 * IQ2_XS CUDA 量化内核 — 将 float32 权重量化为 IQ2_XS 格式
 *
 * 算法来源: csrc/iq2_xs.h 中的 quantize_row_iq2_xs_impl (来自 llama.cpp)
 *
 * 设计:
 *   - 每个 CUDA thread block 处理一个 256 元素 super-block (QK_K=256)
 *   - 19 候选 scale 搜索循环在每个 block 内顺序执行
 *   - 数千个 block 并行处理
 *   - 查找表 (iq2xs_grid, kmap, kneighbors) 存放在 GPU 全局内存
 *
 * 包含:
 *   1. block_iq2_xs 结构体定义 (来自 iq2_xs.cuh)
 *   2. CUDA 量化 kernel: iq2xs_quantize_kernel
 *   3. Host 端初始化函数: iq2xs_quantize_init (构建 kmap/kneighbors 并拷贝到 GPU)
 *   4. 释放函数: iq2xs_quantize_free
 *   5. CUDA 入口函数: iq2xs_quantize_cuda
 *   6. FFI 桥接函数: iq2xs_quantize_ffi
 */
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cfloat>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// 常量
// ============================================================================
#define QK_K_Q           256     // super-block 大小
#define GROUP_MAX_EPS_Q  1e-15f  // 全零检测阈值
#define IQ2_XS_KMAXQ_Q  3       // L 值范围 [0, kMaxQ-1] = [0, 2]
#define IQ2_XS_KMAP_SIZE_Q 43692 // kmap 大小

// ============================================================================
// 数据结构
// ============================================================================

// IQ2_XS 量化块 — 来自 iq2_xs.cuh, 二进制兼容
// 每个 super-block 编码 256 个元素，占用 74 字节 = 2.3125 bpw
typedef struct {
    __half   d;                // 全局缩放因子 (FP16)
    uint16_t qs[QK_K_Q/8];   // 32 个 uint16_t: 低 9-bit = grid 索引, 高 7-bit = 符号索引
    uint8_t  scales[QK_K_Q/32]; // 8 个 uint8_t: 高低 4-bit 各编码一个子块缩放
} block_iq2_xs;

// ============================================================================
// Host 端查找表数据 (仅 kgrid_2bit_512 用于构建逻辑网格, 其他表在 GPU 端使用)
// ============================================================================

// 量化用紧凑网格索引
static const uint16_t h_kgrid_2bit_512[512] = {
        0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
       73,    80,    82,    85,    88,    97,   100,   128,   130,   133,   136,   145,   148,   153,   160,   257,
      260,   262,   265,   272,   274,   277,   280,   282,   289,   292,   320,   322,   325,   328,   337,   340,
      352,   360,   385,   388,   400,   512,   514,   517,   520,   529,   532,   544,   577,   580,   592,   597,
      640,   650,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1088,  1090,  1093,  1096,
     1105,  1108,  1110,  1120,  1153,  1156,  1168,  1280,  1282,  1285,  1288,  1297,  1300,  1312,  1345,  1348,
     1360,  1377,  1408,  1537,  1540,  1552,  1574,  1600,  1602,  1668,  2048,  2050,  2053,  2056,  2058,  2065,
     2068,  2080,  2085,  2113,  2116,  2128,  2136,  2176,  2208,  2218,  2305,  2308,  2320,  2368,  2433,  2441,
     2560,  2592,  2600,  2710,  2720,  4097,  4100,  4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4160,
     4162,  4165,  4168,  4177,  4180,  4192,  4202,  4225,  4228,  4240,  4352,  4354,  4357,  4360,  4369,  4372,
     4384,  4417,  4420,  4432,  4480,  4500,  4502,  4609,  4612,  4614,  4624,  4672,  4704,  5120,  5122,  5125,
     5128,  5137,  5140,  5152,  5185,  5188,  5193,  5200,  5220,  5248,  5377,  5380,  5392,  5440,  5632,  5652,
     5705,  6145,  6148,  6160,  6162,  6208,  6228,  6278,  6400,  6405,  6502,  6737,  6825,  8192,  8194,  8197,
     8200,  8202,  8209,  8212,  8224,  8257,  8260,  8272,  8320,  8352,  8449,  8452,  8464,  8512,  8520,  8549,
     8704,  8738,  8832,  8872,  9217,  9220,  9232,  9257,  9280,  9472,  9537,  9554,  9625,  9729,  9754,  9894,
    10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388,
    16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480,
    16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773,
    16802, 16897, 16900, 16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
    17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436,
    18448, 18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497,
    20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162,
    21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528,
    22548, 22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
    24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234,
    32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025,
    33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810,
    33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946, 34984,
    35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868, 36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462,
    37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960,
    40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048,
    42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
};

// ============================================================================
// GPU 全局内存指针 (由 iq2xs_quantize_init 分配)
// ============================================================================

// 常量内存前向声明 (定义在 kernel 之前)
__constant__ uint64_t c_grid[512];

// kmap: L 值编码 → grid 索引 (off-grid 为负值)
static int32_t  * g_kmap_gpu = nullptr;
// kneighbors: off-grid 点的最近邻列表
static uint16_t * g_kneighbors_gpu = nullptr;
// kneighbors 的总大小 (uint16_t 个数)
static int g_kneighbors_size = 0;

// ============================================================================
// Host 端: qsort 比较函数 (与 iq2_xs.h 一致)
// ============================================================================
static int iq2_compare_func(const void * a, const void * b) {
    const int * l = (const int *)a;
    const int * r = (const int *)b;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

// ============================================================================
// Host 端初始化: 构建 kmap 和 kneighbors, 拷贝到 GPU
// 与 iq2_xs.h 中的 iq2xs_init() 算法完全一致
// ============================================================================

// 返回 0 成功, 非 0 失败
int iq2xs_quantize_init_impl(void) {
    if (g_kmap_gpu) return 0; // 已初始化

    const int grid_size = 512;
    const int kmap_size = IQ2_XS_KMAP_SIZE_Q;
    const int nwant = 2;

    // 步骤 1: 从 kgrid_2bit_512 展开逻辑网格 (值域 {1, 3, 5})
    uint64_t * the_grid = (uint64_t *)malloc(grid_size * sizeof(uint64_t));
    for (int k = 0; k < grid_size; ++k) {
        int8_t * pos = (int8_t *)(the_grid + k);
        for (int i = 0; i < 8; ++i) {
            int l = (h_kgrid_2bit_512[k] >> 2 * i) & 0x3;
            pos[i] = 2 * l + 1;
        }
    }

    // 拷贝逻辑网格到 GPU 常量内存 (4KB, 只读广播缓存)
    cudaError_t err = cudaMemcpyToSymbol(c_grid, the_grid, grid_size * sizeof(uint64_t));
    if (err != cudaSuccess) { free(the_grid); return -1; }

    // 步骤 2: 构建 kmap
    int * kmap = (int *)malloc(kmap_size * sizeof(int));
    for (int i = 0; i < kmap_size; ++i) kmap[i] = -1;

    uint64_t aux64;
    uint8_t * aux8 = (uint8_t *)&aux64;
    for (int i = 0; i < grid_size; ++i) {
        aux64 = the_grid[i];
        uint16_t index = 0;
        for (int k = 0; k < 8; ++k) {
            uint16_t q = (aux8[k] - 1) / 2;
            index |= (q << 2 * k);
        }
        kmap[index] = i;
    }

    // 步骤 3: 为 off-grid 点构建 kneighbors
    int8_t pos[8];
    int * dist2 = (int *)malloc(2 * grid_size * sizeof(int));
    int num_neighbors = 0, num_not_in_map = 0;

    // 第一遍: 计算总邻居数
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap[i] >= 0) continue;
        ++num_not_in_map;
        for (int k = 0; k < 8; ++k) {
            int l = (i >> 2 * k) & 0x3;
            pos[k] = 2 * l + 1;
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(the_grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), iq2_compare_func);
        int n = 0, d2 = dist2[0], nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2 * j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2 * j];
                ++nhave;
            }
            ++n;
        }
        num_neighbors += n;
    }

    // 第二遍: 填充 kneighbors 数组
    uint16_t * kneighbors = (uint16_t *)malloc((num_neighbors + num_not_in_map) * sizeof(uint16_t));
    int counter = 0;

    for (int i = 0; i < kmap_size; ++i) {
        if (kmap[i] >= 0) continue;
        for (int k = 0; k < 8; ++k) {
            int l = (i >> 2 * k) & 0x3;
            pos[k] = 2 * l + 1;
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(the_grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), iq2_compare_func);
        kmap[i] = -(counter + 1);
        int d2 = dist2[0];
        uint16_t * start = &kneighbors[counter++];
        int n = 0, nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2 * j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2 * j];
                ++nhave;
            }
            kneighbors[counter++] = (uint16_t)dist2[2 * j + 1];
            ++n;
        }
        *start = (uint16_t)n;
    }
    free(dist2);
    free(the_grid);

    g_kneighbors_size = num_neighbors + num_not_in_map;

    // 拷贝 kmap 到 GPU
    err = cudaMalloc(&g_kmap_gpu, kmap_size * sizeof(int32_t));
    if (err != cudaSuccess) {
        free(kmap); free(kneighbors); return -1;
    }
    cudaMemcpy(g_kmap_gpu, kmap, kmap_size * sizeof(int32_t), cudaMemcpyHostToDevice);
    free(kmap);

    // 拷贝 kneighbors 到 GPU
    err = cudaMalloc(&g_kneighbors_gpu, g_kneighbors_size * sizeof(uint16_t));
    if (err != cudaSuccess) {
        free(kneighbors); return -1;
    }
    cudaMemcpy(g_kneighbors_gpu, kneighbors, g_kneighbors_size * sizeof(uint16_t), cudaMemcpyHostToDevice);
    free(kneighbors);

    return 0;
}

// 释放 GPU 内存
void iq2xs_quantize_free_impl(void) {
    if (g_kmap_gpu)      { cudaFree(g_kmap_gpu);      g_kmap_gpu = nullptr; }
    if (g_kneighbors_gpu){ cudaFree(g_kneighbors_gpu); g_kneighbors_gpu = nullptr; }
    g_kneighbors_size = 0;
}

// ============================================================================
// 设备端辅助函数
// ============================================================================

// 快速取整 (与 iq2_xs.h nearest_int 一致)
static __device__ __forceinline__ int nearest_int_cuda(float fval) {
    float val = fval + 12582912.f;
    int i = __float_as_int(val);
    return (i & 0x007fffff) - 0x00400000;
}

// 在邻居列表中找最佳 grid 索引 (与 iq2_xs.h iq2_find_best_neighbour 一致)
static __device__ int iq2_find_best_neighbour_cuda(
    const uint16_t * neighbours,
    const uint64_t * grid,
    const float    * xval,
    const float    * weight,
    float            scale,
    int8_t         * L)
{
    int num_neighbors = neighbours[0];
    float best_d2 = FLT_MAX;
    int grid_index = -1;

    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }

    // 从最佳 grid 点反推 L 值
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1) / 2;

    return grid_index;
}

// ============================================================================
// 优化版 CUDA kernel: 16 threads/block, 子块并行 + 共享内存协作
// ============================================================================
// 设计:
//   - 16 threads/block, 每个 thread (tid 0-15) 处理一个 16 元素子块
//   - 子块间独立: 19 候选搜索在子块内串行, 子块间并行
//   - Warp shuffle 用于 sumx2 归约 (16 线程在同一 warp 内)
//   - 共享内存: s_scales[16] 和 s_q2[32] 用于最终编码汇聚
//   - grid 查找表放常量内存 (4KB, 只读, 广播缓存)
//   - maxrregcount=80 限制寄存器使用, 提高 occupancy (编译选项控制)
//   - 相比 1 thread/block 版本, occupancy 从 ~0.07% 提升到 ~1%
// ============================================================================

__global__ void iq2xs_quantize_kernel(
    const float*    __restrict__ x,        // [n_blocks, QK_K] float32 输入
    block_iq2_xs*  __restrict__ out,       // [n_blocks] 输出
    int                          n_blocks,
    const int32_t*  __restrict__ kmap,
    const uint16_t* __restrict__ kneighbors)
{
    const int ibl = blockIdx.x;
    if (ibl >= n_blocks) return;

    const int tid = threadIdx.x;  // 0-15, 每个 thread 处理一个子块
    if (tid >= 16) return;

    const int kMaxQ = IQ2_XS_KMAXQ_Q;
    const float * xbl = x + QK_K_Q * ibl;

    // 共享内存: 子块间数据汇聚 (仅 196 字节)
    __shared__ float    s_scales[16];   // 每个子块的 scale
    __shared__ uint16_t s_q2[32];       // 每个子块的 2 个 qs 编码值

    // 初始化输出 (thread 0 负责)
    if (tid == 0) {
        out[ibl].d = __float2half(0.f);
        for (int i = 0; i < QK_K_Q / 8; ++i)  out[ibl].qs[i] = 0;
        for (int i = 0; i < QK_K_Q / 32; ++i) out[ibl].scales[i] = 0;
    }

    // 步骤 0: 计算 sigma2 (warp shuffle 归约, 无需共享内存)
    const float * xb = xbl + 16 * tid;
    float sumx2 = 0;
    for (int i = 0; i < 16; ++i) sumx2 += xb[i] * xb[i];

    // Warp shuffle 归约: 16 线程的 sumx2 求和
    // mask=0xffff: 仅低 16 线程参与; 归约后 thread 0 持有总和
    for (int offset = 8; offset > 0; offset >>= 1) {
        sumx2 += __shfl_down_sync(0xffff, sumx2, offset);
    }
    // 广播 thread 0 的总 sumx2, 各 thread 自行计算 sigma2
    float sigma2 = __shfl_sync(0xffff, sumx2, 0) / QK_K_Q;

    // 步骤 1-5: 每个 thread 独立处理自己的 16 元素子块
    float  w[16];
    float  xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float  waux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];

    // 重要性权重
    for (int i = 0; i < 16; ++i) {
        w[i] = sqrtf(sigma2 + xb[i] * xb[i]);
    }
    for (int i = 0; i < 16; ++i) waux[i] = sqrtf(w[i]);

    // 步骤 1: 符号提取 — 将每 8 个元素分为 2 组，确保每组负数个数为偶数
    for (int k = 0; k < 2; ++k) {
        int nflip = 0;
        uint8_t s = 0;
        for (int i = 0; i < 8; ++i) {
            if (xb[8 * k + i] >= 0) {
                xval[8 * k + i] = xb[8 * k + i];
            } else {
                xval[8 * k + i] = -xb[8 * k + i];
                ++nflip;
                s |= (1 << i);
            }
        }
        if (nflip % 2) {
            int imin = 0;
            float min_val = w[8 * k] * xb[8 * k] * xb[8 * k];
            for (int i = 1; i < 8; ++i) {
                float ax = w[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                if (ax < min_val) { min_val = ax; imin = i; }
            }
            xval[8 * k + imin] = -xval[8 * k + imin];
            s ^= (1 << imin);
        }
        block_signs[k] = s & 127;
    }

    // 步骤 2: 19 候选搜索
    float max_val = xval[0];
    for (int i = 1; i < 16; ++i) max_val = fmaxf(max_val, xval[i]);
    for (int i = 0; i < 16; ++i) L[i] = 0;

    float scale = 0;
    if (max_val >= GROUP_MAX_EPS_Q) {
        float best = 0;
        scale = max_val / (2 * kMaxQ - 1);
        is_on_grid[0] = is_on_grid[1] = true;

        for (int is = -9; is <= 9; ++is) {
            float id = (2 * kMaxQ - 1 + is * 0.1f) / max_val;
            float this_scale = 1.0f / id;

            for (int k = 0; k < 2; ++k) {
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int_cuda(0.5f * (id * xval[8 * k + i] - 1));
                    Laux[8 * k + i] = (int8_t)max(0, min(kMaxQ - 1, l));
                }
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (Laux[8 * k + i] << 2 * i);
                int grid_index = kmap[u];
                is_on_grid_aux[k] = true;
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    const uint16_t * neighbours = kneighbors - kmap[u] - 1;
                    grid_index = iq2_find_best_neighbour_cuda(
                        neighbours, c_grid, xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
                }
            }

            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 16; ++i) {
                float wi = w[i];
                float q = 2 * Laux[i] + 1;
                sumqx += wi * xval[i] * q;
                sumq2 += wi * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                for (int k = 0; k < 2; ++k) is_on_grid[k] = is_on_grid_aux[k];
            }
        }

        // 步骤 3: off-grid 修正
        int n_not_ongrid = 0;
        for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
        if (n_not_ongrid > 0 && scale > 0) {
            float id = 1.0f / scale;
            for (int k = 0; k < 2; ++k) {
                if (is_on_grid[k]) continue;
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int_cuda(0.5f * (id * xval[8 * k + i] - 1));
                    l = max(0, min(kMaxQ - 1, l));
                    u |= (l << 2 * i);
                    L[8 * k + i] = (int8_t)l;
                }
                int grid_index = kmap[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = kneighbors - kmap[u] - 1;
                    grid_index = iq2_find_best_neighbour_cuda(
                        neighbours, c_grid, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 16; ++i) {
                float wi = w[i];
                float q = 2 * L[i] + 1;
                sumqx += wi * xval[i] * q;
                sumq2 += wi * q * q;
            }
            if (sumq2 > 0) scale = sumqx / sumq2;
        }

        // 步骤 4: 负 scale 处理
        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 2; ++k) block_signs[k] = (~block_signs[k]) & 127;
        }

        // 步骤 5: 编码 qs (写入共享内存)
        for (int k = 0; k < 2; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) u |= (L[8 * k + i] << 2 * i);
            int grid_index = kmap[u];
            if (grid_index < 0) grid_index = 0;
            s_q2[2 * tid + k] = (uint16_t)(grid_index | (block_signs[k] << 9));
        }
    } else {
        // 全零子块: q2 保持 0
        s_q2[2 * tid + 0] = 0;
        s_q2[2 * tid + 1] = 0;
    }

    // 将子块 scale 写入共享内存
    s_scales[tid] = scale;

    // 同步: 确保所有 thread 完成子块处理, 共享内存数据就绪
    __syncthreads();

    // 步骤 6: thread 0 编码 d 和 scale_4bit, 写出最终结果
    if (tid == 0) {
        float max_scale = 0;
        for (int i = 0; i < 16; ++i) max_scale = fmaxf(max_scale, s_scales[i]);

        if (!max_scale) {
            for (int i = 0; i < QK_K_Q / 8; ++i) out[ibl].qs[i] = 0;
            return;
        }

        float d = max_scale / 31.0f;
        out[ibl].d = __float2half(d);
        float id = 1.0f / d;

        for (int ib = 0; ib < QK_K_Q / 16; ++ib) {
            int l = nearest_int_cuda(0.5f * (id * s_scales[ib] - 1));
            l = max(0, min(15, l));
            if (ib % 2 == 0) out[ibl].scales[ib / 2] = (uint8_t)l;
            else out[ibl].scales[ib / 2] |= (uint8_t)(l << 4);
        }

        for (int i = 0; i < QK_K_Q / 8; ++i) out[ibl].qs[i] = s_q2[i];
    }
}

// ============================================================================
// CUDA 入口函数
// ============================================================================

void iq2xs_quantize_cuda(
    const float*    x,
    block_iq2_xs* out,
    int             n_blocks,
    const int32_t*  kmap,
    const uint16_t* kneighbors,
    cudaStream_t    stream)
{
    // 16 threads/block: 每个 thread 处理一个 16 元素子块, 子块间并行
    // 共享内存用于 sigma2 广播和 scales/q2 汇聚
    iq2xs_quantize_kernel<<<n_blocks, 16, 0, stream>>>(
        x, out, n_blocks, kmap, kneighbors);
}

// ============================================================================
// FFI 桥接函数 (extern "C" 避免 C++ name mangling)
// ============================================================================

extern "C" {

// 初始化 CUDA 量化 (构建 kmap/kneighbors 并拷贝到 GPU)
int iq2xs_quantize_init(void);

// 释放 GPU 内存
void iq2xs_quantize_free(void);

// FFI interface for Python/Rust
// src: float32 data on GPU [nrow * n_per_row]
// dst: block_iq2_xs output on GPU [n_blocks]
// nrow: number of rows
// n_per_row: elements per row (must be multiple of 256)
// stream: CUDA stream (NULL = 默认流)
void iq2xs_quantize_ffi(
    const float* src,
    void*        dst,
    int          nrow,
    int          n_per_row,
    void*        stream);

} // extern "C"

// FFI 实现
int iq2xs_quantize_init(void) {
    return iq2xs_quantize_init_impl();
}

void iq2xs_quantize_free(void) {
    iq2xs_quantize_free_impl();
}

void iq2xs_quantize_ffi(
    const float* src,
    void*        dst,
    int          nrow,
    int          n_per_row,
    void*        stream)
{
    // 确保已初始化
    iq2xs_quantize_init();

    int nblock_per_row = n_per_row / QK_K_Q;
    int total_blocks = nrow * nblock_per_row;

    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : 0;

    // 一次性启动所有 block，利用 GPU 大规模并行
    // 数据布局: src 是 [nrow, n_per_row] 连续存储，与 [total_blocks, 256] 等价
    // 因为 n_per_row 是 256 的倍数，所以每行的 block 在内存中连续
    iq2xs_quantize_kernel<<<total_blocks, 16, 0, cu_stream>>>(
        src, (block_iq2_xs*)dst, total_blocks,
        g_kmap_gpu, g_kneighbors_gpu);
}
