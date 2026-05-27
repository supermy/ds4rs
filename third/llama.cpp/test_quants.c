/* test_quants.c — 验证 IQ2_XS / IQ2_XXS / Q2_K 反量化正确性
 *
 * IQ2_XS/XXS 量化需要 importance matrix (校准数据)，无法简单测试。
 * 因此只测试反量化路径：手动构造 block → 反量化 → 检查输出范围。
 * Q2_K 量化不需要 importance matrix，可以完整测试量化→反量化。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"

#define N 256  // QK_K = 256

static float rand_float(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / RAND_MAX);
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float max_diff = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

int main(void) {
    srand(42);

    // 初始化 IQ2 查找表（反量化需要）
    iq2xs_init_impl(17);  // EXTRACTED_GGML_TYPE_IQ2_XS = 17

    float src[N];
    float dst[N];
    int ok = 1;

    // === Q2_K 完整量化→反量化测试 ===
    {
        block_q2_K block;
        for (int i = 0; i < N; i++) src[i] = rand_float(-1.0f, 1.0f);

        quantize_row_q2_K_ref(src, &block, N);
        dequantize_row_q2_K(&block, dst, N);

        float mad = max_abs_diff(src, dst, N);
        float avg = 0;
        for (int i = 0; i < N; i++) avg += fabsf(src[i] - dst[i]);
        avg /= N;

        printf("[Q2_K] max_abs_diff=%.6f avg_abs_diff=%.6f\n", mad, avg);
        if (mad > 1.0f) {
            printf("  FAIL: max_abs_diff too large\n");
            ok = 0;
        } else {
            printf("  PASS\n");
        }
    }

    // === IQ2_XS 反量化测试：非零 block ===
    // 注意：iq2xs_grid[0] = 0x0808080808080808（非零），所以 qs=0 不代表输出为零
    {
        block_iq2_xs block;
        memset(&block, 0, sizeof(block));
        block.d = 2.0f;
        for (int i = 0; i < QK_K/8; i++) block.qs[i] = 1;
        for (int i = 0; i < QK_K/32; i++) block.scales[i] = 1;

        dequantize_row_iq2_xs(&block, dst, N);

        // 检查输出不全为零
        int any_nonzero = 0;
        for (int i = 0; i < N; i++) {
            if (dst[i] != 0.0f) { any_nonzero = 1; break; }
        }
        printf("[IQ2_XS] dequant nonzero block: %s\n", any_nonzero ? "PASS" : "FAIL");
        if (!any_nonzero) ok = 0;

        // 检查输出值在合理范围内
        float max_val = 0;
        for (int i = 0; i < N; i++) {
            float v = fabsf(dst[i]);
            if (v > max_val) max_val = v;
        }
        printf("[IQ2_XS] max output value: %.6f (expected < 10)\n", max_val);
        if (max_val > 10.0f) {
            printf("  FAIL: output values too large\n");
            ok = 0;
        }
    }

    // === IQ2_XXS 反量化测试：非零 block ===
    {
        block_iq2_xxs block;
        memset(&block, 0, sizeof(block));
        block.d = 2.0f;
        for (int i = 0; i < QK_K/8; i++) block.qs[i] = 1;

        dequantize_row_iq2_xxs(&block, dst, N);

        int any_nonzero = 0;
        for (int i = 0; i < N; i++) {
            if (dst[i] != 0.0f) { any_nonzero = 1; break; }
        }
        printf("[IQ2_XXS] dequant nonzero block: %s\n", any_nonzero ? "PASS" : "FAIL");
        if (!any_nonzero) ok = 0;

        float max_val = 0;
        for (int i = 0; i < N; i++) {
            float v = fabsf(dst[i]);
            if (v > max_val) max_val = v;
        }
        printf("[IQ2_XXS] max output value: %.6f (expected < 10)\n", max_val);
        if (max_val > 10.0f) {
            printf("  FAIL: output values too large\n");
            ok = 0;
        }
    }

    iq2xs_free_impl(17);

    printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
