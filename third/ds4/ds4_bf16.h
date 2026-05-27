#ifndef DS4_BF16_H
#define DS4_BF16_H

#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#define DS4_BF16_HD __host__ __device__
#else
#define DS4_BF16_HD
#endif

static DS4_BF16_HD inline uint16_t ds4_f32_to_bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    u = (u + 0x8000) & 0xFFFF0000u;
    return (uint16_t)(u >> 16);
}

static DS4_BF16_HD inline float ds4_bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

#endif
