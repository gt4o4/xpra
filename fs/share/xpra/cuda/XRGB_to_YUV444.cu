/*
 * This file is part of Xpra.
 * Copyright (C) 2013-2024 Antoine Martin <antoine@xpra.org>
 * Xpra is released under the terms of the GNU GPL v2, or, at your option, any
 * later version. See the file COPYING for details.
 */

#include <stdint.h>

__device__ __forceinline__ uint8_t quant(float v)
{
    return (uint8_t)min(max(__float2int_rn(v), 0), 255);
}

// Y = 0.299 * R + 0.587 * G + 0.114 * B + 0
#define YR 0.299
#define YG 0.587
#define YB 0.114
#define YC 0
// U = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
#define UR -0.168736
#define UG -0.331264
#define UB 0.5
#define UC 128
// V = 0.5 * R - 0.418688 * G - 0.081312 * B + 128
#define VR 0.5
#define VG -0.418688
#define VB -0.081312
#define VC 128

/*
 * XRGB -> planar YUV444, same argument contract as the NV12 kernels:
 * src_w x src_h is the VALID source region, dst_w x dst_h the padded
 * coded frame, and w x h the CONTENT region within it (== dst when
 * the caller scales into the padding, == the true source size when
 * not scaling - the padding is then edge-extended by the coordinate
 * clamp).  Nearest-neighbor with pixel-center mapping.
 */
extern "C" __global__ void XRGB_to_YUV444(uint8_t *srcImage, int src_w, int src_h, int srcPitch,
                             uint8_t *dstImage, int dst_w, int dst_h, int dstPitch,
                             int w, int h)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h)
        return;
    // clamp into the content region (edge extension for the padding),
    // then map by pixel centers and round to the nearest source pixel
    const int cx = min(gx, w - 1);
    const int cy = min(gy, h - 1);
    int src_x = (int)(((float)cx + 0.5f) * (float)src_w / (float)w);
    int src_y = (int)(((float)cy + 0.5f) * (float)src_h / (float)h);
    src_x = min(max(src_x, 0), src_w - 1);
    src_y = min(max(src_y, 0), src_h - 1);

    //one 32-bit RGB pixel at a time:
    const uint32_t si = ((uint32_t)src_y * srcPitch) + (uint32_t)src_x * 4;
    const uint8_t R = srcImage[si+1];
    const uint8_t G = srcImage[si+2];
    const uint8_t B = srcImage[si+3];

    uint32_t di = ((uint32_t)gy * dstPitch) + (uint32_t)gx;
    // clamped quantization: pure blue/red hit U/V = 255.5 exactly,
    // and a bare __float2int_rn rounds that to 256 = wrap to 0
    dstImage[di] = quant(YR * R + YG * G + YB * B + YC);
    di += dstPitch*dst_h;
    dstImage[di] = quant(UR * R + UG * G + UB * B + UC);
    di += dstPitch*dst_h;
    dstImage[di] = quant(VR * R + VG * G + VB * B + VC);
}
