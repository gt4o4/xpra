/*
 * This file is part of Xpra.
 * Copyright (C) 2013-2026 Antoine Martin <antoine@xpra.org>
 * Xpra is released under the terms of the GNU GPL v2, or, at your option, any
 * later version. See the file COPYING for details.
 */

#include <stdint.h>

// input pixel byte order (XRGB)
#define PX_B 3
#define PX_G 2
#define PX_R 1

// full-range BT.601 R'G'B' -> Y'CbCr (JFIF), matching the SPS VUI
// declaration (videoFullRangeFlag=1, colourMatrix=6):
// Y = 0.299 R' + 0.587 G' + 0.114 B'
#define YR 0.299f
#define YG 0.587f
#define YB 0.114f
// U = -0.168736 R' - 0.331264 G' + 0.5 B' + 128
#define UR -0.168736f
#define UG -0.331264f
#define UB 0.5f
// V = 0.5 R' - 0.418688 G' - 0.081312 B' + 128
#define VR 0.5f
#define VG -0.418688f
#define VB -0.081312f

#define PI_F 3.14159265358979f

/* Resampling happens in LINEAR light: sRGB is decoded before the
 * filter and re-encoded after it, so the filter averages photons,
 * not gamma codes (gamma-domain averaging darkens fine bright
 * detail - exactly the content, thin text, this path carries). */
__device__ __forceinline__ float srgb_oetf(float c)
{
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

/* Lanczos3: sinc(x)*sinc(x/3) for |x|<3 */
__device__ __forceinline__ float lanczos3(float x)
{
    x = fabsf(x);
    if (x >= 3.0f)
        return 0.0f;
    if (x < 1e-5f)
        return 1.0f;
    const float px = PI_F * x;
    return 3.0f * sinf(px) * sinf(px * (1.0f / 3.0f)) / (px * px);
}

__device__ __forceinline__ void read_lin(const uint8_t *src, int pitch,
                                         int vw, int vh, int x, int y,
                                         const float *lut, float rgb[3])
{
    x = min(max(x, 0), vw - 1);         // clamp-to-edge on the VALID region
    y = min(max(y, 0), vh - 1);
    const uint8_t *p = src + (size_t)y * (size_t)pitch + (size_t)x * 4;
    rgb[0] = lut[p[PX_R]];
    rgb[1] = lut[p[PX_G]];
    rgb[2] = lut[p[PX_B]];
}

/* Separable Lanczos3 at continuous source position (sx,sy), kernel
 * stretched by (fsx,fsy) >= 1 so a downscale is band-limited (the
 * stretch IS the low-pass).  Windows wider than MAX_TAPS (filter
 * scale > 2.5; chroma runs at 2x the luma scale, so any downscale
 * beyond ~4/5 for chroma / 2/5 for luma) are truncated to MAX_TAPS
 * taps CENTERED on the sample position and renormalized: a centered
 * cut keeps the (all-positive) main lobe dominant - no shift, no
 * negative-sum inversion - trading only sharpness of the cutoff.
 * (A one-sided cut here mis-sited chroma by up to 1.75px at 1/3 and
 * inverted the filter outright at 1/4 - found by proofread.)
 * Anti-ringing: blend toward the clamp against the nearest-2x2
 * extremes with strength `ar` (0 disables). */
#define MAX_TAPS 16
__device__ void sample_lanczos(const uint8_t *src, int pitch, int vw, int vh,
                               const float *lut, float sx, float sy,
                               float fsx, float fsy, float ar, float out[3])
{
    int x0 = (int)ceilf(sx - 3.0f * fsx);
    int x1 = (int)floorf(sx + 3.0f * fsx);
    int y0 = (int)ceilf(sy - 3.0f * fsy);
    int y1 = (int)floorf(sy + 3.0f * fsy);
    if (x1 - x0 >= MAX_TAPS) {
        x0 = (int)floorf(sx) - (MAX_TAPS / 2 - 1);
        x1 = x0 + MAX_TAPS - 1;
    }
    if (y1 - y0 >= MAX_TAPS) {
        y0 = (int)floorf(sy) - (MAX_TAPS / 2 - 1);
        y1 = y0 + MAX_TAPS - 1;
    }
    const int nx = x1 - x0 + 1;
    const int ny = y1 - y0 + 1;

    float wx[MAX_TAPS], wy[MAX_TAPS];
    float sum = 0.0f;
    for (int i = 0; i < nx; i++) {
        wx[i] = lanczos3((sx - (float)(x0 + i)) / fsx);
        sum += wx[i];
    }
    const float inx = 1.0f / sum;
    sum = 0.0f;
    for (int j = 0; j < ny; j++) {
        wy[j] = lanczos3((sy - (float)(y0 + j)) / fsy);
        sum += wy[j];
    }
    const float iny = 1.0f / sum;

    float acc[3] = { 0.0f, 0.0f, 0.0f };
    for (int j = 0; j < ny; j++) {
        float row[3] = { 0.0f, 0.0f, 0.0f };
        for (int i = 0; i < nx; i++) {
            float rgb[3];
            read_lin(src, pitch, vw, vh, x0 + i, y0 + j, lut, rgb);
            row[0] += wx[i] * rgb[0];
            row[1] += wx[i] * rgb[1];
            row[2] += wx[i] * rgb[2];
        }
        acc[0] += wy[j] * row[0];
        acc[1] += wy[j] * row[1];
        acc[2] += wy[j] * row[2];
    }
    out[0] = acc[0] * inx * iny;
    out[1] = acc[1] * inx * iny;
    out[2] = acc[2] * inx * iny;

    if (ar > 0.0f) {
        const int xa = (int)floorf(sx);
        const int ya = (int)floorf(sy);
        float lo[3] = { 1e9f, 1e9f, 1e9f };
        float hi[3] = { -1e9f, -1e9f, -1e9f };
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++) {
                float rgb[3];
                read_lin(src, pitch, vw, vh, xa + i, ya + j, lut, rgb);
                for (int c = 0; c < 3; c++) {
                    lo[c] = fminf(lo[c], rgb[c]);
                    hi[c] = fmaxf(hi[c], rgb[c]);
                }
            }
        for (int c = 0; c < 3; c++) {
            const float cl = fminf(fmaxf(out[c], lo[c]), hi[c]);
            out[c] = out[c] + (cl - out[c]) * ar;
        }
    }
}

__device__ __forceinline__ uint8_t quant(float v)
{
    return (uint8_t)min(max(__float2int_rn(v), 0), 255);
}

/*
 * XRGB -> NV12 with fused Lanczos3 linear-light scaling.
 *
 * The source VALID region is src_w x src_h (srcPitch bytes per row);
 * the destination is the FULL padded coded frame dst_w x dst_h, and
 * w x h is the CONTENT region within it:
 *
 *  - When the server scales, the caller passes w,h = dst_w,dst_h:
 *    content deliberately fills the 32-alignment padding ("scale
 *    into the padding" - every coded pixel carries content), and
 *    the scaled_size the client receives is the padded size, so its
 *    crop matches.
 *  - When the server does NOT scale, w,h = the true source size:
 *    content stays 1:1 (bit-stable, no resampling) and the padding
 *    beyond it is EDGE-EXTENDED - deterministic, cheap to code, and
 *    the client (which received no scaled_size) crops exactly the
 *    1:1 region.  Stretching unscaled content into the padding
 *    would desync it from the client's crop by up to 31 columns.
 *
 * Mapping uses pixel centers: src = (dst + 0.5) * (src_w / w) - 0.5,
 * with output coordinates clamped into the content region first
 * (the clamp IS the edge extension).  Chroma is sited per H.264
 * chroma_sample_loc_type 0 (MPEG-2): co-sited horizontally with
 * even luma columns, centered vertically - the chroma sample
 * (cx,cy) lives at luma coords (2cx, 2cy+0.5) and is band-limited
 * at twice the luma filter scale.
 *
 * Launch geometry (exec_kernel): one thread per 2x2 luma block =
 * one chroma sample; the grid covers the PADDED dst dims.
 */
extern "C" __global__ void XRGB_to_NV12(uint8_t *srcImage, int src_w, int src_h, int srcPitch,
                          uint8_t *dstImage, int dst_w, int dst_h, int dstPitch,
                          int w, int h)
{
    __shared__ float lut[256];
    {
        // cooperative sRGB EOTF LUT (exact piecewise decode)
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int nthreads = blockDim.x * blockDim.y;
        for (int i = tid; i < 256; i += nthreads) {
            const float c = (float)i / 255.0f;
            lut[i] = c <= 0.04045f ? c * (1.0f / 12.92f)
                                   : powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
        }
    }
    __syncthreads();

    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int ox = gx * 2;
    const int oy = gy * 2;
    if (ox >= dst_w || oy >= dst_h)
        return;

    uint8_t *yplane = dstImage;
    uint8_t *cplane = dstImage + (size_t)dst_h * (size_t)dstPitch;

    if (src_w == w && src_h == h) {
        // 1:1 content - no resampling: direct gamma-domain conversion
        // (an unscaled stream is untouched by any filter); output
        // coordinates clamp into the content region = edge extension
        // for the 32-alignment padding
        float px0[2][3];
        for (int j = 0; j < 2; j++) {
            const int y = min(min(oy + j, h - 1), src_h - 1);
            for (int i = 0; i < 2; i++) {
                const int x = min(min(ox + i, w - 1), src_w - 1);
                const uint8_t *p = srcImage + (size_t)y * (size_t)srcPitch + (size_t)x * 4;
                const float R = (float)p[PX_R], G = (float)p[PX_G], B = (float)p[PX_B];
                if (ox + i < dst_w && oy + j < dst_h)
                    yplane[(size_t)(oy + j) * dstPitch + (ox + i)] =
                        quant(YR * R + YG * G + YB * B);
                if (i == 0) {
                    px0[j][0] = R;
                    px0[j][1] = G;
                    px0[j][2] = B;
                }
            }
        }
        // chroma, type-0 siting at 1:1: co-sited column 2gx, rows
        // 2gy/2gy+1 averaged (nonlinear averaging, codec-conventional)
        const float R = 0.5f * (px0[0][0] + px0[1][0]);
        const float G = 0.5f * (px0[0][1] + px0[1][1]);
        const float B = 0.5f * (px0[0][2] + px0[1][2]);
        uint8_t *c = cplane + (size_t)gy * dstPitch + (size_t)gx * 2;
        c[0] = quant(UR * R + UG * G + UB * B + 128.0f);
        c[1] = quant(VR * R + VG * G + VB * B + 128.0f);
        return;
    }

    const float rx = (float)src_w / (float)w;
    const float ry = (float)src_h / (float)h;
    const float fsx = fmaxf(rx, 1.0f);
    const float fsy = fmaxf(ry, 1.0f);

    // 4 luma samples, anti-ringing 0.8 (text rings hard under a
    // negative-lobe filter; the clamp keeps edges clean); output
    // coordinates clamp into the content region first (padding =
    // edge extension)
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            const int dx = ox + i;
            const int dy = oy + j;
            if (dx >= dst_w || dy >= dst_h)
                continue;
            const int cx = min(dx, w - 1);
            const int cy = min(dy, h - 1);
            const float sx = ((float)cx + 0.5f) * rx - 0.5f;
            const float sy = ((float)cy + 0.5f) * ry - 0.5f;
            float lin[3];
            sample_lanczos(srcImage, srcPitch, src_w, src_h, lut, sx, sy, fsx, fsy, 0.8f, lin);
            const float R = srgb_oetf(lin[0]) * 255.0f;
            const float G = srgb_oetf(lin[1]) * 255.0f;
            const float B = srgb_oetf(lin[2]) * 255.0f;
            yplane[(size_t)dy * dstPitch + dx] = quant(YR * R + YG * G + YB * B);
        }
    }

    // 1 chroma sample at the type-0 sited position, band-limited at
    // twice the luma filter scale (half the output rate); no AR -
    // chroma ringing is not visible and the clamp costs extra reads
    {
        const float lumx = fminf((float)(2 * gx), (float)(w - 1));
        const float lumy = fminf((float)(2 * gy) + 0.5f, (float)(h - 1));
        const float sx = (lumx + 0.5f) * rx - 0.5f;
        const float sy = (lumy + 0.5f) * ry - 0.5f;
        float lin[3];
        sample_lanczos(srcImage, srcPitch, src_w, src_h, lut, sx, sy, 2.0f * fsx, 2.0f * fsy, 0.0f, lin);
        const float R = srgb_oetf(lin[0]) * 255.0f;
        const float G = srgb_oetf(lin[1]) * 255.0f;
        const float B = srgb_oetf(lin[2]) * 255.0f;
        uint8_t *c = cplane + (size_t)gy * dstPitch + (size_t)gx * 2;
        c[0] = quant(UR * R + UG * G + UB * B + 128.0f);
        c[1] = quant(VR * R + VG * G + VB * B + 128.0f);
    }
}
