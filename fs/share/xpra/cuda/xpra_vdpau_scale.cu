/*
 * xpra_vdpau_scale.cu - VDPAU field textures -> H-filtered 10:10:10 strip
 *
 * Ships as SOURCE here like the encoder-side CSC kernels; the client
 * build compiles it to an sm_11 cubin (nvcc 6.5 - the only compiler
 * that still targets the GeForce GT 130's compute 1.1) installed
 * beside the fatbins under share/xpra/cuda/.  Loaded at runtime by
 * xpra/codecs/cuda_vdpau.pyx via pycuda's driver API (module load,
 * texref binding, launch) inside the cudart PRIMARY context the
 * cuda_vdpau bootstrap has VDPAU-bound; the interop registration and
 * mapping stay in xpra/codecs/gpujpeg/decoder.pyx (cudart).  There is
 * deliberately NO host launcher in this file: geometry, validation,
 * texture state and the launch all live Python-side.
 *
 * DRIVER-API TEXTURING NOTE: a texture's READ MODE is host-side
 * state, not cubin metadata - the runtime API carries it in the
 * host-side texture<> object, the driver API does not.  The loader
 * must therefore set each texref's format after every set_array
 * (CU_TRSA_OVERRIDE_FORMAT adopts the array's format) and leave the
 * flags at 0: these references are cudaReadModeNormalizedFloat, which
 * IS the driver default (TRSF_READ_AS_INTEGER would be wrong).
 *
 * THE HYBRID SHAPE: this file provides the CUDA half of the scaled
 * VDPAU paint - ONE kernel that converts and HORIZONTALLY filters
 * each band's source rows into an H-strip inside a CUDA-registered
 * GL buffer; xpra's lanczos-v GL shader applies the vertical taps,
 * anti-ringing and the display epilogue while rastering into the
 * (tiled) offscreen texture.  Why the split is exactly here:
 *  - conversion (~6 transcendentals/texel) must run ONCE per source
 *    texel, not per filter tap (36x);
 *  - the horizontal taps share data only ALONG a row - a block's
 *    column window fits shared memory, so H fuses with conversion at
 *    zero structural cost;
 *  - the vertical taps share data ACROSS rows - that sharing needs a
 *    materialization boundary on sm_11 (no grid-wide sync, no
 *    self-visible cache), and the GL raster pass, which is mandatory
 *    anyway (only GL writes tiled textures here), provides it free.
 * Measured on the GT 130 at 1856x1088 -> 1920x1173: this kernel
 * ~19ms/frame (conversion-bound - tap arrays, multi-row weight
 * reuse and arrayless forms all measured identical, the ~6
 * transcendentals per source texel are the floor) + the GL vertical
 * pass ~1ms/band-of-128, ~29ms total vs 37ms for the previous
 * 2-kernel + raster design - and the BGRX band PBO and its extra
 * VRAM round-trip are gone.
 *
 * MATH CONTRACT - the P1+H math here is validated against the golden
 * test's CPU model (scratchpad vdpau-repro/scale-golden.c, <=1 LSB in
 * 10-bit, driving the SAME cubin the client ships); the vertical
 * half's contract lives in the GLSL shader
 * (xpra/opengl/shaders.py LANCZOS_V_SHADER), validated live.  The
 * encoder implements the same resampling contract in its own CUDA
 * CSC kernels (linear light, Lanczos3, type-0 chroma siting,
 * pixel-centre mapping).  Per strip texel:
 *   1. source texel: field weave (luma row clamp BEFORE the parity
 *      split), chroma bilinear at MPEG-2 4:2:0 siting (BOTH axes
 *      clamped to the VIDEO chroma region - the sampler stops at the
 *      32-ALIGNED surface edge), Y'CbCr->R'G'B' BT.601, per-channel
 *      clamp, sRGB EOTF, sigmoid forward
 *   2. 6-tap horizontal Lanczos3: weights on UNCLAMPED tap
 *      coordinates, coordinates clamped to the video edge (edge
 *      replication with original weights), normalized
 *   3. 10:10:10 pack, clamping to [0,1] - this IS the between-passes
 *      row clamp the shader's anti-ringing depends on
 *
 * Sigmoid constants are duplicated from shaders.py (_SIG_*): centre
 * 0.75, slope 6.5.  Keep the two in sync.
 */

#include <stdint.h>
#include <math.h>

/* --- sigmoid (shaders.py _SIG_CENTER/_SIG_SLOPE and derivations) --- */
#define SIG_CENTER  0.75f
#define SIG_SLOPE   6.5f
/* _SIG_OFFSET = 1/(1+exp(slope*centre)) */
#define SIG_OFFSET  0.007577241f
/* _SIG_SCALE = 1/(1+exp(slope*(centre-1))) - _SIG_OFFSET */
#define SIG_SCALE   0.827906296f
#define SIG_INV_SLOPE (1.0f / SIG_SLOPE)

/* BT.601, full and studio range (CS_MULTIPLIERS["bt601"]) */
#define BT601_E   1.402f
#define BT601_D   1.772f
#define BT601_F  (-0.114f * 1.772f / 0.587f)
#define BT601_G  (-0.299f * 1.402f / 0.587f)
#define YMULT_STUDIO   1.1643835616438356f
#define UVMULT_STUDIO  1.1383928571428572f
#define YOFFSET_STUDIO 0.062745098f

/* sm_11: texture REFERENCES (texture objects need sm_30) */
texture<unsigned char, 2, cudaReadModeNormalizedFloat> tex_y0;
texture<unsigned char, 2, cudaReadModeNormalizedFloat> tex_y1;
texture<uchar2, 2, cudaReadModeNormalizedFloat> tex_c0;
texture<uchar2, 2, cudaReadModeNormalizedFloat> tex_c1;

__device__ __forceinline__ float lz(float x)
{
    /* Lanczos3 kernel, matching _LANCZOS_FN in shaders.py */
    x = fabsf(x);
    if (x >= 3.0f)
        return 0.0f;
    if (x < 1e-5f)
        return 1.0f;
    /* __sinf: SFU approximation - the accurate sinf is a ~60-insn
     * software expansion on sm_1x and this is the kernel's hottest
     * transcendental; abs error ~1e-5 over |px| < 3pi, invisible
     * through the 10-bit tile (golden test: max delta 1 LSB) */
    const float px = 3.14159265358979f * x;
    return 3.0f * __sinf(px) * __sinf(px / 3.0f) / (px * px);
}

__device__ __forceinline__ float srgb_eotf(float c)
{
    return (c <= 0.04045f) ? (c / 12.92f) : __powf((c + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ float sig_forward(float c)
{
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    return SIG_CENTER - __logf(1.0f / (c * SIG_SCALE + SIG_OFFSET) - 1.0f) * SIG_INV_SLOPE;
}

/* luma at absolute video (col, row): clamp the ROW first, then split by
 * parity into the half-height field texture (fY in shaders.py) */
__device__ __forceinline__ float fetch_y(int x, int r, int vid_h)
{
    r = min(max(r, 0), vid_h - 1);
    const float p = (float)(r >> 1) + 0.5f;
    return (r & 1) ? tex2D(tex_y1, (float)x + 0.5f, p)
                   : tex2D(tex_y0, (float)x + 0.5f, p);
}

/* chroma-plane texel (cx, cr): clamp BOTH axes to the video chroma
 * region, then the same parity split (fC in shaders.py) */
__device__ __forceinline__ float2 fetch_c(int cx, int cr, int vid_w, int vid_h)
{
    cx = min(max(cx, 0), (vid_w - 1) >> 1);
    cr = min(max(cr, 0), (vid_h - 1) >> 1);
    const float p = (float)(cr >> 1) + 0.5f;
    return (cr & 1) ? tex2D(tex_c1, (float)cx + 0.5f, p)
                    : tex2D(tex_c0, (float)cx + 0.5f, p);
}

/* one source texel of the (never materialized) linear+sigmoid plane.
 * MPEG-2 4:2:0 siting at integer (col, row) makes the bilinear
 * fractions constants: tx = col&1 ? 0.5 : 0, ty = row&1 ? 0.25 : 0.75
 * (crf = row/2 - 1/4), and the base cells are col>>1, (row-1)>>1 */
__device__ __forceinline__ float3 source_texel(int col, int row,
                                               int vid_w, int vid_h, int full_range)
{
    float y = fetch_y(col, row, vid_h);
    const int cbx = col >> 1;
    const int cby = (row - 1) >> 1;
    const float tx = (col & 1) ? 0.5f : 0.0f;
    const float ty = (row & 1) ? 0.25f : 0.75f;
    const float2 c00 = fetch_c(cbx,     cby,     vid_w, vid_h);
    const float2 c10 = fetch_c(cbx + 1, cby,     vid_w, vid_h);
    const float2 c01 = fetch_c(cbx,     cby + 1, vid_w, vid_h);
    const float2 c11 = fetch_c(cbx + 1, cby + 1, vid_w, vid_h);
    const float u_top = c00.x + (c10.x - c00.x) * tx;
    const float v_top = c00.y + (c10.y - c00.y) * tx;
    const float u_bot = c01.x + (c11.x - c01.x) * tx;
    const float v_bot = c01.y + (c11.y - c01.y) * tx;
    float u = u_top + (u_bot - u_top) * ty;
    float v = v_top + (v_bot - v_top) * ty;

    if (full_range) {
        u -= 0.5f;
        v -= 0.5f;
    } else {
        y = (y - YOFFSET_STUDIO) * YMULT_STUDIO;
        u = (u - 0.5f) * UVMULT_STUDIO;
        v = (v - 0.5f) * UVMULT_STUDIO;
    }
    float r = fminf(fmaxf(y + BT601_E * v, 0.0f), 1.0f);
    float g = fminf(fmaxf(y + BT601_F * u + BT601_G * v, 0.0f), 1.0f);
    float b = fminf(fmaxf(y + BT601_D * u, 0.0f), 1.0f);
    float3 out;
    out.x = sig_forward(srgb_eotf(r));
    out.y = sig_forward(srgb_eotf(g));
    out.z = sig_forward(srgb_eotf(b));
    return out;
}

/* strip texel: 10:10:10 unorm in a uint32 - a third of float3's
 * bandwidth, and 10 bits are enough BECAUSE the values are in sigmoid
 * space (the inverse sigmoid is compressive exactly where the sRGB
 * OETF is steep, keeping the display error under an 8-bit LSB) */
__device__ __forceinline__ unsigned int pack10(float3 v)
{
    const unsigned int r = (unsigned int)(fminf(fmaxf(v.x, 0.0f), 1.0f) * 1023.0f + 0.5f);
    const unsigned int g = (unsigned int)(fminf(fmaxf(v.y, 0.0f), 1.0f) * 1023.0f + 0.5f);
    const unsigned int b = (unsigned int)(fminf(fmaxf(v.z, 0.0f), 1.0f) * 1023.0f + 0.5f);
    return (r << 20) | (g << 10) | b;
}

/* ---- P1 + horizontal Lanczos in ONE kernel ------------------------
 * The horizontal 6-tap filter only shares data ALONG a row, and a row
 * segment fits a thread block's shared memory - so unlike the
 * vertical taps (cross-row sharing = grid-wide = a kernel boundary),
 * H can fuse with the conversion at zero structural cost: each block
 * converts its source-column window once into smem, then every thread
 * produces one H-filtered output column.  The strip this kernel
 * writes is OUT_W wide (H already applied), 10:10:10 packed - the
 * pack's [0,1] clamp IS the contract's between-passes row clamp - and
 * the GL raster pass finishes the job with the 6 vertical taps,
 * anti-ringing and the epilogue (measured 0.97ms/band on the GT 130
 * vs 2.3-3.1ms for any fused 36-tap variant; see scratchpad tbo-fused-probe.c).
 */
#define P1H_BLOCK_X  128
/* source-column capacity: covers 128 output columns at inv_x <= ~2.0
 * plus the 6-tap halo; deeper downscale transients (window shrunk far
 * below the video before the server re-encodes) take the direct
 * per-tap path below - correct, briefly slower */
#define P1H_SEG_CAP  268

extern "C" __global__ void __launch_bounds__(P1H_BLOCK_X)
xpra_p1h_kernel(unsigned int *strip, int pitch_words,
                int vid_w, int vid_h, int out_w,
                int src_y0, float inv_scale_x, int full_range)
{
    __shared__ float3 seg[P1H_SEG_CAP];
    const int ox0 = blockIdx.x * P1H_BLOCK_X;
    /* strip row blockIdx.y holds the CLAMPED video row (edge rows
     * replicate, keeping the row index an affine rebase downstream) */
    const int row = min(max(src_y0 + (int)blockIdx.y, 0), vid_h - 1);
    const float sx0 = (ox0 + 0.5f) * inv_scale_x - 0.5f;
    const int last_ox = min(ox0 + P1H_BLOCK_X, out_w) - 1;
    const float sx1 = (last_ox + 0.5f) * inv_scale_x - 0.5f;
    const int src_x0 = (int)floorf(sx0) - 2;
    const int span = (int)floorf(sx1) + 3 - src_x0 + 1;
    const int use_seg = span <= P1H_SEG_CAP;   /* uniform per block */

    if (use_seg) {
        for (int t = threadIdx.x; t < span; t += P1H_BLOCK_X) {
            const int col = min(max(src_x0 + t, 0), vid_w - 1);
            seg[t] = source_texel(col, row, vid_w, vid_h, full_range);
        }
        __syncthreads();
    }

    const int ox = ox0 + (int)threadIdx.x;
    if (ox >= out_w)
        return;
    const float sx = (ox + 0.5f) * inv_scale_x - 0.5f;
    const float bx = floorf(sx);
    /* NO tap arrays: everything inline in the unrolled loop - an
     * indexed local array would spill to DRAM on sm_11 */
    float3 acc = make_float3(0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;
#pragma unroll
    for (int i = -2; i <= 3; i++) {
        const float t = bx + (float)i;
        const float w = lz(sx - t);
        wsum += w;
        /* weights on UNCLAMPED taps, coordinates clamped to the video
         * edge (edge replication with original weights) */
        const int col = (int)fminf(fmaxf(t, 0.0f), (float)(vid_w - 1));
        float3 sv;
        if (use_seg) {
            int l = col - src_x0;
            l = l < 0 ? 0 : (l >= span ? span - 1 : l);
            sv = seg[l];
        } else {
            sv = source_texel(col, row, vid_w, vid_h, full_range);
        }
        acc.x += w * sv.x;
        acc.y += w * sv.y;
        acc.z += w * sv.z;
    }
    const float rcp = 1.0f / wsum;
    acc.x *= rcp;
    acc.y *= rcp;
    acc.z *= rcp;
    /* pack10 clamps to [0,1]: the between-passes row clamp the
     * anti-ringing step depends on, fused with the quantization */
    strip[(size_t)blockIdx.y * pitch_words + ox] = pack10(acc);
}

