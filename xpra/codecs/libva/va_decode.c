/* This file is part of Xpra.
 * Copyright (C) 2026 Netflix, Inc.
 * Xpra is released under the terms of the GNU GPL v2, or, at your option, any
 * later version. See the file COPYING for details.
 * ABOUTME: libva decoder - C implementation.
 * ABOUTME: Minimal VA-API H.264 decoder, zero-copy VdpVideoSurface export. */

#include "va_decode.h"
#include "va_common.h"

#include <va/va.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>     /* close() */
#else
#include <dirent.h>
#include <unistd.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <dlfcn.h>
#include <time.h>

#define LIBVA_LOG2_MAX_FRAME_NUM_MINUS4 4
#define LIBVA_LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4 4

static libva_log_fn g_log_fn = NULL;
static char g_device[256] = "";
static char g_vendor[256] = "";
static char g_error[256] = "";
static int g_major = 0;
static int g_minor = 0;
static int g_h264_420_supported = 0;
static VAProfile g_h264_420_profile = VAProfileH264ConstrainedBaseline;

struct H264Params;

/* H.264 decoded picture buffer: up to 16 reference frames (spec DPB
 * ceiling), progressive only (field coding is rejected at parse). */
#define H264_DPB_SIZE 16
/* export slack: frames handed to the GL painter (zero-copy path) not
 * yet released.  Steady-state is 1-2 in flight, but a window resize
 * stalls the paint pipeline (GL backing re-init) while the server
 * keeps sending, so decode runs ahead and pins pile up.  MEASURED
 * (two-window gears, maximize/unmaximize cycles): slack 3 = a
 * restart storm on every maximize (8 pool-exhaustions + 38 mid-GOP
 * slice errors cascading until the next IDR, ~15 decoder
 * generations); slack 5 = occasional single exhaustion per couple
 * of cycles; slack 8 = clean.  Exhaustion stays a decode error +
 * restart, never corruption.
 * 2026-07-25: dialed 8 -> 5 -> 3 (VRAM audit: a maximized window
 * cost ~295MB of the GT 130's 512; each slack frame is ~2.9MB at
 * 1856x1088).  The slack-3/5 numbers above are from the ORIGINAL
 * 2026-07-20 measurement; the pipeline has since gained the
 * partial-vaCreateSurfaces fix, per-decoder displays and faster
 * paints, and slack 5 retested CLEAN under the same churn (12
 * resize transitions incl. 1228x861: 0 exhaustions).  Slack 3 is
 * the next probe of the new floor - exhaustion is a clean decoder
 * restart (loud decode error, never corruption); restore 5 if the
 * maximize-churn gate logs exhaustions again. */
#define H264_EXPORT_SLACK 3
/* the surface ARRAYS are sized for the worst case (a full spec DPB);
 * the pool actually ALLOCATED is sized from the stream's own SPS at
 * first decode: max_num_ref_frames + 1 current + slack - an nvenc
 * nrf=3 stream gets 12 surfaces instead of 25 (VRAM: each surface is
 * a full NV12 frame) */
#define H264_NUM_SURFACES (H264_DPB_SIZE + 1 + H264_EXPORT_SLACK)
_Static_assert(H264_NUM_SURFACES <= 32, "pinned_bits and dpb_used are 32-bit sets");

/* describes the reference picture held by surfaces[i] - the surface
 * id itself lives ONLY in surfaces[] (same index, no duplication) */
struct H264DPBEntry {
    int frame_num;              /* FrameNum as coded */
    int top_foc;
    int bottom_foc;
    int is_long_term;
    int long_term_frame_idx;
};

struct LibVADecoder {
    int             fd;
    VADisplay       display;
    VAConfigID      config;
    VAContextID     context;
    VASurfaceID     surfaces[H264_NUM_SURFACES];
    int             num_surfaces;
    int             surface_index;
    /* zero-copy export path (vendor extension of the VDPAU-backed VA
     * driver): bit i of pinned_bits marks a slot handed to the
     * painter.  ONE ATOMIC WORD, no lock, safe by construction:
     * pins (0->1) are set only by the decode thread, which is also
     * the only thread that scans the set (h264_pick_surface) - so a
     * pick can never race a pin; the sole cross-thread transition is
     * a release (1->0) from the UI/GC thread, and a stale set-bit
     * merely makes the picker skip a just-freed slot (conservative).
     * seq_cst operations are used for zero reasoning burden;
     * release-on-clear / acquire-on-load would suffice (the clear
     * publishes the painter's completed UnmapSurfacesNV to the
     * decode thread's reuse of the slot).
     * export_fn is the dlsym'd vdpau_va_export_v1. */
    _Atomic uint32_t pinned_bits;
    void           *export_fn;
    int             width;
    int             height;
    int             surface_width;
    int             surface_height;
    LibVACodec      codec;
    VAProfile       profile;
    unsigned int    rt_format;
    /* DPB metadata, parallel to surfaces[]: dpb[i] describes the
     * reference picture held by surfaces[i] (which alone owns the
     * surface id - VA wants surfaces[] contiguous, so the entry does
     * not duplicate it).  The 1:1 mapping is guaranteed by
     * progressive-only decode (a frame owns its whole surface; a
     * field pair sharing one would break bit-per-surface) */
    struct H264DPBEntry dpb[H264_NUM_SURFACES];
    /* bit i set = surfaces[i] holds a reference picture - the SAME
     * index space as pinned_bits, so the picker's busy set is one OR.
     * Plain (not atomic): the DPB is decode-thread-private - unlike
     * pinned_bits this is data compaction, not a concurrency fix. */
    uint32_t        dpb_used;
    /* picture order count state (spec 8.2.1) */
    int             poc_prev_lsb;
    int             poc_prev_msb;
    int             poc_prev_frame_num;
    int             poc_prev_frame_num_offset;
    struct H264Params *h264_params;
    int             full_range;     /* colour range from the last parsed bitstream headers */
    int             last_status;
    char            last_error[256];
    char            device[256];
    char            vendor[256];
};

struct BitReader {
    const uint8_t *data;
    int size;
    int byte_pos;
    int bit_pos;
    int zeros;
    int bits_read;
};

struct H264Params {
    int valid_sps;
    int valid_pps;
    int chroma_format_idc;
    int separate_colour_plane_flag;
    int log2_max_frame_num_minus4;
    int pic_order_cnt_type;
    int log2_max_pic_order_cnt_lsb_minus4;
    int delta_pic_order_always_zero_flag;
    int max_num_ref_frames;
    int gaps_in_frame_num_value_allowed_flag;
    int width_mbs_minus1;
    int height_mbs_minus1;
    int frame_mbs_only_flag;
    int mb_adaptive_frame_field_flag;
    int direct_8x8_inference_flag;
    int video_full_range_flag;
    int entropy_coding_mode_flag;
    int weighted_pred_flag;
    int weighted_bipred_idc;
    int transform_8x8_mode_flag;
    int pic_order_present_flag;
    int deblocking_filter_control_present_flag;
    int redundant_pic_cnt_present_flag;
    int constrained_intra_pred_flag;
    int num_ref_idx_l0_active_minus1;
    int num_ref_idx_l1_active_minus1;
    int pic_init_qp_minus26;
    int pic_init_qs_minus26;
    int chroma_qp_index_offset;
    int second_chroma_qp_index_offset;
};

static void init_h264_params(struct H264Params *params);

struct H264SliceInfo {
    int offset;
    int size;
    int nal_type;
    int nal_ref_idc;
    int first_mb;
    int slice_type;
    int frame_num;
    int poc_lsb;
    int num_ref_idx_l0_active_minus1;
    int num_ref_idx_l1_active_minus1;
    int cabac_init_idc;
    int slice_qp_delta;
    int disable_deblocking_filter_idc;
    int slice_alpha_c0_offset_div2;
    int slice_beta_offset_div2;
    int luma_log2_weight_denom;
    int chroma_log2_weight_denom;
    uint8_t luma_weight_l0_flag;
    int16_t luma_weight_l0[32];
    int16_t luma_offset_l0[32];
    uint8_t chroma_weight_l0_flag;
    int16_t chroma_weight_l0[32][2];
    int16_t chroma_offset_l0[32][2];
    int delta_poc_bottom;             /* delta_pic_order_cnt_bottom (poc type 0) */
    /* dec_ref_pic_marking() */
    int idr_long_term_reference_flag;
    int adaptive_ref_pic_marking;
    int n_mmco;
    struct { int op; int arg1; int arg2; } mmco[H264_DPB_SIZE + 2];
    /* ref_pic_list_modification(), list 0 */
    int n_ref_mod_l0;
    struct { int idc; int val; } ref_mod_l0[32];
    int bit_offset;
};

void libva_decode_set_log(libva_log_fn fn) {
    g_log_fn = fn;
}

static void libva_log(const char *fmt, ...) {
    if (!g_log_fn)
        return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_log_fn(buf);
}

static long long usec_now(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (long long)(count.QuadPart * 1000000LL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
#endif
}

static LibVADecodeStatus set_error(LibVADecoder *dec, VAStatus status, const char *context) {
    if (dec) {
        dec->last_status = (int)status;
        snprintf(dec->last_error, sizeof(dec->last_error), "%s failed: %s (%d)",
                 context, vaErrorStr(status), (int)status);
        libva_log("libva decode error: %s", dec->last_error);
    }
    return LIBVA_DEC_ERROR;
}

static LibVADecodeStatus set_message(LibVADecoder *dec, LibVADecodeStatus status, const char *message) {
    if (dec) {
        dec->last_status = (int)status;
        snprintf(dec->last_error, sizeof(dec->last_error), "%s", message);
        libva_log("libva decode error: %s", dec->last_error);
    } else {
        snprintf(g_error, sizeof(g_error), "%s", message);
    }
    return status;
}

const char* libva_decode_status_str(LibVADecodeStatus status) {
    switch (status) {
        case LIBVA_DEC_OK:            return "ok";
        case LIBVA_DEC_ERROR:         return "error";
        case LIBVA_DEC_NOT_AVAILABLE: return "not_available";
        case LIBVA_DEC_UNSUPPORTED:   return "unsupported";
        default:                      return "unknown";
    }
}

static int vld_supported(VADisplay display, VAProfile profile, unsigned int rt_format) {
    VAEntrypoint entrypoints[32];
    int nentrypoints = 0;
    VAConfigAttrib attr;
    VAStatus status = vaQueryConfigEntrypoints(display, profile, entrypoints, &nentrypoints);
    if (status != VA_STATUS_SUCCESS) {
        snprintf(g_error, sizeof(g_error), "vaQueryConfigEntrypoints(%s) failed: %s (%d)",
                 h264_profile_name(profile), vaErrorStr(status), (int)status);
        return 0;
    }
    if (!entrypoint_supported(entrypoints, nentrypoints, VAEntrypointVLD))
        return 0;
    attr.type = VAConfigAttribRTFormat;
    attr.value = 0;
    status = vaGetConfigAttributes(display, profile, VAEntrypointVLD, &attr, 1);
    if (status != VA_STATUS_SUCCESS) {
        snprintf(g_error, sizeof(g_error), "vaGetConfigAttributes(%s, VLD) failed: %s (%d)",
                 h264_profile_name(profile), vaErrorStr(status), (int)status);
        return 0;
    }
    return (attr.value & rt_format) != 0;
}

static int try_device(const char *device) {
    int fd = -1, major = 0, minor = 0;
    VADisplay display = NULL;
    char vendor[256] = "";
    VAProfile profiles[64];
    int nprofiles = 0;
    int h264_420 = 0;
    VAProfile h264_420_profile = VAProfileH264ConstrainedBaseline;
    VAStatus status;

    if (!libva_open_display(device, &fd, &display, &major, &minor, vendor, sizeof(vendor),
                            g_error, sizeof(g_error)))
        return 0;

    status = vaQueryConfigProfiles(display, profiles, &nprofiles);
    if (status != VA_STATUS_SUCCESS) {
        snprintf(g_error, sizeof(g_error), "vaQueryConfigProfiles failed: %s (%d)",
                 vaErrorStr(status), (int)status);
        vaTerminate(display);
        libva_x11_close(display);
        if (fd >= 0)
            close(fd);
        return 0;
    }

    static const VAProfile h264_profiles[] = {
        VAProfileH264High,
        VAProfileH264Main,
        VAProfileH264ConstrainedBaseline,
    };
    for (unsigned int i = 0; i < sizeof(h264_profiles) / sizeof(h264_profiles[0]); i++) {
        VAProfile profile = h264_profiles[i];
        if (profile_supported(profiles, nprofiles, profile) &&
            vld_supported(display, profile, VA_RT_FORMAT_YUV420)) {
            h264_420 = 1;
            h264_420_profile = profile;
            break;
        }
    }
    vaTerminate(display);
    libva_x11_close(display);
    if (fd >= 0)
        close(fd);
    if (h264_420) {
        snprintf(g_device, sizeof(g_device), "%s", device);
        snprintf(g_vendor, sizeof(g_vendor), "%s", vendor);
        g_major = major;
        g_minor = minor;
        g_h264_420_supported = h264_420;
        g_h264_420_profile = h264_420_profile;
        libva_log("libva decode: selected %s (%s), h264-420=%d",
                  g_device, g_vendor, h264_420);
    }
    return h264_420;
}

#ifdef _WIN32
LibVADecodeStatus libva_decode_startup(void) {
    g_error[0] = 0;
    if (try_device(""))
        return LIBVA_DEC_OK;
    if (!g_error[0])
        snprintf(g_error, sizeof(g_error), "no VA-API adapter found");
    return LIBVA_DEC_NOT_AVAILABLE;
}
#else
LibVADecodeStatus libva_decode_startup(void) {
    const char *env_device = getenv("XPRA_LIBVA_DEVICE");
    DIR *dir;
    struct dirent *entry;

    g_error[0] = 0;
    if (env_device && env_device[0]) {
        if (try_device(env_device))
            return LIBVA_DEC_OK;
        return LIBVA_DEC_NOT_AVAILABLE;
    }
    if (try_device("/dev/dri/renderD128"))
        return LIBVA_DEC_OK;
    dir = opendir("/dev/dri");
    if (dir) {
        while ((entry = readdir(dir))) {
            char path[256];
            if (strncmp(entry->d_name, "renderD", 7) != 0)
                continue;
            snprintf(path, sizeof(path), "/dev/dri/%.200s", entry->d_name);
            if (strcmp(path, "/dev/dri/renderD128") == 0)
                continue;
            if (try_device(path)) {
                closedir(dir);
                return LIBVA_DEC_OK;
            }
        }
        closedir(dir);
    }
    /* no render node worked: try an X11 VA display (VDPAU-backed VA
     * drivers have no DRM path at all) */
    if (try_device("x11"))
        return LIBVA_DEC_OK;
    if (!g_error[0])
        snprintf(g_error, sizeof(g_error), "no VA-API render node found");
    return LIBVA_DEC_NOT_AVAILABLE;
}
#endif

void libva_decode_shutdown(void) {
    libva_log("libva decode shutdown");
}

const char *libva_decode_get_device(void) {
    return g_device;
}

const char *libva_decode_get_vendor(void) {
    return g_vendor;
}

const char *libva_decode_get_last_error(void) {
    return g_error;
}

int libva_decode_get_major(void) {
    return g_major;
}

int libva_decode_get_minor(void) {
    return g_minor;
}

int libva_decode_supports(const char *encoding, const char *colorspace) {
    LibVACodec codec;
    if (!g_device[0] && libva_decode_startup() != LIBVA_DEC_OK)
        return 0;
    if (!codec_from_name(encoding, &codec))
        return 0;
    if (codec == LIBVA_CODEC_H264 && strcmp(colorspace, "YUV420P") == 0)
        return g_h264_420_supported;
    return 0;
}

static void fill_invalid_picture(VAPictureH264 *pic) {
    memset(pic, 0, sizeof(*pic));
    pic->picture_id = VA_INVALID_SURFACE;
    pic->flags = VA_PICTURE_H264_INVALID;
}

static void destroy_buffers(LibVADecoder *dec, VABufferID *buffers, int count) {
    for (int i = 0; i < count; i++) {
        if (buffers[i] != VA_INVALID_ID) {
            vaDestroyBuffer(dec->display, buffers[i]);
            buffers[i] = VA_INVALID_ID;
        }
    }
}

/* allocate dec->num_surfaces surfaces + the VA context - runs at
 * FIRST DECODE, once the pool has been sized from the stream's SPS */
static LibVADecodeStatus alloc_surfaces_and_context(LibVADecoder *dec) {
    VASurfaceAttrib surface_attrs[2];
    VAStatus status;
    memset(surface_attrs, 0, sizeof(surface_attrs));
    surface_attrs[0].type = VASurfaceAttribPixelFormat;
    surface_attrs[0].flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrs[0].value.type = VAGenericValueTypeInteger;
    surface_attrs[0].value.value.i = VA_FOURCC_NV12;
    surface_attrs[1].type = VASurfaceAttribUsageHint;
    surface_attrs[1].flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrs[1].value.type = VAGenericValueTypeInteger;
    surface_attrs[1].value.value.i = VA_SURFACE_ATTRIB_USAGE_HINT_DECODER;
    status = vaCreateSurfaces(dec->display, dec->rt_format,
                              (unsigned int)dec->surface_width,
                              (unsigned int)dec->surface_height,
                              dec->surfaces, (unsigned int)dec->num_surfaces,
                              surface_attrs, 2);
    if (status != VA_STATUS_SUCCESS) {
        /* VA-API leaves the output array UNDEFINED on failure, and the
         * VDPAU bridge (pre-fix) left the ids of partially-created,
         * already-destroyed surfaces in it - our teardown would then
         * re-destroy them (the vdpau_DestroySurfaces assert = SIGABRT
         * under VRAM pressure).  Never trust the array on error. */
        for (int i = 0; i < H264_NUM_SURFACES; i++)
            dec->surfaces[i] = VA_INVALID_SURFACE;
        return set_error(dec, status, "vaCreateSurfaces");
    }
    status = vaCreateContext(dec->display, dec->config,
                             dec->surface_width, dec->surface_height,
                             VA_PROGRESSIVE, dec->surfaces, dec->num_surfaces,
                             &dec->context);
    if (status != VA_STATUS_SUCCESS) {
        /* decode re-entry would allocate a fresh pool over this
         * array, orphaning these surfaces in the driver */
        vaDestroySurfaces(dec->display, dec->surfaces, dec->num_surfaces);
        for (int i = 0; i < H264_NUM_SURFACES; i++)
            dec->surfaces[i] = VA_INVALID_SURFACE;
        dec->context = VA_INVALID_ID;
        return set_error(dec, status, "vaCreateContext");
    }
    return LIBVA_DEC_OK;
}

LibVADecodeStatus libva_decoder_create(LibVADecoder **out, const char *encoding,
                                       int width, int height, const char *colorspace) {
    LibVACodec codec;
    LibVADecoder *dec;
    VAStatus status;
    VAConfigAttrib attr;
    int major = 0, minor = 0;

    if (!out)
        return LIBVA_DEC_ERROR;
    *out = NULL;
    if (!codec_from_name(encoding, &codec))
        return LIBVA_DEC_NOT_AVAILABLE;
    if (width <= 0 || height <= 0)
        return LIBVA_DEC_ERROR;
    if (!g_device[0] && libva_decode_startup() != LIBVA_DEC_OK)
        return LIBVA_DEC_NOT_AVAILABLE;
    if (!libva_decode_supports(encoding, colorspace))
        return LIBVA_DEC_NOT_AVAILABLE;

    dec = (LibVADecoder *)calloc(1, sizeof(LibVADecoder));
    if (!dec)
        return LIBVA_DEC_ERROR;
    dec->h264_params = (struct H264Params *)calloc(1, sizeof(*dec->h264_params));
    if (!dec->h264_params) {
        free(dec);
        return LIBVA_DEC_ERROR;
    }
    init_h264_params(dec->h264_params);
    dec->fd = -1;
    dec->config = VA_INVALID_ID;
    dec->context = VA_INVALID_ID;
    /* the pool is sized from the stream's SPS at first decode */
    dec->num_surfaces = 0;
    for (int i = 0; i < H264_NUM_SURFACES; i++)
        dec->surfaces[i] = VA_INVALID_SURFACE;
    dec->width = width;
    dec->height = height;
    /* surfaces must hold the CODED picture, and encoders pad the coded
     * size beyond macroblock alignment - nvenc to 32-aligned dims (a
     * 1228x861 target arrives coded 1248x864).  16-aligned surfaces
     * made the hardware write 32-aligned rows past the row end: whole-
     * frame mosaic garbage for every size where the two alignments
     * differ (all common test sizes happened to be 32-aligned - this
     * hid the bug until a 31/32-scaled window landed on 1228 wide).
     * Oversized surfaces are safe: decode fills the top-left, the
     * display rectangle crops. */
    dec->surface_width = roundup(width, 32);
    dec->surface_height = roundup(height, 32);
    dec->codec = codec;
    dec->rt_format = VA_RT_FORMAT_YUV420;
    dec->profile = g_h264_420_profile;
    dec->last_status = VA_STATUS_SUCCESS;
    atomic_init(&dec->pinned_bits, 0);
    /* zero-copy export: H264 decode is export-only - the frames stay
     * on the GPU as VdpVideoSurfaces for GL_NV_vdpau_interop; the
     * VDPAU-backed VA driver must provide the vendor export symbol */
    if (codec == LIBVA_CODEC_H264) {
        dec->export_fn = dlsym(RTLD_DEFAULT, "vdpau_va_export_v1");
        if (!dec->export_fn) {
            libva_decoder_destroy(dec);
            snprintf(g_error, sizeof(g_error),
                     "vdpau_va_export_v1 not exported by the VA driver");
            return LIBVA_DEC_NOT_AVAILABLE;
        }
    }
    snprintf(dec->device, sizeof(dec->device), "%s", g_device);

    if (!libva_open_display(dec->device, &dec->fd, &dec->display, &major, &minor,
                            dec->vendor, sizeof(dec->vendor),
                            g_error, sizeof(g_error))) {
        libva_decoder_destroy(dec);
        return LIBVA_DEC_NOT_AVAILABLE;
    }

    attr.type = VAConfigAttribRTFormat;
    attr.value = dec->rt_format;
    status = vaCreateConfig(dec->display, dec->profile, VAEntrypointVLD, &attr, 1, &dec->config);
    if (status != VA_STATUS_SUCCESS) {
        set_error(dec, status, "vaCreateConfig");
        libva_decoder_destroy(dec);
        return LIBVA_DEC_ERROR;
    }

    {
        VASurfaceAttrib qattrs[32];
        unsigned int nq = 32;
        /* zero the array: drivers only write the entries they report,
         * and a stale stack entry whose type happened to equal
         * PixelFormat printed garbage fourccs (seen live with the
         * VDPAU bridge) - zeroed entries have type VASurfaceAttribNone */
        memset(qattrs, 0, sizeof(qattrs));
        VAStatus qst = vaQuerySurfaceAttributes(dec->display, dec->config, qattrs, &nq);
        libva_log("surface probe: vaQuerySurfaceAttributes status=%d nattrs=%u", (int)qst, nq);
        if (qst == VA_STATUS_SUCCESS && nq > 32)
            nq = 32;
        for (unsigned int i = 0; qst == VA_STATUS_SUCCESS && i < nq; i++) {
            if (qattrs[i].type == VASurfaceAttribPixelFormat) {
                unsigned int fourcc = (unsigned int)qattrs[i].value.value.i;
                char c[4];
                for (int b = 0; b < 4; b++) {
                    unsigned char ch = (unsigned char)((fourcc >> (8 * b)) & 0xff);
                    c[b] = (ch >= 32 && ch < 127) ? (char)ch : '?';
                }
                libva_log("surface probe:   supported pixel format: %c%c%c%c (0x%08x)",
                          c[0], c[1], c[2], c[3], fourcc);
            }
        }
    }
    libva_log("libva %s decoder create: %dx%d surface=%dx%d colorspace=%s profile=%s device=%s vendor=%s",
              codec_name(dec->codec), width, height, dec->surface_width, dec->surface_height,
              colorspace, h264_profile_name(dec->profile), dec->device, dec->vendor);
    *out = dec;
    return LIBVA_DEC_OK;
}

void libva_decoder_destroy(LibVADecoder *dec) {
    if (!dec)
        return;
    if (dec->display) {
        if (dec->context != VA_INVALID_ID)
            vaDestroyContext(dec->display, dec->context);
        for (int i = 0; i < H264_NUM_SURFACES; i++) {
            if (dec->surfaces[i] != VA_INVALID_SURFACE)
                vaDestroySurfaces(dec->display, &dec->surfaces[i], 1);
        }
        if (dec->config != VA_INVALID_ID)
            vaDestroyConfig(dec->display, dec->config);
        vaTerminate(dec->display);
        libva_x11_close(dec->display);
    }
    if (dec->fd >= 0)
        close(dec->fd);
    free(dec->h264_params);
    free(dec);
}

static void br_init(struct BitReader *br, const uint8_t *data, int size) {
    memset(br, 0, sizeof(*br));
    br->data = data;
    br->size = size;
}

static int br_read_bit(struct BitReader *br) {
    int bit;
    if (br->byte_pos >= br->size)
        return 0;
    if (br->zeros >= 2 && br->data[br->byte_pos] == 0x03) {
        br->byte_pos++;
        br->zeros = 0;
        if (br->byte_pos >= br->size)
            return 0;
    }
    bit = (br->data[br->byte_pos] >> (7 - br->bit_pos)) & 1;
    br->bit_pos++;
    br->bits_read++;
    if (br->bit_pos == 8) {
        uint8_t b = br->data[br->byte_pos];
        br->zeros = b == 0 ? br->zeros + 1 : 0;
        br->byte_pos++;
        br->bit_pos = 0;
    }
    return bit;
}

static int br_bits_left(const struct BitReader *br) {
    int bits = br->size * 8 - br->byte_pos * 8 - br->bit_pos;
    return bits > 0 ? bits : 0;
}

static unsigned int br_bits(struct BitReader *br, int bits) {
    unsigned int v = 0;
    for (int i = 0; i < bits; i++)
        v = (v << 1) | (unsigned int)br_read_bit(br);
    return v;
}

static unsigned int br_ue(struct BitReader *br) {
    int zeros = 0;
    while (zeros < 32 && br_read_bit(br) == 0)
        zeros++;
    if (zeros == 0)
        return 0;
    return ((1U << zeros) - 1U) + br_bits(br, zeros);
}

static int br_se(struct BitReader *br) {
    unsigned int ue = br_ue(br);
    int v = (int)((ue + 1) >> 1);
    return (ue & 1) ? v : -v;
}

static int br_more_rbsp_data(const struct BitReader *br) {
    struct BitReader rb = *br;
    while (br_bits_left(&rb) > 0) {
        if (br_read_bit(&rb)) {
            while (br_bits_left(&rb) > 0) {
                if (br_read_bit(&rb))
                    return 1;
            }
            return 0;
        }
    }
    return 0;
}

static int find_start_code(const uint8_t *data, int size, int offset, int *prefix) {
    for (int i = offset; i + 3 < size; i++) {
        if (data[i] == 0 && data[i + 1] == 0) {
            if (data[i + 2] == 1) {
                *prefix = 3;
                return i;
            }
            if (i + 4 < size && data[i + 2] == 0 && data[i + 3] == 1) {
                *prefix = 4;
                return i;
            }
        }
    }
    return -1;
}

static void skip_h264_scaling_list(struct BitReader *br, int size) {
    int last_scale = 8;
    int next_scale = 8;
    for (int j = 0; j < size; j++) {
        if (next_scale != 0) {
            int delta_scale = br_se(br);
            next_scale = (last_scale + delta_scale + 256) & 0xff;
        }
        last_scale = next_scale == 0 ? last_scale : next_scale;
    }
}

static int parse_h264_sps(const uint8_t *nal, int size, struct H264Params *params) {
    struct BitReader br;
    int profile_idc;
    if (size < 2)
        return 0;
    br_init(&br, nal + 1, size - 1);
    profile_idc = (int)br_bits(&br, 8);
    br_bits(&br, 8);                  /* constraint flags + reserved */
    br_bits(&br, 8);                  /* level_idc */
    br_ue(&br);                       /* seq_parameter_set_id */
    params->chroma_format_idc = 1;
    params->separate_colour_plane_flag = 0;
    if (profile_idc == 100 || profile_idc == 110 || profile_idc == 122 ||
        profile_idc == 244 || profile_idc == 44 || profile_idc == 83 ||
        profile_idc == 86 || profile_idc == 118 || profile_idc == 128 ||
        profile_idc == 138 || profile_idc == 144) {
        params->chroma_format_idc = (int)br_ue(&br);
        if (params->chroma_format_idc == 3)
            params->separate_colour_plane_flag = (int)br_bits(&br, 1);
        br_ue(&br);                   /* bit_depth_luma_minus8 */
        br_ue(&br);                   /* bit_depth_chroma_minus8 */
        br_bits(&br, 1);              /* qpprime_y_zero_transform_bypass_flag */
        if (br_bits(&br, 1)) {        /* seq_scaling_matrix_present_flag */
            int count = params->chroma_format_idc != 3 ? 8 : 12;
            for (int i = 0; i < count; i++) {
                if (br_bits(&br, 1))
                    skip_h264_scaling_list(&br, i < 6 ? 16 : 64);
            }
        }
    }
    params->log2_max_frame_num_minus4 = (int)br_ue(&br);
    params->pic_order_cnt_type = (int)br_ue(&br);
    if (params->pic_order_cnt_type == 0) {
        params->log2_max_pic_order_cnt_lsb_minus4 = (int)br_ue(&br);
    } else if (params->pic_order_cnt_type == 1) {
        params->delta_pic_order_always_zero_flag = (int)br_bits(&br, 1);
        br_se(&br);                   /* offset_for_non_ref_pic */
        br_se(&br);                   /* offset_for_top_to_bottom_field */
        int count = (int)br_ue(&br);
        for (int i = 0; i < count; i++)
            br_se(&br);
    }
    params->max_num_ref_frames = (int)br_ue(&br);
    params->gaps_in_frame_num_value_allowed_flag = (int)br_bits(&br, 1);
    params->width_mbs_minus1 = (int)br_ue(&br);
    params->height_mbs_minus1 = (int)br_ue(&br);
    params->frame_mbs_only_flag = (int)br_bits(&br, 1);
    if (!params->frame_mbs_only_flag)
        params->mb_adaptive_frame_field_flag = (int)br_bits(&br, 1);
    params->direct_8x8_inference_flag = (int)br_bits(&br, 1);
    /* continue into the VUI to recover the colour range (video_full_range_flag);
     * br handles emulation-prevention bytes and returns 0 past the end of the RBSP: */
    if (br_bits(&br, 1)) {            /* frame_cropping_flag */
        br_ue(&br);                   /* frame_crop_left_offset */
        br_ue(&br);                   /* frame_crop_right_offset */
        br_ue(&br);                   /* frame_crop_top_offset */
        br_ue(&br);                   /* frame_crop_bottom_offset */
    }
    params->video_full_range_flag = 0;    /* default (studio) when not signalled */
    if (br_bits(&br, 1)) {            /* vui_parameters_present_flag */
        if (br_bits(&br, 1)) {        /* aspect_ratio_info_present_flag */
            if ((int)br_bits(&br, 8) == 255) {  /* aspect_ratio_idc == Extended_SAR */
                br_bits(&br, 16);     /* sar_width */
                br_bits(&br, 16);     /* sar_height */
            }
        }
        if (br_bits(&br, 1))          /* overscan_info_present_flag */
            br_bits(&br, 1);          /* overscan_appropriate_flag */
        if (br_bits(&br, 1)) {        /* video_signal_type_present_flag */
            br_bits(&br, 3);          /* video_format */
            params->video_full_range_flag = (int)br_bits(&br, 1);
        }
    }
    params->valid_sps = 1;
    return 1;
}

static int parse_h264_pps(const uint8_t *nal, int size, struct H264Params *params) {
    struct BitReader br;
    int num_slice_groups_minus1;
    if (size < 2)
        return 0;
    br_init(&br, nal + 1, size - 1);
    br_ue(&br);                       /* pic_parameter_set_id */
    br_ue(&br);                       /* seq_parameter_set_id */
    params->entropy_coding_mode_flag = (int)br_bits(&br, 1);
    params->pic_order_present_flag = (int)br_bits(&br, 1);
    num_slice_groups_minus1 = (int)br_ue(&br);
    if (num_slice_groups_minus1 != 0)
        return 0;
    params->num_ref_idx_l0_active_minus1 = (int)br_ue(&br);
    params->num_ref_idx_l1_active_minus1 = (int)br_ue(&br);
    params->weighted_pred_flag = (int)br_bits(&br, 1);
    params->weighted_bipred_idc = (int)br_bits(&br, 2);
    params->pic_init_qp_minus26 = br_se(&br);
    params->pic_init_qs_minus26 = br_se(&br);
    params->chroma_qp_index_offset = br_se(&br);
    params->deblocking_filter_control_present_flag = (int)br_bits(&br, 1);
    params->constrained_intra_pred_flag = (int)br_bits(&br, 1);
    params->redundant_pic_cnt_present_flag = (int)br_bits(&br, 1);
    params->second_chroma_qp_index_offset = params->chroma_qp_index_offset;
    if (br_more_rbsp_data(&br)) {
        params->transform_8x8_mode_flag = (int)br_bits(&br, 1);
        if (br_bits(&br, 1)) {        /* pic_scaling_matrix_present_flag */
            int count = 6;
            if (params->transform_8x8_mode_flag)
                count += params->chroma_format_idc == 3 ? 6 : 2;
            for (int i = 0; i < count; i++) {
                if (br_bits(&br, 1))
                    skip_h264_scaling_list(&br, i < 6 ? 16 : 64);
            }
        }
        params->second_chroma_qp_index_offset = br_se(&br);
    }
    params->valid_pps = 1;
    return 1;
}

static void init_h264_params(struct H264Params *params) {
    memset(params, 0, sizeof(*params));
    params->chroma_format_idc = 1;
    params->log2_max_frame_num_minus4 = LIBVA_LOG2_MAX_FRAME_NUM_MINUS4;
    params->log2_max_pic_order_cnt_lsb_minus4 = LIBVA_LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4;
    params->max_num_ref_frames = 1;
    params->frame_mbs_only_flag = 1;
    params->direct_8x8_inference_flag = 1;
    params->deblocking_filter_control_present_flag = 1;
}

static void fill_h264_default_iq_matrix(VAIQMatrixBufferH264 *iq) {
    memset(iq, 0, sizeof(*iq));
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 16; j++)
            iq->ScalingList4x4[i][j] = 16;
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 64; j++)
            iq->ScalingList8x8[i][j] = 16;
    }
}

static void parse_h264_params(const uint8_t *data, int size, struct H264Params *params) {
    int prefix = 0;
    int start;
    start = find_start_code(data, size, 0, &prefix);
    while (start >= 0) {
        int nal = start + prefix;
        int next_prefix = 0;
        int next = find_start_code(data, size, nal + 1, &next_prefix);
        int end = next >= 0 ? next : size;
        if (nal < end) {
            int type = data[nal] & 0x1f;
            if (type == 7)
                parse_h264_sps(data + nal, end - nal, params);
            else if (type == 8)
                parse_h264_pps(data + nal, end - nal, params);
        }
        start = next;
        prefix = next_prefix;
    }
}

static int parse_h264_ref_pic_list_modification(struct BitReader *br, int slice_type_mod,
                                                struct H264SliceInfo *si) {
    si->n_ref_mod_l0 = 0;
    if (slice_type_mod == 2 || slice_type_mod == 4)
        return 1;                     /* I/SI slices carry no lists */
    if (br_bits(br, 1)) {             /* ref_pic_list_modification_flag_l0 */
        unsigned int idc;
        do {
            idc = br_ue(br);          /* modification_of_pic_nums_idc */
            if (idc == 3)
                break;
            if (idc > 2)
                return 0;             /* MVC ops (4/5) not supported */
            int val = (int)br_ue(br); /* abs_diff_pic_num_minus1 / long_term_pic_num */
            if (si->n_ref_mod_l0 < 32) {
                si->ref_mod_l0[si->n_ref_mod_l0].idc = (int)idc;
                si->ref_mod_l0[si->n_ref_mod_l0].val = val;
                si->n_ref_mod_l0++;
            }
        } while (br_bits_left(br) > 0);
    }
    /* B slices would carry an L1 modification loop here; B slices are
     * rejected at slice-header parse (this decoder is IPP-only) */
    return 1;
}

static int parse_h264_dec_ref_pic_marking(struct BitReader *br, int nal_type,
                                          struct H264SliceInfo *si) {
    si->idr_long_term_reference_flag = 0;
    si->adaptive_ref_pic_marking = 0;
    si->n_mmco = 0;
    if (!si->nal_ref_idc)
        return 1;
    if (nal_type == 5) {
        br_bits(br, 1);               /* no_output_of_prior_pics_flag */
        si->idr_long_term_reference_flag = (int)br_bits(br, 1);
        return 1;
    }
    si->adaptive_ref_pic_marking = (int)br_bits(br, 1);
    if (si->adaptive_ref_pic_marking) {
        unsigned int op;
        do {
            op = br_ue(br);           /* memory_management_control_operation */
            if (op == 0)
                break;
            if (op > 6)
                return 0;
            int arg1 = 0, arg2 = 0;
            if (op == 1 || op == 3)
                arg1 = (int)br_ue(br);    /* difference_of_pic_nums_minus1 */
            if (op == 2)
                arg1 = (int)br_ue(br);    /* long_term_pic_num */
            if (op == 3 || op == 6)
                arg2 = (int)br_ue(br);    /* long_term_frame_idx */
            if (op == 4)
                arg1 = (int)br_ue(br);    /* max_long_term_frame_idx_plus1 */
            if (si->n_mmco < (int)(sizeof(si->mmco) / sizeof(si->mmco[0]))) {
                si->mmco[si->n_mmco].op = (int)op;
                si->mmco[si->n_mmco].arg1 = arg1;
                si->mmco[si->n_mmco].arg2 = arg2;
                si->n_mmco++;
            }
        } while (br_bits_left(br) > 0);
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* H.264 decoded picture buffer (progressive frames only)              */
/* ------------------------------------------------------------------ */

static void h264_dpb_flush(LibVADecoder *dec) {
    /* emptiness is the bitmap alone - entry fields are void while
     * their dpb_used bit is clear */
    dec->dpb_used = 0;
}

/* PicNum for a frame (spec 8.2.4.1: FrameNumWrap, since for frames
 * PicNum == FrameNumWrap) relative to the current frame_num */
static int h264_pic_num(const struct H264DPBEntry *e, int cur_frame_num, int max_frame_num) {
    if (e->frame_num > cur_frame_num)
        return e->frame_num - max_frame_num;
    return e->frame_num;
}

/* spec 8.2.1: picture order count, types 0 and 2 (type 1 unsupported).
 * Progressive only. Returns 0 and fills *top_foc / *bottom_foc. */
static int h264_compute_poc(LibVADecoder *dec, const struct H264Params *params,
                            const struct H264SliceInfo *si, int is_idr,
                            int *top_foc, int *bottom_foc) {
    if (params->pic_order_cnt_type == 0) {
        int max_lsb = 1 << (params->log2_max_pic_order_cnt_lsb_minus4 + 4);
        int prev_lsb = is_idr ? 0 : dec->poc_prev_lsb;
        int prev_msb = is_idr ? 0 : dec->poc_prev_msb;
        int msb;
        if (si->poc_lsb < prev_lsb && (prev_lsb - si->poc_lsb) >= max_lsb / 2)
            msb = prev_msb + max_lsb;
        else if (si->poc_lsb > prev_lsb && (si->poc_lsb - prev_lsb) > max_lsb / 2)
            msb = prev_msb - max_lsb;
        else
            msb = prev_msb;
        *top_foc = msb + si->poc_lsb;
        *bottom_foc = *top_foc + si->delta_poc_bottom;
        if (si->nal_ref_idc) {        /* prev state tracks reference pictures */
            dec->poc_prev_lsb = si->poc_lsb;
            dec->poc_prev_msb = msb;
        }
        return 0;
    }
    if (params->pic_order_cnt_type == 2) {
        int max_frame_num = 1 << (params->log2_max_frame_num_minus4 + 4);
        int offset;
        if (is_idr)
            offset = 0;
        else if (dec->poc_prev_frame_num > si->frame_num)
            offset = dec->poc_prev_frame_num_offset + max_frame_num;
        else
            offset = dec->poc_prev_frame_num_offset;
        int poc = 2 * (offset + si->frame_num);
        if (!si->nal_ref_idc)
            poc -= 1;
        *top_foc = poc;
        *bottom_foc = poc;
        dec->poc_prev_frame_num = si->frame_num;
        dec->poc_prev_frame_num_offset = offset;
        return 0;
    }
    return -1;                        /* type 1: no real encoder emits it here */
}

/* pick a surface that is neither referenced by the DPB nor pinned by
 * an exported (in-flight) frame; the pool = DPB size + 1 + export
 * slack, so exhaustion needs a full DPB AND more unreleased exports
 * than the slack - that fails the decode and restarts the decoder */
static int h264_pick_surface(LibVADecoder *dec) {
    /* one snapshot is enough: pins are only ever SET by this thread,
     * so the snapshot cannot go stale in the unsafe direction (a
     * concurrent release just means we skip a slot that became free).
     * dpb_used shares the surface index space, so busy is one OR. */
    const uint32_t busy = atomic_load(&dec->pinned_bits) | dec->dpb_used;
    /* 64-bit shift: with the pool at 32 surfaces a 32-bit 1u << 32
     * would be undefined behaviour (x86 masks it to a no-op shift) */
    const uint32_t free_mask = (uint32_t)((1ULL << dec->num_surfaces) - 1) & ~busy;
    /* round-robin: first free slot at index >= surface_index, wrapping.
     * __builtin_ffs returns 1 + the lowest set bit, 0 for no bits -
     * the wrap and the all-busy case fall out of its semantics.
     * (ctz would need a zero guard; clz scans from the top - no use
     * in an ascending scan.) */
    uint32_t cand = free_mask & (~0u << dec->surface_index);
    if (!cand)
        cand = free_mask;
    const int ffs = __builtin_ffs((int)cand);
    if (ffs == 0)
        return -1;
    const int idx = ffs - 1;
    dec->surface_index = (idx + 1) % dec->num_surfaces;
    return idx;
}

/* spec 8.2.5.3: sliding-window marking - make room for one more
 * reference by unmarking the short-term entry with the smallest
 * FrameNumWrap */
static void h264_dpb_sliding_window(LibVADecoder *dec, const struct H264Params *params,
                                    int cur_frame_num) {
    int max_frame_num = 1 << (params->log2_max_frame_num_minus4 + 4);
    int num_ref = params->max_num_ref_frames > 0 ? params->max_num_ref_frames : 1;
    if (num_ref > H264_DPB_SIZE)
        num_ref = H264_DPB_SIZE;
    int used = __builtin_popcount(dec->dpb_used);
    while (used >= num_ref) {
        int victim = -1, victim_pn = 0;
        for (uint32_t refs = dec->dpb_used; refs; refs &= refs - 1) {
            int i = __builtin_ctz(refs);
            struct H264DPBEntry *e = &dec->dpb[i];
            if (e->is_long_term)
                continue;
            int pn = h264_pic_num(e, cur_frame_num, max_frame_num);
            if (victim < 0 || pn < victim_pn) {
                victim = i;
                victim_pn = pn;
            }
        }
        if (victim < 0)
            break;                    /* only long-term refs left */
        dec->dpb_used &= ~(1u << victim);
        used--;
    }
}

/* spec 8.2.5.4: adaptive memory control (MMCO ops 1-6) */
static void h264_dpb_apply_mmco(LibVADecoder *dec, const struct H264Params *params,
                                const struct H264SliceInfo *si, int cur_frame_num,
                                int *cur_is_long_term, int *cur_lt_idx, int *had_mmco5) {
    int max_frame_num = 1 << (params->log2_max_frame_num_minus4 + 4);
    *had_mmco5 = 0;
    for (int m = 0; m < si->n_mmco; m++) {
        int op = si->mmco[m].op;
        if (op == 1) {                /* unmark short-term */
            int pic_num = cur_frame_num - (si->mmco[m].arg1 + 1);
            for (uint32_t refs = dec->dpb_used; refs; refs &= refs - 1) {
                const int i = __builtin_ctz(refs);
                struct H264DPBEntry *e = &dec->dpb[i];
                if (!e->is_long_term &&
                    h264_pic_num(e, cur_frame_num, max_frame_num) == pic_num)
                    dec->dpb_used &= ~(1u << i);
            }
        } else if (op == 2) {         /* unmark long-term by LongTermPicNum */
            for (uint32_t refs = dec->dpb_used; refs; refs &= refs - 1) {
                const int i = __builtin_ctz(refs);
                struct H264DPBEntry *e = &dec->dpb[i];
                if (e->is_long_term &&
                    e->long_term_frame_idx == si->mmco[m].arg1)
                    dec->dpb_used &= ~(1u << i);
            }
        } else if (op == 3) {         /* short-term -> long-term */
            int pic_num = cur_frame_num - (si->mmco[m].arg1 + 1);
            for (uint32_t refs = dec->dpb_used; refs; refs &= refs - 1) {
                const int i = __builtin_ctz(refs);
                struct H264DPBEntry *e = &dec->dpb[i];
                if (e->is_long_term &&
                    e->long_term_frame_idx == si->mmco[m].arg2)
                    dec->dpb_used &= ~(1u << i);
            }
            for (uint32_t refs = dec->dpb_used; refs; refs &= refs - 1) {
                struct H264DPBEntry *e = &dec->dpb[__builtin_ctz(refs)];
                if (!e->is_long_term &&
                    h264_pic_num(e, cur_frame_num, max_frame_num) == pic_num) {
                    e->is_long_term = 1;
                    e->long_term_frame_idx = si->mmco[m].arg2;
                }
            }
        } else if (op == 4) {         /* max_long_term_frame_idx */
            int max_idx = si->mmco[m].arg1 - 1;
            for (uint32_t refs = dec->dpb_used; refs; refs &= refs - 1) {
                const int i = __builtin_ctz(refs);
                struct H264DPBEntry *e = &dec->dpb[i];
                if (e->is_long_term && e->long_term_frame_idx > max_idx)
                    dec->dpb_used &= ~(1u << i);
            }
        } else if (op == 5) {         /* unmark everything, reset numbering */
            h264_dpb_flush(dec);
            *had_mmco5 = 1;
        } else if (op == 6) {         /* current picture becomes long-term */
            for (uint32_t refs = dec->dpb_used; refs; refs &= refs - 1) {
                const int i = __builtin_ctz(refs);
                struct H264DPBEntry *e = &dec->dpb[i];
                if (e->is_long_term &&
                    e->long_term_frame_idx == si->mmco[m].arg2)
                    dec->dpb_used &= ~(1u << i);
            }
            *cur_is_long_term = 1;
            *cur_lt_idx = si->mmco[m].arg2;
        }
    }
}

static void h264_dpb_insert(LibVADecoder *dec, int surface_index,
                            int frame_num, int top_foc, int bottom_foc,
                            int is_long_term, int lt_idx) {
    /* the slot IS the surface index (dpb[] is parallel to surfaces[]);
     * its bit is clear by construction: the picker refuses surfaces
     * that are still referenced or pinned */
    struct H264DPBEntry *e = &dec->dpb[surface_index];
    e->frame_num = frame_num;
    e->top_foc = top_foc;
    e->bottom_foc = bottom_foc;
    e->is_long_term = is_long_term;
    e->long_term_frame_idx = lt_idx;
    dec->dpb_used |= 1u << surface_index;
}

static void h264_fill_va_picture(VAPictureH264 *p, const LibVADecoder *dec, int slot) {
    const struct H264DPBEntry *e = &dec->dpb[slot];
    p->picture_id = dec->surfaces[slot];
    if (e->is_long_term) {
        p->frame_idx = (uint32_t)e->long_term_frame_idx;
        p->flags = VA_PICTURE_H264_LONG_TERM_REFERENCE;
    } else {
        p->frame_idx = (uint32_t)e->frame_num;
        p->flags = VA_PICTURE_H264_SHORT_TERM_REFERENCE;
    }
    p->TopFieldOrderCnt = e->top_foc;
    p->BottomFieldOrderCnt = e->bottom_foc;
}

/* RefPicList0 (spec 8.2.4.2.1 default order + 8.2.4.3 modification).
 * Note the VDPAU-bridge driver ignores this list entirely (VDPAU
 * re-derives it from the slice headers); it is filled correctly for
 * VA drivers that do consume it.  Simplification vs spec: a picture
 * appears at most once in the produced list. */
static int h264_build_ref_list_l0(LibVADecoder *dec, const struct H264Params *params,
                                  const struct H264SliceInfo *si,
                                  struct H264DPBEntry **list, int max_out) {
    int max_frame_num = 1 << (params->log2_max_frame_num_minus4 + 4);
    int cur = si->frame_num;
    int n = 0;
    /* short-term references by descending PicNum */
    for (uint32_t refs = dec->dpb_used; refs && n < max_out; refs &= refs - 1) {
        struct H264DPBEntry *e = &dec->dpb[__builtin_ctz(refs)];
        if (e->is_long_term)
            continue;
        int pn = h264_pic_num(e, cur, max_frame_num);
        int j = n;
        while (j > 0 && h264_pic_num(list[j - 1], cur, max_frame_num) < pn) {
            list[j] = list[j - 1];
            j--;
        }
        list[j] = e;
        n++;
    }
    /* then long-term references by ascending LongTermFrameIdx */
    int st_count = n;
    for (uint32_t refs = dec->dpb_used; refs && n < max_out; refs &= refs - 1) {
        struct H264DPBEntry *e = &dec->dpb[__builtin_ctz(refs)];
        if (!e->is_long_term)
            continue;
        int j = n;
        while (j > st_count && list[j - 1]->long_term_frame_idx > e->long_term_frame_idx) {
            list[j] = list[j - 1];
            j--;
        }
        list[j] = e;
        n++;
    }
    if (si->n_ref_mod_l0 && n > 1) {
        int pred = cur;               /* picNumL0Pred = CurrPicNum */
        int ref_idx = 0;
        for (int m = 0; m < si->n_ref_mod_l0 && ref_idx < n; m++) {
            struct H264DPBEntry *target = NULL;
            if (si->ref_mod_l0[m].idc <= 1) {
                int abs_diff = si->ref_mod_l0[m].val + 1;
                int nowrap;
                if (si->ref_mod_l0[m].idc == 0) {
                    nowrap = pred - abs_diff;
                    if (nowrap < 0)
                        nowrap += max_frame_num;
                } else {
                    nowrap = pred + abs_diff;
                    if (nowrap >= max_frame_num)
                        nowrap -= max_frame_num;
                }
                pred = nowrap;
                int picnum = nowrap > cur ? nowrap - max_frame_num : nowrap;
                for (int i = 0; i < n; i++) {
                    if (!list[i]->is_long_term &&
                        h264_pic_num(list[i], cur, max_frame_num) == picnum) {
                        target = list[i];
                        break;
                    }
                }
            } else {                  /* idc == 2: long-term */
                for (int i = 0; i < n; i++) {
                    if (list[i]->is_long_term &&
                        list[i]->long_term_frame_idx == si->ref_mod_l0[m].val) {
                        target = list[i];
                        break;
                    }
                }
            }
            if (!target)
                continue;             /* unresolvable op in a broken stream */
            int from = 0;
            while (from < n && list[from] != target)
                from++;
            if (from >= n)
                continue;
            for (int i = from; i > ref_idx; i--)
                list[i] = list[i - 1];
            list[ref_idx] = target;
            ref_idx++;
        }
    }
    return n;
}

static void init_h264_weight_defaults(struct H264SliceInfo *si) {
    int luma_weight = 1 << si->luma_log2_weight_denom;
    int chroma_weight = 1 << si->chroma_log2_weight_denom;
    for (int i = 0; i < 32; i++) {
        si->luma_weight_l0[i] = (int16_t)luma_weight;
        si->luma_offset_l0[i] = 0;
        for (int c = 0; c < 2; c++) {
            si->chroma_weight_l0[i][c] = (int16_t)chroma_weight;
            si->chroma_offset_l0[i][c] = 0;
        }
    }
}

static void parse_h264_pred_weight_table(struct BitReader *br, const struct H264Params *params,
                                         struct H264SliceInfo *si) {
    int l0_refs = clamp_int(si->num_ref_idx_l0_active_minus1, 0, 31);
    si->luma_log2_weight_denom = (int)br_ue(br);
    if (params->chroma_format_idc != 0)
        si->chroma_log2_weight_denom = (int)br_ue(br);
    init_h264_weight_defaults(si);
    for (int i = 0; i <= l0_refs; i++) {
        if (br_bits(br, 1)) {
            if (i == 0)
                si->luma_weight_l0_flag = 1;
            si->luma_weight_l0[i] = (int16_t)br_se(br);
            si->luma_offset_l0[i] = (int16_t)br_se(br);
        }
        if (params->chroma_format_idc != 0 && br_bits(br, 1)) {
            if (i == 0)
                si->chroma_weight_l0_flag = 1;
            for (int c = 0; c < 2; c++) {
                si->chroma_weight_l0[i][c] = (int16_t)br_se(br);
                si->chroma_offset_l0[i][c] = (int16_t)br_se(br);
            }
        }
    }
}

static int parse_h264_slice_header(const uint8_t *slice, int size, int nal_type,
                                   const struct H264Params *params,
                                   struct H264SliceInfo *si) {
    struct BitReader br;
    int pic_parameter_set_id;
    int slice_type_mod;
    if (size < 2)
        return 0;
    br_init(&br, slice + 1, size - 1);
    si->first_mb = (int)br_ue(&br);
    si->slice_type = (int)br_ue(&br);
    slice_type_mod = si->slice_type % 5;
    if (slice_type_mod == 1)
        return 0;                     /* B slices: single-direction DPB only */
    pic_parameter_set_id = (int)br_ue(&br);
    if (pic_parameter_set_id != 0)
        return 0;
    if (params->separate_colour_plane_flag)
        br_bits(&br, 2);              /* colour_plane_id */
    si->frame_num = (int)br_bits(&br, params->log2_max_frame_num_minus4 + 4);
    if (!params->frame_mbs_only_flag) {
        int field_pic_flag = (int)br_bits(&br, 1);
        if (field_pic_flag)
            return 0;
    }
    if (nal_type == 5) {
        br_ue(&br);                   /* idr_pic_id */
    }
    if (params->pic_order_cnt_type == 0) {
        si->poc_lsb = (int)br_bits(&br, params->log2_max_pic_order_cnt_lsb_minus4 + 4);
        if (params->pic_order_present_flag)
            si->delta_poc_bottom = br_se(&br);
    } else if (params->pic_order_cnt_type == 1 && !params->delta_pic_order_always_zero_flag) {
        br_se(&br);                   /* delta_pic_order_cnt[0] */
        if (params->pic_order_present_flag)
            br_se(&br);               /* delta_pic_order_cnt[1] */
    }
    if (params->redundant_pic_cnt_present_flag)
        br_ue(&br);                   /* redundant_pic_cnt */
    si->num_ref_idx_l0_active_minus1 = params->num_ref_idx_l0_active_minus1;
    si->num_ref_idx_l1_active_minus1 = params->num_ref_idx_l1_active_minus1;
    /* B slices were rejected above, so only P (0/3) can override, and
     * only the L0 count exists in the bitstream */
    if (slice_type_mod == 0 || slice_type_mod == 3) {
        int num_ref_idx_active_override_flag = (int)br_bits(&br, 1);
        if (num_ref_idx_active_override_flag)
            si->num_ref_idx_l0_active_minus1 = (int)br_ue(&br);
    }
    if (!parse_h264_ref_pic_list_modification(&br, slice_type_mod, si))
        return 0;
    if (params->weighted_pred_flag && (slice_type_mod == 0 || slice_type_mod == 3))
        parse_h264_pred_weight_table(&br, params, si);
    else
        init_h264_weight_defaults(si);
    if (!parse_h264_dec_ref_pic_marking(&br, nal_type, si))
        return 0;
    if (params->entropy_coding_mode_flag && slice_type_mod != 2 && slice_type_mod != 4)
        si->cabac_init_idc = (int)br_ue(&br);
    si->slice_qp_delta = br_se(&br);
    if (params->deblocking_filter_control_present_flag) {
        si->disable_deblocking_filter_idc = (int)br_ue(&br);
        if (si->disable_deblocking_filter_idc != 1) {
            si->slice_alpha_c0_offset_div2 = br_se(&br);
            si->slice_beta_offset_div2 = br_se(&br);
        }
    }
    si->bit_offset = 8 + br.bits_read;
    return 1;
}

static int collect_h264_slices(const uint8_t *data, int size, const struct H264Params *params,
                               struct H264SliceInfo *slices, int max_slices) {
    int prefix = 0;
    int count = 0;
    int start = find_start_code(data, size, 0, &prefix);
    while (start >= 0) {
        int nal = start + prefix;
        int next_prefix = 0;
        int next = find_start_code(data, size, nal + 1, &next_prefix);
        int end = next >= 0 ? next : size;
        if (nal < end) {
            int type = data[nal] & 0x1f;
            if (type == 1 || type == 5) {
                if (count >= max_slices)
                    return -1;
                memset(&slices[count], 0, sizeof(slices[count]));
                slices[count].offset = nal;
                slices[count].size = end - nal;
                slices[count].nal_type = type;
                slices[count].nal_ref_idc = (data[nal] >> 5) & 3;
                if (!parse_h264_slice_header(data + nal, end - nal, type, params, &slices[count]))
                    return -1;
                count++;
            }
        }
        start = next;
        prefix = next_prefix;
    }
    return count;
}

typedef int (*vdpau_va_export_v1_fn)(VADisplay, VASurfaceID,
                                     uintptr_t *, void **, uint32_t *);

/* pin a slot for the painter; returns 0 on success */
static int h264_pin_surface(LibVADecoder *dec, int surface_index) {
    if (surface_index < 0 || surface_index >= dec->num_surfaces)
        return -1;
    atomic_fetch_or(&dec->pinned_bits, 1u << surface_index);
    return 0;
}

static LibVADecodeStatus h264_decoder_decode(LibVADecoder *dec,
                                             const uint8_t *data, int data_len,
                                             LibVADecodedFrame *frame) {
    VABufferID buffers[130];
    int nbuf = 0;
    struct H264SliceInfo slices[64];
    int nslices = 0;
    struct H264SliceInfo *first;
    int is_idr;
    int surface_index;
    VASurfaceID surface;
    VAStatus status;
    VAPictureParameterBufferH264 pic;
    VAIQMatrixBufferH264 iq;
    VASliceParameterBufferH264 slice;
    struct H264Params params;
    long long t0, t1, t2;

    if (!dec->h264_params)
        return set_message(dec, LIBVA_DEC_ERROR, "missing H.264 decoder parameters");
    params = *dec->h264_params;
    parse_h264_params(data, data_len, &params);
    dec->full_range = params.video_full_range_flag;
    nslices = collect_h264_slices(data, data_len, &params, slices,
                                  (int)(sizeof(slices) / sizeof(slices[0])));
    if (nslices <= 0)
        return set_message(dec, nslices < 0 ? LIBVA_DEC_UNSUPPORTED : LIBVA_DEC_ERROR,
                           nslices < 0 ? "unsupported H.264 slice header" : "no H.264 slice NAL found");
    *dec->h264_params = params;
    first = &slices[0];

    for (int i = 0; i < (int)(sizeof(buffers) / sizeof(buffers[0])); i++)
        buffers[i] = VA_INVALID_ID;
    {
        /* the pool is sized from the stream itself: SPS
         * max_num_ref_frames + the current picture + export slack.
         * Surfaces are full NV12 frames of VRAM, so provisioning the
         * spec worst case (16 refs) for every stream is expensive -
         * nvenc emits 3. */
        int refs = params.valid_sps ? params.max_num_ref_frames : 0;
        if (refs < 1)
            refs = 1;
        if (refs > H264_DPB_SIZE)
            refs = H264_DPB_SIZE;
        int needed = refs + 1 + H264_EXPORT_SLACK;
        if (dec->context == VA_INVALID_ID) {
            if (!params.valid_sps)
                return set_message(dec, LIBVA_DEC_ERROR,
                                   "no SPS before the first slice");
            dec->num_surfaces = needed;
            LibVADecodeStatus astatus = alloc_surfaces_and_context(dec);
            if (astatus != LIBVA_DEC_OK)
                return astatus;
            libva_log("libva h264: allocated %d surfaces (%d reference frames + 1 + %d export slack)",
                      dec->num_surfaces, refs, H264_EXPORT_SLACK);
        } else if (needed > dec->num_surfaces) {
            /* a mid-stream SPS raised max_num_ref_frames beyond the
             * pool: fail loudly - the CodecStateException restart
             * re-sizes from the new SPS */
            snprintf(dec->last_error, sizeof(dec->last_error),
                     "stream now declares %d reference frames, pool has %d surfaces",
                     refs, dec->num_surfaces);
            dec->last_status = (int)LIBVA_DEC_ERROR;
            libva_log("libva decode error: %s", dec->last_error);
            return LIBVA_DEC_ERROR;
        }
    }
    is_idr = first->nal_type == 5;
    if (params.pic_order_cnt_type == 1)
        return set_message(dec, LIBVA_DEC_UNSUPPORTED,
                           "H.264 pic_order_cnt_type 1 is not supported");
    if (is_idr) {
        /* spec 8.2.5.1: IDR unmarks all reference pictures */
        h264_dpb_flush(dec);
        dec->poc_prev_lsb = 0;
        dec->poc_prev_msb = 0;
        dec->poc_prev_frame_num = 0;
        dec->poc_prev_frame_num_offset = 0;
    }
    int top_foc = 0, bottom_foc = 0;
    if (h264_compute_poc(dec, &params, first, is_idr, &top_foc, &bottom_foc) != 0)
        return set_message(dec, LIBVA_DEC_UNSUPPORTED,
                           "H.264 pic_order_cnt_type 1 is not supported");
    surface_index = h264_pick_surface(dec);
    if (surface_index < 0)
        return set_message(dec, LIBVA_DEC_ERROR, "no free H.264 surface");
    surface = dec->surfaces[surface_index];

    memset(&pic, 0, sizeof(pic));
    pic.CurrPic.picture_id = surface;
    pic.CurrPic.frame_idx = (uint32_t)first->frame_num;
    pic.CurrPic.flags = first->nal_ref_idc ? VA_PICTURE_H264_SHORT_TERM_REFERENCE : 0;
    pic.CurrPic.TopFieldOrderCnt = top_foc;
    pic.CurrPic.BottomFieldOrderCnt = bottom_foc;
    for (int i = 0; i < 16; i++)
        fill_invalid_picture(&pic.ReferenceFrames[i]);
    {
        int n = 0;
        for (uint32_t refs = dec->dpb_used; refs && n < 16; refs &= refs - 1)
            h264_fill_va_picture(&pic.ReferenceFrames[n++], dec, __builtin_ctz(refs));
    }
    if (params.valid_sps &&
        ((params.width_mbs_minus1 + 1) * 16 > dec->surface_width ||
         (params.height_mbs_minus1 + 1) * 16 > dec->surface_height)) {
        /* an encoder padding beyond our 32-aligned pool would overflow
         * surface rows again - fail the decode loudly (the caller
         * raises CodecStateException; persistent failure falls back to
         * software decode via the setup-cost ratchet) */
        dec->last_status = (int)LIBVA_DEC_ERROR;
        snprintf(dec->last_error, sizeof(dec->last_error),
                 "coded size %dx%d exceeds surface pool %dx%d",
                 (params.width_mbs_minus1 + 1) * 16, (params.height_mbs_minus1 + 1) * 16,
                 dec->surface_width, dec->surface_height);
        libva_log("libva decode error: %s", dec->last_error);
        return LIBVA_DEC_ERROR;
    }
    pic.picture_width_in_mbs_minus1 = (uint16_t)(params.valid_sps ?
                                                params.width_mbs_minus1 :
                                                (roundup(dec->width, 16) / 16 - 1));
    pic.picture_height_in_mbs_minus1 = (uint16_t)(params.valid_sps ?
                                                 params.height_mbs_minus1 :
                                                 (roundup(dec->height, 16) / 16 - 1));
    pic.bit_depth_luma_minus8 = 0;
    pic.bit_depth_chroma_minus8 = 0;
    pic.num_ref_frames = (uint8_t)params.max_num_ref_frames;
    pic.seq_fields.bits.chroma_format_idc = (uint32_t)params.chroma_format_idc;
    pic.seq_fields.bits.residual_colour_transform_flag = (uint32_t)params.separate_colour_plane_flag;
    pic.seq_fields.bits.gaps_in_frame_num_value_allowed_flag = (uint32_t)params.gaps_in_frame_num_value_allowed_flag;
    pic.seq_fields.bits.frame_mbs_only_flag = (uint32_t)params.frame_mbs_only_flag;
    pic.seq_fields.bits.mb_adaptive_frame_field_flag = (uint32_t)params.mb_adaptive_frame_field_flag;
    pic.seq_fields.bits.direct_8x8_inference_flag = (uint32_t)params.direct_8x8_inference_flag;
    pic.seq_fields.bits.log2_max_frame_num_minus4 = (uint32_t)params.log2_max_frame_num_minus4;
    pic.seq_fields.bits.pic_order_cnt_type = (uint32_t)params.pic_order_cnt_type;
    pic.seq_fields.bits.log2_max_pic_order_cnt_lsb_minus4 = (uint32_t)params.log2_max_pic_order_cnt_lsb_minus4;
    pic.seq_fields.bits.delta_pic_order_always_zero_flag = (uint32_t)params.delta_pic_order_always_zero_flag;
    pic.pic_init_qp_minus26 = (int8_t)params.pic_init_qp_minus26;
    pic.pic_init_qs_minus26 = (int8_t)params.pic_init_qs_minus26;
    pic.chroma_qp_index_offset = (int8_t)params.chroma_qp_index_offset;
    pic.second_chroma_qp_index_offset = (int8_t)params.second_chroma_qp_index_offset;
    pic.pic_fields.bits.entropy_coding_mode_flag = (uint32_t)params.entropy_coding_mode_flag;
    pic.pic_fields.bits.weighted_pred_flag = (uint32_t)params.weighted_pred_flag;
    pic.pic_fields.bits.weighted_bipred_idc = (uint32_t)params.weighted_bipred_idc;
    pic.pic_fields.bits.transform_8x8_mode_flag = (uint32_t)params.transform_8x8_mode_flag;
    pic.pic_fields.bits.pic_order_present_flag = (uint32_t)params.pic_order_present_flag;
    pic.pic_fields.bits.deblocking_filter_control_present_flag =
        (uint32_t)params.deblocking_filter_control_present_flag;
    pic.pic_fields.bits.redundant_pic_cnt_present_flag = (uint32_t)params.redundant_pic_cnt_present_flag;
    pic.pic_fields.bits.constrained_intra_pred_flag = (uint32_t)params.constrained_intra_pred_flag;
    pic.pic_fields.bits.reference_pic_flag = (uint32_t)(first->nal_ref_idc != 0);
    pic.frame_num = (uint16_t)first->frame_num;
    status = vaCreateBuffer(dec->display, dec->context, VAPictureParameterBufferType,
                            sizeof(pic), 1, &pic, &buffers[nbuf++]);
    if (status != VA_STATUS_SUCCESS) {
        destroy_buffers(dec, buffers, nbuf);
        return set_error(dec, status, "vaCreateBuffer(H264 picture)");
    }

    fill_h264_default_iq_matrix(&iq);
    status = vaCreateBuffer(dec->display, dec->context, VAIQMatrixBufferType,
                            sizeof(iq), 1, &iq, &buffers[nbuf++]);
    if (status != VA_STATUS_SUCCESS) {
        destroy_buffers(dec, buffers, nbuf);
        return set_error(dec, status, "vaCreateBuffer(H264 iq)");
    }

    for (int s = 0; s < nslices; s++) {
        struct H264SliceInfo *si = &slices[s];
        memset(&slice, 0, sizeof(slice));
        slice.slice_data_size = (uint32_t)si->size;
        slice.slice_data_offset = 0;
        slice.slice_data_flag = VA_SLICE_DATA_FLAG_ALL;
        slice.slice_data_bit_offset = (uint16_t)si->bit_offset;
        slice.first_mb_in_slice = (uint16_t)si->first_mb;
        slice.slice_type = (uint8_t)(si->slice_type % 5);
        slice.direct_spatial_mv_pred_flag = 0;    /* B slices rejected at parse */
        slice.num_ref_idx_l0_active_minus1 = (uint8_t)si->num_ref_idx_l0_active_minus1;
        slice.num_ref_idx_l1_active_minus1 = (uint8_t)si->num_ref_idx_l1_active_minus1;
        slice.cabac_init_idc = (uint8_t)si->cabac_init_idc;
        slice.slice_qp_delta = (int8_t)si->slice_qp_delta;
        slice.disable_deblocking_filter_idc = (uint8_t)si->disable_deblocking_filter_idc;
        slice.slice_alpha_c0_offset_div2 = (int8_t)si->slice_alpha_c0_offset_div2;
        slice.slice_beta_offset_div2 = (int8_t)si->slice_beta_offset_div2;
        slice.luma_log2_weight_denom = (uint8_t)si->luma_log2_weight_denom;
        slice.chroma_log2_weight_denom = (uint8_t)si->chroma_log2_weight_denom;
        slice.luma_weight_l0_flag = si->luma_weight_l0_flag;
        slice.chroma_weight_l0_flag = si->chroma_weight_l0_flag;
        memcpy(slice.luma_weight_l0, si->luma_weight_l0, sizeof(slice.luma_weight_l0));
        memcpy(slice.luma_offset_l0, si->luma_offset_l0, sizeof(slice.luma_offset_l0));
        memcpy(slice.chroma_weight_l0, si->chroma_weight_l0, sizeof(slice.chroma_weight_l0));
        memcpy(slice.chroma_offset_l0, si->chroma_offset_l0, sizeof(slice.chroma_offset_l0));
        for (int i = 0; i < 32; i++) {
            fill_invalid_picture(&slice.RefPicList0[i]);
            fill_invalid_picture(&slice.RefPicList1[i]);
        }
        if ((si->slice_type % 5) == 0 || (si->slice_type % 5) == 3) {
            struct H264DPBEntry *l0[32];
            int nl0 = h264_build_ref_list_l0(dec, &params, si, l0, 32);
            for (int i = 0; i < nl0; i++)
                h264_fill_va_picture(&slice.RefPicList0[i], dec, (int)(l0[i] - dec->dpb));
        }
        status = vaCreateBuffer(dec->display, dec->context, VASliceParameterBufferType,
                                sizeof(slice), 1, &slice, &buffers[nbuf++]);
        if (status != VA_STATUS_SUCCESS) {
            destroy_buffers(dec, buffers, nbuf);
            return set_error(dec, status, "vaCreateBuffer(H264 slice)");
        }
        status = vaCreateBuffer(dec->display, dec->context, VASliceDataBufferType,
                                (unsigned int)si->size, 1, (void *)(data + si->offset), &buffers[nbuf++]);
        if (status != VA_STATUS_SUCCESS) {
            destroy_buffers(dec, buffers, nbuf);
            return set_error(dec, status, "vaCreateBuffer(H264 data)");
        }
    }

    t0 = usec_now();
    status = vaBeginPicture(dec->display, dec->context, surface);
    if (status == VA_STATUS_SUCCESS)
        status = vaRenderPicture(dec->display, dec->context, buffers, nbuf);
    if (status == VA_STATUS_SUCCESS)
        status = vaEndPicture(dec->display, dec->context);
    t1 = usec_now();
    destroy_buffers(dec, buffers, nbuf);
    if (status != VA_STATUS_SUCCESS)
        return set_error(dec, status, "VA H264 decode submit");

    status = vaSyncSurface(dec->display, surface);
    t2 = usec_now();
    if (status != VA_STATUS_SUCCESS)
        return set_error(dec, status, "vaSyncSurface");

    frame->us_submit = (int)(t1 - t0);
    frame->us_sync = (int)(t2 - t1);
    if (first->nal_ref_idc) {
        int cur_is_lt = 0, cur_lt_idx = 0, had_mmco5 = 0;
        int ins_frame_num = first->frame_num;
        if (is_idr) {
            /* DPB already flushed before decode */
            cur_is_lt = first->idr_long_term_reference_flag;
        } else if (first->adaptive_ref_pic_marking) {
            h264_dpb_apply_mmco(dec, &params, first, first->frame_num,
                                &cur_is_lt, &cur_lt_idx, &had_mmco5);
        } else {
            h264_dpb_sliding_window(dec, &params, first->frame_num);
        }
        if (had_mmco5) {
            /* spec 8.2.1: after MMCO5 the current picture counts as
             * frame_num 0 with its POC rebased to zero */
            int temp = top_foc < bottom_foc ? top_foc : bottom_foc;
            top_foc -= temp;
            bottom_foc -= temp;
            ins_frame_num = 0;
            dec->poc_prev_lsb = top_foc;
            dec->poc_prev_msb = 0;
            dec->poc_prev_frame_num = 0;
            dec->poc_prev_frame_num_offset = 0;
        }
        h264_dpb_insert(dec, surface_index, ins_frame_num,
                        top_foc, bottom_foc, cur_is_lt, cur_lt_idx);
    }

    /* zero-copy export: the frame never left the GPU - hand the
     * caller the GL_NV_vdpau_interop handle triple and pin the slot
     * (pinned only AFTER a successful export, so a refusal cannot
     * leak a pin) */
    vdpau_va_export_v1_fn fn = (vdpau_va_export_v1_fn)dec->export_fn;
    int rc = fn(dec->display, dec->surfaces[surface_index],
                &frame->vdp_device, &frame->get_proc_address, &frame->vdp_surface);
    if (rc != 0)
        return set_message(dec, LIBVA_DEC_ERROR, "surface export refused");
    h264_pin_surface(dec, surface_index);
    frame->surface_index = surface_index;
    frame->width = dec->width;
    frame->height = dec->height;
    frame->full_range = dec->full_range;
    return LIBVA_DEC_OK;
}

/* unpin a slot so the decode-target picker may reuse it; lifetime is
 * the caller's concern (the decoder must stay alive while any
 * exported frame or GL registration references it) */
void libva_decoder_release_surface(LibVADecoder *dec, int surface_index) {
    if (!dec || surface_index < 0 || surface_index >= dec->num_surfaces)
        return;
    atomic_fetch_and(&dec->pinned_bits, ~(1u << surface_index));
}

int libva_decoder_pinned_count(LibVADecoder *dec) {
    if (!dec)
        return 0;
    return __builtin_popcount(atomic_load(&dec->pinned_bits));
}

LibVADecodeStatus libva_decoder_decode(LibVADecoder *dec,
                                       const uint8_t *data, int data_len,
                                       LibVADecodedFrame *frame) {
    if (!dec || !data || data_len <= 0 || !frame)
        return LIBVA_DEC_ERROR;
    memset(frame, 0, sizeof(*frame));
    if (dec->codec == LIBVA_CODEC_H264)
        return h264_decoder_decode(dec, data, data_len, frame);
    return set_message(dec, LIBVA_DEC_UNSUPPORTED, "unknown VA decode codec");
}

int libva_decoder_get_last_status(LibVADecoder *dec) {
    return dec ? dec->last_status : 0;
}

const char* libva_decoder_get_last_error(LibVADecoder *dec) {
    return dec ? dec->last_error : "no decoder";
}
