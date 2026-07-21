/* This file is part of Xpra.
 * Copyright (C) 2026 Netflix, Inc.
 * Xpra is released under the terms of the GNU GPL v2, or, at your option, any
 * later version. See the file COPYING for details.
 * ABOUTME: libva decoder C API header.
 * ABOUTME: Flat C interface wrapping VA-API for use from Cython. */

#ifndef XPRA_LIBVA_DECODE_H
#define XPRA_LIBVA_DECODE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LibVADecoder LibVADecoder;

typedef enum {
    LIBVA_DEC_OK              =  0,
    LIBVA_DEC_ERROR           = -1,
    LIBVA_DEC_NOT_AVAILABLE   = -2,
    LIBVA_DEC_UNSUPPORTED     = -3,
} LibVADecodeStatus;

typedef enum {
    LIBVA_DEC_FMT_UNKNOWN = 0,
    LIBVA_DEC_FMT_NV12    = 1,
    LIBVA_DEC_FMT_YUV444P = 2,
    LIBVA_DEC_FMT_XYUV    = 3,
    LIBVA_DEC_FMT_AYUV    = 4,
} LibVADecodeFormat;

typedef struct {
    uint8_t *planes[3];
    int      strides[3];
    int      sizes[3];
    int      nplanes;
    int      width;
    int      height;
    int      depth;
    int      bytes_per_pixel;
    int      full_range;        /* colour range parsed from the bitstream (1=full, 0=studio) */
    LibVADecodeFormat format;
    int      us_submit;
    int      us_sync;
    int      us_map;
    int      us_copy;
    /* zero-copy GPU frame (h264): valid when nplanes == 0 - the frame
     * never left the GPU; these are the GL_NV_vdpau_interop handles.
     * The surface pool slot stays pinned until
     * libva_decoder_release_surface(surface_index). */
    uintptr_t vdp_device;
    void     *get_proc_address;
    uint32_t  vdp_surface;
    int       surface_index;
} LibVADecodedFrame;

typedef void (*libva_log_fn)(const char *msg);

void              libva_decode_set_log(libva_log_fn fn);
LibVADecodeStatus libva_decode_startup(void);
void              libva_decode_shutdown(void);
const char       *libva_decode_get_device(void);
const char       *libva_decode_get_vendor(void);
const char       *libva_decode_get_last_error(void);
int               libva_decode_get_major(void);
int               libva_decode_get_minor(void);
int               libva_decode_supports(const char *encoding, const char *colorspace);

LibVADecodeStatus libva_decoder_create(LibVADecoder **out, const char *encoding,
                                       int width, int height, const char *colorspace);
void              libva_decoder_destroy(LibVADecoder *dec);
LibVADecodeStatus libva_decoder_decode(LibVADecoder *dec,
                                       const uint8_t *data, int data_len,
                                       LibVADecodedFrame *frame);

/* Zero-copy surface export (vendor path, VDPAU-backed VA driver only):
 * H264 decode is EXPORT-ONLY - the driver must export
 * vdpau_va_export_v1 (checked at decoder create) and every decoded
 * h264 frame comes back from libva_decoder_decode as a GPU frame
 * (nplanes == 0, the vdp_* fields above); there is no CPU readback
 * for h264.
 *
 * Slot pins are DECODE-REUSE protection only, not lifetime
 * management: an exported frame's pool slot stays pinned (skipped by
 * the decode-target picker) until libva_decoder_release_surface.
 * The decoder's LIFETIME is owned by the caller - every consumer of
 * exported frames and GL registrations must keep the decoder alive
 * (the Cython layer does this with plain Python references) and
 * destroy it only via libva_decoder_destroy once nothing uses it. */
void libva_decoder_release_surface(LibVADecoder *dec, int surface_index);
int  libva_decoder_pinned_count(LibVADecoder *dec);

int               libva_decoder_get_width(LibVADecoder *dec);
int               libva_decoder_get_height(LibVADecoder *dec);
int               libva_decoder_get_last_status(LibVADecoder *dec);
const char       *libva_decoder_get_last_error(LibVADecoder *dec);
const char       *libva_decode_status_str(LibVADecodeStatus status);
const char       *libva_decode_format_str(LibVADecodeFormat format);

#ifdef __cplusplus
}
#endif

#endif /* XPRA_LIBVA_DECODE_H */
