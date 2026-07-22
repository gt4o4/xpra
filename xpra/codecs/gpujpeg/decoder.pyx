# This file is part of Xpra.
# Copyright (C) 2026 Antoine Martin <antoine@xpra.org>
# Xpra is released under the terms of the GNU GPL v2, or, at your option, any
# later version. See the file COPYING for details.

"""
JPEG decoding on the GPU via CESNET libgpujpeg (CUDA).

The decoder is a single persistent module-level object: libgpujpeg
uses the CUDA primary context of the current device (no context of
its own) and `gpujpeg_decoder_decode` re-initializes itself from each
image's headers, growing its workspace buffers monotonically - so one
decoder serves every size and never churns device allocations on the
steady state.  The CUDA context is created once, on the first decode,
and deliberately never destroyed: context lifecycle is the expensive
(and on old drivers, fragile) part; a resident context is the cheap
part.  gpujpeg_device_reset() must NEVER be called from here - it
cudaDeviceReset()s the whole process, nuking every other CUDA/GL
consumer.

LOCK-FREE: CUDA compute shares no state with the graphics engines
that needs caller-side serialization (validated under concurrent
VDPAU decode + GL churn), and the CUDA primary context is created
exactly once.  libgpujpeg's own constraint (one global __constant__
quantization table = two in-process decodes must not overlap) is
satisfied structurally: xpra decodes all draw packets on a single
decode thread, so gpujpeg calls never overlap.

Output is BGRX produced ON the GPU (444/4444 interleaving + channel
remap + 0xFF fill in the postprocessor kernel) and copied back into a
per-image host buffer: the CPU never touches entropy decoding, IDCT
or color conversion.
"""

import os
from time import monotonic
from typing import Any
from collections.abc import Sequence

from xpra.codecs.image import ImageWrapper
from xpra.log import Logger
from xpra.util.env import envint
from xpra.util.objects import typedict

log = Logger("decoder", "gpujpeg")

from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS, PyObject_GetBuffer, PyBuffer_Release
from xpra.buffers.membuf cimport getbuf, MemBuf  # pylint: disable=syntax-error

cdef extern from "cuda_runtime.h":
    int cudaSetDevice(int device) nogil
    int cudaGetDeviceCount(int *count) nogil

# CUDA-GL interop, prototyped by hand: cuda_gl_interop.h drags in the
# system GL headers, which this build environment does not carry (the
# client's GL is PyOpenGL at runtime).  These are plain cudart
# exports; GLuint is unsigned int by definition.
cdef extern from *:
    """
    #include <cuda_runtime.h>
    typedef struct cudaGraphicsResource *xpra_cudaGraphicsResource_t;
    extern cudaError_t cudaGraphicsGLRegisterBuffer(xpra_cudaGraphicsResource_t *resource,
                                                    unsigned int buffer, unsigned int flags);
    extern cudaError_t cudaGraphicsMapResources(int count, xpra_cudaGraphicsResource_t *resources,
                                                cudaStream_t stream);
    extern cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size,
                                                            xpra_cudaGraphicsResource_t resource);
    extern cudaError_t cudaGraphicsUnmapResources(int count, xpra_cudaGraphicsResource_t *resources,
                                                  cudaStream_t stream);
    extern cudaError_t cudaGraphicsUnregisterResource(xpra_cudaGraphicsResource_t resource);
    """
    ctypedef void *xpra_cudaGraphicsResource_t
    int cudaGraphicsGLRegisterBuffer(xpra_cudaGraphicsResource_t *resource,
                                     unsigned int buffer, unsigned int flags) nogil
    int cudaGraphicsMapResources(int count, xpra_cudaGraphicsResource_t *resources,
                                 void *stream) nogil
    int cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size,
                                             xpra_cudaGraphicsResource_t resource) nogil
    int cudaGraphicsUnmapResources(int count, xpra_cudaGraphicsResource_t *resources,
                                   void *stream) nogil
    int cudaGraphicsUnregisterResource(xpra_cudaGraphicsResource_t resource) nogil

DEF cudaGraphicsRegisterFlagsWriteDiscard = 2

cdef extern from "libgpujpeg/gpujpeg_type.h":
    enum gpujpeg_color_space:
        GPUJPEG_NONE
        GPUJPEG_RGB
    enum gpujpeg_pixel_format:
        GPUJPEG_PIXFMT_NONE
        GPUJPEG_U8
        GPUJPEG_444_U8_P012
        GPUJPEG_4444_U8_P0123

cdef extern from "libgpujpeg/gpujpeg_common.h":
    struct gpujpeg_image_parameters:
        int width
        int height
        gpujpeg_color_space color_space
        gpujpeg_pixel_format pixel_format
        int width_padding
    struct gpujpeg_parameters:
        pass

cdef extern from "libgpujpeg/gpujpeg_decoder.h":
    struct gpujpeg_decoder:
        pass
    enum gpujpeg_decoder_output_type:
        GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER
        GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER
    struct gpujpeg_decoder_output:
        gpujpeg_decoder_output_type type
        uint8_t *data
        size_t data_size
        gpujpeg_image_parameters param_image
    void gpujpeg_decoder_output_set_custom_cuda(gpujpeg_decoder_output *output,
                                                uint8_t *custom_cuda_buffer)
    gpujpeg_decoder* gpujpeg_decoder_create(void *stream)
    int gpujpeg_decoder_decode(gpujpeg_decoder *decoder, uint8_t *image, size_t image_size,
                               gpujpeg_decoder_output *output) nogil
    void gpujpeg_decoder_output_set_custom(gpujpeg_decoder_output *output, uint8_t *custom_buffer)
    void gpujpeg_decoder_set_output_format(gpujpeg_decoder *decoder,
                                           gpujpeg_color_space color_space,
                                           gpujpeg_pixel_format pixel_format)
    int gpujpeg_decoder_set_option(gpujpeg_decoder *decoder, const char *opt, const char *val)
    int gpujpeg_decoder_get_image_info(uint8_t *image, size_t image_size,
                                       gpujpeg_image_parameters *param_image,
                                       gpujpeg_parameters *param, int *segment_count)
    int gpujpeg_decoder_destroy(gpujpeg_decoder *decoder)


# the workspace grows monotonically with the largest image seen, and
# VRAM on the target hardware is precious: refuse anything beyond the
# cap and let the caller fall back to the CPU decoder
MAX_PIXELS: int = envint("XPRA_GPUJPEG_MAX_PIXELS", 6 * 1024 * 1024)

cdef gpujpeg_decoder *g_decoder = NULL
cdef gpujpeg_decoder *g_alpha_decoder = NULL
cdef int g_frames = 0


def get_version() -> tuple[int, int]:
    return (0, 27)


def get_type() -> str:
    return "gpujpeg"


def get_encodings() -> Sequence[str]:
    return ("jpeg", "jpega")


def get_info() -> dict[str, Any]:
    return {
        "version": get_version(),
        "encodings": get_encodings(),
        "max-pixels": MAX_PIXELS,
        "frames": g_frames,
    }


def init_module(options: typedict) -> None:
    log("gpujpeg.init_module(%s)", options)
    cdef int count = 0
    if cudaGetDeviceCount(&count) != 0 or count <= 0:
        raise ImportError("no CUDA device")


def cleanup_module() -> None:
    # the decoder and its primary context are deliberately left alone:
    # destroying a CUDA context at process teardown is all risk and no
    # reward, and cleanup_module may be called from atexit contexts
    log("gpujpeg.cleanup_module() frames=%i", g_frames)


cdef gpujpeg_decoder *get_decoder() except NULL:
    # single-decode-thread calls only (see the module docstring)
    global g_decoder
    if g_decoder != NULL:
        return g_decoder
    if cudaSetDevice(0) != 0:
        raise RuntimeError("gpujpeg: cudaSetDevice failed")
    cdef gpujpeg_decoder *dec = gpujpeg_decoder_create(NULL)
    if dec == NULL:
        raise RuntimeError("gpujpeg: decoder create failed")
    # BGRX/BGRA, produced on the GPU: 4-byte interleaved output with
    # the channels remapped to B,G,R and the fourth byte filled with
    # 0xFF (jpega overwrites it from the alpha plane afterwards)
    gpujpeg_decoder_set_output_format(dec, GPUJPEG_RGB, GPUJPEG_4444_U8_P0123)
    if gpujpeg_decoder_set_option(dec, b"dec_opt_channel_remap", b"210F") != 0:
        gpujpeg_decoder_destroy(dec)
        raise RuntimeError("gpujpeg: channel remap rejected")
    log.info("gpujpeg: CUDA decoder initialized (BGRX output, max %i pixels)", MAX_PIXELS)
    g_decoder = dec
    return g_decoder


cdef gpujpeg_decoder *get_alpha_decoder() except NULL:
    # a second persistent decoder configured for the GRAYSCALE alpha
    # jpeg of the "jpega" convention (its workspace is small: one
    # component at frame size)
    global g_alpha_decoder
    if g_alpha_decoder != NULL:
        return g_alpha_decoder
    if cudaSetDevice(0) != 0:
        raise RuntimeError("gpujpeg: cudaSetDevice failed")
    cdef gpujpeg_decoder *dec = gpujpeg_decoder_create(NULL)
    if dec == NULL:
        raise RuntimeError("gpujpeg: alpha decoder create failed")
    gpujpeg_decoder_set_output_format(dec, GPUJPEG_NONE, GPUJPEG_U8)
    log.info("gpujpeg: CUDA alpha decoder initialized")
    g_alpha_decoder = dec
    return g_alpha_decoder


def decompress(rgb_format: str, img_data, options=None) -> ImageWrapper:
    # BGRX for plain jpeg; BGRA for xpra's "jpega" convention (an RGB
    # jpeg at [0:alpha-offset] + a GRAYSCALE jpeg of the alpha plane
    # at [alpha-offset:] - jpeg itself has no alpha channel)
    cdef unsigned int alpha_offset = 0
    if options:
        alpha_offset = int(options.get("alpha-offset", 0))
    if rgb_format == "BGRX":
        if alpha_offset:
            raise ValueError("alpha data but BGRX requested")
    elif rgb_format == "BGRA":
        if not alpha_offset:
            raise ValueError("BGRA requested without alpha data")
    else:
        raise ValueError(f"unsupported rgb format {rgb_format!r}")
    cdef Py_buffer py_buf
    if PyObject_GetBuffer(img_data, &py_buf, PyBUF_ANY_CONTIGUOUS):
        raise ValueError(f"failed to read compressed data from {type(img_data)}")
    cdef uint8_t *src = <uint8_t *> py_buf.buf
    cdef size_t src_len = py_buf.len
    cdef size_t main_len = src_len
    cdef gpujpeg_image_parameters info
    cdef gpujpeg_parameters params
    cdef int segments = 0
    cdef gpujpeg_decoder *dec
    cdef gpujpeg_decoder_output out
    cdef int width, height, stride
    cdef int r
    cdef MemBuf membuf
    cdef MemBuf alphabuf
    cdef uint8_t *dst
    cdef uint8_t *adst
    cdef size_t i, npixels
    cdef double start = monotonic()
    global g_frames
    try:
        if alpha_offset:
            if alpha_offset >= src_len:
                raise ValueError(f"alpha offset {alpha_offset} beyond {src_len} bytes of data")
            main_len = alpha_offset
        if gpujpeg_decoder_get_image_info(src, main_len, &info, &params, &segments) != 0:
            raise RuntimeError("gpujpeg: cannot parse jpeg header")
        width = info.width
        height = info.height
        if width <= 0 or height <= 0:
            raise RuntimeError(f"gpujpeg: invalid dimensions {width}x{height}")
        if width * height > MAX_PIXELS:
            # graceful size gate: the caller falls back to the CPU decoder
            raise RuntimeError(f"gpujpeg: {width}x{height} exceeds the {MAX_PIXELS}-pixel cap")
        stride = width * 4
        membuf = getbuf(stride * height, 0)
        dst = <uint8_t *> membuf.get_mem()
        dec = get_decoder()
        gpujpeg_decoder_output_set_custom(&out, dst)
        with nogil:
            r = gpujpeg_decoder_decode(dec, src, main_len, &out)
        if r != 0:
            raise RuntimeError(f"gpujpeg: decode failed ({r})")
        if alpha_offset:
            # decode the alpha plane on the GPU too, then merge the
            # bytes into the 4th channel (the only CPU step: the
            # heavy Huffman+IDCT+CSC work of both images is CUDA)
            if gpujpeg_decoder_get_image_info(src + alpha_offset, src_len - alpha_offset,
                                              &info, &params, &segments) != 0:
                raise RuntimeError("gpujpeg: cannot parse alpha jpeg header")
            if info.width != width or info.height != height:
                raise RuntimeError(f"gpujpeg: alpha plane is {info.width}x{info.height},"
                                   f" expected {width}x{height}")
            npixels = <size_t> width * <size_t> height
            alphabuf = getbuf(npixels, 0)
            adst = <uint8_t *> alphabuf.get_mem()
            dec = get_alpha_decoder()
            gpujpeg_decoder_output_set_custom(&out, adst)
            with nogil:
                r = gpujpeg_decoder_decode(dec, src + alpha_offset,
                                           src_len - alpha_offset, &out)
                if r == 0:
                    for i in range(npixels):
                        dst[4 * i + 3] = adst[i]
            if r != 0:
                raise RuntimeError(f"gpujpeg: alpha decode failed ({r})")
        g_frames += 1
        log("gpujpeg decoded %ix%i %s in %ims", width, height, rgb_format,
            round(1000 * (monotonic() - start)))
        return ImageWrapper(0, 0, width, height, memoryview(membuf), rgb_format, 32, stride,
                            bytesperpixel=4, planes=ImageWrapper.PACKED)
    finally:
        PyBuffer_Release(&py_buf)


def decode_into_gl_buffer(rgb_format: str, img_data, options, pbo: int,
                          expected_w: int, expected_h: int) -> tuple[int, int]:
    """decode a plain jpeg DIRECTLY into a GL pixel-unpack buffer: the
    PBO is CUDA-registered and mapped, and the decoder's postprocessor
    writes its BGRX output straight to the mapped device pointer - no
    intermediate device buffer, no host bytes, no copies at all.
    Must run on the thread that owns the current GL context."""
    if rgb_format != "BGRX":
        raise ValueError(f"unsupported rgb format {rgb_format!r}")
    if options and int(options.get("alpha-offset", 0)):
        raise ValueError("jpega is host-path only")
    cdef Py_buffer py_buf
    if PyObject_GetBuffer(img_data, &py_buf, PyBUF_ANY_CONTIGUOUS):
        raise ValueError(f"failed to read compressed data from {type(img_data)}")
    cdef uint8_t *src = <uint8_t *> py_buf.buf
    cdef size_t src_len = py_buf.len
    cdef gpujpeg_image_parameters info
    cdef gpujpeg_parameters params
    cdef int segments = 0
    cdef gpujpeg_decoder *dec
    cdef gpujpeg_decoder_output out
    cdef int width, height
    cdef size_t need
    cdef unsigned int glbuf = pbo
    cdef xpra_cudaGraphicsResource_t res = NULL
    cdef void *mapped = NULL
    cdef size_t mapped_size = 0
    cdef int r
    cdef double start = monotonic()
    global g_frames
    try:
        if gpujpeg_decoder_get_image_info(src, src_len, &info, &params, &segments) != 0:
            raise RuntimeError("gpujpeg: cannot parse jpeg header")
        width = info.width
        height = info.height
        if width != expected_w or height != expected_h:
            raise RuntimeError(f"gpujpeg: jpeg is {width}x{height},"
                               f" expected {expected_w}x{expected_h}")
        if width * height > MAX_PIXELS:
            raise RuntimeError(f"gpujpeg: {width}x{height} exceeds the {MAX_PIXELS}-pixel cap")
        need = <size_t> width * <size_t> height * 4
        with nogil:
            r = cudaGraphicsGLRegisterBuffer(&res, glbuf, cudaGraphicsRegisterFlagsWriteDiscard)
        if r != 0:
            raise RuntimeError(f"gpujpeg: GL buffer registration failed ({r})")
        try:
            with nogil:
                r = cudaGraphicsMapResources(1, &res, NULL)
            if r != 0:
                raise RuntimeError(f"gpujpeg: map failed ({r})")
            try:
                with nogil:
                    r = cudaGraphicsResourceGetMappedPointer(&mapped, &mapped_size, res)
                if r != 0 or mapped == NULL:
                    raise RuntimeError(f"gpujpeg: mapped pointer failed ({r})")
                if mapped_size < need:
                    raise RuntimeError(f"gpujpeg: pbo is {mapped_size} bytes, need {need}")
                dec = get_decoder()
                gpujpeg_decoder_output_set_custom_cuda(&out, <uint8_t *> mapped)
                with nogil:
                    r = gpujpeg_decoder_decode(dec, src, src_len, &out)
                if r != 0:
                    raise RuntimeError(f"gpujpeg: decode failed ({r})")
            finally:
                with nogil:
                    cudaGraphicsUnmapResources(1, &res, NULL)
        finally:
            with nogil:
                cudaGraphicsUnregisterResource(res)
        g_frames += 1
        log("gpujpeg decoded %ix%i into a GL buffer in %ims", width, height,
            round(1000 * (monotonic() - start)))
        return width, height
    finally:
        PyBuffer_Release(&py_buf)


# a 16x16 uniform-grey 4:4:4 q90 JPEG generated by gpujpeg itself
TEST_DATA = bytes.fromhex(
    "ffd8ffe000104a46494600010101012c012c0000ffdb00430003020203020203"
    "03030304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d"
    "0e110e0b0b1016101113141515150c0f171816141812141514ffdb0043010304"
    "0405040509050509140d0b0d1414141414141414141414141414141414141414"
    "141414141414141414141414141414141414141414141414141414141414ffc0"
    "0011080010001003011100021101031101ffc4001f0000010501010101010100"
    "000000000000000102030405060708090a0bffc400b510000201030302040305"
    "0504040000017d01020300041105122131410613516107227114328191a10823"
    "42b1c11552d1f02433627282090a161718191a25262728292a3435363738393a"
    "434445464748494a535455565758595a636465666768696a737475767778797a"
    "838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7"
    "b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1"
    "f2f3f4f5f6f7f8f9faffc4001f01000301010101010101010100000000000001"
    "02030405060708090a0bffc400b5110002010204040304070504040001027700"
    "0102031104052131061241510761711322328108144291a1b1c109233352f015"
    "6272d10a162434e125f11718191a262728292a35363738393a43444546474849"
    "4a535455565758595a636465666768696a737475767778797a82838485868788"
    "898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4"
    "c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9"
    "faffdd0004000cfffe002143524541544f523a204750554a5045472c20717561"
    "6c697479203d20393000ffda0008010100003f00d5a28a2bffda000801021100"
    "3f00fa30003fffda0008010311003f00f9f0003fffd9"
)


def selftest(full=False) -> None:
    img = decompress("BGRX", TEST_DATA)
    assert img.get_width() == 16 and img.get_height() == 16, "unexpected decoded size"
    pixels = bytes(img.get_pixels())
    # uniform grey source (120,140,160 RGB) -> BGRX ~ (160,140,120,255)
    b, g, r, x = pixels[0], pixels[1], pixels[2], pixels[3]
    assert x == 255, f"X byte not filled: {x}"
    assert abs(b - 160) < 12 and abs(g - 140) < 12 and abs(r - 120) < 12, \
        f"unexpected color: bgrx={(b, g, r, x)}"
