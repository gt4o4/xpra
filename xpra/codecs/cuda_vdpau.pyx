# This file is part of Xpra.
# Copyright (C) 2026 Antoine Martin <antoine@xpra.org>
# Xpra is released under the terms of the GNU GPL v2, or, at your option, any
# later version. See the file COPYING for details.

"""
The CUDA <-> VDPAU stack bootstrap: (1) the device bind, (2) the
scaled-paint kernel loader.

PART 1 - THE BIND.

cudaVDPAUSetVDPAUDevice must run BEFORE the process creates any CUDA
context (it configures how the primary context will be built; there is
no retrofit call - a late bind fails with "cannot set while device is
active in this process"), and it binds exactly ONE VdpDevice: only
surfaces belonging to that device can ever register with
cudaGraphicsVDPAURegisterVideoSurface (handles are meaningless outside
their owning device's handle table).

The VA->VDPAU bridge (gt4o4/vdpau-va-driver-vp9) therefore shares one
process-global, never-destroyed VdpDevice across all VADisplays, and
exports it through the vdpau_va_export_device_v1 vendor extension the
moment any VADisplay is initialized.  This module opens a bootstrap
VADisplay, pulls the device pair out through that export, binds, and
terminates the bootstrap display - the shared device outlives it by
design.  It is COMPILED so the FFI is honest linkage: libva /
libva-x11 / libX11 are its own DT_NEEDED (a nix closure has them on
no default loader path), cudaVDPAUSetVDPAUDevice resolves through
libgpujpeg's cudart, and only the vendor export stays a runtime dlsym
- it exists solely in the bridge driver (the vdpau_va_export_v1
precedent in va_decode.c).  The build gates this module on libva +
gpujpeg being available: where it does not exist, there is nothing to
bind for.

Every CUDA consumer must call ensure_vdpau_bind() before its first
context-creating call.  Failure keeps jpeg decoding fully working (it
needs no VDPAU) but makes libva h264 decoding unavailable: scaled
VDPAU paints are CUDA-only, so a session that cannot bind must not
decode h264 in hardware at all - the decoder-create gate raises
EncodingNotSupported and openh264 takes over.

PART 2 - THE KERNEL LOADER.

The scaled-paint kernel ships as source (fs/share/xpra/cuda/
xpra_vdpau_scale.cu) and is compiled at package build to an sm_11
cubin installed beside the encoder fatbins.  The loader half of this
module loads that cubin through pycuda's driver API INSIDE the
primary context part 1 bound: Context.attach() (cuCtxAttach - adopt
the current context, never create one), module_from_buffer, per-paint
texref binding, prepared cuLaunchKernel.  The interop registration
and mapping stay in the gpujpeg codec (cudart); this module only
binds texture references and launches.

THREAD CONFINEMENT: pycuda's context bookkeeping is thread-local, so
ensure_scale_module()/bind_surface()/scale_band() must only ever run
on the paint (GTK main) thread - which every caller does by
construction (paints marshal through with_gfx_context).
scale_available() is context-free and safe from any thread: the
libva decoder-create gate calls it on the decode thread.

DRIVER-API TEXTURING: a texture's read mode is HOST-side state, not
cubin metadata (the runtime API carries it in the host texture<>
object).  set_array uses CU_TRSA_OVERRIDE_FORMAT, which adopts the
array's format - so set_format must follow EVERY set_array.  The
field references are cudaReadModeNormalizedFloat = the driver default
flags 0; setting TRSF_READ_AS_INTEGER would be wrong.

The attached context is process-global and lives until exit; an
atexit hook detaches it because pycuda's thread-local context stack
aborts the process if non-empty at TSD cleanup.  cuCtxDetach on the
primary context only decrements the attach count - the shared
VdpDevice-bound context itself is never destroyed, per doctrine.
"""

import os
import atexit
import threading

from xpra.log import Logger

from libc.stdint cimport uintptr_t

log = Logger("cuda")


cdef extern from "X11/Xlib.h":
    ctypedef struct Display:
        pass
    Display *XOpenDisplay(const char *display_name) nogil

cdef extern from "va/va.h":
    ctypedef void *VADisplay
    int vaInitialize(VADisplay dpy, int *major_version, int *minor_version) nogil
    int vaTerminate(VADisplay dpy) nogil

cdef extern from "va/va_x11.h":
    VADisplay vaGetDisplay(Display *dpy) nogil

cdef extern from *:
    """
    #include <dlfcn.h>
    #include <stdint.h>
    /* the vendor export lives in the bridge DRIVER (dlopen'd by libva
       at vaInitialize with RTLD_GLOBAL), never in libva itself - so it
       is looked up at runtime, exactly like vdpau_va_export_v1 in
       va_decode.c */
    typedef int (*xpra_export_device_fn)(void *va_dpy, uintptr_t *vdp_device,
                                         void **get_proc_address);
    static xpra_export_device_fn xpra_find_export_device(void) {
        return (xpra_export_device_fn) dlsym(RTLD_DEFAULT, "vdpau_va_export_device_v1");
    }
    /* cuda_vdpau_interop.h needs vdpau/vdpau.h, which this build env
       does not carry - hand prototype (VdpDevice is uint32_t,
       VdpGetProcAddress* passes as void*); the symbol resolves through
       libgpujpeg's cudart at load time */
    extern int cudaVDPAUSetVDPAUDevice(int device, unsigned int vdpDevice,
                                       void *vdpGetProcAddress);
    """
    ctypedef int (*xpra_export_device_fn)(void *va_dpy, uintptr_t *vdp_device,
                                          void **get_proc_address) nogil
    xpra_export_device_fn xpra_find_export_device() nogil
    int cudaVDPAUSetVDPAUDevice(int device, unsigned int vdpDevice,
                                void *vdpGetProcAddress) nogil


_lock = threading.Lock()
_result = None


def _try_bind() -> bool:
    cdef Display *xdpy
    cdef VADisplay vadpy
    cdef int major = 0
    cdef int minor = 0
    cdef int r
    cdef xpra_export_device_fn export_device
    cdef uintptr_t dev = 0
    cdef void *gpa = NULL

    # the bootstrap X connection is DELIBERATELY never closed: the
    # bridge normally creates the shared VdpDevice on its own dedicated
    # X connection, but its fallback path adopts the CALLER's display -
    # this one - and the immortal device would then live on it.  One fd
    # for the process lifetime.
    with nogil:
        xdpy = XOpenDisplay(NULL)
    if xdpy == NULL:
        log.warn("Warning: cuda_vdpau bind skipped: cannot open the X display")
        return False
    vadpy = vaGetDisplay(xdpy)
    if vadpy == NULL:
        log.warn("Warning: cuda_vdpau bind skipped: vaGetDisplay failed")
        return False
    with nogil:
        r = vaInitialize(vadpy, &major, &minor)
    if r != 0:
        log.warn("Warning: cuda_vdpau bind skipped: vaInitialize failed (%i)", r)
        return False
    try:
        # the bridge is loaded now - find its vendor extension
        export_device = xpra_find_export_device()
        if export_device == NULL:
            log("cuda_vdpau: no vdpau_va_export_device_v1 (bridge too old, or not the vdpau driver)")
            return False
        with nogil:
            r = export_device(<void *> vadpy, &dev, &gpa)
        if r != 0:
            log.warn("Warning: cuda_vdpau bind skipped: device export failed (%i)", r)
            return False
        with nogil:
            r = cudaVDPAUSetVDPAUDevice(0, <unsigned int> dev, gpa)
        if r != 0:
            log.warn("Warning: cudaVDPAUSetVDPAUDevice failed (%i)", r)
            return False
        log.info("CUDA bound to shared VdpDevice %#x", <unsigned long> dev)
        return True
    finally:
        # the shared device is process-global and immortal: the
        # bootstrap VADisplay is free to go (its X connection stays,
        # see above)
        with nogil:
            vaTerminate(vadpy)


def ensure_vdpau_bind() -> bool:
    """idempotent; must be called before any context-creating CUDA
    call.  Returns True if the process CUDA context is (or will be)
    VDPAU-bound."""
    global _result
    with _lock:
        if _result is None:
            if not os.environ.get("DISPLAY"):
                _result = False
            else:
                try:
                    _result = _try_bind()
                except Exception as e:
                    log("cuda_vdpau bind failed", exc_info=True)
                    log.warn("Warning: cuda_vdpau bind failed: %s", e)
                    _result = False
        return _result


# ---------------------------------------------------------------------
# part 2: the scaled-paint kernel loader

CUBIN_NAME = "xpra_vdpau_scale.cubin"
KERNEL_NAME = "xpra_p1h_kernel"
# must equal P1H_BLOCK_X in xpra_vdpau_scale.cu: the kernel indexes by
# the MACRO (ox0 = blockIdx.x * P1H_BLOCK_X), not blockDim.x, so a
# divergence corrupts geometry silently rather than just perf
BLOCK_X = 128

_scale_lock = threading.Lock()
_scale_state = None      # (ctx, module, prepared function, texrefs)
_scale_available = None


def _cubin_path() -> str:
    from xpra.platform.paths import get_resources_dir
    return os.path.join(get_resources_dir(), "cuda", CUBIN_NAME)


def scale_available() -> bool:
    """context-free, any-thread: is the scaled-paint kernel loadable?
    The cubin check is the load-bearing half on hosts that carry
    pycuda for other codecs (the nvenc server) but no sm_11 cubin."""
    global _scale_available
    if _scale_available is None:
        try:
            path = _cubin_path()
            if not os.path.exists(path):
                log("cuda_vdpau: no cubin at %r", path)
                _scale_available = False
            else:
                import pycuda.driver  # noqa: F401  (lazy - never at module scope)
                _scale_available = True
        except ImportError as e:
            log("cuda_vdpau: no pycuda: %s", e)
            _scale_available = False
    return _scale_available


def ensure_scale_module():
    """paint-thread-only; idempotent.  Adopts the CURRENT context -
    the caller guarantees a cudart call preceded on this thread (the
    strip-buffer registration or the surface map, both cudart)."""
    global _scale_state
    with _scale_lock:
        if _scale_state is not None:
            return _scale_state
        import pycuda.driver as drv
        drv.init()
        ctx = drv.Context.attach()
        mod = fn = texrefs = None
        try:
            with open(_cubin_path(), "rb") as f:
                data = f.read()
            mod = drv.module_from_buffer(data)
            fn = mod.get_function(KERNEL_NAME)
            # (strip ptr, pitch_words, vid_w, vid_h, out_w, src_y0,
            #  inv_scale_x, full_range)
            fn.prepare("Piiiiifi")
            texrefs = tuple(mod.get_texref(n) for n in ("tex_y0", "tex_y1", "tex_c0", "tex_c1"))
        except Exception:
            # drop the module reference BEFORE the detach so its
            # finalizer (cuModuleUnload) runs while the context wrapper
            # is still valid - after detach the dead-context guard
            # skips the unload and the module code would leak into the
            # still-alive primary context, once per retry
            fn = texrefs = None
            mod = None
            ctx.detach()
            raise
        # pycuda's thread-local context stack aborts if non-empty at
        # TSD cleanup: detach at exit (attach-count decrement only)
        atexit.register(ctx.detach)
        _scale_state = (ctx, mod, fn, texrefs)
        log.info("cuda_vdpau: %s loaded (%i bytes)", CUBIN_NAME, len(data))
        return _scale_state


def bind_surface(y0: int, y1: int, c0: int, c1: int) -> None:
    """bind the 4 mapped field arrays (foreign cudaArray_t handles
    from the gpujpeg codec's surface map) to the module texrefs.
    Once per paint - the handles are only valid while mapped."""
    import pycuda.driver as drv
    texrefs = ensure_scale_module()[3]
    fmt = drv.array_format.UNSIGNED_INT8
    clamp = drv.address_mode.CLAMP
    point = drv.filter_mode.POINT
    for texref, handle, channels in zip(texrefs, (y0, y1, c0, c1), (1, 1, 2, 2)):
        texref.set_array(drv.array_from_handle(handle, managed=False))
        # AFTER set_array: OVERRIDE_FORMAT adopted the array's format
        texref.set_format(fmt, channels)
        texref.set_filter_mode(point)
        texref.set_address_mode(0, clamp)
        texref.set_address_mode(1, clamp)


def scale_band(buf_ptr: int, buf_size: int,
               vid_w: int, vid_h: int, out_w: int,
               src_y0: int, strip_rows: int,
               pitch_words: int, full_range: bool) -> None:
    """launch the P1+H kernel for one band into the mapped strip
    buffer.  The caller owns the band geometry; the size check here
    guards the write into the mapped buffer."""
    need = pitch_words * 4 * strip_rows
    if buf_size < need:
        raise RuntimeError(f"cuda_vdpau: strip buffer is {buf_size} bytes, need {need}"
                           f" (pitch_words={pitch_words} strip_rows={strip_rows})")
    if vid_w <= 0 or vid_h <= 0 or out_w <= 0 or strip_rows <= 0 or pitch_words < out_w:
        raise RuntimeError(f"cuda_vdpau: invalid geometry {vid_w}x{vid_h}->{out_w}"
                           f" rows={strip_rows} pitch={pitch_words}")
    fn = ensure_scale_module()[2]
    fn.prepared_call(((out_w + BLOCK_X - 1) // BLOCK_X, strip_rows), (BLOCK_X, 1, 1),
                     buf_ptr, pitch_words, vid_w, vid_h, out_w,
                     src_y0, vid_w / out_w, int(full_range))
