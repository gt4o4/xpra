# This file is part of Xpra.
# Copyright (C) 2026 Antoine Martin <antoine@xpra.org>
# Xpra is released under the terms of the GNU GPL v2, or, at your option, any
# later version. See the file COPYING for details.

"""Video scaling decisions over explicit settings and a snapshot of window state."""

from dataclasses import dataclass
from math import sqrt, gcd

Scaling = tuple[int, int | float]
DEFAULT_SCALING_OPTIONS = (1, 10), (1, 5), (1, 4), (1, 3), (1, 2), (2, 3), (1, 1)


@dataclass(frozen=True, slots=True)
class ScalingSettings:
    enabled: bool = True
    forced_scaling: Scaling | None = None
    options: tuple[Scaling, ...] = DEFAULT_SCALING_OPTIONS
    pps_target: int = 25*1920*1080
    min_pps: int = 25*320*240
    downscale: bool = True
    downscale_threshold: int = 20
    text_quality: int = 100


@dataclass(frozen=True, slots=True)
class ScalingState:
    width: int
    height: int
    max_width: int = 4096
    max_height: int = 4096
    max_pixels: int = 0
    client_render_size: tuple[int, int] = (0, 0)
    control: int | None = None
    requested_scaling: Scaling | None = None
    actual_scaling: Scaling = (1, 1)
    time_since_resize: float = float("inf")
    time_since_scroll: float = float("inf")
    time_since_pipeline: float = float("inf")
    damage_events_count: int = 0
    content_types: tuple[str, ...] = ()
    quality: int = 100
    speed: int = 50
    bandwidth_limit: int = 0
    is_shadow: bool = False
    image_depth: int = 24
    fullscreen: bool = False
    maximized: bool = False


def fits_pixel_budget(width: int, height: int, max_pixels: int) -> bool:
    # measured on 32-aligned dimensions: hardware decoder area limits
    # are macroblock counts of the CODED size (ie: VDPAU feature-set-A
    # caps H264 at 8192 16x16 macroblocks = 2097152 pixels), and
    # encoders pad the coded size - nvenc to 32-aligned dimensions
    # (verified: a 1884x1104 scale target was coded as 1888x1120 =
    # 8260 macroblocks, busting an 8192 budget the unpadded size met).
    # 32 covers the worst common alignment; 16-aligning encoders just
    # get a slightly conservative fit.
    if max_pixels <= 0:
        return True
    aligned_w = (width + 31) & ~31
    aligned_h = (height + 31) & ~31
    return aligned_w * aligned_h <= max_pixels


def best_fit_scaling(width: int, height: int, mw: int, mh: int, max_pixels: int) -> tuple[int, int] | None:
    # the largest exact scaling fraction that satisfies the size and
    # (32-aligned) pixel-budget constraints - computed directly rather
    # than quantized to the scaling options, so an oversized window keeps
    # the maximum resolution the limits allow.  Uses den=width so the
    # scaled width IS the numerator (integer-exact); the fit predicate
    # is monotone in num, so binary search applies.
    def fits(num: int) -> bool:
        sw = num
        sh = height * num // width
        return sw <= mw and sh <= mh and sw >= 1 and sh >= 1 \
            and fits_pixel_budget(sw, sh, max_pixels)
    if fits(width):
        return 1, 1
    lo, hi = 0, width      # lo: always fits (0=sentinel), hi: does not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    if lo == 0:
        return None
    g = gcd(lo, width)
    return lo // g, width // g


def get_min_required_scaling(state: ScalingState, settings: ScalingSettings, default: Scaling = (1, 1)) -> Scaling:
    mw, mh = state.max_width, state.max_height
    if settings.downscale and state.client_render_size != (0, 0):
        # There is no need to send more pixels than the client will render.
        crsw, crsh = state.client_render_size
        mw = min(mw, crsw)
        mh = min(mh, crsh)
    if state.width <= mw and state.height <= mh and fits_pixel_budget(state.width, state.height, state.max_pixels):
        return default
    # most encoders can't deal with that!
    # compute the largest exact fraction that fits (not quantized to the
    # scaling options - keep the maximum resolution the limits allow):
    best = best_fit_scaling(state.width, state.height, mw, mh, state.max_pixels)
    if best:
        return best
    raise ValueError(f"BUG: failed to find a scaling value for window size {state.width}x{state.height}")


def get_scaling_override(state: ScalingState, settings: ScalingSettings) -> Scaling | None:
    """Resolve controls and holdoffs before the caller measures video activity."""
    if state.width == 0 or state.height == 0 or not settings.enabled:
        return 1, 1
    if settings.forced_scaling:
        return get_min_required_scaling(state, settings, settings.forced_scaling)
    if state.control == 0:
        return get_min_required_scaling(state, settings)
    if state.requested_scaling:
        return get_min_required_scaling(state, settings, state.requested_scaling)
    if state.time_since_resize < 0.5 or state.time_since_scroll < 5:
        return get_min_required_scaling(state, settings, state.actual_scaling)
    if state.damage_events_count <= 50 or ("text" in state.content_types and settings.text_quality > 90):
        return get_min_required_scaling(state, settings)
    return None


def calculate_scaling(state: ScalingState, settings: ScalingSettings, *, fps: int = 0,
                      matches_subregion: bool = False, subregion_is_video: bool = False) -> Scaling:
    override = get_scaling_override(state, settings)
    if override is not None:
        return override
    video = "video" in state.content_types or (matches_subregion and subregion_is_video)
    q = state.quality
    if state.control is None:
        # Auto mode derives the target from quality, speed and measured activity.
        if q >= 65 + int(video)*30 or fps == 0:
            scaling = get_min_required_scaling(state, settings)
        else:
            scaling = _get_auto_scaling(state, settings, fps, video)
    else:
        scaling = _get_controlled_scaling(state, fps, matches_subregion, video)
    num, den = scaling
    if num/den > 1.0:
        return 1, 1
    if num/den < 0.1:
        return 1, 10
    return scaling


def _get_auto_scaling(state: ScalingState, settings: ScalingSettings, fps: int, video: bool) -> Scaling:
    pps = fps*state.width*state.height
    if state.bandwidth_limit > 0:
        # Assume 95% compression and convert the bandwidth limit to bytes/s.
        target = max(settings.min_pps, state.bandwidth_limit//8*20)
    else:
        target = settings.pps_target
    if state.is_shadow:
        target *= 16
    elif "text" in state.content_types:
        target *= 4
    elif not video:
        target *= 2
    if state.image_depth == 30:
        target *= 10
    target = target * (100+max(0, state.quality-video*30))**2 // 200**2
    target = target * 60**2 // (state.speed+20)**2
    min_num, min_den = get_min_required_scaling(state, settings)
    min_ratio = min_num/min_den
    denom_mult = 1
    if settings.downscale:
        crsw, crsh = state.client_render_size
        if crsw > 0 and crsh > 0 and state.width-crsw > settings.downscale_threshold and state.height-crsh > settings.downscale_threshold:
            denom_mult = max(state.width/crsw, state.height/crsh)
    scores = {}
    for num, denom in settings.options:
        if denom_mult > 1.0 and 0.5 < num/denom < 1.0:
            # Match the client's downscaling with whole multiples, skipping 2/3.
            continue
        denom *= denom_mult
        spps = pps*(num**2)/(denom**2)
        score = round(abs(1-target/spps)*100)
        if state.actual_scaling == (num, denom):
            elapsed = min(16.0, max(0.0, state.time_since_pipeline))
            score = round(score / sqrt(17-elapsed))
        if num/denom > min_ratio:
            score = round(score*100)
        scores[score] = num, denom
    if scores:
        return scores[min(scores)]
    return get_min_required_scaling(state, settings)


def _get_controlled_scaling(state: ScalingState, fps: int, matches_subregion: bool, video: bool) -> Scaling:
    # Manual controls range from 1 to 100, from least to most aggressive.
    if matches_subregion:
        control = (state.control+50)*2 if video else state.control+25
    else:
        control = max(0, (state.control-50)//2)
    resistance = control/50.0
    if state.actual_scaling != (1, 1):
        resistance += 0.5*state.actual_scaling[1]/state.actual_scaling[0]
    qs = state.speed > state.quality-resistance*10 and state.quality < 50+resistance*15
    if state.fullscreen and (qs or fps >= max(2, round(10-resistance*3))):
        return 1, 3
    if state.maximized and (qs or fps >= max(2, round(10-resistance*3))):
        return 1, 2
    pixels = state.width*state.height
    if pixels >= (2560-resistance*768)*1600 and (qs or fps >= max(4, round(25-resistance*5))):
        return 1, 3
    if pixels >= (1920-resistance*384)*1200 and (qs or fps >= max(5, round(30-resistance*10))):
        return 2, 3
    if pixels >= (1200-resistance*256)*1024 and (qs or fps >= max(10, round(50-resistance*15))):
        return 2, 3
    return 1, 1
