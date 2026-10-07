"""
Low-level image math shared by every stage of the beauty pipeline.

Everything here works on float32 BGR images in [0, 1] unless a docstring says
otherwise, because chaining eight uint8 stages quantises the picture to death:
skin smoothing, tone mapping and warping each lose a little, and the losses are
what make a retouched video look "processed". One conversion in, one out.
"""
from __future__ import annotations

import numpy as np

import cv2

EPS = 1e-6

# Rec.709 luma weights in OpenCV's BGR channel order.
LUMA_BGR = np.float32([0.0722, 0.7152, 0.2126])


# ---------------------------------------------------------------- conversions

def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 BGR -> float32 BGR in [0, 1] (a float input is passed through)."""
    if img.dtype == np.float32:
        return img
    return img.astype(np.float32) / 255.0


def to_u8(img: np.ndarray) -> np.ndarray:
    """float32 BGR in [0, 1] -> uint8 BGR, rounded rather than truncated."""
    if img.dtype == np.uint8:
        return img
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def luminance(bgr: np.ndarray) -> np.ndarray:
    """Single-channel Rec.709 luma of a float BGR image."""
    return np.tensordot(bgr, LUMA_BGR, axes=([2], [0])).astype(np.float32)


def as3(mask: np.ndarray) -> np.ndarray:
    """Broadcast a HxW mask to HxWx3 without copying the data three times."""
    if mask.ndim == 3:
        return mask
    return mask[:, :, None]


# ------------------------------------------------------------------- filtering

def box_filter(img: np.ndarray, radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REPLICATE)


def gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur by sigma, kernel size derived from it."""
    if sigma <= 0:
        return img
    k = int(max(3, round(sigma * 4) | 1))
    return cv2.GaussianBlur(img, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float,
                  subsample: int = 4) -> np.ndarray:
    """
    Fast guided filter (He et al.), used everywhere a bilateral filter would
    normally go. Two reasons: it is O(1) in the radius, and it does not produce
    the "plastic" gradient reversal bilateral filtering leaves on skin.

    `subsample` computes the coefficients on a downscaled copy - the standard
    fast-guided-filter trick. The coefficients are smooth by construction, so
    the only thing lost is a little precision at the very edges.
    """
    radius = max(1, int(radius))
    s = max(1, int(subsample))
    h, w = guide.shape[:2]
    if s > 1:
        gs = cv2.resize(guide, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
        ss = cv2.resize(src, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
        r = max(1, radius // s)
    else:
        gs, ss, r = guide, src, radius

    mean_g = box_filter(gs, r)
    mean_s = box_filter(ss, r)
    corr_gg = box_filter(gs * gs, r)
    corr_gs = box_filter(gs * ss, r)
    var_g = np.maximum(corr_gg - mean_g * mean_g, 0.0)
    cov_gs = corr_gs - mean_g * mean_s

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)
    if s > 1:
        mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
    return mean_a * guide + mean_b


def edge_preserving_smooth(bgr: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Guided-filter smoothing of a colour image, guided by its own luma."""
    guide = luminance(bgr)
    out = np.empty_like(bgr)
    for c in range(3):
        out[:, :, c] = guided_filter(guide, bgr[:, :, c], radius, eps)
    return out


def unsharp(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """Classic unsharp mask. `amount` 0 = unchanged, 1 = a strong sharpen."""
    if amount <= 0 or sigma <= 0:
        return img
    blurred = gaussian(img, sigma)
    return img + (img - blurred) * amount


# ------------------------------------------------------------------- blending

def soft_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Photoshop's soft-light. Gentler than overlay, which is why it is used
    for hair gloss and highlight shaping instead of a straight multiply."""
    b = np.clip(base, 0.0, 1.0)
    s = np.clip(blend, 0.0, 1.0)
    return np.where(s <= 0.5,
                    b - (1 - 2 * s) * b * (1 - b),
                    b + (2 * s - 1) * (np.sqrt(np.maximum(b, 0.0)) - b))


def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - np.clip(a, 0, 1)) * (1.0 - np.clip(b, 0, 1))


def blend(base: np.ndarray, layer: np.ndarray, mask) -> np.ndarray:
    """base*(1-m) + layer*m where m is a scalar, HxW mask or HxWx3 mask."""
    if np.isscalar(mask):
        if mask <= 0:
            return base
        return base * (1.0 - mask) + layer * mask
    m = as3(mask)
    return base * (1.0 - m) + layer * m


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, EPS), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a 0..1 mask so a composite has no cut-out edge."""
    return np.clip(gaussian(mask.astype(np.float32), sigma), 0.0, 1.0)


# ----------------------------------------------------------------- masks/geom

def poly_mask(shape, polys, value: float = 1.0) -> np.ndarray:
    """Filled polygon mask (float32 HxW) from a list of Nx2 integer arrays."""
    m = np.zeros(shape[:2], np.float32)
    pts = [np.round(p).astype(np.int32) for p in polys if p is not None and len(p) >= 3]
    if pts:
        cv2.fillPoly(m, pts, value, lineType=cv2.LINE_AA)
    return m


def hull_mask(shape, points, dilate_px: int = 0) -> np.ndarray:
    """Convex hull of a point set, optionally dilated, as a float32 mask."""
    if points is None or len(points) < 3:
        return np.zeros(shape[:2], np.float32)
    hull = cv2.convexHull(np.round(np.asarray(points)).astype(np.int32))
    m = np.zeros(shape[:2], np.float32)
    cv2.fillConvexPoly(m, hull, 1.0, lineType=cv2.LINE_AA)
    if dilate_px > 0:
        k = int(dilate_px) * 2 + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m


def skin_likelihood(bgr_float: np.ndarray) -> np.ndarray:
    """
    0..1 probability that a pixel is skin, from chroma alone.

    Chroma-only on purpose: any brightness term makes the mask fall apart the
    moment the subject walks into shade, and a smoothing mask that flickers
    with the lighting is worse than no mask at all. The Cr/Cb window is wide
    enough to cover the full range of human skin tones - it is there to reject
    hair, lips, clothing and background inside the face oval, not to decide who
    counts as skin.
    """
    u8 = to_u8(bgr_float)
    ycrcb = cv2.cvtColor(u8, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    # Soft window instead of a hard threshold, so the mask has no stair-step.
    p_cr = smoothstep(128.0, 137.0, cr) * (1.0 - smoothstep(168.0, 180.0, cr))
    p_cb = smoothstep(70.0, 80.0, cb) * (1.0 - smoothstep(125.0, 138.0, cb))
    return np.clip(p_cr * p_cb, 0.0, 1.0)


# ------------------------------------------------------------------ temporal

class EMA:
    """
    Exponential moving average with a "reset if it jumped" escape hatch.

    Every per-frame measurement in this pipeline (landmarks, silhouette width,
    exposure percentiles) goes through one of these. Without it the effects are
    individually correct and the video still looks wrong, because a mask that
    wobbles by two pixels per frame reads as a crawling edge.
    """

    def __init__(self, alpha: float = 0.35, reset_distance: float | None = None):
        self.alpha = float(alpha)
        self.reset_distance = reset_distance
        self.value = None

    def update(self, x):
        x = np.asarray(x, np.float32)
        if self.value is None or self.value.shape != x.shape:
            self.value = x.copy()
            return self.value
        if self.reset_distance is not None:
            # A cut, a new subject, or a tracker relock: snap instead of
            # smearing the old state across the change over ten frames.
            if float(np.mean(np.abs(x - self.value))) > self.reset_distance:
                self.value = x.copy()
                return self.value
        self.value = self.value * (1.0 - self.alpha) + x * self.alpha
        return self.value

    def reset(self):
        self.value = None


# ------------------------------------------------------------------- sizing

def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if max_side <= 0 or m <= max_side:
        return img
    s = max_side / float(m)
    return cv2.resize(img, (max(2, int(round(w * s))), max(2, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


def even(n: int) -> int:
    """H.264 wants even dimensions; odd ones fail the encoder, not the resize."""
    n = int(n)
    return n if n % 2 == 0 else n - 1
