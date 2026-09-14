"""
Hair enhancement: definition, gloss, colour depth, flyaway control and a
little volume.

Hair has no landmarks, so the mask is built by elimination: take the person
silhouette, keep the part inside a head-shaped region around the tracked face,
drop anything that reads as skin, and drop the face itself. What is left is
hair (plus, occasionally, a hat or a collar - which is why every operation here
is a texture or tone adjustment that does no harm when it lands on one).

The volume pass is the only geometric one, and it is a masked composite rather
than a warp: it grows the silhouette by re-sampling the hair slightly outward
in a feathered ring outside the existing edge. That way a fuller outline never
drags the background or the shoulders with it.
"""
from __future__ import annotations

import numpy as np

import cv2

from .imaging import (blend, feather, gaussian, luminance, skin_likelihood,
                      smoothstep, unsharp)
from .landmarks import Body, Face
from .reshape import _base_grid


def head_region(shape, f: Face, body: Body | None) -> np.ndarray:
    """A head-and-hair shaped region: rotated ellipse over the skull, plus a
    column below it for long hair, stopped at the shoulders."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    angle = float(np.degrees(np.arctan2(f.axis[0], -f.axis[1])))
    centre = f.centre - f.axis * (f.height * 0.22)
    cv2.ellipse(m, (int(centre[0]), int(centre[1])),
                (int(f.width * 1.25), int(f.height * 1.45)),
                angle, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)

    # Long hair: a band under the skull, cut off at the shoulder line (or a
    # face-height below the chin when there is no pose to ask).
    chin_y = float(f.p(152)[1])
    bottom = body.shoulder_y + abs(body.hip_y - body.shoulder_y) * 0.10 \
        if body is not None else chin_y + f.height * 1.1
    bottom = int(np.clip(bottom, 0, h))
    top = int(np.clip(centre[1], 0, h))
    x0 = int(np.clip(f.centre[0] - f.width * 1.35, 0, w))
    x1 = int(np.clip(f.centre[0] + f.width * 1.35, 0, w))
    if bottom > top and x1 > x0:
        m[top:bottom, x0:x1] = np.maximum(m[top:bottom, x0:x1], 1.0)
    return m


def hair_mask(bgr: np.ndarray, f: Face, person: np.ndarray | None,
              body: Body | None) -> np.ndarray:
    """0..1 hair mask for one head."""
    region = head_region(bgr.shape, f, body)
    if person is not None:
        region = region * person
    skin_geo = f.skin_mask(bgr.shape, feather_px=max(2.0, f.width * 0.05))
    grow = max(2, int(f.width * 0.05))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
    skin_geo = cv2.dilate(skin_geo, k)
    chroma_skin = gaussian(skin_likelihood(bgr), max(2.0, f.width * 0.03))
    m = region * (1.0 - np.clip(skin_geo, 0, 1)) * (1.0 - 0.85 * chroma_skin)
    return feather(np.clip(m, 0.0, 1.0), max(2.0, f.width * 0.035))


def enhance_hair(bgr: np.ndarray, faces: list[Face], person: np.ndarray | None,
                 body: Body | None, s) -> np.ndarray:
    if not faces or not s.touches_hair():
        return bgr
    out = bgr
    for f in faces:
        m = hair_mask(out, f, person, body)
        if float(m.max()) < 0.05:
            continue
        out = _enhance_one(out, f, m, s)
    return np.clip(out, 0.0, 1.0)


def _enhance_one(bgr: np.ndarray, f: Face, m: np.ndarray, s) -> np.ndarray:
    out = bgr
    fw = f.width

    if s.hair_frizz > 0:
        # Flyaways live in the outer ring, where the silhouette meets the
        # background. Smoothing there - and only there - tidies the outline
        # without touching the strand detail inside the hair.
        er = max(1, int(fw * 0.04))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er * 2 + 1, er * 2 + 1))
        core = cv2.erode(m, k)
        ring = feather(np.clip(m - core, 0, 1), max(1.5, fw * 0.02))
        smooth = cv2.bilateralFilter(out, d=0, sigmaColor=0.09,
                                     sigmaSpace=max(3.0, fw * 0.03))
        out = blend(out, smooth, ring * s.hair_frizz * 0.8)

    if s.hair_detail > 0:
        # Strand definition. Small sigma: the structure we want to bring out
        # is a few pixels wide, and a larger radius just adds halo.
        sharp = unsharp(out, sigma=max(0.7, fw * 0.005), amount=s.hair_detail * 1.35)
        out = blend(out, sharp, m * 0.9)

    if s.hair_richness > 0:
        lum = luminance(out)
        chroma = out - lum[:, :, None]
        # More colour, and a slightly deeper shadow, which is what reads as
        # "healthy" rather than "flat".
        richer = lum[:, :, None] + chroma * (1.0 + s.hair_richness * 0.45)
        shade = 1.0 - smoothstep(0.12, 0.55, lum)
        richer = richer * (1.0 - (shade * s.hair_richness * 0.12))[:, :, None]
        out = blend(out, richer, m)

    if s.hair_shine > 0:
        # Gloss follows the light that is already there: take the top of the
        # hair's own luminance range, blur it into soft bands, soft-light it
        # back in. Inventing highlights where the light is not produces the
        # painted-on look.
        lum = luminance(out)
        inside = lum[m > 0.35]
        if inside.size > 32:
            hi = float(np.percentile(inside, 88.0))
            band = smoothstep(hi * 0.92, min(hi * 1.18, 1.0), lum) * m
            band = gaussian(band, max(2.0, fw * 0.02))
            gloss = out + band[:, :, None] * (s.hair_shine * 0.28)
            out = blend(out, gloss, m)

    if s.hair_volume > 0:
        out = _volume(out, f, m, s.hair_volume)

    return out


def _volume(bgr: np.ndarray, f: Face, m: np.ndarray, amount: float) -> np.ndarray:
    """
    Grow the hair silhouette outward by a few pixels.

    The ring just outside the current edge is filled by sampling the image
    along a vector pointing back into the head, so the new pixels are real
    hair, with its real colour and direction, pulled outward - then feathered
    in. It is capped at ~2.5% of the face width, which is a fuller outline
    rather than a wig.
    """
    h, w = bgr.shape[:2]
    grow_px = max(1.0, f.width * 0.025 * amount)
    g = int(round(grow_px))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g * 2 + 1, g * 2 + 1))
    grown = cv2.dilate(m, k)
    ring = np.clip(grown - m, 0.0, 1.0)
    if float(ring.max()) < 0.05:
        return bgr

    centre = f.centre - f.axis * (f.height * 0.35)
    bx, by = _base_grid(w, h)
    vx = bx - float(centre[0])
    vy = by - float(centre[1])
    n = np.sqrt(vx * vx + vy * vy) + 1e-3
    # Sample inward: the destination ring reads from where the hair already is.
    mapx = bx - (vx / n) * (grow_px * 1.15)
    mapy = by - (vy / n) * (grow_px * 1.15)
    pulled = cv2.remap(bgr, mapx.astype(np.float32), mapy.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha = feather(ring, max(1.5, grow_px * 0.9)) * 0.9
    return blend(bgr, pulled, alpha)
