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

from .imaging import (EMA, blend, feather, gaussian, luminance,
                      skin_likelihood, smoothstep, unsharp)
from .settings import HAIR_COLOURS
from .landmarks import Body, Face
from .reshape import _base_grid


def head_region(shape, f: Face, body: Body | None) -> np.ndarray:
    """A head-and-hair shaped region: rotated ellipse over the skull, plus the
    columns either side of the face for long hair.

    The band below the face is LATERAL only. Directly under the chin is the
    throat, the collar, and - on this project's own test frame - a space
    helmet's neck ring, none of which is hair. Hair that hangs down does so
    beside the face, not out of it.
    """
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    angle = float(np.degrees(np.arctan2(f.axis[0], -f.axis[1])))
    # + axis, not -: the axis runs chin -> forehead, so adding it moves up
    # the head. Subtracting put this ellipse's centre below the chin, which is
    # how a space helmet's visor came to be treated as hair.
    centre = f.centre + f.axis * (f.height * 0.18)
    cv2.ellipse(m, (int(centre[0]), int(centre[1])),
                (int(f.width * 1.25), int(f.height * 1.30)),
                angle, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)

    chin_y = float(f.p(152)[1])
    bottom = body.shoulder_y + abs(body.hip_y - body.shoulder_y) * 0.10 \
        if body is not None else chin_y + f.height * 1.1
    bottom = int(np.clip(bottom, 0, h))
    top = int(np.clip(centre[1], 0, h))
    if bottom > top:
        inner = f.width * 0.46          # the throat column, kept out
        outer = f.width * 1.35
        cx = float(f.centre[0])
        for x0, x1 in ((cx - outer, cx - inner), (cx + inner, cx + outer)):
            a, b = int(np.clip(x0, 0, w)), int(np.clip(x1, 0, w))
            if b > a:
                m[top:bottom, a:b] = np.maximum(m[top:bottom, a:b], 1.0)
    return m


def skull_core(shape, f: Face) -> np.ndarray:
    """The part of the head that is hair beyond argument: the top of the
    skull. Used as the seed for keeping only what is joined to it."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    angle = float(np.degrees(np.arctan2(f.axis[0], -f.axis[1])))
    centre = f.centre + f.axis * (f.height * 0.62)
    cv2.ellipse(m, (int(centre[0]), int(centre[1])),
                (int(f.width * 0.80), int(f.height * 0.55)),
                angle, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)
    return m


def _grow_from_seed(candidate: np.ndarray, seed: np.ndarray, lab: np.ndarray,
                    tolerance: float) -> np.ndarray:
    """
    Keep the parts of `candidate` that are both joined to `seed` and coloured
    like it.

    Connectivity alone is not enough: the region either side of the face runs
    from the skull down to the shoulders in one piece, so a helmet ring or a
    collar inside that column is joined to the hair by the column itself.
    Colour is what separates them - hair a few centimetres below the ear is
    the same colour as hair on the crown, and a visor is not.
    """
    core = seed > 0.5
    if not core.any():
        return candidate * 0.0

    ref = np.float32([np.median(lab[:, :, i][core]) for i in range(3)])
    d = np.sqrt(((lab[:, :, 0] - ref[0]) * 0.6) ** 2 +
                (lab[:, :, 1] - ref[1]) ** 2 +
                (lab[:, :, 2] - ref[2]) ** 2)
    similar = d < tolerance

    binary = ((candidate > 0.15) & similar).astype(np.uint8)
    binary[core] = 1
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    if count <= 1:
        return candidate * 0.0
    keep_ids = set(np.unique(labels[core]).tolist()) - {0}
    if not keep_ids:
        return candidate * 0.0
    keep = np.isin(labels, list(keep_ids)).astype(np.float32)
    return candidate * keep


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
    not_skin = 1.0 - np.clip(skin_geo, 0, 1)

    # The chroma test used to carry most of the weight here, at 0.85. That is
    # wrong for light hair: blonde sits squarely inside the skin-tone window,
    # so the test suppressed the very thing it was meant to find - on the
    # reference frame it left real hair at 0.13 while a visor stayed at 1.0.
    # The face oval already excludes skin geometrically; chroma is now a light
    # touch, and only near the face, where the neck and ears actually are.
    chroma_skin = gaussian(skin_likelihood(bgr), max(2.0, f.width * 0.03))
    near = cv2.GaussianBlur(f.skin_mask(bgr.shape), (0, 0), max(3.0, f.width * 0.22))
    near = np.clip(near / max(float(near.max()), 1e-3), 0.0, 1.0)
    candidate = np.clip(region * not_skin * (1.0 - 0.45 * chroma_skin * near), 0.0, 1.0)

    # Seed: the crown, which is hair beyond argument, then grow outward
    # through pixels that are the same colour and joined to it.
    seed = skull_core(bgr.shape, f) * not_skin
    if person is not None:
        seed = seed * person
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    m = _grow_from_seed(candidate, seed, lab, tolerance=26.0)
    return feather(np.clip(m, 0.0, 1.0), max(2.0, f.width * 0.03))


class HairColourist:
    """
    Recolours hair, and remembers what colour it was.

    The per-frame measurement that drives the transform is the hair's own
    average colour, and measuring that independently each frame makes the
    result breathe: the mask wobbles by a few pixels, the average moves with
    it, and the applied shift moves the other way. So the average is smoothed
    over time and the transform is computed from the smoothed one.
    """

    def __init__(self, stabilise: bool = True):
        self._mean = EMA(0.15 if stabilise else 1.0, reset_distance=18.0)

    def target_lab(self, s) -> np.ndarray | None:
        """The requested colour, in Lab."""
        name = str(s.hair_colour).strip().lower()
        if name in ("", "none") or s.hair_colour_amount <= 0:
            return None
        rgb = HAIR_COLOURS.get(name)
        if rgb is None and name == "custom":
            # A hue with plausible hair saturation and mid lightness; the
            # transform carries the subject's own lightness range anyway.
            hsv = np.float32([[[float(s.hair_hue) / 2.0, 150, 120]]])
            bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            patch = bgr.astype(np.float32) / 255.0
        elif rgb is None:
            return None
        else:
            patch = np.float32([[[rgb[2], rgb[1], rgb[0]]]]) / 255.0
        return cv2.cvtColor(patch, cv2.COLOR_BGR2Lab)[0, 0]

    def apply(self, bgr: np.ndarray, mask: np.ndarray, s) -> np.ndarray:
        target = self.target_lab(s)
        if target is None:
            return bgr
        sel = mask > 0.3
        if int(sel.sum()) < 64:
            return bgr

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        w = mask[sel]
        mean = np.float32([
            float(np.average(L[sel], weights=w)),
            float(np.average(a[sel], weights=w)),
            float(np.average(b[sel], weights=w)),
        ])
        mean = self._mean.update(mean).copy()
        amount = float(s.hair_colour_amount)

        # Chroma: shift the average to the target and leave every pixel's own
        # departure from it alone. That is what keeps the variation between
        # strands instead of painting the hair one flat colour.
        a_new = a + (target[1] - mean[1]) * amount
        b_new = b + (target[2] - mean[2]) * amount

        # Lightness: scale about the mean rather than offsetting it. Dark hair
        # genuinely has a narrower range of lightness than blonde does, so
        # darkening compresses the range and lightening opens it - an offset
        # alone gives grey hair with blonde contrast, which reads as paint.
        gain = float(np.clip((target[0] + 12.0) / max(mean[0] + 12.0, 8.0), 0.35, 1.8))
        L_new = target[0] + (L - mean[0]) * gain
        L_new = L * (1.0 - amount) + L_new * amount

        # Specular highlights are where the light is, not where the pigment
        # is: hold them back so wet-looking gloss survives a recolour instead
        # of flattening into the new shade.
        gloss = smoothstep(mean[0] + 14.0, mean[0] + 34.0, L)
        hold = (1.0 - 0.55 * gloss)[:, :, None]

        out_lab = lab.copy()
        out_lab[:, :, 0] = np.clip(L + (L_new - L) * hold[:, :, 0], 0.0, 100.0)
        out_lab[:, :, 1] = np.clip(a + (a_new - a) * hold[:, :, 0], -127.0, 127.0)
        out_lab[:, :, 2] = np.clip(b + (b_new - b) * hold[:, :, 0], -127.0, 127.0)
        recoloured = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)
        return blend(bgr, np.clip(recoloured, 0.0, 1.0), mask)


def enhance_hair(bgr: np.ndarray, faces: list[Face], person: np.ndarray | None,
                 body: Body | None, s, colourist: "HairColourist | None" = None) -> np.ndarray:
    if not faces or not s.touches_hair():
        return bgr
    out = bgr
    for f in faces:
        m = hair_mask(out, f, person, body)
        if float(m.max()) < 0.05:
            continue
        if colourist is not None:
            # Colour first: the detail, gloss and volume passes below should
            # work on the hair as it will look, not as it arrived.
            out = colourist.apply(out, m, s)
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
