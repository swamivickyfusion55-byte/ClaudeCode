"""
The HDR grade.

"HDR look" in a phone app usually means a crushed, halo-ringed, over-saturated
image. What people actually want when they say it is the thing an HDR display
gives you: open shadows, highlights that keep their detail, and local contrast
that makes the picture feel three-dimensional - without the whole frame
turning into a poster.

So the core here is a real local tone-mapping operator (Durand-style
base/detail separation in the log domain, with a guided filter as the
edge-preserving base) rather than a contrast curve pretending to be one. That
is what recovers a blown window and an underlit face in the same shot. Curves,
vibrance and bloom sit on top as grading, not as the mechanism.

Every auto-measured quantity is smoothed over time. An exposure statistic
recomputed independently per frame is how you get a video that breathes.
"""
from __future__ import annotations

import numpy as np

import cv2

from .imaging import (EMA, gaussian, guided_filter, luminance,
                      skin_likelihood, smoothstep, unsharp)


class Grader:
    """Stateful so that the auto-exposure and tone-map anchors can be smoothed
    across frames. Construct one per render, call `apply` per frame."""

    def __init__(self, stabilise: bool = True):
        a = 0.12 if stabilise else 1.0
        self._lo = EMA(a, reset_distance=0.10)
        self._hi = EMA(a, reset_distance=0.10)
        self._anchor = EMA(a, reset_distance=0.10)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _texture_weight(lum: np.ndarray) -> np.ndarray:
        """
        Where is there real detail to enhance?

        Local contrast and sharpening only pay off on actual structure. Run
        them flat across a frame and the places with nothing in them - a sky,
        a wall, a studio backdrop - hand back sensor noise and JPEG blocking
        instead, which is the single clearest tell of an "HDR" filter. This
        weight is near zero in those areas and near one on texture, and it
        gates every detail-boosting stage below.
        """
        h, w = lum.shape[:2]
        small = cv2.resize(lum, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
        energy = np.abs(small - gaussian(small, 2.0))
        energy = gaussian(energy, 3.0)
        weight = smoothstep(0.004, 0.030, energy)
        return cv2.resize(weight, (w, h), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------ main
    def apply(self, bgr: np.ndarray, s) -> np.ndarray:
        out = bgr
        if s.warmth or s.tint:
            out = self._white_balance(out, s.warmth, s.tint)
        if s.hdr_strength > 0:
            out = self._tone_map(out, s)
        if s.shadows or s.highlights:
            out = self._shadow_highlight(out, s.shadows, s.highlights)
        if s.clarity > 0:
            out = self._clarity(out, s.clarity)
        if s.contrast:
            out = self._contrast(out, s.contrast)
        if s.vibrance > 0 or s.saturation:
            out = self._colour(out, s.vibrance, s.saturation, s.protect_skin_colour)
        if s.bloom > 0:
            out = self._bloom(out, s.bloom)
        if s.sharpen > 0:
            out = self._sharpen(out, s.sharpen)
        return np.clip(out, 0.0, 1.0)

    # -------------------------------------------------------------- stages
    @staticmethod
    def _white_balance(bgr: np.ndarray, warmth: float, tint: float) -> np.ndarray:
        """Channel gains, normalised so the overall exposure does not drift."""
        r_gain = 1.0 + 0.16 * warmth
        b_gain = 1.0 - 0.16 * warmth
        g_gain = 1.0 - 0.10 * tint
        gains = np.float32([b_gain, g_gain, r_gain])
        gains = gains / float(np.dot(gains, [0.0722, 0.7152, 0.2126]))
        return np.clip(bgr * gains, 0.0, 1.0)

    def _tone_map(self, bgr: np.ndarray, s) -> np.ndarray:
        """
        Log-domain base/detail tone mapping.

        base    = edge-preserving (guided) filter of log luma -> the lighting
        detail  = log luma - base                             -> the texture

        Compressing `base` and leaving `detail` alone is what separates this
        from a contrast curve: the dynamic range between the lit and shadowed
        parts of the frame shrinks, while the local texture inside each region
        keeps (or slightly gains) its punch.
        """
        strength = float(s.hdr_strength)
        h, w = bgr.shape[:2]
        lum = np.maximum(luminance(bgr), 1e-4)
        log_l = np.log(lum)

        # Radius ~4% of the long edge: large enough to read as "the lighting",
        # small enough not to cost the local contrast we are here to gain.
        radius = max(4, int(round(max(h, w) * 0.04)))
        base = guided_filter(log_l, log_l, radius, eps=0.012, subsample=4)
        detail = log_l - base

        # Percentiles come off a thumbnail. They are scene statistics, not
        # pixel values - a 256-wide sample lands within a hundredth of the
        # full-resolution answer for a fraction of the cost.
        thumb = cv2.resize(base, (256, max(2, int(256 * h / max(w, 1)))),
                           interpolation=cv2.INTER_AREA)
        lo = float(self._lo.update(np.percentile(thumb, 2.0)))
        hi = float(self._hi.update(np.percentile(thumb, 98.0)))
        anchor = float(self._anchor.update(np.percentile(thumb, 60.0)))
        span = max(hi - lo, 0.25)

        # Compression factor: 1.0 keeps the range, 0.45 halves it in stops.
        # Scaled by how wide the scene's range actually is, so an already-flat
        # shot is not flattened further.
        range_factor = float(np.clip(span / 2.2, 0.35, 1.0))
        compress = 1.0 - 0.55 * strength * range_factor
        detail_gain = 1.0 + 0.30 * strength

        new_base = anchor + (base - anchor) * compress
        # The detail layer is boosted only where there is detail; in flat
        # regions the extra gain would land on compression artefacts.
        gain_map = 1.0 + (detail_gain - 1.0) * self._texture_weight(lum)
        new_log = new_base + detail * gain_map
        # Put the mid-tone back where it was: tone mapping should change the
        # relationship between light and shade, not the overall exposure.
        new_lum = np.exp(new_log)
        ratio = new_lum / lum

        # Cap the gain so a near-black pixel cannot be multiplied into noise.
        ratio = np.clip(ratio, 0.35, 3.0)
        out = bgr * ratio[:, :, None]

        # Highlights that go over 1.0 after the lift are rolled off rather than
        # clipped, which is what keeps skies and windows from turning to paper.
        out = self._roll_off(out)
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _roll_off(bgr: np.ndarray, knee: float = 0.82) -> np.ndarray:
        """Soft shoulder above `knee` instead of a hard clip at 1.0."""
        x = np.maximum(bgr, 0.0)
        over = x > knee
        if not np.any(over):
            return x
        # Clamped at 0 because np.where evaluates both branches: below the
        # knee t goes negative and passes through -1, which is a divide by
        # zero in an expression whose result is then thrown away.
        t = np.maximum((x - knee) / max(1.0 - knee, 1e-3), 0.0)
        # Reinhard-style shoulder on the part above the knee only.
        shoulder = knee + (1.0 - knee) * (t / (1.0 + t))
        return np.where(over, shoulder, x)

    @staticmethod
    def _shadow_highlight(bgr: np.ndarray, shadows: float, highlights: float) -> np.ndarray:
        lum = np.maximum(luminance(bgr), 1e-4)
        out = bgr
        if shadows:
            # Weight peaks in the shadows and is gone by the mid-tones.
            w = (1.0 - smoothstep(0.02, 0.55, lum)) ** 1.5
            lift = 1.0 + 0.85 * shadows * w
            out = out * lift[:, :, None]
        if highlights:
            w = smoothstep(0.55, 1.0, lum)
            gain = 1.0 + 0.55 * highlights * w
            out = out * gain[:, :, None]
        return np.clip(out, 0.0, 1.2)

    @staticmethod
    def _clarity(bgr: np.ndarray, amount: float) -> np.ndarray:
        """
        Mid-frequency local contrast on luma only.

        Luma only because clarity applied per channel shifts hue in every
        transition; and the effect is damped at both ends of the tone scale,
        which is the halo that gives cheap HDR away.
        """
        lum = np.maximum(luminance(bgr), 1e-4)
        h, w = bgr.shape[:2]
        sigma = max(3.0, max(h, w) * 0.012)
        soft = guided_filter(lum, lum, int(sigma), eps=0.02, subsample=4)
        detail = lum - soft
        protect = smoothstep(0.03, 0.18, lum) * (1.0 - smoothstep(0.86, 0.99, lum))
        protect = protect * Grader._texture_weight(lum)
        new_lum = lum + detail * (amount * 1.3) * protect
        ratio = np.clip(np.maximum(new_lum, 1e-4) / lum, 0.5, 2.0)
        return bgr * ratio[:, :, None]

    @staticmethod
    def _contrast(bgr: np.ndarray, amount: float) -> np.ndarray:
        """Gentle S-curve around 0.5, expressed as a smooth pivot so there is
        no visible kink where the two halves meet."""
        x = np.clip(bgr, 0.0, 1.0)
        k = 0.55 * amount
        return np.clip(x + k * (x - 0.5) * (1.0 - np.abs(x - 0.5) * 2.0) * 1.6 +
                       k * 0.35 * (x - 0.5), 0.0, 1.0)

    @staticmethod
    def _colour(bgr: np.ndarray, vibrance: float, saturation: float,
                protect_skin: bool) -> np.ndarray:
        """
        Vibrance first (weighted toward already-dull colours), then a flat
        saturation trim. Skin is held back from the vibrance pass, because the
        single fastest way to make a graded face look artificial is to let a
        global saturation boost run over it.
        """
        lum = luminance(bgr)
        chroma = bgr - lum[:, :, None]
        sat = np.clip(np.max(np.abs(chroma), axis=2) * 2.0, 0.0, 1.0)

        gain = np.ones_like(sat)
        if vibrance:
            w = (1.0 - sat) ** 1.6          # dull pixels gain most, vivid ones least
            gain = gain + vibrance * 0.9 * w
        if saturation:
            gain = gain * (1.0 + 0.8 * saturation)

        if protect_skin and (vibrance or saturation > 0):
            # Half resolution: the skin term is a soft weight, and computing
            # it on every pixel of a 4K frame buys nothing visible.
            h, w = bgr.shape[:2]
            half = cv2.resize(bgr, (max(2, w // 2), max(2, h // 2)), interpolation=cv2.INTER_AREA)
            skin = gaussian(skin_likelihood(half), 2.0)
            skin = cv2.resize(skin, (w, h), interpolation=cv2.INTER_LINEAR)
            gain = gain * (1.0 - 0.65 * skin) + 1.0 * (0.65 * skin)

        out = lum[:, :, None] + chroma * gain[:, :, None]
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _bloom(bgr: np.ndarray, amount: float) -> np.ndarray:
        """Highlight glow - the optical part of the HDR look, not a curve."""
        h, w = bgr.shape[:2]
        # Bloom is a wide, soft, low-frequency layer, so it is built at
        # quarter resolution: same picture, a sixteenth of the blur.
        small = cv2.resize(bgr, (max(2, w // 4), max(2, h // 4)), interpolation=cv2.INTER_AREA)
        mask = smoothstep(0.72, 1.0, luminance(small))
        glow = gaussian(small * mask[:, :, None], max(2.0, max(h, w) * 0.015 / 4.0))
        glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(bgr + glow * (amount * 0.55), 0.0, 1.0)

    @staticmethod
    def _sharpen(bgr: np.ndarray, amount: float) -> np.ndarray:
        """Output sharpening on luma, damped in flat areas so noise in a sky
        or on a wall is not amplified along with the edges we want."""
        lum = np.maximum(luminance(bgr), 1e-4)
        sharp = unsharp(lum, sigma=1.1, amount=amount * 1.1)
        edge = cv2.Laplacian(lum, cv2.CV_32F, ksize=3)
        edge_w = smoothstep(0.008, 0.06, np.abs(edge)) * Grader._texture_weight(lum)
        new_lum = lum + (sharp - lum) * edge_w
        ratio = np.clip(np.maximum(new_lum, 1e-4) / lum, 0.6, 1.8)
        return np.clip(bgr * ratio[:, :, None], 0.0, 1.0)


def grade_once(bgr: np.ndarray, s) -> np.ndarray:
    """Single-image convenience wrapper (no temporal state)."""
    return Grader(stabilise=False).apply(bgr, s)
