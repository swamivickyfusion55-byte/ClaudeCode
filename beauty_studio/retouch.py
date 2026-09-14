"""
Skin, eyes, teeth and lips.

The whole stage is built around one idea: frequency separation. Split the face
into a smooth "colour and light" layer and a high-frequency "texture" layer,
clean up the first, and put a controllable amount of the second back. Blur-
based beauty filters skip the second half, which is why they turn faces into
mannequins - and why the `texture` slider here defaults to keeping two thirds
of the real pores.

Everything runs inside the face's bounding box rather than on the full frame.
At 1080p that is a tenth of the pixels, and the masks are all face-relative
anyway.
"""
from __future__ import annotations

import numpy as np

import cv2

from .imaging import (edge_preserving_smooth, gaussian, guided_filter,
                      luminance, skin_likelihood, smoothstep, to_u8, unsharp)
from .landmarks import Face


def _roi(shape, face: Face, pad: float = 0.35) -> tuple[int, int, int, int]:
    """Face bounding box, padded, clamped to the frame."""
    h, w = shape[:2]
    x0, y0, x1, y1 = face.box()
    px = int(face.width * pad)
    py = int(face.height * pad)
    return (max(0, x0 - px), max(0, y0 - py),
            min(w, x1 + px), min(h, y1 + py))


def _shifted(face: Face, ox: int, oy: int) -> Face:
    """Copy of a face with its landmarks moved into ROI coordinates."""
    pts = face.points.copy()
    pts[:, 0] -= ox
    pts[:, 1] -= oy
    return Face(points=pts, width=face.width, height=face.height,
                centre=face.centre - np.float32([ox, oy]), axis=face.axis,
                has_iris=face.has_iris)


class Retoucher:
    """Stateless per frame - all temporal smoothing already happened on the
    landmarks, so the same call is used for photos and for video."""

    def apply(self, bgr: np.ndarray, faces: list[Face], s) -> np.ndarray:
        if not faces or not s.touches_face():
            return bgr
        out = bgr
        for face in faces:
            x0, y0, x1, y1 = _roi(bgr.shape, face)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            sub = out[y0:y1, x0:x1].copy()
            f = _shifted(face, x0, y0)
            sub = self._retouch_face(sub, f, s)
            out = out.copy() if out is bgr else out
            out[y0:y1, x0:x1] = sub
        return out

    # ------------------------------------------------------------------ face
    def _retouch_face(self, sub: np.ndarray, f: Face, s) -> np.ndarray:
        skin = self._skin_mask(sub, f)

        if s.blemish > 0:
            sub = self._blemish(sub, f, skin, s.blemish)
        if s.skin_smooth > 0:
            sub = self._smooth_skin(sub, f, skin, s.skin_smooth, s.texture)
        if s.skin_even > 0:
            sub = self._even_tone(sub, f, skin, s.skin_even)
        if s.under_eye > 0:
            sub = self._under_eye(sub, f, s.under_eye)
        if s.glow > 0:
            sub = self._glow(sub, skin, s.glow)
        if s.eye_brighten > 0:
            sub = self._eyes(sub, f, s.eye_brighten)
        if s.teeth_whiten > 0:
            sub = self._teeth(sub, f, s.teeth_whiten)
        if s.lip_enhance > 0:
            sub = self._lips(sub, f, s.lip_enhance)
        return np.clip(sub, 0.0, 1.0)

    # ----------------------------------------------------------------- masks
    @staticmethod
    def _skin_mask(sub: np.ndarray, f: Face) -> np.ndarray:
        """
        Geometry AND colour.

        The landmark oval says where the face is; the chroma term says which
        pixels inside it are actually skin. Using only the first smooths beard,
        glasses and stray hair; using only the second is unstable frame to
        frame. The chroma term is floored at 0.35 so a strong colour cast or an
        unusual light never erases the mask entirely.
        """
        geo = f.skin_mask(sub.shape)
        chroma = gaussian(skin_likelihood(sub), max(2.0, f.width * 0.02))
        return np.clip(geo * (0.35 + 0.65 * chroma), 0.0, 1.0)

    # ---------------------------------------------------------------- stages
    @staticmethod
    def _blemish(sub: np.ndarray, f: Face, skin: np.ndarray, amount: float) -> np.ndarray:
        """
        Suppress small dark spots by comparing against a median of the skin.

        Only darker-than-median pixels are touched, and only ones smaller than
        the median kernel, so this removes a spot without removing the shadow
        under the nose or the eyelashes the oval mask already excluded.
        """
        k = int(max(3, round(f.width * 0.030)))
        k = k + 1 if k % 2 == 0 else k
        k = min(k, 31)
        med = cv2.medianBlur(to_u8(sub), k).astype(np.float32) / 255.0
        lum, lum_med = luminance(sub), luminance(med)
        # How much darker than its surroundings this pixel is, normalised.
        darkness = np.clip((lum_med - lum) / 0.09, 0.0, 1.0)
        w = skin * darkness * amount
        return sub * (1.0 - w[:, :, None]) + med * w[:, :, None]

    @staticmethod
    def _smooth_skin(sub: np.ndarray, f: Face, skin: np.ndarray,
                     amount: float, texture: float) -> np.ndarray:
        """Frequency separation: smooth the base, restore `texture` of the detail."""
        radius = int(max(3, f.width * 0.045))
        base = edge_preserving_smooth(sub, radius, eps=0.0022)
        detail = sub - base
        # Fine detail is kept in full; the mid-frequency blotchiness is what
        # gets flattened. Splitting the detail layer again is what keeps this
        # from reading as a blur even at high strength.
        fine = detail - gaussian(detail, max(1.2, f.width * 0.008))
        retouched = base + fine * 1.0 + (detail - fine) * texture
        w = skin * amount
        return sub * (1.0 - w[:, :, None]) + retouched * w[:, :, None]

    @staticmethod
    def _even_tone(sub: np.ndarray, f: Face, skin: np.ndarray, amount: float) -> np.ndarray:
        """
        Flatten colour blotches (redness round the nose, uneven patches) while
        leaving luminance alone, so the modelling of the face is untouched.
        """
        lab = cv2.cvtColor(sub, cv2.COLOR_BGR2Lab)
        l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        sigma = max(4.0, f.width * 0.09)
        a_s = guided_filter(l / 100.0, a, int(sigma), eps=0.01, subsample=2)
        b_s = guided_filter(l / 100.0, b, int(sigma), eps=0.01, subsample=2)
        w = skin * amount * 0.85
        lab[:, :, 1] = a * (1 - w) + a_s * w
        lab[:, :, 2] = b * (1 - w) + b_s * w
        return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    @staticmethod
    def _under_eye(sub: np.ndarray, f: Face, amount: float) -> np.ndarray:
        """
        Dark circles: lift the shadow and pull its colour back toward the
        surrounding skin, weighted by how dark the pixel actually is. Applying
        a flat lift to the whole band is what produces the pale rectangles
        under the eyes you see in over-retouched stills.
        """
        mask = f.under_eye_mask(sub.shape)
        if mask.max() <= 0:
            return sub
        lab = cv2.cvtColor(sub, cv2.COLOR_BGR2Lab)
        l = lab[:, :, 0]
        # Reference: the cheek just below, read through a heavy blur.
        ref = guided_filter(l / 100.0, l, int(max(6, f.width * 0.12)), eps=0.02, subsample=2)
        deficit = np.clip((ref - l) / 22.0, 0.0, 1.0)
        w = mask * deficit * amount
        lab[:, :, 0] = l + (ref - l) * w * 0.85
        # Circles are blue/purple; nudge b (yellow-blue) back toward the ref.
        b = lab[:, :, 2]
        b_ref = guided_filter(l / 100.0, b, int(max(6, f.width * 0.12)), eps=0.02, subsample=2)
        lab[:, :, 2] = b + (b_ref - b) * w * 0.7
        return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    @staticmethod
    def _glow(sub: np.ndarray, skin: np.ndarray, amount: float) -> np.ndarray:
        """Soft luminosity - a diffusion layer over skin only, screened in at
        low opacity. This is the 'lit by a softbox' part of the look."""
        soft = gaussian(sub, max(3.0, min(sub.shape[:2]) * 0.03))
        glow = 1.0 - (1.0 - np.clip(sub, 0, 1)) * (1.0 - np.clip(soft, 0, 1) * 0.55)
        w = skin * amount * 0.6
        return sub * (1.0 - w[:, :, None]) + glow * w[:, :, None]

    @staticmethod
    def _eyes(sub: np.ndarray, f: Face, amount: float) -> np.ndarray:
        """Whiten the sclera, add definition to the iris and lashes."""
        mask = f.eye_mask(sub.shape)
        if mask.max() <= 0:
            return sub
        lab = cv2.cvtColor(sub, cv2.COLOR_BGR2Lab)
        l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        # Sclera: bright and nearly neutral. Both conditions matter - bright
        # alone catches the catchlight on the iris and flattens it.
        neutral = 1.0 - smoothstep(6.0, 20.0, np.sqrt(a * a + b * b))
        bright = smoothstep(0.35, 0.72, l / 100.0)
        sclera = mask * neutral * bright
        lab[:, :, 0] = l + sclera * amount * 9.0
        lab[:, :, 1] = a * (1.0 - sclera * amount * 0.55)
        lab[:, :, 2] = b * (1.0 - sclera * amount * 0.55)
        out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        # Definition on the whole eye region, including lashes and iris edge.
        sharp = unsharp(out, sigma=max(0.8, f.width * 0.006), amount=amount * 0.9)
        w = mask * amount
        return out * (1.0 - w[:, :, None]) + sharp * w[:, :, None]

    @staticmethod
    def _teeth(sub: np.ndarray, f: Face, amount: float) -> np.ndarray:
        """Inside the mouth only, and only on the bright pixels - a closed
        mouth has no bright interior, so this is a no-op rather than a
        lightened lip line."""
        mouth = f.lips_mask(sub.shape, inner=True)
        if mouth.max() <= 0:
            return sub
        lab = cv2.cvtColor(sub, cv2.COLOR_BGR2Lab)
        l, b = lab[:, :, 0], lab[:, :, 2]
        teeth = mouth * smoothstep(0.30, 0.55, l / 100.0)
        lab[:, :, 0] = l + teeth * amount * 7.0
        lab[:, :, 2] = b - teeth * amount * 6.0      # pull the yellow out
        return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    @staticmethod
    def _lips(sub: np.ndarray, f: Face, amount: float) -> np.ndarray:
        """Definition and colour depth on the lips, no shape change."""
        lips = f.lips_mask(sub.shape) - f.lips_mask(sub.shape, inner=True) * 0.6
        lips = np.clip(lips, 0.0, 1.0)
        if lips.max() <= 0:
            return sub
        lum = luminance(sub)
        chroma = sub - lum[:, :, None]
        boosted = lum[:, :, None] + chroma * (1.0 + amount * 0.55)
        boosted = unsharp(boosted, sigma=max(0.8, f.width * 0.006), amount=amount * 0.5)
        w = lips * amount
        return sub * (1.0 - w[:, :, None]) + boosted * w[:, :, None]
