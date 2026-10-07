"""
Complexion: moving the skin's own colour, not replacing it.

A complexion is two things at once - how deep it is, and which way it leans:
golden, olive, rosy, neutral. Those are the axes this works on, because they
are the axes the picture actually has. The named looks below are points on
them.

They are NOT nationalities, and this file deliberately has none. There is no
"Indian" or "Brazilian" or "Italian" skin colour: each of those names covers
most of the human range, so a preset carrying one would be one guess frozen
into three numbers - wrong for the great majority of the people it claims to
describe, and a stereotype taught to the software besides. Anyone reaching for
such a label wants a depth and an undertone, and those are here by name.

The transform is the same one the hair colourist uses, for the same reason:
the subject's average skin colour is moved to the target while every pixel
keeps its own departure from that average. Skin is not one colour - a cheek is
not a forehead, and the shadow under a jaw is not either - and a transform that
lands every pixel on one value gives a mask, not a face. Two differences from
hair:

  * both ends are held back, not just the top. Specular highlights are where
    the light is rather than where the pigment is, and the deepest shadows
    carry almost no pigment information at all; recolouring either flattens
    the face into a cut-out.
  * it runs on the whole person, not the face. Shift a face without its neck,
    shoulders and arms and you have given someone a mask - which is the single
    most common way this effect is got wrong.
"""
from __future__ import annotations

import logging

import numpy as np

import cv2

from .imaging import EMA, blend, feather, poly_mask, smoothstep
from .landmarks import (FACE_OVAL, LEFT_BROW, LEFT_EYE, LIPS_OUTER,
                        RIGHT_BROW, RIGHT_EYE)
from .settings import COMPLEXIONS
from .spots import _skin_like

log = logging.getLogger(__name__)



# How far a named complexion is allowed to carry the skin, as a fraction of
# the way to the target. Short of 1.0 on purpose: the last stretch is where a
# face stops looking lit and starts looking painted, and nobody asks for that.
MAX_SHIFT = 0.90


def complexion_mask(bgr: np.ndarray, faces, person, shape) -> np.ndarray | None:
    """
    Every bit of this person's skin, feathered - face, neck, shoulders, arms.

    The face mask comes from the mesh and already has eyes, brows, lips and
    nostrils cut out of it. The body comes from the person mask narrowed to
    the colour of this subject's own face, which is what keeps a shirt or a
    background out of it without assuming anything about the skin tone.
    """
    h, w = shape[:2]
    face_px = np.zeros((h, w), np.float32)
    for f in faces or []:
        face_px = np.maximum(face_px, _face_tone_mask(f, shape))
    out = face_px.copy()

    if person is not None:
        body = np.asarray(person, np.float32)
        if body.shape[:2] != (h, w):
            body = cv2.resize(body, (w, h), interpolation=cv2.INTER_LINEAR)
        skin = _skin_like(bgr, face_px)
        if skin is not None:
            body = (body > 0.5).astype(np.float32) * skin
            # Off the silhouette edge, so a rim of background never gets
            # dragged along with the skin.
            grow = max(2, int(min(h, w) * 0.006))
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (grow * 2 + 1,) * 2)
            body = cv2.erode(body.astype(np.uint8), k).astype(np.float32)
            out = np.maximum(out, feather(body, max(2.0, min(h, w) * 0.006)))
    return out if float(out.max()) > 0.02 else None


def _face_tone_mask(f, shape) -> np.ndarray:
    """A face mask for colour, which is not the same mask as for retouching.

    The retouch mask cuts eyes, brows, lips and nostrils out cleanly, because
    smoothing any of them is how a face melts. Colour wants the opposite kind
    of mask: smooth, and without holes. Use the retouch mask for a complexion
    shift and every one of those exclusions shows up as an un-shifted blotch
    the moment the shift is large - which is exactly what the deep end of the
    range looked like before this.

    So: the whole oval, eyes taken out properly because an iris is not skin,
    brows and lips only damped - they do take some of a complexion with them,
    just less than a cheek does - and the result closed and feathered wide
    enough that no edge of any of it is visible as an edge.
    """
    m = poly_mask(shape, [f.poly(FACE_OVAL)])
    eyes = poly_mask(shape, [f.poly(LEFT_EYE), f.poly(RIGHT_EYE)])
    damp = poly_mask(shape, [f.poly(LEFT_BROW), f.poly(RIGHT_BROW),
                             f.poly(LIPS_OUTER)])
    grow = max(1, int(f.width * 0.01))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1,) * 2)
    m = np.clip(m - cv2.dilate(eyes, k) - 0.55 * damp, 0.0, 1.0)
    # Close before feathering: a nostril or a lip corner left as a hole
    # becomes a visible speck once the colour around it has moved.
    fill = max(2, int(f.width * 0.035))
    kf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill * 2 + 1,) * 2)
    m = np.maximum(m, cv2.morphologyEx(m, cv2.MORPH_CLOSE, kf) * 0.85)
    return feather(m, max(2.0, f.width * 0.035))


class Complexion:
    """Shifts skin depth and undertone, and remembers what they were.

    The measurement that drives the shift is the skin's own average, and
    measuring it independently each frame makes the result breathe: the mask
    moves a few pixels, the average follows, and the applied shift moves the
    other way. So the average is smoothed over time and the transform is built
    from the smoothed one - the same treatment every other per-frame statistic
    in this app gets.
    """

    def __init__(self, stabilise: bool = True):
        self._mean = EMA(0.12 if stabilise else 1.0, reset_distance=14.0)

    @staticmethod
    def target_lab(s) -> np.ndarray | None:
        name = str(getattr(s, "skin_tone", "none")).strip().lower()
        rgb = COMPLEXIONS.get(name)
        if rgb is None:
            return None
        patch = np.float32([[[rgb[2], rgb[1], rgb[0]]]]) / 255.0
        return cv2.cvtColor(patch, cv2.COLOR_BGR2Lab)[0, 0]

    def apply(self, bgr: np.ndarray, mask: np.ndarray, s) -> np.ndarray:
        target = self.target_lab(s)
        amount = float(getattr(s, "skin_tone_amount", 0.0)) * MAX_SHIFT
        depth = float(getattr(s, "skin_depth", 0.0))
        if (target is None or amount <= 0) and abs(depth) < 1e-3:
            return bgr
        sel = mask > 0.25
        if int(sel.sum()) < 256:
            return bgr

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        w = mask[sel]
        mean = np.float32([float(np.average(L[sel], weights=w)),
                           float(np.average(a[sel], weights=w)),
                           float(np.average(b[sel], weights=w))])
        mean = self._mean.update(mean).copy()

        L_new, a_new, b_new = L, a, b
        if target is not None and amount > 0:
            # Depth: scale about the mean rather than offsetting it. Deeper
            # skin genuinely holds a narrower range of lightness than fair
            # skin does, so an offset alone gives a deep tone with fair-skin
            # contrast, which reads as a filter laid over the top.
            gain = float(np.clip((target[0] + 14.0) / max(mean[0] + 14.0, 8.0), 0.45, 1.7))
            shifted = target[0] + (L - mean[0]) * gain
            L_new = L * (1.0 - amount) + shifted * amount

            # Undertone: move the average onto the target, and carry every
            # pixel's own departure from it across - scaled WITH the depth,
            # not held flat. Chroma and lightness do not move independently in
            # skin: carry fair-skin chroma down to a deep lightness and the
            # face comes out grey and ashy, which is exactly what the first
            # version of this did at the bottom of the range.
            cgain = float(np.clip(gain ** -0.45, 0.75, 1.6))
            a_full = target[1] + (a - mean[1]) * cgain
            b_full = target[2] + (b - mean[2]) * cgain
            a_new = a * (1.0 - amount) + a_full * amount
            b_new = b * (1.0 - amount) + b_full * amount

        if abs(depth) > 1e-3:
            # The fine control, usable with a named look or on its own:
            # lighter to the right, deeper to the left, about the same mean.
            lift = 16.0 * depth
            gain = float(np.clip(1.0 + 0.22 * depth, 0.6, 1.5))
            ref = float(np.average(L_new[sel], weights=w)) if target is not None else mean[0]
            L_new = ref + lift + (L_new - ref) * gain
            # Deeper skin is a little richer, lighter skin a little less so -
            # holding chroma flat through a depth change is what makes a
            # lightened face look washed out and a deepened one look muddy.
            k = float(np.clip(1.0 - 0.14 * depth, 0.8, 1.25))
            a_new = mean[1] + (a_new - mean[1]) * k
            b_new = mean[2] + (b_new - mean[2]) * k

        # Hold both ends. A specular highlight is where the light is, not
        # where the pigment is; the deepest shadows carry almost no pigment
        # information at all. Recolour either at full strength and the face
        # flattens into a cut-out.
        hi = smoothstep(mean[0] + 13.0, mean[0] + 32.0, L)
        lo = 1.0 - smoothstep(mean[0] - 30.0, mean[0] - 11.0, L)
        # Chroma is held hard at both ends and lightness only gently. A
        # specular highlight genuinely carries no pigment, so its COLOUR
        # should barely move - but its brightness still has to come down with
        # the rest of the face, or darkening leaves chalky blown patches
        # sitting on a deep complexion, which was the other half of what made
        # the bottom of the range look wrong.
        hold_ab = np.clip(1.0 - 0.60 * hi - 0.45 * lo, 0.25, 1.0)
        hold_L = np.clip(1.0 - 0.22 * hi - 0.18 * lo, 0.60, 1.0)

        out_lab = lab.copy()
        out_lab[:, :, 0] = np.clip(L + (L_new - L) * hold_L, 0.0, 100.0)
        out_lab[:, :, 1] = np.clip(a + (a_new - a) * hold_ab, -127.0, 127.0)
        out_lab[:, :, 2] = np.clip(b + (b_new - b) * hold_ab, -127.0, 127.0)
        shifted_bgr = np.clip(cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR), 0.0, 1.0)
        return blend(bgr, shifted_bgr, mask)
