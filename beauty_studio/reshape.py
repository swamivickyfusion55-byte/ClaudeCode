"""
Face and body reshaping.

Everything in this file is one mechanism: a single smooth displacement field
that every requested adjustment writes into, applied to the frame with one
`cv2.remap` at the end. That matters for two reasons.

  * Quality. Warping eight times in a row for eight adjustments resamples the
    picture eight times, and each resample costs sharpness. One field, one
    resample, no compounding softness.
  * Honesty of the result. Because the field is built on a coarse grid and
    upsampled, it is smooth by construction - it can narrow a jaw or a waist,
    but it physically cannot produce the wobbling straight lines (door frames,
    railings, patterned backgrounds) that give away a body-shape edit.

The field is a BACKWARD map: for each destination pixel it stores where to
read from in the source. So pushing the sample point outward makes the subject
narrower, and the background pixels that flow in behind the new silhouette are
real background, stretched slightly, rather than a smeared copy of the subject.
"""
from __future__ import annotations

import numpy as np

import cv2

from .imaging import EMA, smoothstep
from .landmarks import (CHIN, JAW_LEFT, JAW_RIGHT, LEFT_EYE, LEFT_IRIS,
                        NOSE_BRIDGE, NOSE_TIP, NOSE_WING_L, NOSE_WING_R,
                        P_SHOULDER_L, P_SHOULDER_R, RIGHT_EYE, RIGHT_IRIS,
                        Body, Face)

# Cache of base coordinate grids, keyed by (w, h). Rebuilding a 1080p meshgrid
# per frame is pure overhead and it is the same array every time.
_BASE_GRID: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _base_grid(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    key = (w, h)
    g = _BASE_GRID.get(key)
    if g is None:
        if len(_BASE_GRID) > 8:
            _BASE_GRID.clear()
        xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
        ys = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
        g = (np.ascontiguousarray(xs), np.ascontiguousarray(ys))
        _BASE_GRID[key] = g
    return g


class WarpField:
    """
    Accumulating backward displacement field, stored on a coarse grid.

    The grid is ~256 cells on the long edge whatever the frame size. Any warp
    worth applying to a person is far lower frequency than that, so the coarse
    grid costs nothing in fidelity and makes the field both fast to build and
    impossible to make jagged.
    """

    def __init__(self, w: int, h: int, cells: int = 256):
        self.w, self.h = int(w), int(h)
        self.step = max(1.0, max(w, h) / float(cells))
        self.gw = max(2, int(round(w / self.step)))
        self.gh = max(2, int(round(h / self.step)))
        # Grid sample positions in full-resolution pixel coordinates.
        self.gx = (np.arange(self.gw, dtype=np.float32) + 0.5) * (w / self.gw)
        self.gy = (np.arange(self.gh, dtype=np.float32) + 0.5) * (h / self.gh)
        self.X, self.Y = np.meshgrid(self.gx, self.gy)
        self.dx = np.zeros((self.gh, self.gw), np.float32)
        self.dy = np.zeros((self.gh, self.gw), np.float32)
        self.dirty = False

    # ------------------------------------------------------------ primitives
    def add_local_translation(self, c, m, radius: float):
        """
        Move the feature at `c` to `m`, falling off to nothing at `radius`.

        Gustafsson's interactive-image-warping kernel - the same one every
        liquify tool uses. It is C1-continuous at the radius, so several of
        these can overlap along a jawline without leaving ridges between them.
        """
        c = np.asarray(c, np.float32)
        m = np.asarray(m, np.float32)
        d = m - c
        dlen2 = float(d[0] * d[0] + d[1] * d[1])
        if dlen2 < 1e-8 or radius <= 1:
            return
        r2 = float(radius) ** 2
        ddx = self.X - c[0]
        ddy = self.Y - c[1]
        dist2 = ddx * ddx + ddy * ddy
        inside = dist2 < r2
        if not inside.any():
            return
        num = np.where(inside, r2 - dist2, 0.0)
        w = (num / (num + dlen2 + 1e-6)) ** 2
        self.dx -= w * d[0]
        self.dy -= w * d[1]
        self.dirty = True

    def add_radial_zoom(self, c, radius: float, zoom: float):
        """`zoom` > 1 enlarges what is at `c`, < 1 shrinks it."""
        if abs(zoom - 1.0) < 1e-4 or radius <= 1:
            return
        c = np.asarray(c, np.float32)
        ddx = self.X - c[0]
        ddy = self.Y - c[1]
        dist = np.sqrt(ddx * ddx + ddy * ddy)
        t = np.clip(dist / float(radius), 0.0, 1.0)
        w = (1.0 - t * t) ** 2
        scale = 1.0 / (1.0 + (zoom - 1.0) * w) - 1.0
        self.dx += ddx * scale
        self.dy += ddy * scale
        self.dirty = True

    def add_row_squeeze(self, centre_x: np.ndarray, half_width: np.ndarray,
                        amount: np.ndarray, valid: np.ndarray):
        """
        Per-row horizontal squeeze - the body reshaper.

        For each row: displacement grows linearly from the body's centre line
        out to the silhouette, peaks there at `amount * half_width`, then
        decays outside it. Peaking at the silhouette is the point: that is the
        edge the viewer reads as "the shape", and the decay outside means the
        background a few centimetres away is barely touched.

        All three inputs are per-grid-row arrays, already smoothed.
        """
        cx = centre_x[:, None]
        hw = np.maximum(half_width[:, None], 1.0)
        a = amount[:, None]
        v = valid[:, None]
        off = self.X - cx
        u = np.abs(off) / hw
        inner = u
        outer = np.exp(-(((u - 1.0) / 0.32) ** 2))
        f = np.where(u <= 1.0, inner, outer)
        self.dx += np.sign(off) * f * hw * a * v
        self.dirty = True

    # ----------------------------------------------------------------- apply
    def max_shift(self) -> float:
        """Largest displacement in pixels - what the render report shows so a
        user can tell "the effect is subtle" from "the effect did not run"."""
        if not self.dirty:
            return 0.0
        return float(max(np.abs(self.dx).max(), np.abs(self.dy).max()))

    def clamp(self, max_fraction: float = 0.045):
        """Hard ceiling on displacement, as a fraction of the long edge.

        A slider is a request, not a licence: past a few percent the result
        stops being a flattering adjustment and starts being a distortion with
        a bent background to match.
        """
        lim = max(self.w, self.h) * float(max_fraction)
        np.clip(self.dx, -lim, lim, out=self.dx)
        np.clip(self.dy, -lim, lim, out=self.dy)

    def apply(self, img: np.ndarray) -> np.ndarray:
        if not self.dirty:
            return img
        h, w = img.shape[:2]
        self.clamp()
        dx = cv2.resize(self.dx, (w, h), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize(self.dy, (w, h), interpolation=cv2.INTER_LINEAR)
        bx, by = _base_grid(w, h)
        mapx = cv2.add(bx, dx)
        mapy = cv2.add(by, dy)
        return cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------- faces

def _perp_inward(f: Face, p: np.ndarray) -> np.ndarray:
    """Unit vector from point `p` toward the face's own vertical midline.

    Face-relative, not image-relative: a head tilted 30 degrees must slim
    toward its own centre line, or the jaw slides sideways instead of in.
    """
    chin = f.p(CHIN)
    v = p - chin
    along = float(np.dot(v, f.axis))
    perp = v - along * f.axis
    n = float(np.linalg.norm(perp))
    if n < 1e-3:
        return np.float32([0.0, 0.0])
    return (-perp / n).astype(np.float32)


def add_face_reshape(field: WarpField, f: Face, s) -> None:
    """Jaw, chin, nose and eye adjustments, written into the shared field."""
    fw, fh = f.width, f.height

    if s.face_round > 0:
        # Outward where face_slim goes inward: a fuller cheek and a softer
        # jaw. Weaker than the slimming equivalent, because widening a face
        # has to push into the background and shows sooner.
        weights = [0.20, 0.50, 0.85, 1.0, 0.95, 0.70, 0.40]
        amount = -s.face_round * fw * 0.055
        for side in (JAW_LEFT, JAW_RIGHT):
            for idx, wgt in zip(side[2:9], weights):
                p = f.p(idx)
                d = _perp_inward(f, p) * (amount * wgt)
                field.add_local_translation(p, p + d, radius=fw * 0.45)

    if s.face_slim > 0:
        # Weighted along the jaw: most at the cheek/jaw corner, least at the
        # chin (moving the chin sideways is what makes a slimmed face look
        # like it melted) and least at the ear.
        weights = [0.25, 0.55, 0.85, 1.0, 0.95, 0.70, 0.45]
        # 0.065 rather than the 0.055 this shipped with, and well short of
        # the 0.085 tried first: these local translations overlap along the
        # jaw and their displacements add, so the face narrows faster than any
        # single point's move suggests. Past this the jaw starts to read as
        # pinched at the strong presets.
        amount = s.face_slim * fw * 0.065
        for side in (JAW_LEFT, JAW_RIGHT):
            pts = side[2:9]
            for idx, wgt in zip(pts, weights):
                p = f.p(idx)
                d = _perp_inward(f, p) * (amount * wgt)
                field.add_local_translation(p, p + d, radius=fw * 0.42)

    if s.chin_shape > 0:
        chin = f.p(CHIN)
        # Up along the face axis (shorter) plus a slight narrowing of the two
        # points either side of it (tapered rather than blunt).
        field.add_local_translation(chin, chin + f.axis * (fh * 0.045 * s.chin_shape),
                                    radius=fw * 0.32)
        for idx in (JAW_LEFT[1], JAW_RIGHT[1]):
            p = f.p(idx)
            d = _perp_inward(f, p) * (fw * 0.040 * s.chin_shape)
            field.add_local_translation(p, p + d, radius=fw * 0.28)

    if s.nose_slim > 0:
        bridge, tip = f.p(NOSE_BRIDGE), f.p(NOSE_TIP)
        axis = tip - bridge
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 1e-3 else f.axis
        nose_w = float(np.linalg.norm(f.p(NOSE_WING_R) - f.p(NOSE_WING_L)))
        for idx in (NOSE_WING_L, NOSE_WING_R):
            p = f.p(idx)
            v = p - tip
            perp = v - float(np.dot(v, axis)) * axis
            n = float(np.linalg.norm(perp))
            if n < 1e-3:
                continue
            d = (-perp / n) * (nose_w * 0.16 * s.nose_slim)
            field.add_local_translation(p, p + d, radius=max(nose_w * 0.9, 8.0))

    if s.eye_enlarge > 0:
        for ring, iris in ((LEFT_EYE, LEFT_IRIS), (RIGHT_EYE, RIGHT_IRIS)):
            pts = f.poly(ring)
            centre = pts.mean(axis=0)
            if f.has_iris:
                centre = f.poly(iris).mean(axis=0)
            eye_w = float(pts[:, 0].max() - pts[:, 0].min())
            if eye_w < 4:
                continue
            field.add_radial_zoom(centre, radius=eye_w * 1.25,
                                  zoom=1.0 + 0.16 * s.eye_enlarge)


# --------------------------------------------------------------------- bodies

class BodyProfiler:
    """
    Turns a person mask into a per-row centre line and half width, smoothed
    along the body and across time.

    The silhouette, not the skeleton, is what a slimming warp has to follow -
    pose landmarks sit inside the body and say nothing about how wide a coat
    is. The pose is still used, but only to locate the shoulders, waist and
    hips along the vertical axis.
    """

    def __init__(self, stabilise: bool = True):
        a = 0.3 if stabilise else 1.0
        self._centre = EMA(a, reset_distance=30.0)
        self._half = EMA(a, reset_distance=30.0)
        self._valid = EMA(a, reset_distance=0.5)

    WORK_ROWS = 192      # the profile is smooth; it does not need full height

    def profile(self, mask: np.ndarray, rows: np.ndarray, w: int,
                centre_hint: float | None = None):
        """`rows` are full-resolution y positions (the warp grid's rows)."""
        h, mw = mask.shape[:2]
        small = cv2.resize(mask, (max(32, mw * self.WORK_ROWS // max(h, 1)), self.WORK_ROWS),
                           interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        binm = small > 0.5
        counts = binm.sum(axis=1).astype(np.float32)
        xs = np.arange(sw, dtype=np.float32)[None, :]
        weighted = (binm * xs).sum(axis=1)
        centroid = np.where(counts > 0, weighted / np.maximum(counts, 1), sw * 0.5)
        hint = (centre_hint * sw / max(mw, 1)) if centre_hint is not None else None

        centre = np.empty(sh, np.float32)
        half = np.empty(sh, np.float32)
        for i in range(sh):
            anchor = hint if hint is not None else centroid[i]
            c, hw = _run_through(binm[i], anchor)
            centre[i] = c if hw > 0 else centroid[i]
            half[i] = max(hw, 1.0)
        valid = (counts > sw * 0.01).astype(np.float32)

        # Smooth down the body so an arm entering the silhouette does not step
        # the profile, then resample onto the warp grid's rows.
        k = max(3, int(sh * 0.05) | 1)
        centre = cv2.GaussianBlur(centre.reshape(-1, 1), (1, k), 0).ravel() * (mw / sw)
        half = cv2.GaussianBlur(half.reshape(-1, 1), (1, k), 0).ravel() * (mw / sw)
        valid = cv2.GaussianBlur(valid.reshape(-1, 1), (1, k), 0).ravel()

        # Back to full-resolution rows. The profile was measured on a 192-row
        # copy, so the sample positions scale with it.
        yy = np.clip(rows * (sh / max(h, 1)), 0, sh - 1)
        c = np.interp(yy, np.arange(sh), centre).astype(np.float32)
        hw = np.interp(yy, np.arange(sh), half).astype(np.float32)
        v = np.clip(np.interp(yy, np.arange(sh), valid), 0.0, 1.0).astype(np.float32)

        return (self._centre.update(c).copy(),
                self._half.update(hw).copy(),
                self._valid.update(v).copy())


def _run_through(row: np.ndarray, anchor: float) -> tuple[float, float]:
    """Centre and half-width of the mask run containing `anchor`.

    This is the fix for arms. Measuring a row's width by counting its mask
    pixels means an arm held away from the body is added to the torso's width,
    so the warp thinks the body is much wider than it is and puts its peak
    displacement out on the arm - which then bends inward with everything
    else. Taking only the connected run through the body's centre line
    measures the torso and leaves a separated arm out of it entirely, where
    the lateral falloff reduces it to almost nothing.
    """
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return 0.0, 0.0
    # Run boundaries: positions where the mask turns on or off.
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    a = int(np.clip(round(anchor), 0, row.size - 1))
    hit = np.flatnonzero((starts <= a) & (ends >= a))
    if hit.size:
        i = int(hit[0])
    else:
        # The anchor is off the body (an arm-only row, or a gap): fall back to
        # the widest run, which is the torso wherever there is one.
        i = int(np.argmax(ends - starts))
    lo, hi = float(starts[i]), float(ends[i])
    return (lo + hi) * 0.5, (hi - lo) * 0.5


def _torso_length(body: Body, frame_h: int) -> float:
    """
    Shoulders-to-hips distance, sanity-checked against shoulder width.

    The pose model reports hip landmarks even when the hips are outside the
    frame, by extrapolation, and it is confident about them. On a
    head-and-shoulders shot that extrapolation can be wildly long or short,
    and since every band position is a fraction of this number, a bad torso
    length puts the waist somewhere that is not the waist. Human proportions
    are reliable enough to catch that: the torso runs roughly 1.2 to 2.2
    shoulder widths.
    """
    torso = abs(body.hip_y - body.shoulder_y)
    plausible_lo = body.shoulder_w * 1.1
    plausible_hi = body.shoulder_w * 2.4
    if not (plausible_lo <= torso <= plausible_hi):
        torso = body.shoulder_w * 1.6
    return max(torso, frame_h * 0.06)


def _torso_bands(body: Body, torso: float, frame_h: int) -> tuple[float, float, float]:
    """Row positions for the bust, waist and hip bands.

    Derived from the shoulder line rather than read straight off the pose, so
    that a subject framed from the chest up still gets a waist in a sensible
    place instead of one extrapolated below the bottom of the picture.
    """
    shoulder = body.shoulder_y
    bust_y = shoulder + torso * 0.30
    waist_y = shoulder + torso * 0.64
    hip_y = shoulder + torso * 1.0
    # If the pose's own hip estimate is on screen and agrees, prefer it.
    if 0 <= body.hip_y < frame_h and abs(body.hip_y - hip_y) < torso * 0.35:
        hip_y = body.hip_y
        waist_y = shoulder + (hip_y - shoulder) * 0.64
        bust_y = shoulder + (hip_y - shoulder) * 0.30
    return waist_y, bust_y, hip_y


def _band(rows: np.ndarray, centre: float, sigma: float) -> np.ndarray:
    if sigma <= 1:
        return np.zeros_like(rows)
    return np.exp(-(((rows - centre) / sigma) ** 2))


def add_body_reshape(field: WarpField, mask: np.ndarray | None, body: Body | None,
                     profiler: BodyProfiler, s, shape, head_y: float | None = None) -> None:
    """
    Silhouette adjustments: overall slimming, waist, and the hourglass.

    Waist and hourglass need to know where the waist is, which needs the pose.
    Overall slimming does not, so it still runs from the mask alone when the
    subject's hips are out of frame - the common case for a talking-head shot.
    """
    if mask is None or not s.touches_body():
        return
    h, w = shape[:2]
    rows = field.gy
    hint = body.centre_x if body is not None else None
    centre, half, valid = profiler.profile(mask, rows, w, centre_hint=hint)
    if float(valid.max()) <= 0.05:
        return

    # Two separate fields, because they answer to different widths.
    #   silhouette: overall slimming or filling, measured on the whole visible
    #               outline, arms included - they are part of "the body".
    #   bands:      waist, bust and hips, which are TORSO features. Measuring
    #               those against a silhouette that includes the arms puts
    #               their peak displacement out on the arms, which then bulge
    #               or pinch with the torso. That is what made a chest-up shot
    #               look bent.
    amount = np.zeros_like(rows)
    band_amount = np.zeros_like(rows)

    # These coefficients are the fraction of the body's own half-width the
    # silhouette moves at full slider. They were raised substantially in
    # v1.2.0: the first calibration produced changes of two or three pixels on
    # a typical subject, which is arithmetically a few percent and visually
    # nothing at all. A reshaping control that cannot be seen at 100 is not a
    # conservative control, it is a broken one.
    if s.body_slim > 0:
        amount += s.body_slim * 0.16
    if s.body_fuller > 0:
        # The same control pointing the other way: negative amount pushes the
        # silhouette out instead of in.
        amount -= s.body_fuller * 0.15

    if body is not None:
        torso = _torso_length(body, h)
        waist_y, bust_y, hip_y = _torso_bands(body, torso, h)

        # A band whose centre is off the picture is not a band, it is a guess
        # with a tail. On a chest-up shot the estimated waist lands below the
        # frame, and that tail was squeezing the shoulders and arms at the
        # bottom edge - the "everything curves inwards" this produced. If the
        # landmark is not in shot, the control that depends on it sits out.
        def in_frame(y: float) -> bool:
            return -h * 0.02 <= y <= h * 1.02

        waist_ok, bust_ok, hip_ok = in_frame(waist_y), in_frame(bust_y), in_frame(hip_y)

        # Band widths are a fifth to a quarter of the torso, not a third.
        # Wider than this and the bands overlap so heavily that the waist
        # pinch bleeds into the bust band and cancels it - which is exactly
        # what made a "curvy" setting narrow the chest instead of filling it.
        if s.waist_shape > 0 and waist_ok:
            band_amount += _band(rows, waist_y, torso * 0.21) * (s.waist_shape * 0.26)
        if s.curve_shape > 0:
            # Hourglass in one control: in at the waist, out at the bust and
            # hips together. The outward halves are weaker than the inward one
            # - widening reads as a distortion sooner, because it has to
            # invent silhouette where background used to be.
            if waist_ok:
                band_amount += _band(rows, waist_y, torso * 0.20) * (s.curve_shape * 0.20)
            if bust_ok:
                band_amount -= _band(rows, bust_y, torso * 0.20) * (s.curve_shape * 0.11)
            if hip_ok:
                band_amount -= _band(rows, hip_y, torso * 0.26) * (s.curve_shape * 0.13)
        # The same two halves as separate controls, for shaping one without
        # the other - a fuller bust with the hips left alone, or the reverse.
        if s.bust_shape > 0 and bust_ok:
            band_amount -= _band(rows, bust_y, torso * 0.21) * (s.bust_shape * 0.17)
        if s.hip_shape > 0 and hip_ok:
            band_amount -= _band(rows, hip_y, torso * 0.27) * (s.hip_shape * 0.19)
    elif s.waist_shape > 0 or s.curve_shape > 0 or s.bust_shape > 0 or s.hip_shape > 0:
        # No pose: put the waist at the narrowest row in the middle of the
        # visible silhouette rather than guessing from a fixed proportion.
        idx = np.where(valid > 0.5)[0]
        if len(idx) > 8:
            lo, hi = idx[0], idx[-1]
            mid = slice(lo + (hi - lo) // 4, hi - (hi - lo) // 4 + 1)
            seg = half[mid]
            if len(seg) > 2:
                waist_row = rows[mid][int(np.argmin(seg))]
                span = float(rows[hi] - rows[lo])
                band_amount += _band(rows, waist_row, max(span * 0.18, h * 0.05)) * \
                    (max(s.waist_shape, s.curve_shape) * 0.18)
                # No pose means no reliable bust or hip line; widening blind
                # would land the fullness in the wrong place, so those two
                # controls sit this frame out rather than guess.

    # Ceiling on the SUM. Each slider alone is capped in settings.py, but three
    # of them pushed up together would otherwise compound into a caricature -
    # and the frame-relative clamp in WarpField is too coarse to catch it,
    # because a small subject can be wildly distorted while moving far fewer
    # pixels than a frame-relative limit allows.
    np.clip(amount, -0.20, 0.24, out=amount)
    np.clip(band_amount, -0.20, 0.24, out=band_amount)

    # Nothing above the shoulders. The person mask includes the head, so a
    # whole-silhouette squeeze was narrowing the skull and jaw along with the
    # body - which is most of what "everything curves inwards" looks like. The
    # gate ramps in over the neck so there is no step at the collar.
    gate_y = None
    if body is not None:
        gate_y = body.shoulder_y
    elif head_y is not None:
        gate_y = head_y
    if gate_y is not None:
        span = max(float(np.ptp(rows)) * 0.04, 8.0)
        gate = smoothstep(gate_y - span, gate_y + span * 2.0, rows)
        amount = amount * gate
        band_amount = band_amount * gate

    if np.any(amount):
        field.add_row_squeeze(centre, half, amount, valid)
    if np.any(band_amount):
        # Torso width for the bands: the shoulder span from the pose, never
        # wider than the silhouette itself. Arms sit outside it, where the
        # lateral falloff leaves them alone.
        torso_half = half
        if body is not None:
            torso_half = np.minimum(half, max(body.shoulder_w * 0.55, 8.0))
        field.add_row_squeeze(centre, torso_half, band_amount, valid)

    if body is not None and s.posture != 0:
        # Shoulders up (or down) a touch. Small radius, small move: this is
        # posture, not a different body.
        lift = -s.posture * abs(body.hip_y - body.shoulder_y) * 0.035
        for idx in (P_SHOULDER_L, P_SHOULDER_R):
            if body.visible(idx, 0.4):
                p = body.points[idx]
                field.add_local_translation(p, p + np.float32([0.0, lift]),
                                            radius=body.shoulder_w * 0.55)
