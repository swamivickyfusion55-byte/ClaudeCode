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

from .imaging import EMA
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
        outer = np.exp(-(((u - 1.0) / 0.45) ** 2))
        f = np.where(u <= 1.0, inner, outer)
        self.dx += np.sign(off) * f * hw * a * v
        self.dirty = True

    # ----------------------------------------------------------------- apply
    def clamp(self, max_fraction: float = 0.06):
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

    if s.face_slim > 0:
        # Weighted along the jaw: most at the cheek/jaw corner, least at the
        # chin (moving the chin sideways is what makes a slimmed face look
        # like it melted) and least at the ear.
        weights = [0.25, 0.55, 0.85, 1.0, 0.95, 0.70, 0.45]
        amount = s.face_slim * fw * 0.055
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
        field.add_local_translation(chin, chin + f.axis * (fh * 0.035 * s.chin_shape),
                                    radius=fw * 0.32)
        for idx in (JAW_LEFT[1], JAW_RIGHT[1]):
            p = f.p(idx)
            d = _perp_inward(f, p) * (fw * 0.030 * s.chin_shape)
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

    def profile(self, mask: np.ndarray, rows: np.ndarray, w: int):
        """`rows` are full-resolution y positions (the warp grid's rows)."""
        h, mw = mask.shape[:2]
        binm = (mask > 0.5)
        counts = binm.sum(axis=1).astype(np.float32)
        xs = np.arange(mw, dtype=np.float32)[None, :]
        weighted = (binm * xs).sum(axis=1)
        centre = np.where(counts > 0, weighted / np.maximum(counts, 1), mw * 0.5)
        # Half width from the pixel count rather than the extreme columns: an
        # outstretched arm should not decide how wide the torso is.
        half = np.maximum(counts * 0.5, 1.0)
        valid = (counts > mw * 0.01).astype(np.float32)

        # Smooth down the body so an arm entering the silhouette does not step
        # the profile, then resample onto the warp grid's rows.
        k = max(3, int(h * 0.05) | 1)
        centre = cv2.GaussianBlur(centre.reshape(-1, 1), (1, k), 0).ravel()
        half = cv2.GaussianBlur(half.reshape(-1, 1), (1, k), 0).ravel()
        valid = cv2.GaussianBlur(valid.reshape(-1, 1), (1, k), 0).ravel()

        yy = np.clip(rows, 0, h - 1)
        c = np.interp(yy, np.arange(h), centre).astype(np.float32)
        hw = np.interp(yy, np.arange(h), half).astype(np.float32)
        v = np.clip(np.interp(yy, np.arange(h), valid), 0.0, 1.0).astype(np.float32)

        return (self._centre.update(c).copy(),
                self._half.update(hw).copy(),
                self._valid.update(v).copy())


def _band(rows: np.ndarray, centre: float, sigma: float) -> np.ndarray:
    if sigma <= 1:
        return np.zeros_like(rows)
    return np.exp(-(((rows - centre) / sigma) ** 2))


def add_body_reshape(field: WarpField, mask: np.ndarray | None, body: Body | None,
                     profiler: BodyProfiler, s, shape) -> None:
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
    centre, half, valid = profiler.profile(mask, rows, w)
    if float(valid.max()) <= 0.05:
        return

    amount = np.zeros_like(rows)

    if s.body_slim > 0:
        # A flat 0..4.5% narrowing wherever there is a subject.
        amount += s.body_slim * 0.045

    if body is not None:
        torso = max(abs(body.hip_y - body.shoulder_y), h * 0.08)
        if s.waist_shape > 0:
            amount += _band(rows, body.waist_y, torso * 0.34) * (s.waist_shape * 0.075)
        if s.curve_shape > 0:
            # Hourglass: in at the waist, out at the bust and hips. The
            # outward bands are deliberately weaker than the inward one -
            # widening reads as a distortion far sooner than narrowing does.
            bust_y = body.shoulder_y + torso * 0.38
            amount += _band(rows, body.waist_y, torso * 0.30) * (s.curve_shape * 0.055)
            amount -= _band(rows, bust_y, torso * 0.26) * (s.curve_shape * 0.032)
            amount -= _band(rows, body.hip_y, torso * 0.34) * (s.curve_shape * 0.038)
    elif s.waist_shape > 0 or s.curve_shape > 0:
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
                amount += _band(rows, waist_row, max(span * 0.18, h * 0.05)) * \
                    (max(s.waist_shape, s.curve_shape) * 0.055)

    field.add_row_squeeze(centre, half, amount, valid)

    if body is not None and s.posture != 0:
        # Shoulders up (or down) a touch. Small radius, small move: this is
        # posture, not a different body.
        lift = -s.posture * abs(body.hip_y - body.shoulder_y) * 0.035
        for idx in (P_SHOULDER_L, P_SHOULDER_R):
            if body.visible(idx, 0.4):
                p = body.points[idx]
                field.add_local_translation(p, p + np.float32([0.0, lift]),
                                            radius=body.shoulder_w * 0.55)
