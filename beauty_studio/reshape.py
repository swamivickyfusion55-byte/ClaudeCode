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
                        P_ANKLE_L, P_ANKLE_R, P_HIP_L, P_HIP_R, P_KNEE_L,
                        P_KNEE_R, P_SHOULDER_L, P_SHOULDER_R, RIGHT_EYE,
                        RIGHT_IRIS, Body, Face)

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

    def add_axis_scale(self, anchor, axis, band_along: float, band_sigma: float,
                       k: float, half_left: float, half_right: float):
        """
        Widen (k > 0) or narrow (k < 0) about a feature's own axis, in a band
        along it, with each side scaled against its OWN half-width.

        This is the right primitive for a fuller or thinner FACE, as opposed to
        a moved jaw line. Dragging the outline outward with local translations
        leaves the nose and mouth where they were, and the picture stretches
        between them - the smeared look. A scale about the axis carries the
        features with the width, which is what a fuller face actually is.

        The per-side normalisation is what keeps a turned head from bulging.
        Displacement proportional to raw distance from the midline means that
        on a yawed face - where one cheek is much further from the midline
        than the other in the picture, because the near one is foreshortened -
        the far cheek gets a far bigger push, and it balloons for exactly the
        few frames where the subject turns. Measured against each side's own
        half-width instead, both cheeks move by the same fraction of the face
        they belong to, and the displacement is bounded at the face's edge
        rather than growing until the lateral falloff cuts it off.
        """
        if abs(k) < 1e-4 or min(half_left, half_right) <= 1:
            return
        anchor = np.asarray(anchor, np.float32)
        axis = np.asarray(axis, np.float32)
        normal = np.float32([axis[1], -axis[0]])        # perpendicular, unit
        vx = self.X - anchor[0]
        vy = self.Y - anchor[1]
        along = vx * axis[0] + vy * axis[1]
        signed = vx * normal[0] + vy * normal[1]

        side_half = np.where(signed >= 0, float(half_right), float(half_left))
        u = np.abs(signed) / side_half
        # Linear to the face's edge, then decaying: the cheek moves by
        # k * its own half-width at most, whichever side it is on.
        shape = np.where(u <= 1.0, u, np.exp(-(((u - 1.0) / 0.35) ** 2)))
        band = np.exp(-(((along - band_along) / max(band_sigma, 1.0)) ** 2))
        mag = k * side_half * shape * band * np.sign(signed)
        self.dx -= normal[0] * mag
        self.dy -= normal[1] * mag
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
    def max_shift(self, box: tuple[int, int, int, int] | None = None,
                  outside: bool = False) -> float:
        """Largest displacement in pixels, optionally restricted to a region.

        The report shows this so a user can tell "the effect is subtle" from
        "the effect did not run" - and, with `box`, can tell which of the two
        happened to the face and to the body separately, since those have
        different causes and different fixes.
        """
        if not self.dirty:
            return 0.0
        mag = np.maximum(np.abs(self.dx), np.abs(self.dy))
        if box is None:
            return float(mag.max())
        x0, y0, x1, y1 = box
        gx0 = int(np.clip(x0 / self.w * self.gw, 0, self.gw - 1))
        gx1 = int(np.clip(x1 / self.w * self.gw, 1, self.gw))
        gy0 = int(np.clip(y0 / self.h * self.gh, 0, self.gh - 1))
        gy1 = int(np.clip(y1 / self.h * self.gh, 1, self.gh))
        if gx1 <= gx0 or gy1 <= gy0:
            return 0.0
        if not outside:
            return float(mag[gy0:gy1, gx0:gx1].max())
        masked = mag.copy()
        masked[gy0:gy1, gx0:gx1] = 0.0
        return float(masked.max())

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

def _side_half_widths(f: Face) -> tuple[float, float]:
    """Perpendicular distance from the face's midline to each cheek edge.

    On a frontal face these are near enough equal; on a turned one they are
    not, and that difference is the whole reason a face-widening warp needs
    to know about them.
    """
    normal = np.float32([f.axis[1], -f.axis[0]])
    chin = f.p(CHIN)
    d = [float(np.dot(f.p(idx) - chin, normal)) for idx in (234, 454)]
    left = max(-min(d), 1.0)
    right = max(max(d), 1.0)
    return left, right


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
        # Each side's own half-width, measured perpendicular to the face axis,
        # so a head turned away from the camera is scaled by what is actually
        # visible of each cheek.
        hl, hr = _side_half_widths(f)
        # And on a strongly turned head, less of it: a 2D widening of a face
        # seen at an angle has no good answer, so the honest thing is to back
        # off rather than invent one.
        asym = max(hl, hr) / max(min(hl, hr), 1e-3)
        yaw_damp = 1.0 / (1.0 + 0.9 * max(asym - 1.35, 0.0))
        # A fuller face is not just the slimming warp reversed. Slimming only
        # has to move the jaw line in; fullness reads from three cues at once,
        # and with only the first it is invisible at any strength that still
        # looks like a face:
        #   - the cheeks and jaw carry outward, widest at the cheekbone,
        #   - the widest points of the face (the ears' line) go with them,
        #   - the chin drops a little, which is what softens the jaw instead
        #     of just making a wide, hard one.
        chin = f.p(CHIN)
        # One scale about the face's own axis, centred on the cheek/jaw third
        # and fading out by the brow. Everything inside widens together, so a
        # fuller face gets a fuller nose and mouth as well - which is what
        # makes it read as a face rather than as a stretched picture.
        field.add_axis_scale(chin, f.axis,
                             band_along=fh * 0.42, band_sigma=fh * 0.38,
                             k=s.face_round * 0.13 * yaw_damp,
                             half_left=hl, half_right=hr)
        # A little more at the jaw itself, and a chin that drops slightly:
        # together they soften the jaw line instead of widening a hard one.
        field.add_axis_scale(chin, f.axis,
                             band_along=fh * 0.12, band_sigma=fh * 0.20,
                             k=s.face_round * 0.07 * yaw_damp,
                             half_left=hl * 0.85, half_right=hr * 0.85)
        field.add_local_translation(chin, chin - f.axis * (fh * 0.022 * s.face_round),
                                    radius=fw * 0.38)

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
                centre_line: np.ndarray | None = None):
        """
        Per-row centre and half-width for the warp.

        `centre_line` is the body's axis in full-resolution x, sampled at
        `rows` - from the pose where there is one. It matters most where the
        silhouette splits in two: below the knees the axis runs down the gap
        BETWEEN the legs, so no run contains it, and a profiler that picks
        "the widest run" instead latches onto one leg. The warp then squeezes
        that leg about its own centre and leaves the other where it was, which
        is the disjointed-legs artefact - and when the wider leg changes from
        frame to frame, the centre jumps and the legs shear apart for those
        frames.

        So the axis is never inferred from a run. Where the row is split, the
        half-width is the distance from the axis out to the outermost edge of
        the runs near it, and both legs move symmetrically about the body's
        real centre line.
        """
        h, mw = mask.shape[:2]
        small = cv2.resize(mask, (max(32, mw * self.WORK_ROWS // max(h, 1)), self.WORK_ROWS),
                           interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        scale = sw / float(mw)
        binm = small > 0.5
        counts = binm.sum(axis=1).astype(np.float32)
        xs = np.arange(sw, dtype=np.float32)[None, :]
        weighted = (binm * xs).sum(axis=1)
        centroid = np.where(counts > 0, weighted / np.maximum(counts, 1), sw * 0.5)

        if centre_line is not None and len(centre_line):
            anchors = np.interp(np.arange(sh),
                                np.linspace(0, sh - 1, num=len(centre_line)),
                                np.asarray(centre_line, np.float32) * scale)
        else:
            anchors = centroid.copy()

        centre = np.empty(sh, np.float32)
        half = np.empty(sh, np.float32)
        reach = max(sw * 0.08, 4.0)      # grows to the body's own width below
        for i in range(sh):
            a = float(anchors[i])
            c, hw = _row_extent(binm[i], a, reach)
            if hw <= 0:
                centre[i], half[i] = a, 1.0
                continue
            centre[i], half[i] = c, hw
            # Carry the body's width downward as the limit on what counts as
            # part of it, so an outstretched arm cannot inflate the torso and
            # a far-off object cannot join the legs.
            reach = max(reach * 0.7 + hw * 2.0 * 0.3, 4.0)
        valid = (counts > sw * 0.01).astype(np.float32)

        # Smooth down the body so an arm entering the silhouette does not step
        # the profile, then resample onto the warp grid's rows.
        k = max(3, int(sh * 0.05) | 1)
        centre = cv2.GaussianBlur(centre.reshape(-1, 1), (1, k), 0).ravel() / scale
        half = cv2.GaussianBlur(half.reshape(-1, 1), (1, k), 0).ravel() / scale
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


def _row_extent(row: np.ndarray, anchor: float, reach: float) -> tuple[float, float]:
    """
    Centre and half-width of the body in one row, measured about `anchor`.

    Runs further than `reach` from the anchor are not this body - that is what
    keeps an outstretched arm from being counted as torso width. Everything
    nearer is, including the second run when the legs are apart, so a split row
    reports one width about one centre and the two legs move together.
    """
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return 0.0, 0.0
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1])).astype(np.float32)
    ends = np.concatenate((idx[breaks], [idx[-1]])).astype(np.float32)

    # Distance from the anchor to each run (zero when the anchor is inside it).
    gap = np.maximum(np.maximum(starts - anchor, anchor - ends), 0.0)
    keep = gap <= reach
    if not keep.any():
        # Nothing close: fall back to the run nearest the anchor, so a frame
        # where the tracker drifts does not silently disable the row.
        keep = gap == gap.min()
    lo = float(starts[keep].min())
    hi = float(ends[keep].max())
    # The centre stays on the anchor whenever the anchor is inside the body's
    # span; only a row entirely to one side re-centres, and then only to the
    # near edge of what it found.
    centre = anchor if lo <= anchor <= hi else (lo + hi) * 0.5
    half = max(abs(hi - centre), abs(centre - lo), 1.0)
    return float(centre), float(half)


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


def _pose_centre_line(body: Body, rows: np.ndarray) -> np.ndarray | None:
    """The body's axis, x per row, from the pose.

    Built from the shoulder, hip, knee and ankle midpoints and interpolated
    between them, so it follows a body that leans or steps sideways - and, for
    the rows where the legs are apart and the silhouette has no middle, it is
    the only thing that knows where the middle is.
    """
    pts = body.points
    vis = body.visibility
    anchors: list[tuple[float, float]] = []

    def pair(a: int, b: int) -> tuple[float, float] | None:
        ok_a, ok_b = vis[a] >= 0.3, vis[b] >= 0.3
        if ok_a and ok_b:
            return ((pts[a][1] + pts[b][1]) * 0.5, (pts[a][0] + pts[b][0]) * 0.5)
        if ok_a:
            return (float(pts[a][1]), float(pts[a][0]))
        if ok_b:
            return (float(pts[b][1]), float(pts[b][0]))
        return None

    for a, b in ((P_SHOULDER_L, P_SHOULDER_R), (P_HIP_L, P_HIP_R),
                 (P_KNEE_L, P_KNEE_R), (P_ANKLE_L, P_ANKLE_R)):
        got = pair(a, b)
        if got is not None:
            anchors.append(got)
    if len(anchors) < 2:
        return None
    anchors.sort(key=lambda t: t[0])
    ys = np.array([a[0] for a in anchors], np.float32)
    xs = np.array([a[1] for a in anchors], np.float32)
    # Flat extrapolation past the ends: above the shoulders and below the feet
    # the axis simply continues, rather than shooting off at the last slope.
    return np.interp(rows, ys, xs).astype(np.float32)


def _leg_taper(body: Body, rows: np.ndarray) -> np.ndarray:
    """1 above the knees, fading to 0 at the ankles.

    Below the knee a leg is narrow, moves fast and sits against background, so
    a warp there buys almost nothing and shows up as calves that do not line up
    with the knees. The thighs - where slimming actually reads - keep the full
    amount.
    """
    knees = [body.points[i][1] for i in (P_KNEE_L, P_KNEE_R) if body.visibility[i] >= 0.3]
    ankles = [body.points[i][1] for i in (P_ANKLE_L, P_ANKLE_R) if body.visibility[i] >= 0.3]
    if not knees:
        return np.ones_like(rows)
    knee_y = float(np.mean(knees))
    ankle_y = float(np.mean(ankles)) if ankles else knee_y + abs(knee_y - body.hip_y)
    if ankle_y <= knee_y + 1:
        return np.ones_like(rows)
    return 1.0 - smoothstep(knee_y, ankle_y, rows)


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
    axis = _pose_centre_line(body, rows) if body is not None else None
    centre, half, valid = profiler.profile(mask, rows, w, centre_line=axis)
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

    if body is not None:
        taper = _leg_taper(body, rows)
        amount = amount * taper
        band_amount = band_amount * taper

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
