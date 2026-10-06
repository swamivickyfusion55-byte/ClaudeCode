"""
Spot removal: tap a mole once, and it is gone for the whole clip.

Two problems, and the second is the one that makes this worth building.

Removing a spot from one frame is the easy half: a mole is a dark patch of
low-frequency colour sitting in skin whose colour is known all around it, so
replacing the colour inside the patch with the skin colour that surrounds it -
exactly as asked - and keeping the skin's own texture on top is enough. It
does not look painted because the texture is never replaced, only the blotch
under it.

Keeping it gone while the subject moves is the hard half. A fixed rectangle
would heal the cheek in frame one and a shoulder in frame two hundred. So a
mark is stored as a position relative to something that moves WITH the
subject:

  * on a face, as a weighted combination of nearby face landmarks. The mesh
    moves, turns and scales with the head, so the mark follows it for free,
    on every frame where a face is found - before the marked frame as well as
    after it, which a tracker cannot do without a second pass.
  * anywhere else, by optical flow, tracked in a cheap pre-pass that runs
    forward and backward from the marked frame before the render starts.

Both give the same thing to the renderer: where this spot is in this frame.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

import cv2

from .imaging import feather, poly_mask, to_u8
from .landmarks import (FACE_OVAL, LEFT_BROW, LEFT_EYE, LIPS_OUTER, RIGHT_BROW,
                        RIGHT_EYE)

log = logging.getLogger(__name__)

# Landmarks used to anchor a face mark. Six is enough to survive a turn of the
# head without being thrown off by one noisy point.
ANCHOR_POINTS = 6

# A spot is healed only when this much of the ring around it is one surface.
# Below it the tap is on an edge, and the fill would match neither side.
MIN_SURROUND_SHARE = 0.62


@dataclass
class SpotMark:
    """One tap. Normalised coordinates, so it survives a change of resolution."""
    x: float                       # 0..1 across the frame, at the marked frame
    y: float                       # 0..1 down the frame
    radius: float                  # 0..1 of the frame's long edge
    frame: int = 0                 # which frame it was marked on
    anchor: str = "flow"           # "face" or "flow"
    weights: list = field(default_factory=list)   # [[landmark index, weight], …]
    # What the weights alone do not reach, in the face's own frame: across the
    # face and along it, as fractions of the face's width.
    offset: list = field(default_factory=lambda: [0.0, 0.0])
    face_width: float = 0.0        # face width when marked, to scale the radius
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpotMark":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def marks_from(value) -> list[SpotMark]:
    """Accept whatever the settings carry - dicts from a job's JSON, or
    objects from the UI - and give back marks."""
    out = []
    for item in value or []:
        if isinstance(item, SpotMark):
            out.append(item)
        elif isinstance(item, dict):
            try:
                out.append(SpotMark.from_dict(item))
            except TypeError:
                continue
    return out


# --------------------------------------------------------------- measurement

def locate_spot(bgr: np.ndarray, x: int, y: int, max_radius: int):
    """
    The patch under the tap: its centre, its size, and whether one was found.

    A tap says where, not how big, and asking someone to size every mark with
    a slider is the kind of busywork that stops a feature being used. The
    patch is whatever differs in COLOUR from the skin around it - not whatever
    is darker. Brightness alone finds a black mole on pale skin and little
    else: a brown patch on brown skin can sit within a couple of levels of its
    surroundings in luminance while being plainly a different colour, and on
    deeper skin tones most marks are of that kind. Distance in Lab catches
    both, and needs no assumption about the skin it is on.

    The centre is returned as well as the size, because nobody taps dead
    centre. Healing about the tap instead of about the mark leaves a crescent
    of mole on the far side - the part that fell outside the full-strength
    core - and that crescent is more noticeable than the mole was.

    The last value says whether a distinct patch was actually found. A tap on
    even skin, or into hair, finds nothing to measure, and the caller can say
    so instead of leaving someone to wonder what the app thought it marked.
    """
    h, w = bgr.shape[:2]
    r = int(np.clip(max_radius, 4, 90))
    x0, x1 = max(0, x - r * 2), min(w, x + r * 2 + 1)
    y0, y1 = max(0, y - r * 2), min(h, y + r * 2 + 1)
    patch = bgr[y0:y1, x0:x1]
    fallback = (float(x), float(y), float(max(3, max_radius // 3)), False)
    if patch.size == 0:
        return fallback
    f32 = patch.astype(np.float32) / 255.0 if patch.dtype == np.uint8 else patch
    lab = cv2.cvtColor(f32, cv2.COLOR_BGR2Lab)

    cx, cy = x - x0, y - y0
    yy, xx = np.ogrid[:lab.shape[0], :lab.shape[1]]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # The skin is read from a close ring, not from the edge of the crop. A
    # mole near a hairline or a collar has those inside any generous ring, and
    # they pull the reference colour off the skin and inflate its spread until
    # the threshold sails over the mole itself - the mark then measures as
    # nothing and the tap appears to do nothing at all.
    ring = (dist > r * 0.45) & (dist < r * 0.95)
    if int(ring.sum()) < 24:
        ring = (dist > r * 0.45) & (dist < r * 1.4)
    if int(ring.sum()) < 24:
        return fallback

    skin = np.float32([np.median(lab[:, :, i][ring]) for i in range(3)])
    # Lightness counts for less than hue here: shading across a limb changes L
    # on its own, while a mark changes a and b.
    delta = np.sqrt((0.55 * (lab[:, :, 0] - skin[0])) ** 2 +
                    (lab[:, :, 1] - skin[1]) ** 2 +
                    (lab[:, :, 2] - skin[2]) ** 2)
    level = float(np.median(delta[ring]))
    # Spread as a median absolute deviation: a few stray pixels of hair or
    # lash left in the ring move a standard deviation a long way and a median
    # one hardly at all.
    spread = 1.4826 * float(np.median(np.abs(delta[ring] - level))) + 1e-4
    different = (delta > level + max(2.2 * spread, 1.6)).astype(np.uint8)

    ix = int(np.clip(cx, 0, different.shape[1] - 1))
    iy = int(np.clip(cy, 0, different.shape[0] - 1))
    count, labels = cv2.connectedComponents(different, connectivity=8)
    lab_id = labels[iy, ix]
    if lab_id == 0:
        # The tap landed beside the mark rather than on it. Nobody taps dead
        # centre, and on a phone a fingertip covers a good fraction of the
        # frame as displayed, so the search reaches a full tap-radius out -
        # and takes the nearest candidate of a plausible size, not the
        # largest, which at that reach would start preferring the eyebrow.
        reach = max(6.0, r * 1.2)
        best_id, best_d = 0, None
        for cand in range(1, count):
            cys, cxs = np.nonzero(labels == cand)
            if len(cxs) < 4 or np.sqrt(len(cxs) / np.pi) > r:
                continue
            d = np.hypot(float(cxs.mean()) - cx, float(cys.mean()) - cy)
            if d <= reach and (best_d is None or d < best_d):
                best_id, best_d = cand, d
        if best_id == 0:
            return fallback
        lab_id = best_id
    ys, xs = np.nonzero(labels == lab_id)
    area = float(len(xs))
    mx, my = float(xs.mean()), float(ys.mean())

    # Two readings of the size, and the larger wins. The equivalent-radius of
    # the area is right for a round mole; the reach from the centre is right
    # for a long or lobed patch, which the area alone would under-call and
    # leave an edge of. The threshold cuts through the middle of the blur the
    # mark fades out with, so a margin covers the rest of that fade.
    reach = float(np.percentile(np.sqrt((xs - mx) ** 2 + (ys - my) ** 2), 92)) if area > 4 else 0.0
    radius = max(np.sqrt(max(area, 4.0) / np.pi), reach)
    # A tap on a large area (a shadow, a tattoo, the edge of a garment) is
    # bounded: this heals marks, not regions.
    return (mx + x0, my + y0, float(np.clip(radius * 1.15, 3.0, r * 1.5)), True)


def measure_spot(bgr: np.ndarray, x: int, y: int, max_radius: int) -> float:
    """Just the radius, for callers that already know where the mark is."""
    return locate_spot(bgr, x, y, max_radius)[2]


# ------------------------------------------------------------------ anchoring

def anchor_to_face(mark: SpotMark, faces, shape) -> SpotMark:
    """
    Tie a mark to the face mesh, if it is on a face.

    The mark becomes a weighted combination of the nearest few landmarks, so
    it translates, rotates and scales with the head for free - no tracking,
    and it works backwards through the clip as readily as forwards.

    The weights have to REPRODUCE the tap, which plain inverse-distance
    weights do not: those give a point somewhere among the landmarks, near
    their centroid, and healing a few pixels off a mole leaves a crescent of
    it behind - which is more noticeable than the mole was.

    Solving the weights for an exact fit is the obvious repair and the wrong
    one: the solution is free to leave the simplex, and weights that sum to
    one while running past ±1 turn a pixel of mesh jitter into several pixels
    of drift. On a sixty-frame clip that put the heal off the mole entirely on
    six of them.

    So the weights stay convex - nearest-landmark dominant, which is what
    makes the mark follow the skin it is on - and whatever they do not reach
    is kept separately, in the face's own frame of reference: across the face
    and along its axis, as fractions of its width. That residual turns and
    scales with the head like everything else, and being a stored constant it
    cannot amplify anything.
    """
    h, w = shape[:2]
    px = np.float32([mark.x * w, mark.y * h])
    best = None
    for face in faces:
        d = np.linalg.norm(face.points - px, axis=1)
        near = float(d.min())
        # Only claim the mark if it is on this face, not merely near it.
        if near < face.width * 0.9 and (best is None or near < best[0]):
            best = (near, face, d)
    if best is None:
        return mark

    _, face, d = best
    idx = np.argsort(d)[:ANCHOR_POINTS]
    w0 = 1.0 / np.maximum(d[idx], 1e-3)
    w0 = (w0 / float(w0.sum())).astype(np.float32)

    base = (np.asarray(face.points, np.float32)[idx] * w0[:, None]).sum(0)
    along, across = _face_frame(face)
    gap = px - base
    scale = max(float(face.width), 1.0)

    mark.anchor = "face"
    mark.weights = [[int(i), float(wt)] for i, wt in zip(idx, w0)]
    mark.offset = [float(np.dot(gap, across) / scale), float(np.dot(gap, along) / scale)]
    mark.face_width = float(face.width)
    return mark


def _face_frame(face):
    """The face's own axes: along it (chin to forehead) and across it."""
    along = np.asarray(face.axis, np.float32).reshape(2)
    n = float(np.linalg.norm(along))
    along = along / n if n > 1e-6 else np.float32([0.0, -1.0])
    across = np.float32([-along[1], along[0]])
    return along, across


def resolve_face(mark: SpotMark, faces, shape):
    """Where this face-anchored mark is in this frame, and how big."""
    if mark.anchor != "face" or not mark.weights or not faces:
        return None
    h, w = shape[:2]
    best = None
    ox, oy = (list(mark.offset) + [0.0, 0.0])[:2]
    for face in faces:
        pos = np.zeros(2, np.float32)
        ok = True
        for i, wt in mark.weights:
            if i >= len(face.points):
                ok = False
                break
            pos += face.points[i] * wt
        if not ok:
            continue
        along, across = _face_frame(face)
        pos = pos + (across * ox + along * oy) * max(float(face.width), 1.0)
        # With several faces, the mark belongs to the one whose geometry it was
        # measured against - judged by how close the reconstruction lands to
        # the landmarks it was built from.
        err = float(np.linalg.norm(face.points[mark.weights[0][0]] - pos))
        if best is None or err < best[0]:
            best = (err, pos, face)
    if best is None:
        return None
    _, pos, face = best
    scale = (face.width / mark.face_width) if mark.face_width > 1 else 1.0
    radius = mark.radius * max(w, h) * float(np.clip(scale, 0.3, 3.0))
    return float(pos[0]), float(pos[1]), float(radius)


class FlowTrack:
    """
    Per-frame positions for marks that are not on a face, from a pre-pass.

    Runs Lucas-Kanade both ways from the marked frame on a downscaled copy of
    the clip - a few points per frame, so the pass costs a decode and very
    little else. Backward matters as much as forward: someone marks a mole
    when they notice it, which is rarely the first frame, and the frames
    before it need healing too.
    """

    def __init__(self, work_width: int = 480):
        self.work_width = int(work_width)
        self.table: dict[int, dict[int, tuple[float, float]]] = {}

    def build(self, frames_iter, marks: list[SpotMark], shape, should_cancel=None):
        """`frames_iter` yields (index, bgr) for the range being rendered."""
        flow_marks = [(i, m) for i, m in enumerate(marks) if m.anchor != "face"]
        if not flow_marks:
            return self
        h, w = shape[:2]
        scale = self.work_width / float(max(w, 1))

        small_frames: dict[int, np.ndarray] = {}
        for idx, frame in frames_iter:
            if should_cancel is not None and should_cancel():
                return self
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_frames[idx] = cv2.resize(g, (0, 0), fx=scale, fy=scale,
                                           interpolation=cv2.INTER_AREA)
        if not small_frames:
            return self
        order = sorted(small_frames)

        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.03))
        for mi, mark in flow_marks:
            start = int(np.clip(mark.frame, order[0], order[-1]))
            if start not in small_frames:
                start = min(order, key=lambda i: abs(i - start))
            p0 = np.float32([[[mark.x * w * scale, mark.y * h * scale]]])
            self.table.setdefault(mi, {})[start] = (mark.x * w, mark.y * h)
            for direction in (1, -1):
                pos = p0.copy()
                prev = start
                seq = [i for i in order if (i > start if direction > 0 else i < start)]
                if direction < 0:
                    seq = list(reversed(seq))
                for idx in seq:
                    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(
                        small_frames[prev], small_frames[idx], pos, None, **lk)
                    if nxt is None or not ok.any():
                        break           # lost it; stop rather than drift
                    pos = nxt
                    self.table[mi][idx] = (float(pos[0, 0, 0]) / scale,
                                           float(pos[0, 0, 1]) / scale)
                    prev = idx
        return self

    def resolve(self, mark_index: int, mark: SpotMark, frame_index: int, shape):
        table = self.table.get(mark_index)
        if not table:
            return None
        if frame_index in table:
            x, y = table[frame_index]
        else:
            nearest = min(table, key=lambda i: abs(i - frame_index))
            if abs(nearest - frame_index) > 12:
                return None
            x, y = table[nearest]
        h, w = shape[:2]
        return float(x), float(y), float(mark.radius * max(w, h))


# -------------------------------------------------------------------- healing

def dominant_surround(roi: np.ndarray, dist: np.ndarray, r: float):
    """
    The one surface around the spot, and how much of the ring belongs to it.

    A tap near an edge - where skin meets a sleeve, a hemline, hair - has a
    ring made of two different things, and a fit over both lands halfway
    between them: the spot gets replaced by a colour that matches neither,
    which is worse than leaving it. So the ring is reduced to its dominant
    surface by re-estimating the median from the pixels closest to it, and the
    share that surface holds is reported. A caller that sees a low share
    should decline rather than produce a blob.
    """
    ring = (dist > r * 1.8) & (dist < r * 3.4)
    if int(ring.sum()) < 40:
        return None, 0.0
    lab = cv2.cvtColor(np.clip(roi, 0, 1), cv2.COLOR_BGR2Lab)
    samples = lab[ring]
    keep = np.ones(len(samples), bool)
    centre = np.median(samples, axis=0)
    for _ in range(3):
        d = np.linalg.norm(samples - centre, axis=1)
        tol = max(float(np.median(d)) * 1.6, 4.0)
        keep = d <= tol
        if int(keep.sum()) < 20:
            break
        centre = np.median(samples[keep], axis=0)
    share = float(keep.mean())
    full = np.linalg.norm(lab - centre, axis=2) <= max(
        float(np.median(np.linalg.norm(samples[keep] - centre, axis=1))) * 2.2, 6.0)
    return (ring & full), share


def _skin_plane(roi: np.ndarray, dist: np.ndarray, r: float, surround=None):
    """
    The colour the skin would have had across the patch, as a shaded plane.

    Skin curves, so the fill cannot be one flat colour, but over a few dozen
    pixels its shading is near enough linear. Fitting a plane per channel to
    the ring of real skin around the spot and evaluating it inside gives a
    fill with the right gradient and no residue - where an edge-inward
    inpaint, seeded on a ring the mole has already darkened, leaves a grey
    disc behind on anything but the smallest spots.

    The fit is trimmed: the darkest and brightest fifth of the ring are
    dropped, so a second mole or a stray hair in the ring cannot drag it.
    """
    ring = surround if surround is not None else ((dist > r * 1.8) & (dist < r * 3.4))
    n = int(ring.sum())
    if n < 40:
        return None
    ys, xs = np.nonzero(ring)
    fx, fy = xs.astype(np.float32), ys.astype(np.float32)
    # Quadratic, not linear: a cheek is curved, and over the width of a spot
    # plus its ring that curvature is visible - a plane leaves a faintly
    # lighter or darker patch where the surface was bending.
    a = np.stack([fx * fx, fy * fy, fx * fy, fx, fy, np.ones(n, np.float32)], 1)
    planes = []
    for c in range(roi.shape[2]):
        v = roi[:, :, c][ring]
        lo, hi = np.percentile(v, [20, 80])
        keep = (v >= lo) & (v <= hi)
        if int(keep.sum()) < 24:
            keep = np.ones_like(v, bool)
        coef, *_ = np.linalg.lstsq(a[keep], v[keep], rcond=None)
        planes.append(coef)
    h, w = roi.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.empty_like(roi)
    for c, coef in enumerate(planes):
        out[:, :, c] = (coef[0] * gx * gx + coef[1] * gy * gy + coef[2] * gx * gy
                        + coef[3] * gx + coef[4] * gy + coef[5])
    return out


def _texture_donor(detail: np.ndarray, cx: float, cy: float, r: float,
                   ref_sd: float):
    """
    Pick a patch of clean skin near the spot to borrow texture from.

    Borrowing real skin beats leaving the healed area textureless, which is
    what gives a smooth disc that reads as a smudge even when the colour is
    perfect. But a donor is only clean skin if it holds nothing else, and a
    mole is often near something that does - a nostril, a hairline, the rim of
    a lip. Take the grain from one of those and the crescent of the feature is
    stamped onto the cheek, which looks worse than the mole did.

    So candidates are judged against the skin that actually surrounds this
    spot: eight directions, scored on how close the candidate's grain is to
    the ring's own, and anything markedly busier is refused outright. The
    returned offset is paired with a gain that brings its grain to the ring's
    level, so a slightly livelier donor is used quietly rather than loudly.
    """
    h, w = detail.shape[:2]
    half = int(max(2, round(r * 1.6)))
    best = None
    for radius in (r * 2.4, r * 3.2):
        step = int(max(3, round(radius)))
        diag = int(max(2, round(radius * 0.7071)))
        for dx, dy in ((0, -step), (0, step), (-step, 0), (step, 0),
                       (diag, diag), (diag, -diag), (-diag, diag), (-diag, -diag)):
            sx, sy = int(cx + dx), int(cy + dy)
            x0, x1 = sx - half, sx + half + 1
            y0, y1 = sy - half, sy + half + 1
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue
            sd = float(detail[y0:y1, x0:x1].std())
            if sd > ref_sd * 1.8:
                continue            # a feature lives here; not skin
            score = abs(sd - ref_sd)
            if best is None or score < best[0]:
                best = (score, (dx, dy), sd)
        if best is not None:
            break
    if best is None:
        return None
    _, offset, sd = best
    gain = float(np.clip(ref_sd / max(sd, 1e-5), 0.0, 1.5))
    return offset, gain


def heal(bgr: np.ndarray, spots, strength: float = 1.0) -> np.ndarray:
    """
    Remove each spot by carrying the surrounding skin across it.

    Colour and texture are taken from different places, because they are not
    equally available. The colour is simply the skin that surrounds the patch,
    carried inward - which is what was asked for, and what an edge-inward
    inpaint does. The texture cannot come from the patch (that is the mole)
    and should not be left out (a textureless disc reads as a smudge), so it
    is borrowed from the cleanest skin a couple of radii away.

    Two details matter. The inpaint mask is dilated well past the spot,
    because a mole blurs into its surroundings and an inpaint seeded on a ring
    the mole has already darkened fills it with a grey disc - which is exactly
    what the first version of this did. And the texture donor is offset, not
    mirrored, so no feature is duplicated.
    """
    if not spots or strength <= 0:
        return bgr
    out = bgr
    h, w = bgr.shape[:2]
    for (x, y, r) in spots:
        r = float(np.clip(r, 2.0, max(w, h) * 0.08))
        pad = int(np.ceil(r * 4.0))
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - pad), min(w, xi + pad + 1)
        y0, y1 = max(0, yi - pad), min(h, yi + pad + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        if out is bgr:
            out = bgr.copy()
        roi = out[y0:y1, x0:x1]

        cx, cy = xi - x0, yi - y0
        yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # Dilated: covers the spot and the halo it blurred into the skin.
        fill_mask = (dist <= r * 1.8).astype(np.uint8)
        if not fill_mask.any():
            continue
        soft = np.clip(1.0 - (dist - r * 1.1) / max(r * 0.9, 1.0), 0.0, 1.0).astype(np.float32)

        sigma = max(1.0, r * 0.5)
        surround, share = dominant_surround(roi, dist, r)
        if share and share < MIN_SURROUND_SHARE:
            # The spot sits on an edge: two surfaces around it, and no single
            # colour to carry across. Healing here would leave a patch
            # matching neither, so it is left alone.
            log.info("spot at (%d,%d) skipped: only %.0f%% of its surroundings "
                     "are one surface", xi, yi, 100 * share)
            continue
        plane = _skin_plane(roi, dist, r, surround)
        if plane is None:
            # Not enough clean skin around it to fit: fall back to an
            # edge-inward fill, which is weaker but needs no ring.
            filled = cv2.inpaint(to_u8(roi), fill_mask, max(3, int(r * 2.2)),
                                 cv2.INPAINT_TELEA).astype(np.float32) / 255.0
            colour = cv2.GaussianBlur(filled, (0, 0), sigma)
        else:
            # The plane inside the patch, the real (blurred) image outside, so
            # the join carries no step.
            outside = cv2.GaussianBlur(roi, (0, 0), sigma)
            fade = np.clip(1.0 - (dist - r * 1.2) / max(r * 1.4, 1.0), 0.0, 1.0)
            colour = outside * (1.0 - fade)[:, :, None] + plane * fade[:, :, None]
            filled = colour
        own = roi - cv2.GaussianBlur(roi, (0, 0), sigma)
        # How lively is the skin that surrounds this spot? Everything borrowed
        # is held to that level, so nothing louder than real skin is laid down.
        clean = surround if surround is not None else (dist > r * 1.9) & (dist < r * 3.0)
        ref_sd = float(own[clean].std()) if int(clean.sum()) > 24 else float(own.std())
        ref_sd = max(ref_sd, 1e-4)
        donor = _texture_donor(own, cx, cy, r, ref_sd)
        if donor is not None:
            (dx, dy), gain = donor
            texture = np.roll(np.roll(own, -dy, axis=0), -dx, axis=1) * gain
            np.clip(texture, -3.0 * ref_sd, 3.0 * ref_sd, out=texture)
        else:
            texture = np.zeros_like(own)
        detail = own * (1.0 - soft)[:, :, None] + texture * soft[:, :, None]
        healed = colour + detail

        alpha = (soft * float(np.clip(strength, 0.0, 1.0)))[:, :, None]
        out[y0:y1, x0:x1] = np.clip(roi * (1.0 - alpha) + healed * alpha, 0.0, 1.0)
    return out


# ------------------------------------------------------------- auto-detection

# Nothing smaller is a mark worth healing (it is grain), and nothing larger is
# one (it is a shadow, a tattoo, a nipple, the shade under a jaw). Both are
# fractions of the frame's long edge, so they mean the same thing at any size.
AUTO_MIN_RADIUS = 0.0030
AUTO_MAX_RADIUS = 0.0180

# A blob has to be roughly round. A hair, an eyelash, a crease and the line of
# a seam all read as "darker than their surroundings" and none of them is a
# mole; the one thing that separates them is that they are long and thin.
AUTO_MIN_ROUNDNESS = 0.42


def detect_spots(bgr: np.ndarray, skin: np.ndarray, sensitivity: float = 0.5,
                 limit: int = 12, reference: np.ndarray | None = None
                 ) -> list[tuple[float, float, float, float]]:
    """
    Find the marks on this frame by themselves: (x, y, radius, strength).

    The same measurement a tap uses, run everywhere at once. A mark is a small
    region that differs in colour from the skin immediately around it, so the
    detector is a local one: the image minus a median of itself over a window
    a few mark-widths across, read in Lab with lightness weighted down, which
    is what lets it find a brown patch on brown skin as readily as a black
    mole on pale skin.

    Three filters do the work of not healing the person's face off:

      * size, in both directions. Below the floor it is grain; above the
        ceiling it is a shadow, a tattoo or the shade under a jaw, and this
        heals marks, not regions.
      * roundness. A hair, a lash, a crease and the edge of a seam are all
        darker than what is around them, and all long and thin.
      * the same surround test a tap goes through. A candidate sitting on an
        edge has no single colour to carry across it, so it is dropped rather
        than filled with something matching neither side.

    `skin` is where it is allowed to look, as a 0..1 mask - the face's own
    skin mask (which already excludes eyes, brows, lips and nostrils) plus
    whatever else is the person. Nothing outside it is ever considered.

    `reference` is where the bar is set FROM, and it is a different mask on
    purpose: the part that is confidently skin, before the search region was
    widened to cover the marks themselves. Setting the bar from the widened
    region lets a sleeve or a jacket raise it, and then the mole on the cheek
    measures as ordinary.
    """
    h, w = bgr.shape[:2]
    long_edge = float(max(w, h))
    r_min = max(1.5, AUTO_MIN_RADIUS * long_edge)
    r_max = max(r_min + 1.0, AUTO_MAX_RADIUS * long_edge)
    allow = (np.asarray(skin, np.float32) > 0.5)
    if not allow.any():
        return []

    f32 = bgr.astype(np.float32) / 255.0 if bgr.dtype == np.uint8 else bgr
    lab = cv2.cvtColor(np.clip(f32, 0, 1), cv2.COLOR_BGR2Lab)
    # A window a few mark-widths across: wide enough that a mole cannot define
    # its own background, narrow enough to follow the shading of a cheek.
    k = int(r_max * 4) | 1
    base = cv2.medianBlur(to_u8(f32), min(k, 31)).astype(np.float32) / 255.0
    base = cv2.cvtColor(base, cv2.COLOR_BGR2Lab)
    delta = np.sqrt((0.55 * (lab[:, :, 0] - base[:, :, 0])) ** 2 +
                    (lab[:, :, 1] - base[:, :, 1]) ** 2 +
                    (lab[:, :, 2] - base[:, :, 2]) ** 2)
    # Darker than its surroundings, or a different colour at the same
    # lightness. Lighter-and-otherwise-identical is a highlight, not a mark.
    darker = lab[:, :, 0] < base[:, :, 0] + 0.5
    chroma = np.sqrt((lab[:, :, 1] - base[:, :, 1]) ** 2 +
                     (lab[:, :, 2] - base[:, :, 2]) ** 2)
    signal = np.where(darker | (chroma > 2.0), delta, 0.0)

    ref = allow if reference is None else (np.asarray(reference, np.float32) > 0.5)
    inside = signal[ref]
    if inside.size < 64:
        inside = signal[allow]
    if inside.size < 64:
        return []
    level = float(np.median(inside))
    spread = 1.4826 * float(np.median(np.abs(inside - level))) + 1e-4
    # Sensitivity moves the bar between "only what is unmistakable" and "every
    # freckle", in robust deviations of the skin's own signal. The floor keeps
    # the top of the range from finding compression noise on flat skin.
    k_sigma = 4.0 - 2.2 * float(np.clip(sensitivity, 0.0, 1.0))
    # And an absolute floor as well as a relative one: a mark somebody would
    # ask to have removed differs from the skin around it by more than this,
    # and nothing that does not is worth a heal.
    thresh = max(level + k_sigma * spread, 4.0)
    mask = ((signal > thresh) & allow).astype(np.uint8)
    if not mask.any():
        return []

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    found = []
    ys_all, xs_all = np.nonzero(labels)
    order = np.argsort(labels[ys_all, xs_all], kind="stable")
    ys_all, xs_all = ys_all[order], xs_all[order]
    ids = labels[ys_all, xs_all]
    bounds = np.searchsorted(ids, np.arange(1, n))
    bounds = np.append(bounds, len(ids))
    start = 0
    for lab_id in range(1, n):
        end = bounds[lab_id - 1]
        xs, ys = xs_all[start:end], ys_all[start:end]
        start = end
        area = float(len(xs))
        if area < 4:
            continue
        equiv = np.sqrt(area / np.pi)
        if equiv < r_min * 0.8 or equiv > r_max:
            continue
        mx, my = float(xs.mean()), float(ys.mean())
        reach = float(np.percentile(np.sqrt((xs - mx) ** 2 + (ys - my) ** 2), 92))
        if reach > r_max * 1.4:
            continue
        if area / (np.pi * max(reach, 0.5) ** 2) < AUTO_MIN_ROUNDNESS:
            continue                              # a hair, a lash, a crease
        radius = float(np.clip(max(equiv, reach) * 1.15, r_min, r_max * 1.3))
        strength = float(signal[ys, xs].mean())
        found.append((mx, my, radius, strength))

    found.sort(key=lambda t: -t[3])
    kept = []
    for (mx, my, radius, strength) in found:
        if len(kept) >= max(1, int(limit)):
            break
        # Two detections on top of each other are one mark seen twice.
        if any((mx - kx) ** 2 + (my - ky) ** 2 < (radius + kr) ** 2
               for kx, ky, kr, _ in kept):
            continue
        pad = int(np.ceil(radius * 4.0))
        x0, x1 = max(0, int(mx) - pad), min(w, int(mx) + pad + 1)
        y0, y1 = max(0, int(my) - pad), min(h, int(my) + pad + 1)
        roi = f32[y0:y1, x0:x1]
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            continue
        yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
        dist = np.sqrt((xx - (mx - x0)) ** 2 + (yy - (my - y0)) ** 2)
        _, share = dominant_surround(roi, dist, radius)
        if share and share < MIN_SURROUND_SHARE:
            continue                              # on an edge: leave it
        kept.append((mx, my, radius, strength))
    return kept


def skin_region(bgr, faces, person, shape, scale: float = 1.0):
    """
    Where auto-detection may look, and where it reads the skin from.

    Two masks come back, and they are not the same one. The first is the
    search region; the second is the part that is confidently skin, which is
    what the bar for "this is a mark" is set from. The search region has to be
    the wider of the two - a mole is not skin-coloured, so a colour gate
    excludes the very thing being looked for - and setting the bar from that
    wider region is what lets a sleeve raise it until the mole on the cheek
    measures as ordinary skin.

    The face mask is the easy part - it already excludes eyes, brows, lips and
    nostrils, which are the four things on a face most reliably darker than
    what surrounds them.

    The body is the part that needs care, because the person mask is a mask of
    the PERSON: it includes their clothes. Run a mark-finder over a dark
    jumper and it finds a dozen marks a frame, every one of them a fold or a
    print, and they crowd out the real ones. So the body half is gated on
    colour - and the colour it is gated on is measured from this subject's own
    face in this frame, not from a table. That is what makes it work at any
    skin tone instead of at the ones a table happened to list.

    With no face in shot there is nothing to calibrate from, so it falls back
    to a broad generic skin window, which is weaker: auto-detection on a body
    is at its best when a face is in the same frame.

    The body outline is also eroded, because the rim of a person against the
    background is the one place a "dark patch" is guaranteed to be the
    background showing through.
    """
    h, w = shape[:2]
    face_px = np.zeros((h, w), np.float32)
    for f in faces or []:
        face_px = np.maximum(face_px, _face_skin(f, shape, scale))
    allow = face_px.copy()
    core = face_px.copy()

    if person is not None:
        body = np.asarray(person, np.float32)
        if body.shape[:2] != (h, w):
            body = cv2.resize(body, (w, h), interpolation=cv2.INTER_LINEAR)
        # Close both masks by a little more than the largest mark this will
        # heal, before anything else is done with them. A mole is not
        # skin-coloured - that is the whole point of it - so a colour gate
        # punches a hole in the mask exactly where the mark is, and some
        # segmenters drop a dark patch out of the person too. Closing fills
        # holes of that size and nothing larger; the result is still bounded
        # by the person's own outline, so nothing leaks into the background.
        #
        # Square structuring elements on uint8, not round ones on float: these
        # are coarse region masks, a square is indistinguishable in the result,
        # and OpenCV runs a rectangle separably. Measured on one 640x480 frame,
        # that one substitution took this from 46 ms to 0.6 ms - and it runs on
        # every frame of the render.
        # Just over the diameter of the largest mark this will heal, and no
        # more. Closing by a generous margin instead bridges the gate's gaps
        # back together and quietly re-admits whatever sat between them - on
        # one test frame, the dark folds of a flight suit.
        fill = int(max(4, AUTO_MAX_RADIUS * 1.15 * max(h, w)))
        kf = cv2.getStructuringElement(cv2.MORPH_RECT, (fill * 2 + 1,) * 2)
        body = cv2.morphologyEx((body > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, kf)
        grow = max(3, int(min(h, w) * 0.012))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (grow * 2 + 1,) * 2)
        body = cv2.erode(body, k).astype(np.float32)
        skin = _skin_like(bgr, face_px)
        if skin is not None:
            core = np.maximum(core, body * skin)
            closed = cv2.morphologyEx((skin > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, kf)
            body = body * closed.astype(np.float32)
        else:
            core = np.maximum(core, body)
        allow = np.maximum(allow, body)
    if not allow.any():
        return None, None
    return allow, (core if core.any() else allow)


def _face_skin(face, shape, scale: float = 1.0):
    """The face's skin mask, built directly at the size wanted.

    `Face.skin_mask` builds it at the frame's own size, which is the right
    thing everywhere else and the wrong thing here: auto-detection runs on a
    downscaled copy, and rasterising a 4K mask only to shrink it costs more
    than everything else in the detector put together. The landmarks are
    pixel coordinates, so scaling them is all it takes.
    """
    if scale >= 0.999:
        return face.skin_mask(shape, feather_px=1.0)
    pts = np.asarray(face.points, np.float32) * float(scale)
    poly = lambda idx: pts[list(idx)]
    m = poly_mask(shape, [poly(FACE_OVAL)])
    excl = np.clip(poly_mask(shape, [poly(LEFT_EYE), poly(RIGHT_EYE),
                                     poly(LEFT_BROW), poly(RIGHT_BROW)])
                   + poly_mask(shape, [poly(LIPS_OUTER)]), 0, 1)
    grow = max(2, int(face.width * scale * 0.015))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1,) * 2)
    return feather(np.clip(m - cv2.dilate(excl, k), 0, 1), 1.0)


def _skin_like(bgr, face_mask):
    """A 0/1 mask of pixels the same colour as this person's face."""
    f32 = bgr.astype(np.float32) / 255.0 if bgr.dtype == np.uint8 else bgr
    lab = cv2.cvtColor(np.clip(f32, 0, 1), cv2.COLOR_BGR2Lab)
    sample = lab[face_mask > 0.5]
    if sample.shape[0] >= 200:
        med = np.median(sample, axis=0)
        ab = np.linalg.norm(sample[:, 1:] - med[1:], axis=1)
        tol = max(3.0 * 1.4826 * float(np.median(np.abs(ab - np.median(ab)))), 7.0)
        near = np.linalg.norm(lab[:, :, 1:] - med[1:], axis=2) <= tol
        # Lightness is allowed to wander - an arm in shade is still the arm -
        # but not all the way to black, where chroma stops meaning anything
        # and a dark fold in a garment passes for skin.
        lit = np.abs(lab[:, :, 0] - med[0]) <= 32.0
        return (near & lit).astype(np.float32)
    # No face to learn from: a generic window, deliberately broad, because a
    # narrow one simply fails on the skin tones it was not built around.
    ycc = cv2.cvtColor(to_u8(f32), cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[:, :, 1].astype(np.int16), ycc[:, :, 2].astype(np.int16)
    y = ycc[:, :, 0].astype(np.int16)
    return ((cr >= 130) & (cr <= 184) & (cb >= 74) & (cb <= 132)
            & (y >= 40)).astype(np.float32)


class AutoSpotter:
    """
    Auto-detect across a clip, without the flicker that healing a per-frame
    detection straight off would give.

    A detector run independently on every frame does not agree with itself
    frame to frame - a mark on the threshold is found, missed, found - and
    healing that switches a patch of skin on and off at twenty-four frames a
    second, which is far more visible than the mark was. So detections are
    carried as tracks, exactly as faces are elsewhere in this app: a track has
    to be seen twice before anything is healed, it is coasted for a few frames
    when it is missed, and its strength ramps in and out instead of
    switching. The first heal of a mark is one frame later than it could be;
    nothing pops.

    Tracks are matched to the face mesh where there is a face, so a track
    survives the head moving between frames rather than being matched by
    pixel position and lost on the first quick turn.
    """

    CONFIRM = 2          # frames a track must be seen on before it is healed
    COAST = 4            # frames it survives being missed
    RAMP = 3             # frames to fade in and out over

    # Detection runs at this long edge, whatever the footage is. The smallest
    # mark this acts on is 0.3% of the long edge, which is two pixels here -
    # still a blob - and the alternative is paying for 4K on every frame to
    # locate something that will be healed at full resolution anyway.
    WORK_EDGE = 640

    def __init__(self, sensitivity: float = 0.5, limit: int = 12,
                 stabilise: bool = True):
        self.sensitivity = float(sensitivity)
        self.limit = int(limit)
        self.stabilise = bool(stabilise)
        self.tracks: list[dict] = []
        self.frames = 0

    def __call__(self, bgr, faces, person, shape) -> list[tuple[float, float, float]]:
        h, w = shape[:2]
        k = min(1.0, self.WORK_EDGE / float(max(h, w, 1)))
        if k < 0.999:
            sw, sh = max(16, int(round(w * k))), max(16, int(round(h * k)))
            small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
            small_person = (None if person is None else
                            cv2.resize(np.asarray(person, np.float32), (sw, sh),
                                       interpolation=cv2.INTER_LINEAR))
            k = sw / float(w)          # the scale actually used, after rounding
        else:
            small, small_person, k = bgr, person, 1.0

        allow, ref = self._where_to_look(small, faces, small_person,
                                         small.shape, scale=k)
        if allow is None:
            return []
        found = detect_spots(small, allow, self.sensitivity, self.limit, reference=ref)
        if k != 1.0:
            found = [(x / k, y / k, r / k, st) for x, y, r, st in found]
        if not self.stabilise:
            return [(x, y, r) for x, y, r, _ in found]
        return self._carry(found, faces, shape)

    # ------------------------------------------------------------- internals
    def _where_to_look(self, bgr, faces, person, shape, scale=1.0):
        return skin_region(bgr, faces, person, shape, scale=scale)

    def _carry(self, found, faces, shape):
        # The first frame is the exception to both rules below. Confirming
        # over two frames and ramping over three exist to stop a mark
        # appearing and vanishing mid-clip; at the very first frame there is
        # nothing to appear from, and waiting leaves the mark visible on the
        # one frame people are most likely to look at - the thumbnail.
        first = self.frames == 0
        self.frames += 1
        face = faces[0] if faces else None
        tol = (face.width * 0.09) if face is not None else max(shape[:2]) * 0.02
        tol = max(float(tol), 4.0)

        for t in self.tracks:
            t["matched"] = False
            if face is not None and t.get("weights"):
                pos = _reconstruct(t, face)
                if pos is not None:
                    t["x"], t["y"] = pos

        for (mx, my, radius, _strength) in found:
            best, best_d = None, None
            for t in self.tracks:
                if t["matched"]:
                    continue
                d = np.hypot(mx - t["x"], my - t["y"])
                if d <= tol + radius and (best_d is None or d < best_d):
                    best, best_d = t, d
            if best is None:
                best = {"seen": 0, "missed": 0, "level": 0.0}
                self.tracks.append(best)
            best.update(x=mx, y=my, r=radius, matched=True, missed=0)
            best["seen"] += 1
            if face is not None:
                best.update(_anchor(mx, my, face))

        out = []
        for t in self.tracks:
            if not t["matched"]:
                t["missed"] += 1
            confirm = 1 if first else self.CONFIRM
            target = 1.0 if (t["matched"] and t["seen"] >= confirm) else 0.0
            if t["missed"] > self.COAST:
                target = 0.0
            if first:
                t["level"] = target
            else:
                step = 1.0 / max(self.RAMP, 1)
                t["level"] = float(np.clip(
                    t["level"] + (step if target > t["level"] else -step), 0.0, 1.0))
            if t["level"] > 0.02:
                out.append((t["x"], t["y"], t["r"] * (0.6 + 0.4 * t["level"])))
        self.tracks = [t for t in self.tracks
                       if t["level"] > 0.02 or t["missed"] <= self.COAST]
        return out


def _anchor(x, y, face):
    """Store a detection against the face mesh, the same way a tap is."""
    px = np.float32([x, y])
    d = np.linalg.norm(face.points - px, axis=1)
    if float(d.min()) > face.width * 0.9:
        return {"weights": None}
    idx = np.argsort(d)[:ANCHOR_POINTS]
    w0 = 1.0 / np.maximum(d[idx], 1e-3)
    w0 = (w0 / float(w0.sum())).astype(np.float32)
    base = (np.asarray(face.points, np.float32)[idx] * w0[:, None]).sum(0)
    along, across = _face_frame(face)
    gap = px - base
    scale = max(float(face.width), 1.0)
    return {"weights": [(int(i), float(v)) for i, v in zip(idx, w0)],
            "offset": (float(np.dot(gap, across) / scale),
                       float(np.dot(gap, along) / scale))}


def _reconstruct(track, face):
    """Where a tracked detection has moved to on this frame."""
    weights = track.get("weights")
    if not weights:
        return None
    pos = np.zeros(2, np.float32)
    for i, wt in weights:
        if i >= len(face.points):
            return None
        pos += face.points[i] * wt
    ox, oy = track.get("offset", (0.0, 0.0))
    along, across = _face_frame(face)
    pos = pos + (across * ox + along * oy) * max(float(face.width), 1.0)
    return float(pos[0]), float(pos[1])
