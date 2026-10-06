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

from .imaging import to_u8

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
        # The tap landed beside the mark rather than on it: take the nearest
        # region instead of giving up, since nobody taps dead centre.
        near = labels[(dist < max(5.0, r * 0.6)) & (labels > 0)]
        if near.size == 0:
            return fallback
        lab_id = int(np.bincount(near).argmax())
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
