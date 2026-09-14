"""
Face, body and person-silhouette tracking, with the temporal smoothing that
makes the difference between "this works on a photo" and "this works on video".

MediaPipe gives per-frame estimates. Per-frame estimates jitter, and every
effect downstream is driven by them, so the jitter becomes a visible crawl on
the jawline and a shimmering edge on the hair. Each tracker here therefore
keeps state: landmarks are EMA-smoothed against the previous frame's, faces are
matched to their previous selves by centroid so the smoothing does not swap two
people's geometry, and the segmentation mask is averaged at low resolution
where the noise actually lives.

If MediaPipe is not installed the module still imports; the trackers report
themselves unavailable and the pipeline falls back to grade-only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

import cv2

from .imaging import EMA, feather, poly_mask

log = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except Exception as e:  # pragma: no cover - environment dependent
    mp = None
    MEDIAPIPE_OK = False
    log.warning("mediapipe unavailable (%s) - face/body features disabled", e)


# ------------------------------------------------------------ landmark groups
# Index sets for MediaPipe's 468/478-point face mesh.

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Jaw contour walked from the chin outward, per side. Order matters: the
# slimming warp weights points by how far along the jaw they are.
JAW_LEFT = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234]
JAW_RIGHT = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LEFT_LOWER_LID = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_LOWER_LID = [263, 249, 390, 373, 374, 380, 381, 382, 362]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
              267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311,
              312, 13, 82, 81, 80, 191]

NOSE_TIP = 4
NOSE_BRIDGE = 6
NOSE_BOTTOM = 2
NOSE_WING_L = 98
NOSE_WING_R = 327
CHIN = 152
FOREHEAD = 10

# Pose landmark indices (BlazePose, 33 points).
P_NOSE, P_SHOULDER_L, P_SHOULDER_R = 0, 11, 12
P_ELBOW_L, P_ELBOW_R = 13, 14
P_WRIST_L, P_WRIST_R = 15, 16
P_HIP_L, P_HIP_R = 23, 24
P_KNEE_L, P_KNEE_R = 25, 26
P_ANKLE_L, P_ANKLE_R = 27, 28


@dataclass
class Face:
    """One tracked face: pixel-space landmarks plus the numbers every stage
    needs so they are computed once instead of five times."""
    points: np.ndarray          # (N, 2) float32, pixel coordinates
    width: float                # cheek-to-cheek distance in pixels
    height: float               # forehead-to-chin distance in pixels
    centre: np.ndarray          # (2,) float32
    axis: np.ndarray            # (2,) float32 unit vector, chin -> forehead
    has_iris: bool
    # 0..1. Below 1 while a face is being acquired or coasted through a
    # dropout; every stage scales its amounts by it so effects ramp instead
    # of switching on and off between frames.
    confidence: float = 1.0

    def p(self, idx) -> np.ndarray:
        return self.points[idx]

    def poly(self, idx_list) -> np.ndarray:
        return self.points[list(idx_list)]

    def box(self) -> tuple[int, int, int, int]:
        x0, y0 = self.points.min(axis=0)
        x1, y1 = self.points.max(axis=0)
        return int(x0), int(y0), int(x1), int(y1)

    # -- region masks ------------------------------------------------------
    def skin_mask(self, shape, feather_px: float | None = None) -> np.ndarray:
        """Face oval minus eyes, brows, lips and nostrils, feathered.

        The exclusions are what keep retouching honest: smoothing that runs
        over an eyelash or a lip line is exactly the "melted" look this tool
        is meant to avoid.
        """
        m = poly_mask(shape, [self.poly(FACE_OVAL)])
        eyes_brows = poly_mask(shape, [
            self.poly(LEFT_EYE), self.poly(RIGHT_EYE),
            self.poly(LEFT_BROW), self.poly(RIGHT_BROW),
        ])
        lips = poly_mask(shape, [self.poly(LIPS_OUTER)])
        excl = np.clip(eyes_brows + lips, 0, 1)
        grow = max(2, int(self.width * 0.015))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
        excl = cv2.dilate(excl, k)
        m = np.clip(m - excl, 0, 1)
        f = feather_px if feather_px is not None else max(2.0, self.width * 0.03)
        return feather(m, f)

    def eye_mask(self, shape, grow: float = 0.0) -> np.ndarray:
        polys = [self.poly(LEFT_EYE), self.poly(RIGHT_EYE)]
        m = poly_mask(shape, polys)
        if grow > 0:
            g = max(1, int(self.width * grow))
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g * 2 + 1, g * 2 + 1)))
        return feather(m, max(1.5, self.width * 0.012))

    def lips_mask(self, shape, inner: bool = False) -> np.ndarray:
        m = poly_mask(shape, [self.poly(LIPS_INNER if inner else LIPS_OUTER)])
        return feather(m, max(1.5, self.width * 0.01))

    def under_eye_mask(self, shape) -> np.ndarray:
        """Band under each eye, built by dropping the lower lid downward."""
        drop = self.height * 0.085
        polys = []
        for lid in (LEFT_LOWER_LID, RIGHT_LOWER_LID):
            top = self.poly(lid)
            bottom = top.copy()
            bottom[:, 1] += drop
            polys.append(np.vstack([top, bottom[::-1]]))
        m = poly_mask(shape, polys)
        return feather(m, max(2.0, self.height * 0.03))


@dataclass
class Body:
    """Pose landmarks in pixel space, with the anchors the reshape stage uses."""
    points: np.ndarray          # (33, 2) float32
    visibility: np.ndarray      # (33,) float32
    shoulder_y: float
    hip_y: float
    waist_y: float
    centre_x: float
    shoulder_w: float

    def visible(self, idx, thresh: float = 0.5) -> bool:
        return bool(self.visibility[idx] >= thresh)


def _centroid(pts: np.ndarray) -> np.ndarray:
    return pts.mean(axis=0).astype(np.float32)


class FaceTracker:
    """
    MediaPipe face mesh, plus the three things that make it usable on video:
    per-face landmark smoothing, identity-stable slot matching, and coasting.

    Coasting matters more than it sounds. Detectors drop a frame here and
    there - a fast turn, motion blur, a hand across the face - and an effect
    stack driven straight off the raw detections switches the retouch off for
    that frame and back on for the next. That single-frame pop is far more
    visible than anything the retouch itself does. So a lost face is held for
    a few frames at decaying confidence, and a newly found one ramps up over
    a few, which turns both transitions into something nobody notices.
    """

    HOLD_FRAMES = 6      # how long a lost face is coasted before it is dropped
    RAMP_FRAMES = 5      # how long a newly acquired face takes to reach full

    def __init__(self, max_faces: int = 3, static: bool = False,
                 stabilise: bool = True, alpha: float = 0.45):
        self.ok = MEDIAPIPE_OK
        self.max_faces = int(max(1, max_faces))
        self.stabilise = bool(stabilise) and not static
        self.alpha = float(alpha)
        self._mesh = None
        self._static = static
        self._slots: list[dict] = []
        if self.ok:
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=static,
                max_num_faces=self.max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )

    def close(self):
        if self._mesh is not None:
            try:
                self._mesh.close()
            except Exception:
                pass
            self._mesh = None

    def __call__(self, bgr: np.ndarray) -> list[Face]:
        if not self.ok or self._mesh is None:
            return []
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            res = self._mesh.process(rgb)
        except Exception as e:
            log.warning("face mesh failed: %s", e)
            return []

        raw = []
        if res.multi_face_landmarks:
            for lm in res.multi_face_landmarks:
                raw.append(np.array([[p.x * w, p.y * h] for p in lm.landmark], np.float32))

        if self._static or not self.stabilise:
            return [self._make_face(pts, 1.0) for pts in raw]
        return self._track(raw)

    # -- internals ---------------------------------------------------------
    def _track(self, raw: list[np.ndarray]) -> list[Face]:
        """Match detections to slots, update or coast each slot, emit faces."""
        centroids = [_centroid(p) for p in raw]
        taken: set[int] = set()
        matched: dict[int, int] = {}

        for i, c in enumerate(centroids):
            best, best_d = None, None
            for j, slot in enumerate(self._slots):
                if j in taken:
                    continue
                d = float(np.linalg.norm(c - slot["centroid"]))
                if best_d is None or d < best_d:
                    best, best_d = j, d
            # A jump wider than a face is a different person, not movement -
            # smoothing one person's jaw toward another's is far louder than
            # the jitter the smoothing is there to remove.
            if best is not None and best_d is not None and best_d < 200.0:
                matched[i] = best
                taken.add(best)

        out: list[Face] = []
        updated: set[int] = set()
        for i, pts in enumerate(raw):
            j = matched.get(i)
            if j is None:
                slot = {"ema": EMA(self.alpha, reset_distance=24.0), "age": 0,
                        "miss": 0, "centroid": centroids[i], "pts": pts}
                self._slots.append(slot)
                j = len(self._slots) - 1
            updated.add(j)
            slot = self._slots[j]
            slot["pts"] = slot["ema"].update(pts).copy()
            slot["centroid"] = centroids[i]
            slot["age"] += 1
            slot["miss"] = 0
            out.append(self._make_face(slot["pts"], self._confidence(slot)))

        # Slots with no detection this frame: coast, then retire.
        survivors = []
        for j, slot in enumerate(self._slots):
            if j in updated:
                survivors.append(slot)
                continue
            slot["miss"] += 1
            if slot["miss"] <= self.HOLD_FRAMES:
                survivors.append(slot)
                conf = self._confidence(slot)
                if conf > 0.02:
                    out.append(self._make_face(slot["pts"], conf))
        self._slots = survivors
        return out

    def _confidence(self, slot: dict) -> float:
        ramp = min(1.0, slot["age"] / float(self.RAMP_FRAMES))
        decay = max(0.0, 1.0 - slot["miss"] / float(self.HOLD_FRAMES + 1))
        return float(ramp * decay)

    @staticmethod
    def _make_face(pts: np.ndarray, confidence: float = 1.0) -> Face:
        left, right = pts[234], pts[454]
        chin, brow = pts[CHIN], pts[FOREHEAD]
        width = float(np.linalg.norm(right - left))
        height = float(np.linalg.norm(brow - chin))
        axis = (brow - chin)
        n = float(np.linalg.norm(axis))
        axis = (axis / n) if n > 1e-3 else np.float32([0, -1])
        centre = (left + right + chin + brow) / 4.0
        return Face(points=pts, width=max(width, 1.0), height=max(height, 1.0),
                    centre=centre.astype(np.float32), axis=axis.astype(np.float32),
                    has_iris=len(pts) >= 478, confidence=float(confidence))


class PoseTracker:
    """MediaPipe pose + landmark smoothing, for body reshaping."""

    HOLD_FRAMES = 8

    def __init__(self, static: bool = False, stabilise: bool = True, alpha: float = 0.35):
        self.ok = MEDIAPIPE_OK
        self._pose = None
        self._sm = EMA(alpha, reset_distance=40.0) if (stabilise and not static) else None
        self._last: Body | None = None
        self._miss = 0
        if self.ok:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=static,
                model_complexity=1,
                smooth_landmarks=not static,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )

    def close(self):
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass
            self._pose = None

    def __call__(self, bgr: np.ndarray) -> Body | None:
        if not self.ok or self._pose is None:
            return None
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            res = self._pose.process(rgb)
        except Exception as e:
            log.warning("pose failed: %s", e)
            return None
        if not res.pose_landmarks:
            # Same reasoning as the face tracker: a body warp that vanishes
            # for one frame and comes back is a visible jolt, so a lost pose
            # is held briefly before the reshape lets go of it.
            self._miss += 1
            if self._last is not None and self._miss <= self.HOLD_FRAMES:
                return self._last
            if self._sm is not None:
                self._sm.reset()
            self._last = None
            return None
        self._miss = 0

        lm = res.pose_landmarks.landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm], np.float32)
        vis = np.array([p.visibility for p in lm], np.float32)
        if self._sm is not None:
            pts = self._sm.update(pts).copy()

        sl, sr = pts[P_SHOULDER_L], pts[P_SHOULDER_R]
        hl, hr = pts[P_HIP_L], pts[P_HIP_R]
        shoulder_y = float((sl[1] + sr[1]) / 2)
        hip_y = float((hl[1] + hr[1]) / 2)
        # The natural waist sits a little above halfway down the torso.
        waist_y = shoulder_y + (hip_y - shoulder_y) * 0.62
        centre_x = float((sl[0] + sr[0] + hl[0] + hr[0]) / 4)
        shoulder_w = float(abs(sl[0] - sr[0]))
        self._last = Body(points=pts, visibility=vis, shoulder_y=shoulder_y,
                          hip_y=hip_y, waist_y=waist_y, centre_x=centre_x,
                          shoulder_w=max(shoulder_w, 1.0))
        return self._last


class PersonSegmenter:
    """
    MediaPipe selfie segmentation with temporal averaging.

    The averaging happens on a 256-wide copy on purpose. That is the scale the
    network actually resolves, it is where the frame-to-frame noise is, and
    smoothing there costs a fraction of what smoothing a 1080p mask would.
    """

    def __init__(self, stabilise: bool = True, alpha: float = 0.4, work_width: int = 256):
        self.ok = MEDIAPIPE_OK
        self._seg = None
        self.work_width = int(work_width)
        self._sm = EMA(alpha, reset_distance=0.22) if stabilise else None
        if self.ok:
            self._seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def close(self):
        if self._seg is not None:
            try:
                self._seg.close()
            except Exception:
                pass
            self._seg = None

    def __call__(self, bgr: np.ndarray) -> np.ndarray | None:
        if not self.ok or self._seg is None:
            return None
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            res = self._seg.process(rgb)
        except Exception as e:
            log.warning("segmentation failed: %s", e)
            return None
        m = np.asarray(res.segmentation_mask, np.float32)
        small = cv2.resize(m, (self.work_width,
                               max(2, int(round(self.work_width * h / max(w, 1))))),
                           interpolation=cv2.INTER_AREA)
        if self._sm is not None:
            small = self._sm.update(small).copy()
        full = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(full, 0.0, 1.0)


class Trackers:
    """The three trackers as one object, so the pipeline opens and closes one
    thing and no MediaPipe graph is shared across threads by accident."""

    def __init__(self, settings, static: bool = False, max_faces: int = 3):
        stabilise = bool(getattr(settings, "stabilise", True)) and not static
        need_face = settings.touches_face() or settings.touches_hair()
        need_body = settings.touches_body()
        need_seg = settings.touches_body() or settings.touches_hair()
        self.face = FaceTracker(max_faces, static=static, stabilise=stabilise) if need_face else None
        self.pose = PoseTracker(static=static, stabilise=stabilise) if need_body else None
        self.seg = PersonSegmenter(stabilise=stabilise) if need_seg else None

    @property
    def available(self) -> bool:
        return MEDIAPIPE_OK

    def close(self):
        for t in (self.face, self.pose, self.seg):
            if t is not None:
                t.close()
