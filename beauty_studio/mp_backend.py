"""
One interface over the two MediaPipe generations.

MediaPipe 1.0 removed the legacy `mp.solutions` API that every tutorial (and
the first version of this app) was written against. A Space that installs
`mediapipe` unpinned now gets 1.x, and the old calls fail at the first frame
with `module 'mediapipe' has no attribute 'solutions'`.

So detection lives behind three small classes here, each of which speaks
whichever API is actually installed:

    solutions  - mediapipe 0.10.x: models ship inside the wheel.
    tasks      - mediapipe 1.x: models are downloaded once and cached.

Everything above this file works in normalised (0..1) landmark coordinates and
never learns which generation answered. If neither works - no MediaPipe, or
1.x with no way to fetch its models - `MEDIAPIPE_OK` is False and `NOTE` says
why in a sentence a user can act on; the app then still grades video, it just
cannot retouch or reshape.
"""
from __future__ import annotations

import logging
import os
import tempfile
import urllib.request

import numpy as np

log = logging.getLogger(__name__)

BACKEND = "none"        # "solutions" | "tasks" | "none"
NOTE = "mediapipe not loaded"
MEDIAPIPE_OK = False
VERSION = "?"

mp = None
_vision = None
_mpt = None

# Model assets for the tasks backend. Pinned to specific published versions so
# a Space that redeploys a year from now gets the same weights it was tested
# against.
MODEL_URLS = {
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "segment": "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
               "selfie_segmenter/float16/1/selfie_segmenter.tflite",
}


def _detect_backend():
    global mp, _vision, _mpt, BACKEND, NOTE, MEDIAPIPE_OK, VERSION
    try:
        import mediapipe as _mp
    except Exception as e:
        BACKEND, NOTE = "none", f"MediaPipe is not installed ({e}) - grade only"
        return
    mp = _mp
    VERSION = getattr(mp, "__version__", "?")

    # Legacy first: when it is present it needs no downloads and no network.
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "face_mesh"):
        BACKEND, NOTE, MEDIAPIPE_OK = "solutions", f"MediaPipe {VERSION}", True
        return

    try:
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
    except Exception as e:
        BACKEND = "none"
        NOTE = (f"MediaPipe {VERSION} has neither the solutions API nor a usable "
                f"tasks API ({e}) - grade only")
        return
    _mpt, _vision = mpt, vision
    BACKEND, NOTE, MEDIAPIPE_OK = "tasks", f"MediaPipe {VERSION} (tasks API)", True


_detect_backend()


# ------------------------------------------------------------- model caching

def _cache_dir() -> str:
    """First writable candidate. Spaces mount different things read-only
    depending on how they are built, so this tries rather than assumes."""
    candidates = [
        os.environ.get("BEAUTY_STUDIO_MODELS"),
        os.path.join(os.environ["HF_HOME"], "beauty_studio") if os.environ.get("HF_HOME") else None,
        os.path.join(os.path.expanduser("~"), ".cache", "beauty_studio", "models"),
        os.path.join(tempfile.gettempdir(), "beauty_studio_models"),
    ]
    for path in candidates:
        if not path:
            continue
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".writable")
            with open(probe, "w") as fh:
                fh.write("1")
            os.remove(probe)
            return path
        except Exception:
            continue
    raise RuntimeError("no writable directory for the MediaPipe model cache")


def model_file(key: str) -> str:
    """Path to a cached model, downloading it on first use."""
    url = MODEL_URLS[key]
    path = os.path.join(_cache_dir(), os.path.basename(url))
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        return path
    tmp = path + ".part"
    log.info("downloading MediaPipe %s model (first run only): %s", key, url)
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    # Rename only once the bytes are all there: an interrupted download that
    # left a short file in place would fail every later run with a confusing
    # model-parse error instead of just downloading again.
    os.replace(tmp, path)
    log.info("cached %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
    return path


def _disable(reason: str) -> None:
    global MEDIAPIPE_OK, NOTE
    MEDIAPIPE_OK = False
    NOTE = reason
    log.warning("face/body features disabled: %s", reason)


def _rgb_image(rgb: np.ndarray):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class _TasksClock:
    """Monotonic millisecond timestamps for the tasks API's video mode, which
    rejects a timestamp that does not advance."""

    def __init__(self, step_ms: int = 33):
        self.t = 0
        self.step = step_ms

    def next(self) -> int:
        self.t += self.step
        return self.t


# ------------------------------------------------------------------- face

class FaceBackend:
    """Detect faces; return a list of (N, 2) arrays of normalised landmarks."""

    def __init__(self, max_faces: int = 3, static: bool = False, confidence: float = 0.4):
        self.ok = MEDIAPIPE_OK
        self.static = bool(static)
        self._impl = None
        self._clock = _TasksClock()
        if not self.ok:
            return
        try:
            if BACKEND == "solutions":
                self._impl = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=static, max_num_faces=int(max_faces),
                    refine_landmarks=True, min_detection_confidence=confidence,
                    min_tracking_confidence=confidence)
            else:
                mode = _vision.RunningMode.IMAGE if static else _vision.RunningMode.VIDEO
                self._impl = _vision.FaceLandmarker.create_from_options(
                    _vision.FaceLandmarkerOptions(
                        base_options=_mpt.BaseOptions(model_asset_path=model_file("face")),
                        running_mode=mode, num_faces=int(max_faces),
                        min_face_detection_confidence=confidence,
                        min_face_presence_confidence=confidence,
                        min_tracking_confidence=confidence))
        except Exception as e:
            self.ok = False
            _disable(f"could not start the face model ({e}) - grade only")

    def detect(self, rgb: np.ndarray) -> list[np.ndarray]:
        if not self.ok or self._impl is None:
            return []
        try:
            if BACKEND == "solutions":
                rgb.flags.writeable = False
                res = self._impl.process(rgb)
                if not res.multi_face_landmarks:
                    return []
                return [np.array([[p.x, p.y] for p in lm.landmark], np.float32)
                        for lm in res.multi_face_landmarks]
            img = _rgb_image(rgb)
            res = (self._impl.detect(img) if self.static
                   else self._impl.detect_for_video(img, self._clock.next()))
            return [np.array([[p.x, p.y] for p in face], np.float32)
                    for face in (res.face_landmarks or [])]
        except Exception as e:
            log.warning("face detection failed: %s", e)
            return []

    def close(self):
        if self._impl is not None:
            try:
                self._impl.close()
            except Exception:
                pass
            self._impl = None


# ------------------------------------------------------------------- pose

class PoseBackend:
    """Return (normalised (33, 2) points, (33,) visibility) or None."""

    def __init__(self, static: bool = False, confidence: float = 0.4):
        self.ok = MEDIAPIPE_OK
        self.static = bool(static)
        self._impl = None
        self._clock = _TasksClock()
        if not self.ok:
            return
        try:
            if BACKEND == "solutions":
                self._impl = mp.solutions.pose.Pose(
                    static_image_mode=static, model_complexity=1,
                    smooth_landmarks=not static, min_detection_confidence=confidence,
                    min_tracking_confidence=confidence)
            else:
                mode = _vision.RunningMode.IMAGE if static else _vision.RunningMode.VIDEO
                self._impl = _vision.PoseLandmarker.create_from_options(
                    _vision.PoseLandmarkerOptions(
                        base_options=_mpt.BaseOptions(model_asset_path=model_file("pose")),
                        running_mode=mode, num_poses=1,
                        min_pose_detection_confidence=confidence,
                        min_pose_presence_confidence=confidence,
                        min_tracking_confidence=confidence))
        except Exception as e:
            self.ok = False
            log.warning("pose model unavailable (%s) - body shaping is off", e)

    def detect(self, rgb: np.ndarray):
        if not self.ok or self._impl is None:
            return None
        try:
            if BACKEND == "solutions":
                rgb.flags.writeable = False
                res = self._impl.process(rgb)
                if not res.pose_landmarks:
                    return None
                lm = res.pose_landmarks.landmark
                return (np.array([[p.x, p.y] for p in lm], np.float32),
                        np.array([p.visibility for p in lm], np.float32))
            img = _rgb_image(rgb)
            res = (self._impl.detect(img) if self.static
                   else self._impl.detect_for_video(img, self._clock.next()))
            if not res.pose_landmarks:
                return None
            lm = res.pose_landmarks[0]
            return (np.array([[p.x, p.y] for p in lm], np.float32),
                    np.array([getattr(p, "visibility", 1.0) for p in lm], np.float32))
        except Exception as e:
            log.warning("pose detection failed: %s", e)
            return None

    def close(self):
        if self._impl is not None:
            try:
                self._impl.close()
            except Exception:
                pass
            self._impl = None


# --------------------------------------------------------------- segmenter

class SegmentBackend:
    """Return a HxW float32 person-probability mask at the input resolution."""

    def __init__(self, static: bool = False):
        self.ok = MEDIAPIPE_OK
        self.static = bool(static)
        self._impl = None
        self._clock = _TasksClock()
        if not self.ok:
            return
        try:
            if BACKEND == "solutions":
                self._impl = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            else:
                mode = _vision.RunningMode.IMAGE if static else _vision.RunningMode.VIDEO
                self._impl = _vision.ImageSegmenter.create_from_options(
                    _vision.ImageSegmenterOptions(
                        base_options=_mpt.BaseOptions(model_asset_path=model_file("segment")),
                        running_mode=mode, output_category_mask=False,
                        output_confidence_masks=True))
        except Exception as e:
            self.ok = False
            log.warning("segmentation model unavailable (%s) - hair and body work is off", e)

    def detect(self, rgb: np.ndarray) -> np.ndarray | None:
        if not self.ok or self._impl is None:
            return None
        try:
            if BACKEND == "solutions":
                rgb.flags.writeable = False
                res = self._impl.process(rgb)
                return np.asarray(res.segmentation_mask, np.float32)
            img = _rgb_image(rgb)
            res = (self._impl.segment(img) if self.static
                   else self._impl.segment_for_video(img, self._clock.next()))
            masks = res.confidence_masks or []
            if not masks:
                return None
            # The selfie segmenter publishes the foreground confidence last;
            # where a model also emits a background channel it comes first.
            m = np.asarray(masks[-1].numpy_view(), np.float32)
            return m[:, :, 0] if m.ndim == 3 else m
        except Exception as e:
            log.warning("segmentation failed: %s", e)
            return None

    def close(self):
        if self._impl is not None:
            try:
                self._impl.close()
            except Exception:
                pass
            self._impl = None


def mediapipe_status() -> str:
    """One line for the UI header and the CLI banner."""
    return NOTE


def mediapipe_ready() -> bool:
    """Live answer, not a copy: a backend can disable itself at first use."""
    return MEDIAPIPE_OK


def diagnostics() -> str:
    """Everything needed to work out why face features are off, in one block.

    A Space is awkward to debug from the outside - this is what turns "it says
    grade only" into a fix without a round trip through the logs.
    """
    import platform
    lines = [
        f"python        {platform.python_version()} ({platform.machine()})",
        f"mediapipe     {VERSION}",
        f"backend       {BACKEND}",
        f"status        {NOTE}",
        f"face features {'ON' if MEDIAPIPE_OK else 'OFF'}",
    ]
    if BACKEND == "tasks":
        try:
            cache = _cache_dir()
            lines.append(f"model cache   {cache}")
            for key, url in MODEL_URLS.items():
                path = os.path.join(cache, os.path.basename(url))
                have = os.path.exists(path) and os.path.getsize(path) > 10_000
                lines.append(f"  {key:<11} {'cached' if have else 'will download'}  "
                             f"{os.path.basename(url)}")
        except Exception as e:
            lines.append(f"model cache   UNAVAILABLE ({e})")
    return "\n".join(lines)
