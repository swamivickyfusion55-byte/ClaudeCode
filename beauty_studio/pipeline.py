"""
Frame orchestration, video rendering and encoding.

The per-frame order is deliberate:

    track -> retouch -> hair -> reshape (one warp) -> grade

Geometry goes near the end because every warp resamples the picture, and
resampling a frame you have already sharpened throws away the sharpening. The
grade goes last because it is a grade: it should see the final image, and its
output sharpening then recovers what the warp's interpolation cost.

Tracking happens once per frame and is shared by all four stages. That is
where the time goes on CPU, and running the face mesh three times because three
stages want landmarks is the easiest way to make this three times slower than
it needs to be.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

import numpy as np

import cv2

from .grade import Grader
from .hair import enhance_hair
from .imaging import even, to_float, to_u8
from .landmarks import MEDIAPIPE_OK, Trackers
from .reshape import BodyProfiler, WarpField, add_body_reshape, add_face_reshape
from .retouch import Retoucher
from .settings import Settings, scale_person_amounts

log = logging.getLogger(__name__)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


# --------------------------------------------------------------------- frames

def _scale_for_naturalness(s: Settings) -> Settings:
    """
    Fold `naturalness` into the individual amounts instead of cross-fading the
    finished frame with the original.

    A cross-fade would be wrong in two ways: it would dilute the grade, which
    is not a "retouch" the user is asking to hold back, and blending a warped
    frame with an unwarped one produces a double edge on every reshaped
    contour - the ghosting you see in cheap body-slimming apps.
    """
    return scale_person_amounts(s, s.naturalness, relax_texture=True)


class FrameProcessor:
    """
    One render's worth of state: trackers, the grader's exposure memory and the
    body profiler. Not thread-safe - MediaPipe graphs are not - so give each
    job its own.
    """

    def __init__(self, settings: Settings, static: bool = False, max_faces: int = 3):
        self.raw = settings.normalised()
        self.s = _scale_for_naturalness(self.raw)
        self.static = bool(static)
        self.trackers = Trackers(self.s, static=static, max_faces=max_faces)
        self.grader = Grader(stabilise=self.s.stabilise and not static)
        self.retoucher = Retoucher()
        self.profiler = BodyProfiler(stabilise=self.s.stabilise and not static)
        self.frames_with_face = 0
        self.frames_seen = 0

    def close(self):
        self.trackers.close()

    # ------------------------------------------------------------------ main
    def process(self, frame_u8: np.ndarray) -> np.ndarray:
        """uint8 BGR in, uint8 BGR out, at the same size."""
        s = self.s
        img = to_float(frame_u8)
        self.frames_seen += 1

        faces, body, person = [], None, None
        if self.trackers.face is not None:
            faces = self.trackers.face(frame_u8)
            if faces:
                self.frames_with_face += 1
        if self.trackers.pose is not None:
            body = self.trackers.pose(frame_u8)
        if self.trackers.seg is not None:
            person = self.trackers.seg(frame_u8)

        if faces and s.touches_face():
            img = self.retoucher.apply(img, faces, s)
        if faces and s.touches_hair():
            img = enhance_hair(img, faces, person, body, s)

        if (faces and (s.face_slim or s.chin_shape or s.nose_slim or s.eye_enlarge)) \
                or (person is not None and s.touches_body()):
            h, w = img.shape[:2]
            field = WarpField(w, h)
            for f in faces:
                add_face_reshape(field, f, s)
            add_body_reshape(field, person, body, self.profiler, s, img.shape)
            img = field.apply(img)

        img = self.grader.apply(img, s)
        return to_u8(img)

    def process_pair(self, frame_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(original, processed) - for the before/after preview."""
        return frame_u8, self.process(frame_u8)


def _work_size(w: int, h: int, long_edge: int) -> tuple[int, int]:
    """Working/output size: capped at `long_edge`, never upscaled, even."""
    if long_edge <= 0 or max(w, h) <= long_edge:
        return even(w), even(h)
    scale = long_edge / float(max(w, h))
    return even(int(round(w * scale))), even(int(round(h * scale)))


def process_image(frame_u8: np.ndarray, settings: Settings) -> np.ndarray:
    """One-shot photo path. Static tracking (no temporal smoothing to do)."""
    s = settings.normalised()
    h, w = frame_u8.shape[:2]
    ow, oh = _work_size(w, h, s.out_long_edge or s.process_scale)
    if (ow, oh) != (w, h):
        frame_u8 = cv2.resize(frame_u8, (ow, oh), interpolation=cv2.INTER_AREA)
    fp = FrameProcessor(s, static=True)
    try:
        return fp.process(frame_u8)
    finally:
        fp.close()


# ---------------------------------------------------------------------- video

@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    has_audio: bool

    @property
    def label(self) -> str:
        return (f"{self.width}x{self.height} · {self.fps:.2f} fps · "
                f"{self.frames} frames · {self.duration:.1f}s"
                f" · {'audio' if self.has_audio else 'no audio'}")


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {os.path.basename(path)}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or fps > 240:
        fps = 30.0
    duration = n / fps if n > 0 else 0.0
    return VideoInfo(path=path, width=w, height=h, fps=fps, frames=max(n, 0),
                     duration=duration, has_audio=_has_audio(path))


def _has_audio(path: str) -> bool:
    if not FFPROBE:
        return False
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "json", path],
            capture_output=True, text=True, timeout=30)
        return bool(json.loads(out.stdout or "{}").get("streams"))
    except Exception:
        return False


def grab_frame(path: str, position: float = 0.35) -> np.ndarray | None:
    """Grab a frame at `position` (0..1) through the file, for previewing."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.clip(position, 0, 0.999) * (n - 1)))
        ok, frame = cap.read()
        if not ok:
            # Seeking fails on some containers; fall back to the first frame.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _open_writer(path: str, fps: float, size) -> cv2.VideoWriter | None:
    """Prefer a real H.264 stream; fall back to MPEG-4 part 2 if the OpenCV
    build has no H.264 encoder. The ffmpeg finish pass fixes either one up."""
    for tag in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*tag), fps, size)
        if writer.isOpened():
            log.info("encoder: %s", tag)
            return writer
        writer.release()
    return None


class Cancelled(Exception):
    pass


def render_video(src: str, settings: Settings, out_path: str | None = None,
                 start: float = 0.0, end: float = 1.0,
                 progress=None, should_cancel=None, max_faces: int = 3) -> dict:
    """
    Render `src` with `settings` and return a report dict.

    `progress(done, total, message)` is called as it goes; `should_cancel()`
    is polled each frame so the UI can stop a long render.
    """
    s = settings.normalised()
    info = probe(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError("could not open the video file")

    ow, oh = _work_size(info.width, info.height, s.out_long_edge or s.process_scale)
    total = info.frames if info.frames > 0 else 0
    f0 = int(max(0.0, min(start, 1.0)) * total) if total else 0
    f1 = int(max(0.0, min(end, 1.0)) * total) if total else 0
    if total and f1 <= f0:
        f1 = total
    n_out = (f1 - f0) if total else 0
    if f0 > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    tmp_dir = tempfile.mkdtemp(prefix="beauty_")
    silent = os.path.join(tmp_dir, "render.mp4")
    writer = _open_writer(silent, info.fps, (ow, oh))
    if writer is None:
        cap.release()
        raise RuntimeError("no usable video encoder in this OpenCV build")

    fp = FrameProcessor(s, static=False, max_faces=max_faces)
    t0 = time.time()
    done = 0
    try:
        while True:
            if total and done >= n_out:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if should_cancel is not None and should_cancel():
                raise Cancelled()
            if (frame.shape[1], frame.shape[0]) != (ow, oh):
                interp = cv2.INTER_AREA if frame.shape[1] > ow else cv2.INTER_CUBIC
                frame = cv2.resize(frame, (ow, oh), interpolation=interp)
            writer.write(fp.process(frame))
            done += 1
            if progress is not None and (done % 3 == 0 or done == 1):
                rate = done / max(time.time() - t0, 1e-3)
                eta = (n_out - done) / rate if n_out and rate > 0 else 0
                progress(done, n_out or done,
                         f"{done}/{n_out or '?'} frames · {rate:.1f} fps · ETA {eta:.0f}s")
    finally:
        writer.release()
        cap.release()
        fp.close()

    elapsed = time.time() - t0
    final = out_path or os.path.join(tmp_dir, "beauty_studio_output.mp4")
    audio_src = src if info.has_audio else None
    encoded, notes = _finish(silent, final, audio_src, s, info.fps, start, end)

    return {
        "path": encoded,
        "frames": done,
        "seconds": elapsed,
        "fps": done / max(elapsed, 1e-3),
        "size": (ow, oh),
        "faces_seen": fp.frames_with_face,
        "frames_seen": fp.frames_seen,
        "notes": notes,
    }


def _finish(silent: str, final: str, audio_src: str | None, s: Settings,
            fps: float, start: float, end: float) -> tuple[str, list[str]]:
    """
    Re-encode to browser-safe H.264 and bring the original audio back.

    OpenCV writes video and only video, so without this pass every render
    would come back silent - which for a talking-head clip makes the whole
    tool useless. When ffmpeg is missing the render still returns, with the
    limitation stated rather than hidden.
    """
    notes: list[str] = []
    if not FFMPEG:
        notes.append("ffmpeg not found - output is re-encoded by OpenCV and has no audio.")
        try:
            shutil.copyfile(silent, final)
            return final, notes
        except Exception:
            return silent, notes

    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", silent]
    if audio_src:
        # Trim the audio to match the frame range that was actually rendered.
        if start > 0 or end < 1:
            dur_cmd = []
            info = probe(audio_src)
            if info.duration > 0:
                ss = info.duration * start
                t = info.duration * (end - start)
                dur_cmd = ["-ss", f"{ss:.3f}", "-t", f"{t:.3f}"]
            cmd += dur_cmd
        cmd += ["-i", audio_src, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]

    if s.hdr10:
        ok, hdr_args = _hdr10_args()
        if ok:
            cmd += hdr_args
            notes.append("HDR10 export: BT.2020 + PQ, tagged for HDR playback (experimental).")
        else:
            notes.append("HDR10 export needs ffmpeg with libx265 and zscale - "
                         "exported as SDR instead.")
            cmd += ["-c:v", "libx264", "-crf", str(int(s.quality)), "-preset", "medium",
                    "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "libx264", "-crf", str(int(s.quality)), "-preset", "medium",
                "-pix_fmt", "yuv420p"]

    cmd += ["-movflags", "+faststart", "-shortest", final]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0 or not os.path.exists(final):
            log.warning("ffmpeg finish failed: %s", (r.stderr or "")[-400:])
            notes.append("ffmpeg re-encode failed - returning the raw render.")
            return silent, notes
    except Exception as e:
        log.warning("ffmpeg finish error: %s", e)
        notes.append(f"ffmpeg re-encode error ({e}) - returning the raw render.")
        return silent, notes
    return final, notes


def _hdr10_args() -> tuple[bool, list[str]]:
    """
    Arguments for a real HDR10 file: BT.709 SDR converted to BT.2020 primaries
    and the PQ transfer function, tagged with the mastering metadata a display
    needs to switch into HDR mode.

    This is an inverse tone map of SDR material, not recovered highlight data -
    the picture gains the HDR container and the wider gamut, not detail that
    was never in the source. Only offered when the local ffmpeg can actually do
    it, because a mis-tagged file looks badly wrong rather than merely flat.
    """
    if not FFMPEG:
        return False, []
    try:
        enc = subprocess.run([FFMPEG, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30).stdout
        filt = subprocess.run([FFMPEG, "-hide_banner", "-filters"],
                              capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return False, []
    if "libx265" not in enc or "zscale" not in filt:
        return False, []
    vf = ("zscale=t=linear:npl=100,format=gbrpf32le,"
          "zscale=p=bt2020:m=bt2020nc:t=smpte2084:r=tv,format=yuv420p10le")
    x265 = ("hdr10=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
            "master-display=G(8500,39850)B(6550,2300)R(35400,14600)"
            "WP(15635,16450)L(10000000,1):max-cll=1000,400")
    return True, ["-vf", vf, "-c:v", "libx265", "-crf", "20", "-preset", "medium",
                  "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-x265-params", x265]


def capability_report() -> str:
    """One line the UI shows on start, so a missing dependency is visible
    before a ten-minute render rather than after it."""
    bits = [f"OpenCV {cv2.__version__}"]
    bits.append("MediaPipe ready" if MEDIAPIPE_OK else "MediaPipe MISSING (grade only)")
    bits.append("ffmpeg ready" if FFMPEG else "ffmpeg MISSING (no audio in output)")
    return " · ".join(bits)
