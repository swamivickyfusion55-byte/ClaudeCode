"""
End-to-end check that does not need a video file.

It builds a short synthetic clip with a real face in it (a moving crop of a
public-domain photo when scikit-image is available, a drawn stand-in
otherwise), renders it, and then reports the two things that actually decide
whether this install works on this machine:

  * how often the face tracker held on to the subject, and
  * how much frame-to-frame change the render added over the source.

The second one is the flicker test. An effect stack can be perfect on stills
and still be unusable on video; a ratio near 1 means the temporal smoothing is
doing its job.
"""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np

import cv2

from .pipeline import capability_report, render_video
from .settings import PRESETS


def _source_image() -> np.ndarray:
    try:
        from skimage import data
        img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
        return cv2.copyMakeBorder(img, 120, 120, 200, 200, cv2.BORDER_REFLECT)
    except Exception:
        # No sample data available: a drawn stand-in still exercises the grade,
        # the encoder and the temporal path, just not the face tracker.
        img = np.full((720, 720, 3), 40, np.uint8)
        cv2.circle(img, (360, 330), 150, (160, 180, 210), -1)
        cv2.circle(img, (305, 300), 18, (255, 255, 255), -1)
        cv2.circle(img, (415, 300), 18, (255, 255, 255), -1)
        cv2.ellipse(img, (360, 400), (60, 30), 0, 0, 180, (90, 90, 140), -1)
        return img


def _make_clip(path: str, frames: int = 36, size=(640, 480)) -> None:
    big = cv2.resize(_source_image(), (1152, 1024), interpolation=cv2.INTER_CUBIC)
    w, h = size
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    rng = np.random.default_rng(0)
    for i in range(frames):
        z = 0.62 * (1.0 + 0.04 * np.sin(i / 12.0))
        cx, cy = 576 + 25 * np.sin(i / 18.0), 470 + 12 * np.cos(i / 15.0)
        m = cv2.getRotationMatrix2D((cx, cy), 1.2 * np.sin(i / 20.0), z)
        m[0, 2] += w / 2 - cx
        m[1, 2] += h / 2 - cy
        f = cv2.warpAffine(big, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        f = np.clip(f.astype(np.float32) + rng.normal(0, 2.5, f.shape), 0, 255).astype(np.uint8)
        vw.write(f)
    vw.release()


def _read(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f.astype(np.float32))
    cap.release()
    return out


def run() -> int:
    print(capability_report())
    tmp = tempfile.mkdtemp(prefix="beauty_selftest_")
    src = os.path.join(tmp, "in.mp4")
    _make_clip(src)
    print(f"synthetic clip: {src}")

    t = time.time()
    res = render_video(src, PRESETS["Professional Portrait"],
                       out_path=os.path.join(tmp, "out.mp4"))
    print(f"rendered {res['frames']} frames in {time.time() - t:.1f}s "
          f"({res['fps']:.1f} fps) -> {res['path']}")

    face_pct = 100.0 * res["faces_seen"] / max(res["frames_seen"], 1)
    print(f"face tracked on {face_pct:.0f}% of frames")

    a, b = _read(src), _read(res["path"])
    n = min(len(a), len(b))
    ok = True
    if n > 2:
        din = [np.abs(a[i] - a[i - 1]).mean() for i in range(1, n)]
        dout = [np.abs(b[i] - b[i - 1]).mean() for i in range(1, n)]
        ratio = float(np.mean(dout) / max(np.mean(din), 1e-3))
        worst = max(o / max(i_, 1e-3) for o, i_ in zip(dout, din))
        print(f"temporal stability: mean ratio {ratio:.2f}, worst frame {worst:.2f} "
              f"(1.0 = the render adds no motion of its own)")
        if worst > 2.0:
            print("  WARNING: a frame changed far more than its source did - "
                  "check that stabilisation is on")
            ok = False
    for note in res["notes"]:
        print(f"note: {note}")
    print("SELFTEST PASSED" if ok else "SELFTEST FINISHED WITH WARNINGS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
