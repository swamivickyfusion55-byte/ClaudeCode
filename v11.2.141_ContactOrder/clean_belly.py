"""One-go belly clean (v11.2.117, SDOS-117; v11.2.140, SDOS-140: the whole belly).

Job setting ``clean_belly`` (the app's Belly clean buttons). On a few frames of the video the tattoos and marks on the uncovered skin of the
whole belly (v11.2.140: from just under the chest down to 0.10 above the hip line) are found, and each set of marks becomes one position of one
Patch Removal mark ("belly" group). The existing body-locked fill then carries it through the video. Nothing outside the band is
painted, the navel and clothing (waistband, shorts) are left alone.

The band, the pose and the mark finder are belly_clear.py (v11.2.110-112), tested on synthetic skin with MediaPipe Pose Tasks
0.10.35. This module replaces the earlier dark-hair detector of this line: on the belly test clip it covered 30-45 % of the
tattoo (its band stopped above the tattoo's lower strokes) and painted the navel.

Never raises into the job: a failure logs its reason and the job continues without the clean-up.
"""
from __future__ import annotations

import logging
import os
import tempfile

import cv2
import numpy as np

import belly_clear as BC

log = logging.getLogger(__name__)

KEY_MIN_PX = 30          # a key frame needs at least this many px of marks to be used
SAMPLES = 12             # frames looked at for the marks (spread over the video)


def _root():
    root = "/data/phoenix_patches" if os.path.isdir("/data") else os.path.join(tempfile.gettempdir(), "phoenix_patches")
    os.makedirs(root, exist_ok=True)
    return root


def build_clean_belly(video_path, max_views=3, length=1.0, width=1.0):
    """-> [{image, mask, t, group}] for the patch tracker (group "belly"), or []. Never raises."""
    if not video_path or not os.path.isfile(video_path):
        return []
    pose = None
    cap = None
    try:
        length = BC.clamp_scale(length, BC.LENGTH_RANGE)
        width = BC.clamp_scale(width, BC.WIDTH_RANGE)
        pose = BC.MediaPipePose(still=True)
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        if n < 2:
            return []
        idxs = sorted(set(int(v) for v in np.linspace(0.05 * n, 0.95 * n, SAMPLES)))
        hits, seen, body = [], 0, 0
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            seen += 1
            lm = pose.detect(frame)
            if lm is None or BC.region_polygon(lm, length, width) is None:
                continue
            body += 1
            mk, info = BC.find_marks(frame, lm, length, width)
            if mk is None or int(mk.sum()) < KEY_MIN_PX:
                continue
            hits.append((int(mk.sum()), i, frame, mk))
        if pose.error:
            log.warning("clean belly: the pose model could not be loaded (%s) - not applied", pose.error)
            return []
        log.info("clean belly: body found on %d of %d sampled frames; marks found on %d", body, seen, len(hits))
        if not hits:
            log.info("clean belly: no marks in the band on the sampled frames - the job continues without it")
            return []
        hits.sort(key=lambda h: h[0], reverse=True)
        chosen = []
        gap = max(8, n // 12)
        for area, i, frame, mk in hits:
            if all(abs(i - j) > gap for _, j, _, _ in chosen):
                chosen.append((area, i, frame, mk))
            if len(chosen) >= max_views:
                break
        if not chosen:
            chosen = hits[:1]
        out = []
        root = _root()
        for k, (area, i, frame, mk) in enumerate(chosen):
            img = os.path.join(root, "belly_%d_%d.jpg" % (os.getpid(), i))
            msk = os.path.join(root, "belly_%d_%d.png" % (os.getpid(), i))
            cv2.imwrite(img, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(msk, mk.astype(np.uint8) * 255)
            out.append({"image": img, "mask": msk, "t": i / fps, "group": "belly"})
            log.info("clean belly: position %d at frame %d (%.2f s), %d px of marks, band length x%.2f, breadth x%.2f",
                     k + 1, i, i / fps, area, length, width)
        return out
    except Exception:
        log.exception("clean belly failed; the job continues without it")
        return []
    finally:
        try:
            if cap is not None:
                cap.release()
            if pose is not None:
                pose.close()
        except Exception:
            pass


def preview_frame(img, lm, length=1.0, width=1.0, remove_fn=None):
    """One frame, for the app's preview: -> (cleaned BGR, the same with the band outlined, info dict).
    lm: the body's landmarks (None: no body found). remove_fn(img, mask_u8 0/255) -> (img, info): the still fill (LaMa, or the
    exemplar fill). Nothing is changed when no body is found."""
    length = BC.clamp_scale(length, BC.LENGTH_RANGE)
    width = BC.clamp_scale(width, BC.WIDTH_RANGE)
    info = {"body": lm is not None, "band": None, "length": length, "width": width}
    poly = BC.region_polygon(lm, length, width)
    if poly is None:
        info["reason"] = ("no body found" if lm is None
                          else "shoulders and hips not clear enough (a side view, or the head out of the frame)")
        return img, img.copy(), info
    info["band"] = [[round(float(x), 1), round(float(y), 1)] for x, y in poly]
    mk, mi = BC.find_marks(img, lm, length, width)
    info.update({k: mi[k] for k in ("band_px", "skin_px", "marks_px", "pieces", "clothing_px", "navel")})
    out = img
    if mk is not None and int(mk.sum()) >= KEY_MIN_PX and remove_fn is not None:
        out, _ = remove_fn(img, mk.astype(np.uint8) * 255)
    outlined = out.copy()
    th = max(2, int(round(max(img.shape[:2]) / 400.0)))
    cv2.polylines(outlined, [np.round(poly).astype(np.int32)], True, (255, 200, 40), th, cv2.LINE_AA)
    return out, outlined, info
