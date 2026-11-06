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


def build_clean_belly(video_path, max_views=3, length=1.0, width=1.0, pentagon_body=None, marks_body=None):
    """-> [{image, mask, t, group}] for the patch tracker (group "belly"), or []. Never raises.
    v11.2.142: pentagon_body (5 x [u, t], body coordinates - the app's adjusted pentagon): on every sampled frame the marks are
    looked for only inside it, laid on that frame's body by its shoulders and hips. marks_body (the app's belly_marks: [u, t, taken,
    px] per mark the last preview found): each of those marks gets the preview's decision on every frame."""
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
        pent_ut = None if pentagon_body is None else BC.clamp_pentagon_body(pentagon_body, length, width)
        found = []                                       # with a pentagon: (i, frame, lm, all band marks) - decided below
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
            if mk is None or not mk.any():
                continue
            if pent_ut is not None:
                found.append((i, frame, lm, mk))
                continue
            if int(mk.sum()) < KEY_MIN_PX:
                continue
            hits.append((int(mk.sum()), i, frame, mk))
        if pent_ut is not None and found:
            hits = _pentagon_hits(found, pent_ut, marks_body)
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


def _pentagon_hits(found, pent_ut, marks_body=None):
    """v11.2.142: the app's pentagon, decided ONCE per mark for the whole video. Each piece of marks on each sampled frame is matched
    to the same piece on the other frames by where it lies on the body (u, t). A mark the last preview found (marks_body) gets the
    preview's decision - what the preview showed is what the video gets, also for a mark on the pentagon's line; any other mark is
    taken when its middle, averaged over the frames it is seen on, lies inside the pentagon. Either way the decision holds on EVERY
    key frame: a mark is never covered in one part of the video and left in another, whatever the pose's few px do to the line.
    found: [(i, frame, lm, band marks)] -> hits [(area, i, frame, marks)]"""
    su = float(np.median([np.linalg.norm(BC.body_affine(lm)[:, 0]) for _, _, lm, _ in found]))     # px per u (one hip width)
    st = float(np.median([np.linalg.norm(BC.body_affine(lm)[:, 1]) for _, _, lm, _ in found]))     # px per t (hips to shoulders)
    marks_ = []                                                       # [[sum u, sum t, n, votes in]] - one per physical mark
    per = []
    for i, frame, lm, mk in found:
        lab, pieces = BC.mark_pieces(mk, lm)
        asg = []
        for k, ut, area in sorted(pieces, key=lambda p: -p[2]):
            r = max(12.0, 0.6 * float(np.sqrt(area)))
            best, bd = None, None
            for j, m in enumerate(marks_):
                du, dt = ut[0] - m[0] / m[2], ut[1] - m[1] / m[2]
                dd = float(np.hypot(du * su, dt * st))
                if dd <= r and (bd is None or dd < bd):
                    best, bd = j, dd
            if best is None:
                marks_.append([0.0, 0.0, 0, 0])
                best = len(marks_) - 1
            m = marks_[best]
            m[0] += float(ut[0]); m[1] += float(ut[1]); m[2] += 1; m[3] += int(BC.in_pentagon_body(ut, pent_ut))
            asg.append((k, best))
        per.append((i, frame, lab, asg))
    take, by_preview = [], 0
    for m in marks_:
        mu, mt = m[0] / m[2], m[1] / m[2]
        dec = None
        if marks_body is not None:
            best, bd = None, None
            for row in np.asarray(marks_body, np.float64):
                dd = float(np.hypot((row[0] - mu) * su, (row[1] - mt) * st))
                r = max(12.0, 0.6 * float(np.sqrt(max(float(row[3]), 1.0))))
                if dd <= r and (bd is None or dd < bd):
                    best, bd = row, dd
            if best is not None:
                dec = bool(best[2] >= 0.5)
                by_preview += 1
        take.append(dec if dec is not None else BC.in_pentagon_body((mu, mt), pent_ut))
    hits = []
    for i, frame, lab, asg in per:
        keep = np.zeros(int(lab.max()) + 1, bool)
        for k, j in asg:
            keep[k] = take[j]
        mk = keep[lab]
        if int(mk.sum()) >= KEY_MIN_PX:
            hits.append((int(mk.sum()), i, frame, mk))
    log.info("clean belly: the app's pentagon - %d of %d marks taken (%d as the preview decided, the rest by their place over %d "
             "sampled frames)", sum(take), len(take), by_preview, len(found))
    return hits


def preview_frame(img, lm, length=1.0, width=1.0, remove_fn=None, pentagon_px=None, pentagon_body=None):
    """One frame, for the app's preview: -> (cleaned BGR, the same with the pentagon and the band outlined, info dict).
    lm: the body's landmarks (None: no body found). remove_fn(img, mask_u8 0/255) -> (img, info): the still fill (LaMa, or the
    exemplar fill). Nothing is changed when no body is found.
    v11.2.142: the pentagon. pentagon_px (5 x [x, y] on this frame, the app's dragged corners) or pentagon_body (5 x [u, t]);
    without either, the automatic pentagon around the marks found in the band (or a default one below the navel). Its corners
    are kept on the belly (above the floor, below the chest); only the marks inside it are filled. info["pentagon"] (px) and
    info["pentagon_body"] (u, t) are what the app draws and sends back with the job."""
    length = BC.clamp_scale(length, BC.LENGTH_RANGE)
    width = BC.clamp_scale(width, BC.WIDTH_RANGE)
    info = {"body": lm is not None, "band": None, "length": length, "width": width}
    poly = BC.region_polygon(lm, length, width)
    if poly is None:
        info["reason"] = ("no body found" if lm is None
                          else "shoulders and hips not clear enough (a side view, or the head out of the frame)")
        return img, img.copy(), info
    info["band"] = [[round(float(x), 1), round(float(y), 1)] for x, y in poly]
    if pentagon_px is None and pentagon_body is not None:
        pentagon_px = BC.from_body(BC.clamp_pentagon_body(pentagon_body, length, width), lm)
        info["pentagon_source"] = "app (body)"
    elif pentagon_px is not None:
        info["pentagon_source"] = "app"
    if pentagon_px is None:
        mk_all, _ = BC.find_marks(img, lm, length, width)
        pentagon_px = BC.auto_pentagon(mk_all, lm)
        info["pentagon_source"] = "auto" if (mk_all is not None and int(mk_all.sum()) >= 6) else "default"
    pent, pent_ut = BC.pentagon_info(pentagon_px, lm, length, width)
    if pent is not None:
        info["pentagon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in pent]
    if pent_ut is not None:
        info["pentagon_body"] = [[round(float(u), 4), round(float(v), 4)] for u, v in pent_ut]
    mk, mi = BC.find_marks(img, lm, length, width, region=pent)
    info.update({k: mi[k] for k in ("band_px", "skin_px", "marks_px", "pieces", "clothing_px", "navel")})
    if pent is not None:                         # each mark of the band: where it lies on the body, and whether it was taken (the job
        mk_band, _ = BC.find_marks(img, lm, length, width)   # gives it the same decision on every frame - app's belly_marks)
        if mk_band is not None and mk_band.any():
            lab, pcs = BC.mark_pieces(mk_band, lm)
            info["marks_body"] = [[round(float(ut[0]), 4), round(float(ut[1]), 4), int(bool((mk & (lab == k)).any())), int(a)]
                                  for k, ut, a in pcs[:200]]
    out = img
    if mk is not None and int(mk.sum()) >= KEY_MIN_PX and remove_fn is not None:
        out, _ = remove_fn(img, mk.astype(np.uint8) * 255)
    outlined = out.copy()
    th = max(2, int(round(max(img.shape[:2]) / 400.0)))
    cv2.polylines(outlined, [np.round(poly).astype(np.int32)], True, (160, 160, 160), max(1, th // 2), cv2.LINE_AA)   # the band
    if pent is not None:                                                                                              # the pentagon
        cv2.polylines(outlined, [np.round(pent).astype(np.int32)], True, (255, 200, 40), th, cv2.LINE_AA)
        for x, y in pent:
            cv2.circle(outlined, (int(round(x)), int(round(y))), th * 2 + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(outlined, (int(round(x)), int(round(y))), th * 2, (255, 200, 40), -1, cv2.LINE_AA)
    return out, outlined, info
