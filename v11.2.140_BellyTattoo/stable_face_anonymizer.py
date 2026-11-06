"""
Phoenix stable face-track utility (privacy/anonymisation use).

This module deliberately does not connect to the identity-replacement pipeline.
It accepts face detections (x1, y1, x2, y2) and returns temporally stabilised
tracks suitable for consistent blur/pixelation masks.

No face recognition or identity inference is performed. Track IDs are temporary
within a video and are based only on geometric continuity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math

Box = Tuple[float, float, float, float]


def _clip_box(box: Sequence[float], width: int, height: int) -> Optional[Box]:
    if len(box) != 4:
        return None
    x1, y1, x2, y2 = map(float, box)
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    bb = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _center(box: Box) -> Tuple[float, float]:
    return ((box[0]+box[2])*0.5, (box[1]+box[3])*0.5)


def _smooth(old: Box, new: Box, alpha: float) -> Box:
    return tuple((1.0-alpha)*a + alpha*b for a, b in zip(old, new))  # type: ignore


@dataclass
class FaceTrack:
    track_id: int
    box: Box
    velocity: Tuple[float, float, float, float]
    age: int = 1
    missed: int = 0
    confidence: float = 1.0


class StableFaceAnonymizerTracker:
    """
    Geometry-only tracker for stable anonymisation regions.

    update(detections, frame_shape) returns active FaceTrack objects.
    detections are iterable of (x1,y1,x2,y2) boxes in pixel coordinates.
    frame_shape is (height, width) or an image-like shape.
    """
    def __init__(self, max_missed: int = 8, smooth_alpha: float = 0.65,
                 min_iou: float = 0.02, max_center_distance: float = 0.22):
        self.max_missed = max(0, int(max_missed))
        self.smooth_alpha = min(1.0, max(0.05, float(smooth_alpha)))
        self.min_iou = min(1.0, max(0.0, float(min_iou)))
        self.max_center_distance = max(0.01, float(max_center_distance))
        self._tracks: Dict[int, FaceTrack] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: Iterable[Sequence[float]], frame_shape) -> List[FaceTrack]:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        dets = [b for raw in detections if (b := _clip_box(raw, width, height)) is not None]
        ids = list(self._tracks)
        predicted: Dict[int, Box] = {}
        for tid in ids:
            t = self._tracks[tid]
            predicted[tid] = tuple(v + dv for v, dv in zip(t.box, t.velocity))  # type: ignore

        # Build candidate matches; greedily select globally lowest normalised cost.
        candidates = []
        diag = max(1.0, math.hypot(width, height))
        for tid in ids:
            pbox = predicted[tid]
            pc = _center(pbox)
            pw = max(2.0, pbox[2]-pbox[0])
            ph = max(2.0, pbox[3]-pbox[1])
            for j, dbox in enumerate(dets):
                dc = _center(dbox)
                dist = math.hypot(pc[0]-dc[0], pc[1]-dc[1]) / diag
                scale = max(pw, ph, dbox[2]-dbox[0], dbox[3]-dbox[1], 2.0)
                norm_dist = math.hypot(pc[0]-dc[0], pc[1]-dc[1]) / scale
                overlap = _iou(pbox, dbox)
                if overlap < self.min_iou and norm_dist > self.max_center_distance * 5.0:
                    continue
                if dist > self.max_center_distance:
                    continue
                cost = 0.62*(1.0-overlap) + 0.38*min(1.0, norm_dist/5.0)
                candidates.append((cost, tid, j))
        candidates.sort()
        used_t, used_d = set(), set()
        matches = []
        for cost, tid, j in candidates:
            if tid not in used_t and j not in used_d:
                used_t.add(tid); used_d.add(j); matches.append((tid, j, cost))

        for tid, j, cost in matches:
            t = self._tracks[tid]
            raw = dets[j]
            # Predictive smoothing is less aggressive after a missed frame.
            alpha = min(0.95, self.smooth_alpha + 0.08*t.missed)
            new_box = _smooth(predicted[tid], raw, alpha)
            t.velocity = tuple(0.55*v + 0.45*(n-o) for v, o, n in zip(
                t.velocity, t.box, new_box))  # type: ignore
            t.box = new_box
            t.age += 1
            t.missed = 0
            t.confidence = max(0.35, min(1.0, 1.0-cost))
        for tid in ids:
            if tid not in used_t:
                t = self._tracks[tid]
                t.box = predicted[tid]
                t.missed += 1
                t.confidence *= 0.78
        for j, box in enumerate(dets):
            if j not in used_d:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = FaceTrack(tid, box, (0.0, 0.0, 0.0, 0.0))

        for tid in list(self._tracks):
            if self._tracks[tid].missed > self.max_missed:
                del self._tracks[tid]
        return [self._tracks[k] for k in sorted(self._tracks)
                if self._tracks[k].missed <= self.max_missed]

    @staticmethod
    def anonymisation_box(track: FaceTrack, frame_shape,
                          margin: float = 0.12) -> Tuple[int, int, int, int]:
        """Return an expanded, clipped integer ROI for blur/pixelation."""
        h, w = int(frame_shape[0]), int(frame_shape[1])
        x1, y1, x2, y2 = track.box
        mx, my = (x2-x1)*margin, (y2-y1)*margin
        return (max(0, int(x1-mx)), max(0, int(y1-my)),
                min(w, int(x2+mx)), min(h, int(y2+my)))
