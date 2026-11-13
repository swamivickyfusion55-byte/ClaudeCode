"""v11.2.205 SkinTone (SDOS-205): brighten or darken the skin of the chosen people only - face, neck, arms, hands, legs.

How a frame is changed
  1. Skin: Google MediaPipe's multiclass selfie segmenter (selfie_multiclass_256x256: background / hair / body skin /
     face skin / clothes / other) gives each pixel's chance of being skin. Once per frame, on a square around the chosen people.
  2. Whose body: MediaPipe's interactive segmenter (MagicTouch), given a point on a person's face, returns that person -
     head, body, arms, hands - as one object. One call per tracked person in that square.
     Inside a person's own object no pixel is background, so skin is weighed against hair and clothes only there.
  3. Where two people's objects overlap (a tight hug), the way from each face decides: the length of the way to the pixel over
     the picture, where background is very costly and every step between skin and not-skin costs about a face size, plus a
     soft body-shaped territory and each person's own skin colour (sampled on their cheeks).
  4. Faces always belong to their own person: another person's face is never changed.
  5. The change: lightness in Lab. Brighter lifts the mid-tones (a gamma curve, highlights kept); darker scales lightness down
     and deepens the colour a little, as a natural tan. Hue is kept.
  6. In a video the mask is smoothed over time (a still part keeps its mask, a moving part takes the new one at once), refined
     to the picture's own edges (guided filter), and reset at a shot cut.
  Without MagicTouch, step 3 alone decides whose skin it is; without the multiclass model, skin is found by the colour of each
  person's own face skin. Both are logged.

amounts: {slot: a}, a in [-1, 1]; slot 0 is the app's replacement slot #1. 0 = unchanged.
"""
import hashlib
import logging
import os
import threading
import time
import urllib.request

import cv2
import numpy as np

_BASE = "https://storage.googleapis.com/mediapipe-models/"
MODELS = {
    "skin": dict(name="selfie_multiclass_256x256.tflite",
                 url=_BASE + "image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
                 sha256="c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0", size=16371837),
    "person": dict(name="magic_touch.tflite",
                   url=_BASE + "interactive_segmenter/magic_touch/float32/1/magic_touch.tflite",
                   sha256="e24338a717c1b7ad8d159666677ef400babb7f33b8ad60c4d96db4ecf694cd25", size=6227884),
}

SEG_SIDE = 256          # the skin model's input size
PERSON_SIDE = 512       # the square given to the person model
GRID_LONG = 480         # the working grid: the frame's long side in pixels
EPS_BG = 0.02           # "nobody's" share: skin no tracked person reaches cheaply stays as it is
TAU = 3.0               # a way of TAU face sizes over skin leaves 1/e of a person's claim
BG_COST = 25.0          # a step through background costs this many steps over skin
SKIN_STEP = 1.0         # a step between skin and not-skin costs this many face sizes
EDGE_DIV = 120.0        # a visible edge of EDGE_DIV levels costs one face size (one limb lying on another)
OVERLAP = 0.25          # two people's objects overlapping more than this somewhere: the way from each face decides there
OVERLAP_MODE = "ways"   # how an overlap is shared: "ways" (way from each face), "prior" (body shape), "share" (person model)
REFINE_EPS = 0.02       # the edge refinement's smoothness (guided filter eps, in 0-1 grey levels)
HOLD_FRAMES = 12        # a chosen person not found for a few frames keeps their last place, fading out
_DL_LOCK = threading.Lock()
_TL = threading.local()
_STATE = {"skin": "", "person": ""}      # why a model is not used ("" = fine / not tried)


def parse_amounts(raw):
    """[a1, a2, a3, a4] (index = slot), or {"1": a, ...} (1-based app slot numbers) -> {slot0: a} with a in [-1, 1]."""
    out = {}
    try:
        if isinstance(raw, str) and raw.strip():
            import json
            raw = json.loads(raw)
        if isinstance(raw, (list, tuple)):
            items = [(i, v) for i, v in enumerate(raw[:4])]
        elif isinstance(raw, dict):
            items = []
            for k, v in raw.items():
                ks = str(k).strip().lower().replace("slot", "").replace("#", "")
                if ks.isdigit() and 1 <= int(ks) <= 4:
                    items.append((int(ks) - 1, v))
        else:
            items = []
        for i, v in items:
            try:
                a = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(a):
                continue
            a = float(np.clip(a, -1.0, 1.0))
            if abs(a) >= 0.005:
                out[int(i)] = a
    except Exception:
        logging.warning("skin_tone: unreadable setting %r - skin tone off", str(raw)[:80])
        return {}
    return out


# ---- the models ----------------------------------------------------------------------------------------------------------
def _model_dirs():
    ds = [os.environ.get("PHOENIX_MODEL_DIR", ""), "/data/phoenix_models", os.path.expanduser("~/.phoenix_models"),
          "/tmp/phoenix_models"]
    return [d for d in ds if d]


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_path(kind, download=True):
    m = MODELS[kind]
    env = os.environ.get("PHOENIX_SKIN_MODEL" if kind == "skin" else "PHOENIX_PERSON_MODEL", "").strip()
    if env and os.path.isfile(env):
        return env

    def _have():
        for d in _model_dirs():
            p = os.path.join(d, m["name"])
            if os.path.isfile(p) and os.path.getsize(p) == m["size"]:
                return p
        return None
    p = _have()
    if p or not download:
        return p
    with _DL_LOCK:
        p = _have()
        if p:
            return p
        for d in _model_dirs():
            tmp = os.path.join(d, m["name"] + ".part%d" % os.getpid())
            try:
                os.makedirs(d, exist_ok=True)
                t0 = time.time()
                with urllib.request.urlopen(m["url"], timeout=60) as r, open(tmp, "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                if _sha256(tmp) != m["sha256"]:
                    os.remove(tmp)
                    logging.warning("skin_tone: the downloaded %s has the wrong checksum - not used", m["name"])
                    return None
                p = os.path.join(d, m["name"])
                os.replace(tmp, p)
                logging.info("skin_tone: %s saved to %s (%.1f s)", m["name"], p, time.time() - t0)
                return p
            except Exception as e:
                logging.warning("skin_tone: could not save %s in %s: %s", m["name"], d, e)
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    return None


def _mp():
    os.environ.setdefault("GLOG_minloglevel", "2")
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    return mp, mpt, vision


def _skin_model():
    seg = getattr(_TL, "skin", None)
    if seg is not None or _STATE["skin"]:
        return seg
    try:
        mp, mpt, vision = _mp()
        p = model_path("skin")
        if not p:
            raise RuntimeError("the skin model is not available")
        seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=mpt.BaseOptions(model_asset_path=p), running_mode=vision.RunningMode.IMAGE,
            output_category_mask=False, output_confidence_masks=True))
        _TL.skin, _TL.mp = seg, mp
        return seg
    except Exception as e:
        _STATE["skin"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        logging.warning("skin_tone: MediaPipe skin segmenter unavailable (%s) - skin is found by each person's own face "
                        "colour instead", _STATE["skin"])
        return None


def _person_model():
    seg = getattr(_TL, "person", None)
    if seg is not None or _STATE["person"]:
        return seg
    try:
        mp, mpt, vision = _mp()
        from mediapipe.tasks.python.components import containers
        p = model_path("person")
        if not p:
            raise RuntimeError("the person model is not available")
        seg = vision.InteractiveSegmenter.create_from_options(vision.InteractiveSegmenterOptions(
            base_options=mpt.BaseOptions(model_asset_path=p), output_category_mask=False, output_confidence_masks=True))
        _TL.person, _TL.mp, _TL.kp = seg, mp, containers.keypoint.NormalizedKeypoint
        _TL.roi = vision.InteractiveSegmenterRegionOfInterest
        return seg
    except Exception as e:
        _STATE["person"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        logging.warning("skin_tone: MediaPipe person segmenter unavailable (%s) - whose skin it is comes from the way from "
                        "each face only", _STATE["person"])
        return None


def _skin_prob(rgb256):
    """256x256 RGB uint8 -> (body skin + face skin chance, background chance), float32 [0, 1]; None without the model."""
    seg = _skin_model()
    if seg is None:
        return None
    mp = _TL.mp
    cm = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb256))).confidence_masks
    if cm is None or len(cm) < 4:
        return None

    def _a(i):
        a = cm[i].numpy_view().astype(np.float32)
        return a[:, :, 0] if a.ndim == 3 else a
    return np.clip(_a(2) + _a(3), 0.0, 1.0), np.clip(_a(0), 0.0, 1.0)


def _person_prob(rgb, pt):
    """RGB uint8 square + a point (pixels) on the person -> that person's chance per pixel; None without the model."""
    seg = _person_model()
    if seg is None:
        return None
    mp, KP, RoI = _TL.mp, _TL.kp, _TL.roi
    h, w = rgb.shape[:2]
    roi = RoI(format=RoI.Format.KEYPOINT, keypoint=KP(float(np.clip(pt[0] / w, 0, 1)), float(np.clip(pt[1] / h, 0, 1))))
    cm = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), roi).confidence_masks
    if not cm:
        return None
    a = cm[0].numpy_view().astype(np.float32)
    return np.clip(a[:, :, 0] if a.ndim == 3 else a, 0.0, 1.0)


def backend():
    return "; ".join([
        "skin: MediaPipe multiclass selfie segmenter" if not _STATE["skin"] else
        "skin: face-colour model (%s)" % _STATE["skin"],
        "person: MediaPipe MagicTouch" if not _STATE["person"] else
        "person: way from each face only (%s)" % _STATE["person"]])


# ---- people --------------------------------------------------------------------------------------------------------------
def _person(rec):
    """kps (5 points: eyes, nose, mouth corners) + bbox -> centre, up vector, face size, the face box."""
    k = np.asarray(rec["kps"], np.float32).reshape(-1, 2)[:5]
    b = np.asarray(rec["bbox"], np.float32).reshape(4)
    bw, bh = float(b[2] - b[0]), float(b[3] - b[1])
    eye, mouth = (k[0] + k[1]) / 2.0, (k[3] + k[4]) / 2.0
    up = eye - mouth
    n = float(np.linalg.norm(up))
    up = np.array([0.0, -1.0], np.float32) if n < 1e-3 else (up / n).astype(np.float32)
    c = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], np.float32)
    return dict(c=c, up=up, s=max(bw, bh, 1.0), bw=max(bw, 1.0), bh=max(bh, 1.0), kps=k)


def _capsule_box(p, W, H):
    c, up, s = p["c"], p["up"], p["s"]
    a, b = c + up * 0.9 * s, c - up * 7.0 * s
    return (max(0.0, min(a[0], b[0]) - 3.4 * s), max(0.0, min(a[1], b[1]) - 1.2 * s),
            min(float(W), max(a[0], b[0]) + 3.4 * s), min(float(H), max(a[1], b[1]) + 1.2 * s))


def _face_e(p, xs, ys):
    """Distance in face-ellipse units (1 = the ellipse around the face box) along the face's own axes."""
    c, up = p["c"], p["up"]
    fx, fy = xs - c[0], ys - c[1]
    u = fx * (-up[1]) + fy * up[0]
    v = fx * up[0] + fy * up[1]
    return np.sqrt((u / (0.62 * p["bw"])) ** 2 + (v / (0.72 * p["bh"])) ** 2)


def _face_claim(p, xs, ys):
    """1 on the face, feathered at its edge, 0 beyond it."""
    F = np.clip((1.25 - _face_e(p, xs, ys)) / 0.35, 0.0, 1.0)
    return (F * F * (3.0 - 2.0 * F)).astype(np.float32)


def _body_prior(p, xs, ys):
    """Soft capsule from the top of the head to the knees, shoulder-and-arm wide."""
    c, up, s = p["c"], p["up"], p["s"]
    a, d, L = c + up * 0.9 * s, -up, 7.9 * s
    rx, ry = xs - a[0], ys - a[1]
    t = np.clip((rx * d[0] + ry * d[1]) / L, 0.0, 1.0)
    qx, qy = rx - t * L * d[0], ry - t * L * d[1]
    w = np.clip((t * L - 0.7 * s) / (1.4 * s), 0.0, 1.0)
    w = w * w * (3 - 2 * w)
    sig = 0.95 * s + (2.4 * s - 0.95 * s) * w
    return np.exp(-0.5 * (qx * qx + qy * qy) / (sig * sig)).astype(np.float32)


def _ways(cost, seeds):
    """The cheapest way from the seed pixels to every pixel over `cost` (skimage's MCP); None if not possible here."""
    try:
        from skimage.graph import MCP_Geometric
    except Exception:
        return None
    pts = np.argwhere(seeds)
    if len(pts) == 0:
        return None
    if len(pts) > 400:
        pts = pts[:: int(np.ceil(len(pts) / 400.0))]
    d, _ = MCP_Geometric(cost, fully_connected=True).find_costs([tuple(q) for q in pts])
    return d.astype(np.float32)


def _cheek_stats(frame, p):
    """Mean and covariance of Cr, Cb on the cheeks (between the eyes and the mouth corners) of the ORIGINAL frame."""
    k = p["kps"]
    poly = np.array([k[0], k[1], k[4], k[3]], np.float32)
    cen = poly.mean(0)
    poly = (cen + (poly - cen) * 0.85).astype(np.int32)
    H, W = frame.shape[:2]
    x0, y0 = max(0, int(poly[:, 0].min())), max(0, int(poly[:, 1].min()))
    x1, y1 = min(W, int(poly[:, 0].max()) + 1), min(H, int(poly[:, 1].max()) + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, poly - np.array([x0, y0], np.int32), 1)
    v = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2YCrCb)[m > 0][:, 1:].astype(np.float32)
    if len(v) < 12:
        return None
    return v.mean(0), np.cov(v.T) + np.eye(2, dtype=np.float32) * 25.0


def _colour_like(crcb, stats, k=2.5, floor=0.15):
    mu, cov = stats
    inv = np.linalg.inv(cov).astype(np.float32)
    dx = crcb - mu.astype(np.float32)
    m2 = dx[..., 0] ** 2 * inv[0, 0] + 2 * dx[..., 0] * dx[..., 1] * inv[0, 1] + dx[..., 1] ** 2 * inv[1, 1]
    return np.maximum(floor, np.exp(-0.5 * m2 / (k * k))).astype(np.float32)


# ---- the tone curve ------------------------------------------------------------------------------------------------------
_LUT_CACHE = {}


def _luts(a):
    key = round(float(a), 3)
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    a = float(np.clip(a, -1.0, 1.0))
    x = np.arange(256, dtype=np.float32)
    if a >= 0:
        L = 255.0 * (x / 255.0) ** (1.0 / (1.0 + 0.65 * a))     # +1: L 128 -> 168, highlights kept
        cs = 1.0 - 0.06 * a
    else:
        L = x * (1.0 - 0.40 * (-a))                              # -1: L 128 -> 77, shadows keep their detail
        cs = 1.0 + 0.14 * (-a)                                   # a little deeper colour, as a tan
    C = 128.0 + (x - 128.0) * cs
    lut = (np.clip(L + 0.5, 0, 255).astype(np.uint8), np.clip(C + 0.5, 0, 255).astype(np.uint8))
    if len(_LUT_CACHE) > 64:
        _LUT_CACHE.clear()
    _LUT_CACHE[key] = lut
    return lut


def toned(bgr, a):
    """The whole region with the tone change applied (no mask)."""
    lL, lC = _luts(a)
    l_, a_, b_ = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge([cv2.LUT(l_, lL), cv2.LUT(a_, lC), cv2.LUT(b_, lC)]), cv2.COLOR_LAB2BGR)


def _refine(guide_bgr, m, factor):
    """Up-sampled mask -> the picture's own edges."""
    r = int(max(3, round(2.2 * factor)))
    try:
        out = cv2.ximgproc.guidedFilter(cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY), m, r, (REFINE_EPS * 255) ** 2)
    except Exception:
        out = cv2.GaussianBlur(m, (0, 0), max(1.0, factor * 0.8))
    return np.clip(out, 0.0, 1.0)


def _blend(frame, ms, box, amounts):
    x0, y0, x1, y1 = box
    base = frame[y0:y1, x0:x1]
    region = base.astype(np.float32)
    for s, m in ms.items():
        region += m[:, :, None] * (toned(base, amounts[s]).astype(np.float32) - region)
    out = frame.copy()
    out[y0:y1, x0:x1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
    return out


# ---- the per-video state -------------------------------------------------------------------------------------------------
class SkinTone:
    def __init__(self, amounts, temporal=True):
        self.amounts = {int(k): float(v) for k, v in (amounts or {}).items() if abs(float(v)) >= 0.005}
        self.temporal = bool(temporal)
        self.prev = {}            # slot -> smoothed mask on the grid
        self.held = {}            # slot -> (person, age)
        self.cstats = {}          # slot -> (mu, cov), EMA
        self.stats = dict(frames=0, ms=0.0, seg_ms=0.0, person_ms=0.0, way_ms=0.0, ways=0, skipped=0, area={})
        self.debug = None         # a dict to fill with the working maps (tests only)

    @property
    def active(self):
        return bool(self.amounts)

    def reset(self):
        self.prev.clear()
        self.held.clear()
        self.cstats.clear()

    def masks(self, out, orig, records):
        """{slot: full-res float32 mask} for the chosen people, and the region it covers (x0, y0, x1, y1)."""
        H, W = out.shape[:2]
        people = {}
        for slot, rec in (records or {}).items():
            try:
                if rec is not None and rec.get("kps") is not None and rec.get("bbox") is not None:
                    people[int(slot)] = _person(rec)
            except Exception:
                continue
        fade = {}
        for slot in self.amounts:
            if slot in people:
                self.held[slot] = (people[slot], 0)
                fade[slot] = 1.0
            elif slot in self.held and self.held[slot][1] < HOLD_FRAMES:
                p, age = self.held[slot]
                self.held[slot] = (p, age + 1)
                people[slot] = p
                fade[slot] = 1.0 - (age + 1) / float(HOLD_FRAMES + 1)
            else:
                self.held.pop(slot, None)
                self.prev.pop(slot, None)
        chosen = [s for s in self.amounts if s in people]
        if not chosen:
            return {}, None
        # the region: every chosen person's body capsule
        bx = [_capsule_box(people[s], W, H) for s in chosen]
        x0, y0 = int(min(b[0] for b in bx)), int(min(b[1] for b in bx))
        x1, y1 = int(np.ceil(max(b[2] for b in bx))), int(np.ceil(max(b[3] for b in bx)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return {}, None
        # the square around it (padded where it leaves the frame) for the models
        side = max(x1 - x0, y1 - y0)
        sx0, sy0 = (x0 + x1) // 2 - side // 2, (y0 + y1) // 2 - side // 2
        pad = [max(0, -sy0), max(0, sy0 + side - H), max(0, -sx0), max(0, sx0 + side - W)]
        sq = out[max(0, sy0):min(H, sy0 + side), max(0, sx0):min(W, sx0 + side)]
        if any(pad):
            sq = cv2.copyMakeBorder(sq, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        t0 = time.perf_counter()
        P256 = B256 = None
        try:
            r_ = _skin_prob(cv2.cvtColor(cv2.resize(sq, (SEG_SIDE, SEG_SIDE), interpolation=cv2.INTER_AREA),
                                         cv2.COLOR_BGR2RGB))
            if r_ is not None:
                P256, B256 = r_
        except Exception as e:
            logging.warning("skin_tone: skin segmenter failed on a frame: %s", e)
        self.stats["seg_ms"] += (time.perf_counter() - t0) * 1000.0
        # the grid over the region
        g = min(1.0, GRID_LONG / float(max(W, H)))
        GW, GH = max(1, int(round(W * g))), max(1, int(round(H * g)))
        gx0, gy0 = int(np.floor(x0 * g)), int(np.floor(y0 * g))
        gx1, gy1 = min(GW, int(np.ceil(x1 * g))), min(GH, int(np.ceil(y1 * g)))
        if gx1 - gx0 < 2 or gy1 - gy0 < 2:
            return {}, None
        ys, xs = np.mgrid[gy0:gy1, gx0:gx1].astype(np.float32)
        xs, ys = (xs + 0.5) / g, (ys + 0.5) / g

        def _to_grid(a, n, border):
            u = ((xs - sx0) * (n / float(side)) - 0.5).astype(np.float32)
            v = ((ys - sy0) * (n / float(side)) - 0.5).astype(np.float32)
            return cv2.remap(a, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
        P = _to_grid(P256, SEG_SIDE, 0) if P256 is not None else None
        Bg = _to_grid(B256, SEG_SIDE, 1) if B256 is not None else None
        small = cv2.resize(out, (GW, GH), interpolation=cv2.INTER_AREA)[gy0:gy1, gx0:gx1]
        crcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)[:, :, 1:].astype(np.float32)
        for s, p in people.items():
            st = _cheek_stats(orig if orig is not None else out, p)
            if st is not None:
                if s in self.cstats and self.temporal:
                    m0, c0 = self.cstats[s]
                    st = (0.8 * m0 + 0.2 * st[0], 0.8 * c0 + 0.2 * st[1])
                self.cstats[s] = st
        if P is None:
            # no skin model: skin = looks like one of the people's own face skin
            P = np.zeros(xs.shape, np.float32)
            for s in people:
                if self.cstats.get(s) is not None:
                    P = np.maximum(P, _colour_like(crcb, self.cstats[s], k=1.6, floor=0.0))
            P = np.clip((P - 0.15) / 0.5, 0.0, 1.0)       # faint likeness (a red shirt is 4 % like skin) is not skin
        # whose body: the person model, from a point on each face in the square
        t1 = time.perf_counter()
        inside = [s for s, p in people.items() if sx0 <= p["c"][0] < sx0 + side and sy0 <= p["c"][1] < sy0 + side]
        I = {}
        if set(chosen) <= set(inside):
            n = min(PERSON_SIDE, side)
            rgb = cv2.cvtColor(cv2.resize(sq, (n, n), interpolation=cv2.INTER_AREA) if n != side else sq,
                               cv2.COLOR_BGR2RGB)
            for s in inside:
                try:
                    a = _person_prob(rgb, ((people[s]["c"][0] - sx0) * n / side, (people[s]["c"][1] - sy0) * n / side))
                except Exception as e:
                    logging.warning("skin_tone: person segmenter failed on a frame: %s", e)
                    a = None
                if a is None:
                    I = {}
                    break
                I[s] = _to_grid(a, n, 0)
        self.stats["person_ms"] += (time.perf_counter() - t1) * 1000.0
        F = {s: _face_claim(p, xs, ys) for s, p in people.items()}
        # where two people's objects overlap (or without the person model): the way from each face decides
        ovl = {}
        if len(I) > 1:
            for s in chosen:
                ovl[s] = I[s] * np.clip(sum(I[j] for j in I if j != s), 0.0, 1.0)
        need_ways = (not I) or any(float(o.max()) > OVERLAP for o in ovl.values())
        G = {}
        if need_ways and I and OVERLAP_MODE != "ways":
            if OVERLAP_MODE == "share":
                tot = sum(I.values()) + 1e-6
                G = {s: I[s] / tot for s in I}
            else:
                pr = {s: _body_prior(p, xs, ys) for s, p in people.items()}
                tot = sum(pr.values()) + EPS_BG
                G = {s: pr[s] / tot for s in people}
        elif need_ways:
            t2 = time.perf_counter()
            fs = float(np.median([p["s"] for p in people.values()])) * g
            gy_, gx_ = np.gradient(cv2.GaussianBlur(P, (0, 0), 0.8))
            cost = 1.0 + SKIN_STEP * fs * np.sqrt(gx_ * gx_ + gy_ * gy_)
            ey_, ex_ = np.gradient(cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8))
            cost += fs * np.sqrt(ex_ * ex_ + ey_ * ey_) / EDGE_DIV
            if Bg is not None:
                cost += BG_COST * Bg
            cost = cost.astype(np.float64)
            claim = {}
            for s, p in people.items():
                prior = _body_prior(p, xs, ys)
                d = _ways(cost, _face_e(p, xs, ys) < 0.45)
                way = np.exp(-d / (TAU * p["s"] * g)).astype(np.float32) if d is not None else prior
                C = _colour_like(crcb, self.cstats[s]) if self.cstats.get(s) is not None else 1.0
                claim[s] = way * (0.25 + 0.75 * prior) * C
            tot = sum(claim.values()) + EPS_BG
            G = {s: claim[s] / tot for s in people}
            self.stats["way_ms"] += (time.perf_counter() - t2) * 1000.0
            self.stats["ways"] += 1
        res = {}
        for s in chosen:
            if I:
                ov = ovl.get(s, 0.0)
                w = I[s] * (1.0 - ov) + (ov * G[s] if G else 0.0)
            else:
                w = G[s]
            # faces: their own person's only
            fo = np.max([F[j] for j in F if j != s], axis=0) if len(F) > 1 else 0.0
            w = np.maximum(w * (1.0 - fo), F[s] * F[s] / (F[s] + fo + 1e-6))
            # inside the person's own object a pixel cannot be background: skin among the rest (shaded arms the skin model
            # took for background)
            Ps = P / np.maximum(1.0 - Bg * I[s], 0.3) if (I and Bg is not None) else P
            m = np.clip(Ps * w * fade.get(s, 1.0), 0.0, 1.0).astype(np.float32)
            if self.temporal:
                full = np.zeros((GH, GW), np.float32)
                full[gy0:gy1, gx0:gx1] = m
                pm = self.prev.get(s)
                if pm is not None and pm.shape == full.shape:
                    still = np.abs(full - pm) < 0.35
                    full = np.where(still, 0.55 * full + 0.45 * pm, full).astype(np.float32)
                self.prev[s] = full
                m = full[gy0:gy1, gx0:gx1]
            res[s] = m
        if self.debug is not None:
            self.debug.update(g=g, box=(gx0, gy0, gx1, gy1), P=P, Bg=Bg, I=I, G=G, F=F, res=res)
        # back to full resolution on the region, refined to the picture's edges
        fx0, fy0 = int(gx0 / g), int(gy0 / g)
        fx1, fy1 = min(W, int(np.ceil(gx1 / g))), min(H, int(np.ceil(gy1 / g)))
        guide = out[fy0:fy1, fx0:fx1]
        full = {}
        for s, m in res.items():
            if float(m.max()) < 0.02:
                continue
            up = cv2.resize(m, (fx1 - fx0, fy1 - fy0), interpolation=cv2.INTER_LINEAR)
            full[s] = _refine(guide, up, 1.0 / g)
        return full, (fx0, fy0, fx1, fy1)

    def apply(self, out, orig=None, records=None, cut=False):
        """The finished frame with the chosen people's skin toned. Never raises; on a problem the frame is returned as is."""
        if not self.amounts:
            return out
        t0 = time.perf_counter()
        try:
            if cut:
                self.reset()
            ms, box = self.masks(out, orig, records)
            if not ms:
                self.stats["skipped"] += 1
                return out
            x0, y0, x1, y1 = box
            for s, m in ms.items():
                a = self.stats["area"]
                a[s] = a.get(s, 0.0) + float(m.mean()) * (y1 - y0) * (x1 - x0) / float(out.shape[0] * out.shape[1])
            return _blend(out, ms, box, self.amounts)
        except Exception:
            logging.exception("skin_tone: frame left as it was")
            return out
        finally:
            self.stats["frames"] += 1
            self.stats["ms"] += (time.perf_counter() - t0) * 1000.0

    def summary(self):
        st = self.stats
        n = max(1, st["frames"])
        return ("SKIN-TONE: %s; %d frames, %.0f ms a frame (skin %.0f, person %.0f; ways %.0f ms on %d frames), "
                "%d frames with none of the chosen people; skin changed (share of the picture): %s" % (
                    backend(), st["frames"], st["ms"] / n, st["seg_ms"] / n, st["person_ms"] / n,
                    st["way_ms"] / max(1, st["ways"]), st["ways"], st["skipped"],
                    ", ".join("#%d %+.0f%%: %.1f%%" % (s + 1, 100 * self.amounts[s], 100 * a / n)
                              for s, a in sorted(st["area"].items())) or "-"))


def preview(frame_bgr, people, amounts):
    """One picture: people = {slot: {"kps", "bbox"}}. Returns (toned frame, {slot: full-frame mask}, info)."""
    t0 = time.perf_counter()
    st = SkinTone(amounts, temporal=False)
    ms, box = st.masks(frame_bgr, frame_bgr, people)
    out = _blend(frame_bgr, ms, box, st.amounts) if ms else frame_bgr
    full = {}
    for s, m in ms.items():
        fm = np.zeros(frame_bgr.shape[:2], np.float32)
        fm[box[1]:box[3], box[0]:box[2]] = m
        full[s] = fm
    info = dict(ms=int((time.perf_counter() - t0) * 1000), backend=backend(),
                skin_share={str(s + 1): round(float(m.mean()), 4) for s, m in full.items()})
    return out, full, info
