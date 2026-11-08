"""Manual painted-mark removal only. The face-swap code is not in this file."""
import logging
import math
import os
import time
from collections import OrderedDict

import cv2
import numpy as np


def _off_face(frame, hole, core, x0, y0):
    """A painted mark must not cover a face. Any overlap skips this frame."""
    boxes = _face_boxes(frame)
    if not boxes:
        return hole, core
    H, W = hole.shape[:2]
    guard = np.zeros((H, W), np.uint8)
    for x1, y1, x2, y2 in boxes:
        dw = int(0.22 * (x2 - x1))
        dh = int(0.28 * (y2 - y1))
        xa = max(0, x1 - dw - x0)
        ya = max(0, y1 - dh - y0)
        xb = min(W, x2 + dw - x0)
        yb = min(H, y2 + int(0.35 * dh) - y0)
        if xb > xa and yb > ya:
            guard[ya:yb, xa:xb] = 1
    if int((hole & (guard == 1)).sum()) >= 8:
        logging.info("PATCH-REMOVAL skipped: the mark overlapped a face")
        return None, None
    return hole, core


_HAAR = {"c": None}


def _load_cascades():
    here = os.path.dirname(os.path.abspath(__file__))
    names = ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")
    out = []
    for name in names:
        path = os.path.join(here, name)
        if not os.path.isfile(path):
            path = os.path.join(cv2.data.haarcascades, name)
        if not os.path.isfile(path):
            continue
        c = cv2.CascadeClassifier(path)
        if not c.empty():
            out.append((name, c))
    if not out:
        logging.warning("PATCH-REMOVAL face guard could not load a face detector")
    else:
        logging.info("PATCH-REMOVAL face guard loaded: %s", ", ".join(n for n, _ in out))
    return out


def _scan(gray, cascades, scale):
    boxes = []
    inv = 1.0 / scale
    for name, c in cascades:
        rects = c.detectMultiScale(gray, 1.08, 4, minSize=(32, 32))
        for x, y, bw, bh in rects:
            boxes.append((int(x * inv), int(y * inv), int((x + bw) * inv), int((y + bh) * inv)))
        if "profile" in name:
            flipped = cv2.flip(gray, 1)
            w = gray.shape[1]
            for x, y, bw, bh in c.detectMultiScale(flipped, 1.08, 4, minSize=(32, 32)):
                x2 = int((w - x) * inv)
                x1 = int((w - (x + bw)) * inv)
                boxes.append((x1, int(y * inv), x2, int((y + bh) * inv)))
    return boxes


def _face_boxes(bgr):
    key = id(bgr)
    hit = _HAAR.get("last")
    if hit is not None and hit[0] == key:
        return hit[1]
    cascades = _HAAR.get("c")
    if cascades is None:
        cascades = _load_cascades()
        _HAAR["c"] = cascades
    if not cascades:
        _HAAR["last"] = (key, [])
        return []
    scale = 1.0
    h, w = bgr.shape[:2]
    small = bgr
    if max(h, w) > 960:
        scale = 960.0 / max(h, w)
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    boxes = _scan(gray, cascades, scale)
    _HAAR["last"] = (key, boxes)
    return boxes

PATCH_CFG = {
    "max_patches": 10,         # marks per job (v11.2.100: was 4 - the config's PATCH_MAX only gated the settings, this one the engine)
    "halo_px": 3,              # the painted hole is grown by this (v11.2.102: 3 - a mark's soft edge is covered too)
    "peek_band": 0.0,          # v11.2.97 edge repair outside the hole; v11.2.98: OFF (nothing outside the painted area changes)
    "bridge_max": 45,          # gaps of up to this many frames between followed frames are bridged (v11.2.102: was 3); a bridged
                               #   pose is "weak": it paints only where the mark itself is found (find_mark)
    "hold_max": 15,            # a followed run's last pose is held this many frames further (weak) - the end of a run is not a loss
    "find_budget_s": 90.0,     # v11.2.104: one search for the painted AREA (the painted frame is not in this video) stops after this many seconds ...
    "find_total_s": 240.0,     #   ... and all of a job's searches together after this many (a 4-mark job spent 946 s on searches that found nothing)
    "redetect_now": True,      # v11.2.103: a step whose followed / predicted pose does not fit looks for the patch at once (near the prediction, then
                               #   in the whole frame), not after two lost frames
    "tm_radius": 160,          # px around the predicted pose searched by that local template match (at the frame's own scale)
    "blur_ok": True,           # a re-detected pose with 85 % of the structure matched and a lighting-free score of 0.72 is accepted although the
                               #   plain correlation is lower (motion blur)
    "smooth_err": 1.0,         # v11.2.103: px the smoothing may bend the followed path (the window shrinks where the motion curves)
    "find_mark": True,         # v11.2.102: on every frame the mark itself is looked for (it differs from the skin around it) near where
                               #   the track puts it; the paint is centred on what is found and covers all of it
    "rescue": True,            #   ... and when the track loses the patch (blur, turn, light) the mark found near its predicted place
                               #   keeps it followed
    "opaque": True,            # the paste is opaque (no partial transparency): only strong evidence of an object holds it back
    "clone_texture": True,     # at the anchor: the fill gets the fine texture of clean skin beside the mark (copied, not synthetic)
    "max_views": 20,           # painted positions in all (several per mark) (v11.2.100: was 12)
    "grow_band": 0.0,          # dark rim around the painted area joins it (v11.2.97; OFF: on a face it ate creases and shadows
                               #   beside the mark - the per-frame mark finder covers the mark's own rim instead)
    "follow_margin": 45,       # v11.2.101: a painted position is followed from its anchor to the NEXT painted position and this many
                               #   frames past it (not through the whole video): 10 positions cost ~2-3 x one, not 10 x
    "cover": 1.5,              # v11.2.101: stronger cover - the fill is matched UP to the frame's own skin on the stroke's
    "cover_max": 10.0,         #   margin (it is never darker than that skin; at most this many levels; 1.5 = over by half)
    "lift": 1.0,               # v11.2.98: no faint dark blob may remain under the mark: the fill is lifted (inside the
    "lift_max": 6.0,           #   painted area only, very smoothly, by at most this many levels) to the level around it
    "feather_px": 1.5,         # soft edge of the fill
    "light_sigma": 4.0,        # lighting change is measured on the ring at this smoothness
    "gate_thr": 22.0,          # object gate: colour distance (0-255) to the anchor's appearance, 2 px tolerance
    "track_ncc_min": 0.55,     # the tracked area must still look like the anchor's (else: lost, nothing painted)
    "anchor_window_s": 1.5,    # the anchor frame is searched within this time of the app's frame first
    "smooth": (1.0, 2.0, 3.0, 2.0, 1.0),   # centred smoothing of the per-frame transform
    "body_smooth": (1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0),  # wider window when the paint must ride the body
    "lock_finder": True,       # v11.2.118: a locked painted mark keeps the mark finder (the one-go belly positions do not)
    "body_lock": True,         # place the removal from the surface AROUND the stroke, not from the dark texture inside it
    "hair_halo": 5,            # extra pixels of fringe covered when the stroke is dark-on-skin
    "detail_track": True,      # v11.2.99: the tracker compares a lighting-free view (a shadow moving across the mark
    "detail_sigma": 8.0,       #   lost it on a third of the frames); scale (px) of the light change it ignores
    "detail_thr": 4.0,         #   inlier tolerance in that view (% of local brightness)
    "detail_tol": 2,           #   and in space (px): skin bends a little, an affine track cannot follow thin strokes exactly
}

PATCH_MODES = {
    "auto": {},
    "classic": {"find_mark": False, "rescue": False, "opaque": False, "clone_texture": False, "cover": 1.0, "halo_px": 2,
               "peek_band": 0.08, "grow_band": 0.06, "hold_max": 0, "bridge_max": 3, "detail_track": False},
}
_PR_MODE_BASE = {}


def set_patch_mode(mode):
    """Only the painted mark. Automatic belly clean is not in this pack."""
    mode = "classic"
    for k, v in _PR_MODE_BASE.items():
        PATCH_CFG[k] = v
    for k, v in PATCH_MODES["classic"].items():
        _PR_MODE_BASE.setdefault(k, PATCH_CFG.get(k))
    for k, v in PATCH_MODES[mode].items():
        PATCH_CFG[k] = v
    return mode


PATCH_EVENTS = {}      # v11.2.105: {mark number: {frame index: code}} - P painted, o painted around an object, R refused, W held pose and the mark
                       #   not seen, L no pose (not followed), X too small / out of the picture


PATCH_STATS = {"frames_found": 0, "frames_shifted": 0, "frames_rescued": 0, "frames_weak_skipped": 0,
               "patches": 0, "anchored": 0, "frames_tracked": 0, "frames_filled": 0, "frames_lost": 0,
               "frames_gated": 0, "frames_refused": 0, "frames_bridged": 0, "frames_peek": 0, "views": 0,
               "fill_ms": 0.0, "track_ms": 0.0}


def _pr_detail(g, inside=None):
    """v11.2.99: a lighting-free view of a gray patch - log brightness minus its smooth version, x100 (so in % of the
    local brightness). A shadow edge or a light change across the patch is smooth and multiplies the brightness: it
    cancels here. The mark, its edges, pores and texture stay. inside: the valid pixels (border handling)."""
    s = float(PATCH_CFG.get("detail_sigma", 8.0))
    lg = np.log(g.astype(np.float32) + 8.0)
    if inside is None:
        lo = cv2.GaussianBlur(lg, (0, 0), s)
    else:
        v = inside.astype(np.float32)
        lo = cv2.GaussianBlur(lg * v, (0, 0), s) / np.maximum(cv2.GaussianBlur(v, (0, 0), s), 1e-3)
    return (lg - lo) * 100.0


def _eyes_hold(new, old):
    """Do these 5 points still have the same eyes? A kiss or a fast move makes
    the detector swap the two eyes, or collapse them. That paste is rejected."""
    try:
        ne = np.asarray(new, np.float32).reshape(-1, 2)
        oe = np.asarray(old, np.float32).reshape(-1, 2)
        if len(ne) < 2 or len(oe) < 2:
            return True
        a = ne[1] - ne[0]
        b = oe[1] - oe[0]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if nb < 2.0 or na < 2.0:
            return False
        if not (0.62 <= na / nb <= 1.55):
            return False
        return float(np.dot(a, b)) / (na * nb) > 0.70
    except Exception:
        return True


def _pr_odd(n):
    n = int(round(n))
    return n if n % 2 else n + 1


def _pr_disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_pr_odd(2 * r + 1),) * 2)


def _mls_to_anchor(v, cur, anc):
    """Where each current-frame point came from on the painted frame.
    Moving least squares, similarity (Schaefer 2006): the cover bends with the
    nearby skin instead of staying one rigid plate. v, cur, anc are (N,2)/(M,2)."""
    v = np.asarray(v, np.float64).reshape(-1, 2)
    cur = np.asarray(cur, np.float64).reshape(-1, 2)
    anc = np.asarray(anc, np.float64).reshape(-1, 2)
    d = v[:, None, :] - cur[None, :, :]
    w = 1.0 / (np.sum(d * d, axis=2) + 4.0)
    wsum = np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
    pstar = (w[..., None] * cur[None, :, :]).sum(axis=1) / wsum
    qstar = (w[..., None] * anc[None, :, :]).sum(axis=1) / wsum
    ph = cur[None, :, :] - pstar[:, None, :]
    qh = anc[None, :, :] - qstar[:, None, :]
    mu = np.maximum((w * np.sum(ph * ph, axis=2)).sum(axis=1), 1e-6)
    vp = v - pstar
    phx, phy = ph[:, :, 0], ph[:, :, 1]
    dot = phx * vp[:, 0:1] + phy * vp[:, 1:2]
    dotp = (-phy) * vp[:, 0:1] + phx * vp[:, 1:2]
    qhx, qhy = qh[:, :, 0], qh[:, :, 1]
    accx = (w * (dot * qhx + dotp * (-qhy))).sum(axis=1)
    accy = (w * (dot * qhy + dotp * qhx)).sum(axis=1)
    return np.stack([qstar[:, 0] + accx / mu, qstar[:, 1] + accy / mu], 1)



def _seal_split(hole, inside):
    """Close a gap that opens down the middle of a skin stroke.
    The old 6 px close left a wider split. This bridges up to about a fifth of
    the stroke, then fills any hole that is still enclosed. The outer edge is
    not grown."""
    u8 = np.where(hole, 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(u8)
    if len(ys) < 8:
        return hole
    span = int(min(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
    r = int(np.clip(round(0.22 * span), 8, 28))
    sealed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, _pr_disk(r))
    inv = cv2.bitwise_not(sealed)
    flood = inv.copy()
    cv2.floodFill(flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 0)
    sealed[flood > 0] = 255
    return (sealed > 0) & inside




# --------------------------------------------------------------------------------------------------- smooth fills
def _pr_lowpass(img, valid, sigma):
    """Normalised convolution: blur of the VALID pixels only."""
    v = valid.astype(np.float32)
    k = _pr_odd(max(3, sigma * 3))
    num = cv2.GaussianBlur(img * (v[..., None] if img.ndim == 3 else v), (k, k), sigma)
    den = cv2.GaussianBlur(v, (k, k), sigma)
    return num / (np.maximum(den, 1e-4)[..., None] if img.ndim == 3 else np.maximum(den, 1e-4)), den


def _pr_pushpull(img, w):
    """Fill where w == 0 from where w == 1 (pull-push pyramid + a little relaxation): smooth, seamless."""
    squeeze = img.ndim == 2
    if squeeze:
        img = img[..., None]
    w = np.clip(w.astype(np.float32), 0.0, 1.0)
    pyr = [(img * w[..., None], w)]
    while True:
        p, q = pyr[-1]
        if q.shape[0] <= 2 or q.shape[1] <= 2 or float(q.min()) > 1e-3:
            break
        nh, nw = (q.shape[0] + 1) // 2, (q.shape[1] + 1) // 2
        p2 = cv2.resize(p, (nw, nh), interpolation=cv2.INTER_AREA)
        q2 = cv2.resize(q, (nw, nh), interpolation=cv2.INTER_AREA)
        pyr.append((p2.reshape(nh, nw, -1), q2))
    p, q = pyr[-1]
    est = p / np.maximum(q, 1e-6)[..., None]
    if float(q.min()) <= 1e-6:
        est[q <= 1e-6] = p.reshape(-1, p.shape[-1]).sum(0) / max(float(q.sum()), 1e-6)
    for p, q in reversed(pyr[:-1]):
        up = cv2.resize(est, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(q.shape[0], q.shape[1], -1)
        qn = np.clip(q * 2.0, 0.0, 1.0)
        est = p / np.maximum(q, 1e-6)[..., None] * qn[..., None] + up * (1.0 - qn)[..., None]
    hole = w < 0.5
    if hole.any():
        for _ in range(10):
            b = cv2.blur(est, (5, 5)).reshape(est.shape)
            est[hole] = b[hole]
    return est[..., 0] if squeeze else est


# --------------------------------------------------------------------------------------------------- exemplar fill
def _pr_exemplar(img, hole, srcmask=None):
    """Fill ``hole`` of float32 BGR ``img`` with patches of the known area (structure first, nearest sources).

    srcmask: where source patches may come from (default: everything not near the hole)."""
    H, W = hole.shape
    ys, xs = np.nonzero(hole)
    if len(ys) == 0:
        return img.copy()
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    B = int(np.clip(_pr_odd(size / 5.0), 9, 19))
    r = B // 2
    out = img.copy()
    known = ~hole
    ok = known & ~cv2.dilate(hole.astype(np.uint8), _pr_disk(2)).astype(bool)
    if srcmask is not None:
        ok &= srcmask
    # a source window is valid only if all of it is allowed
    bad = cv2.integral((~ok).astype(np.uint8)).astype(np.int64)
    Ws, Hs = W - B + 1, H - B + 1
    if Ws < 1 or Hs < 1:
        return out
    badwin = (bad[B:B + Hs, B:B + Ws] - bad[0:Hs, B:B + Ws] - bad[B:B + Hs, 0:Ws] + bad[0:Hs, 0:Ws]) > 0
    if badwin.all():
        return out
    # structure term: edges in the known area (smoothed), so lines are carried into the hole first
    g = cv2.GaussianBlur(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    gm = gm / (float(np.percentile(gm[known], 95)) + 1e-3)
    # guidance: the smooth fill says what colour belongs where in the hole; a source patch must agree with it
    # (weakly) as well as with the pixels already known (strongly) - an eye is never copied onto a cheek
    v0 = ~hole
    lowg, deng = _pr_lowpass(img, v0, 3.0)
    guide = _pr_pushpull(lowg, (v0 & (deng > 0.25)).astype(np.float32))
    R = int(max(3 * B, 1.0 * size))                      # search radius around the target
    yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
    lam = 0.15 * B * B * 3.0 / float(R * R)              # gentle preference for nearby sources (same lighting)
    unk = hole.copy()
    guard = 0
    while unk.any() and guard < 4000:
        guard += 1
        kn = (~unk).astype(np.float32)
        conf = cv2.boxFilter(kn, -1, (B, B), normalize=True)
        front = unk & (cv2.dilate(kn.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        if not front.any():
            front = unk
        pri = conf * (0.25 + np.minimum(cv2.dilate(gm * kn, np.ones((3, 3), np.uint8)), 1.5))
        pri[~front] = -1.0
        py, px = np.unravel_index(int(np.argmax(pri)), pri.shape)
        y0, x0 = int(np.clip(py - r, 0, H - B)), int(np.clip(px - r, 0, W - B))
        kn_t = ~unk[y0:y0 + B, x0:x0 + B]
        tpl = np.where(kn_t[..., None], out[y0:y0 + B, x0:x0 + B], guide[y0:y0 + B, x0:x0 + B]).astype(np.float32)
        m = np.where(kn_t, 1.0, 0.45).astype(np.float32)   # SQDIFF weight = mask^2: known 1, guidance 0.2
        sy0, sy1 = max(0, y0 - R), min(Hs, y0 + R + 1)
        sx0, sx1 = max(0, x0 - R), min(Ws, x0 + R + 1)
        cand = ~badwin[sy0:sy1, sx0:sx1]
        if not cand.any():                                 # nothing valid nearby: search everywhere
            sy0, sy1, sx0, sx1 = 0, Hs, 0, Ws
            cand = ~badwin
        area = out[sy0:sy1 + B - 1, sx0:sx1 + B - 1]
        cost = cv2.matchTemplate(area, tpl, cv2.TM_SQDIFF, mask=np.repeat(m[..., None], 3, axis=2))
        cost = cost / max(float((m * m).sum()), 1.0)
        cost = cost + lam * ((yy[sy0:sy1, sx0:sx1] - y0) ** 2 + (xx[sy0:sy1, sx0:sx1] - x0) ** 2)
        cost[~cand] = np.inf
        j = int(np.argmin(cost))
        sy, sx = np.unravel_index(j, cost.shape)
        sy, sx = sy + sy0, sx + sx0
        src = img[sy:sy + B, sx:sx + B].copy()
        mk = kn_t
        if mk.sum() >= 6:                                  # small colour correction towards the target's known part
            src += (tpl[mk].mean(0) - src[mk].mean(0)) * 0.6
        wr = unk[y0:y0 + B, x0:x0 + B]
        blk = out[y0:y0 + B, x0:x0 + B]
        blk[wr] = src[wr]
        unk[y0:y0 + B, x0:x0 + B] = False
    return out


def _pr_seams(filled, hole):
    """Even out the patch seams inside the hole: keep the texture, pull the low frequencies towards a smooth fill
    from the hole's edge."""
    lo_in = cv2.GaussianBlur(filled, (0, 0), 3.0)
    edge = ~hole
    lo_ref = _pr_pushpull(_pr_lowpass(filled, edge, 3.0)[0], edge.astype(np.float32))
    corr = cv2.GaussianBlur(lo_ref - lo_in, (0, 0), 2.0)
    out = filled.copy()
    out[hole] = filled[hole] + 0.5 * corr[hole]
    return out


def _pr_grow(img, hole):
    """Join to the hole the dark rim of the mark the finger missed (only darker-than-skin pixels touching it)."""
    ys, xs = np.nonzero(hole)
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    bw = int(round(size * PATCH_CFG["grow_band"])) + 3
    band = cv2.dilate(hole.astype(np.uint8), _pr_disk(bw)).astype(bool) & ~hole
    valid = ~hole
    low, den = _pr_lowpass(img, valid, 3.0)
    lf = _pr_pushpull(low, (valid & (den > 0.25)).astype(np.float32))
    d = (img - lf).mean(axis=2)
    ring = cv2.dilate(hole.astype(np.uint8), _pr_disk(max(6, bw * 2))).astype(bool) & ~hole
    sig = float(np.median(np.abs(d[ring] - np.median(d[ring])))) * 1.4826 + 0.5 if ring.any() else 3.0
    dark = band & (d < -max(10.0, 3.5 * sig))
    if not dark.any():
        return hole
    n, lab = cv2.connectedComponents((dark | hole).astype(np.uint8), connectivity=8)
    touch = np.unique(lab[hole])
    add = np.isin(lab, touch[touch > 0]) & dark
    return hole | cv2.dilate(add.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def _pr_alpha(hole):
    """1 inside the painted area, a soft edge INSIDE it, 0 outside (v11.2.98: nothing outside the painted area changes)."""
    a = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), PATCH_CFG["feather_px"] * 0.7)
    return np.clip((a - 0.35) / 0.4, 0.0, 1.0) * hole.astype(np.float32)


def _pr_lift(around, filled, hole, core=None):
    """v11.2.98: no faint darkness may remain where the mark was. Inside the painted area, the fill over the mark's
    CORE (the pixels that were dark) is compared with the fill around it in the painted area and the untouched skin
    just outside; where the core is darker, it is lifted to that level - smoothly (a level, not detail), by at most
    lift_max levels. The painted area's overall brightness is left as it is; nothing outside it changes."""
    k = float(PATCH_CFG.get("lift", 0.0) or 0.0)
    if k <= 0 or not hole.any():
        return filled
    ys, xs = np.nonzero(hole)
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    q = 4 if size >= 64 else (2 if size >= 32 else 1)          # a smooth level: solved at reduced size for big marks
    if q > 1:
        H0, W0 = hole.shape
        sm = (max(4, W0 // q), max(4, H0 // q))
        dn = lambda x, interp=cv2.INTER_AREA: cv2.resize(x.astype(np.float32), sm, interpolation=interp)
        a2, f2 = dn(around), dn(filled)
        h2 = dn(hole.astype(np.float32)) > 0.5
        c2 = None if core is None else dn(core.astype(np.float32)) > 0.3
        lifted = _pr_lift_at(a2, f2, h2, c2, k)
        lift = cv2.resize((lifted - f2).mean(axis=2), (W0, H0), interpolation=cv2.INTER_LINEAR)
        return filled + (lift * hole)[..., None]
    return _pr_lift_at(around, filled, hole, core, k)


def _pr_lift_at(around, filled, hole, core, k):
    ys, xs = np.nonzero(hole)
    if len(ys) == 0:
        return filled
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    sig = max(1.5, size / 8.0)
    ag = around.mean(axis=2) if around.ndim == 3 else around
    fg = filled.mean(axis=2)
    if core is None:                                     # the mark: pixels clearly darker than the fill made for them
        core = hole & (ag < fg - 6.0)
    core = cv2.dilate(core.astype(np.uint8), _pr_disk(2)).astype(bool) & hole
    if core.sum() < 6:
        return filled
    outside = ~cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ring = cv2.dilate(hole.astype(np.uint8), _pr_disk(max(3, size // 6))).astype(bool) & outside
    refv = (hole & ~core) | ring                         # the fill around the mark + the skin just outside
    if refv.sum() < 20:
        return filled
    src = np.where(hole, fg, ag).astype(np.float32)
    lv, den = _pr_lowpass(src, refv, sig)
    ref = _pr_pushpull(lv, (refv & (den > 0.2)).astype(np.float32))
    sh = max(1.2, sig * 0.4)                             # the spot's own shape (not one level for the whole core)
    hv, _ = _pr_lowpass(fg, core, sh)                    # the fill's level over the core
    d = np.where(core, ref - hv, 0.0).astype(np.float32)
    w = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sh)
    d = cv2.GaussianBlur(d, (0, 0), sh) / np.maximum(w, 1e-3)
    lift = np.clip(d, 0.0, float(PATCH_CFG.get("lift_max", 8.0))) * k * np.clip(w * 1.5, 0.0, 1.0) * hole
    LIFT_STATS["n"] += 1
    LIFT_STATS["sum"] += float(lift[core].mean())
    return filled + lift[..., None]


LIFT_STATS = {"n": 0, "sum": 0.0}


def _pr_cover(roi, res, hole, core, ok):
    """v11.2.100: stronger cover. The stroke covers plain skin around the mark, and in the CURRENT frame that skin is
    there to be seen: it is the best reference for the level the fill should have. Over those margin pixels (inside the
    painted area, 2 px from its edge, away from the mark's core, not in front of an object) the fill is compared with
    the frame; where the fill is darker (a faint ghost of the mark, lighting that did not quite follow) its level is
    brought up to the frame's, smoothly and also over the mark itself. Brightening only, at most cover_max levels,
    fading to nothing at the painted area's edge; nothing outside the painted area changes."""
    k = float(PATCH_CFG.get("cover", 0.0) or 0.0)
    if k <= 0 or not hole.any():
        return res
    ys, xs = np.nonzero(hole)
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    Hf, Wf = hole.shape
    y0, y1 = max(0, int(ys.min()) - 6), min(Hf, int(ys.max()) + 7)          # work on the painted area's own box (the ROI is
    x0, x1 = max(0, int(xs.min()) - 6), min(Wf, int(xs.max()) + 7)          #  several times larger): same result, a fraction of the time
    if (y1 - y0) * (x1 - x0) < 0.6 * Hf * Wf:
        out = res.copy()
        out[y0:y1, x0:x1] = _pr_cover(roi[y0:y1, x0:x1], res[y0:y1, x0:x1], hole[y0:y1, x0:x1],
                                      None if core is None else core[y0:y1, x0:x1], ok[y0:y1, x0:x1])
        return out
    H, W = hole.shape
    inner = cv2.erode(hole.astype(np.uint8), _pr_disk(2)).astype(bool)
    if core is not None and core.any():
        inner &= ~cv2.dilate(core.astype(np.uint8), _pr_disk(3)).astype(bool)
    mv = inner & ok
    if mv.sum() < 30:
        return res
    e = (roi - res).mean(axis=2)                          # > 0: the fill is darker than the frame's own skin
    med = float(np.median(e[mv]))
    mad = float(np.median(np.abs(e[mv] - med))) * 1.4826 + 0.5
    mv &= np.abs(e - med) < max(6.0, 3.0 * mad)         # specks, hairs, a misplaced edge are not level
    if mv.sum() < 30:
        return res
    q = 4 if size >= 64 else (2 if size >= 32 else 1)
    sig = max(1.5, size / 8.0)
    if q > 1:
        sm = (max(4, W // q), max(4, H // q))
        v_lo = cv2.resize(mv.astype(np.float32), sm, interpolation=cv2.INTER_AREA)
        e_lo = cv2.resize(e * mv, sm, interpolation=cv2.INTER_AREA) / np.maximum(v_lo, 1e-3)
        low, den = _pr_lowpass(e_lo.astype(np.float32), v_lo > 0.5, sig / q + 0.5)
        fld = _pr_pushpull(low, ((v_lo > 0.5) & (den > 0.2)).astype(np.float32))
        fld = cv2.resize(fld.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        low, den = _pr_lowpass(e * mv, mv, sig)
        fld = _pr_pushpull(low, (mv & (den > 0.2)).astype(np.float32))
    ramp = np.clip((cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3) - 0.5) / 5.0, 0.0, 1.0)   # full 5.5 px in
    add = np.clip(fld, 0.0, float(PATCH_CFG.get("cover_max", 8.0))) * k * ramp * hole
    COVER_STATS["n"] += 1
    COVER_STATS["sum"] += float(add[hole].mean())
    return res + add[..., None]


COVER_STATS = {"n": 0, "sum": 0.0}


# Optional learned inpainter (set by the pipeline): fn(crop_bgr_uint8, mask_bool) -> filled crop_bgr_uint8, or None.
# v11.2.96 uses LaMa (Suvorov et al. 2022, "Resolution-robust Large Mask Inpainting with Fourier Convolutions",
# Apache-2.0; ONNX export from the OpenCV model zoo). It runs once per mark (anchor frame / preview), never per frame.
PATCH_INPAINTER = None


def _pr_model_fill(img, hole):
    """Fill with the learned inpainter on a crop around the hole (about 2.6x the mark). None if unavailable."""
    fn = PATCH_INPAINTER
    if fn is None:
        return None
    H, W = hole.shape
    ys, xs = np.nonzero(hole)
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    side = int(max(160, 2.6 * size))
    ch, cw = min(H, side), min(W, side)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    y0 = int(np.clip(cy - ch // 2, 0, H - ch))
    x0 = int(np.clip(cx - cw // 2, 0, W - cw))
    crop = np.clip(img[y0:y0 + ch, x0:x0 + cw] + 0.5, 0, 255).astype(np.uint8)
    mk = hole[y0:y0 + ch, x0:x0 + cw]
    if mk.sum() < hole.sum():                              # hole bigger than the crop: give up, use the fallback
        return None
    try:
        res = fn(crop, mk)
    except Exception as e:
        logging.warning("patch inpainter failed (%s) - exemplar fill instead", e)
        return None
    if res is None or res.shape != crop.shape:
        return None
    out = img.copy()
    blk = out[y0:y0 + ch, x0:x0 + cw]
    blk[mk] = res.astype(np.float32)[mk]
    s = 512.0 / max(ch, cw)
    if s < 0.9:                                            # the model saw it smaller: add back the skin's fine grain
        ring = cv2.dilate(hole.astype(np.uint8), _pr_disk(max(6, size // 4))).astype(bool) & ~hole
        hp = (img - cv2.GaussianBlur(img, (0, 0), 1.0)).mean(axis=2)[ring]
        sig = float(np.median(np.abs(hp - np.median(hp)))) * 1.4826 if hp.size > 20 else 0.0
        grain = cv2.GaussianBlur(np.random.default_rng(5).normal(0, 1, hole.shape).astype(np.float32), (0, 0), 0.6)
        grain *= sig / (float(grain.std()) + 1e-6) * math.sqrt(max(0.0, 1.0 - s * s))
        out[hole] += grain[hole][:, None]
    return out


_PR_EVID = {"v": 1.0}          # how much like the mark the predicted core looks (share of its anchor contrast), set by _pr_find_mark


def _pr_find_mark(g, hole, core, pol, con, area_exp, reach, zone_r=3):
    """v11.2.102: where is the MARK ITSELF in this picture? A mole, a tattoo, a birthmark differs from the skin around it by
    about ``con`` levels (darker when pol = +1, lighter when -1). Around the PREDICTED place (``hole``, searching ``reach`` px
    further) the skin level is taken from OUTSIDE the search area (a smooth interpolation of it); what departs from it with the
    mark's polarity by half of ``con`` or more, in pieces that touch the mark's own predicted core, are the mark's strong
    pixels (seeds), and the mark is those plus its soft edge (candidate pixels a few px from a seed at 30 % of ``con``).
    Works on a crop around the predicted place. -> (mask, (x, y) centroid, area, seeds) in ``g``'s coordinates, or None."""
    Hg, Wg = hole.shape
    ys, xs = np.nonzero(hole)
    if len(ys) < 4:
        return None
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    pad = int(reach) + 12
    cy0, cy1 = max(0, int(ys.min()) - pad), min(Hg, int(ys.max()) + 1 + pad)
    cx0, cx1 = max(0, int(xs.min()) - pad), min(Wg, int(xs.max()) + 1 + pad)
    hl = hole[cy0:cy1, cx0:cx1]
    gc = g[cy0:cy1, cx0:cx1].astype(np.float32)
    cr = None if core is None else core[cy0:cy1, cx0:cx1]
    dist = cv2.distanceTransform((~hl).astype(np.uint8), cv2.DIST_L2, 3)
    srch = dist <= reach
    outer = (dist > reach) & (dist <= reach + 6)
    if outer.sum() < 40:
        return None
    q = 2 if size >= 48 else 1                                  # the skin's level is smooth: work at half size on big marks
    if q > 1:
        sm = (max(4, gc.shape[1] // q), max(4, gc.shape[0] // q))
        o_lo = cv2.resize(outer.astype(np.float32), sm, interpolation=cv2.INTER_AREA) > 0.5
        g_lo = cv2.resize(gc, sm, interpolation=cv2.INTER_AREA)
        lo, den = _pr_lowpass(g_lo, o_lo, max(2.0, size / (6.0 * q)))
        sl = cv2.resize(_pr_pushpull(lo, (o_lo & (den > 0.15)).astype(np.float32)).astype(np.float32),
                        (gc.shape[1], gc.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        lo, den = _pr_lowpass(gc, outer, max(3.0, size / 6.0))
        sl = _pr_pushpull(lo, (outer & (den > 0.15)).astype(np.float32))
    c = cv2.GaussianBlur((pol * (sl - gc)).astype(np.float32), (0, 0), 0.8)
    med = float(np.median(c[outer]))
    if cr is not None and int(cr.sum()) >= 4:
        _PR_EVID["v"] = float(np.percentile(c[cr] - med, 75)) / max(con, 1e-3)
    # the threshold follows the picture's NOISE (its fine residual), not the scene's structure (shading, a nostril, the lips)
    fine = gc - cv2.GaussianBlur(gc, (0, 0), 1.5)
    sn = float(np.median(np.abs(fine[outer] - float(np.median(fine[outer]))))) * 1.4826
    thr = max(0.30 * con, 3.0 * sn + 1.5)
    cand = ((c - med) > thr) & srch
    if not cand.any():
        return None
    seed = ((c - med) > max(0.5 * con, thr)) & srch
    n, lab, st, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    zsrc = cr if cr is not None and int(cr.sum()) >= 4 else hl
    zone = cv2.distanceTransform((~zsrc).astype(np.uint8), cv2.DIST_L2, 3) <= zone_r
    amin = max(4.0, 0.03 * area_exp)
    pieces = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= amin and zone[lab == k].any()]
    if zone_r > 12 and len(pieces) > 1:                         # a wide search: the piece nearest the expected size
        pieces = [min(pieces, key=lambda k: abs(math.log(max(1.0, st[k, cv2.CC_STAT_AREA]) / max(1.0, area_exp))))]
    if not pieces:
        return None
    seeds = np.isin(lab, pieces)
    rg = max(1, int(round(0.25 * math.sqrt(float(seeds.sum()) / math.pi))))             # the soft edge: a quarter of the seeds' radius
    keep = cand & cv2.dilate(seeds.astype(np.uint8), _pr_disk(rg)).astype(bool)
    keep |= seeds
    area = int(keep.sum())
    if area < 0.12 * area_exp or area > 2.4 * area_exp:          # (a blob 3x the mark is the scene's structure, not the mark)
        return None
    y2, x2 = np.nonzero(keep)
    full = np.zeros((Hg, Wg), bool)
    full[cy0:cy1, cx0:cx1] = keep
    fseed = np.zeros((Hg, Wg), bool)
    fseed[cy0:cy1, cx0:cx1] = seeds
    return full, (float(x2.mean()) + cx0, float(y2.mean()) + cy0), area, fseed


def _pr_texture_clone(img, filled, hole):
    """v11.2.102: the fill's fine texture is the SKIN'S OWN. The inpainter returns the right colour and shading but a smoother
    skin than the skin around it. Its fine detail is replaced by the fine detail of a patch of clean skin beside the mark (same
    shape as the hole, taken from the best of a ring of candidate places: wholly plain skin, similar texture strength), copied
    pixel for pixel. Zero-mean detail only, so the level and shading of the fill are untouched. Made once, at the anchor; it
    is then carried with the skin."""
    if not PATCH_CFG.get("clone_texture", False) or hole.sum() < 30:
        return filled
    ys, xs = np.nonzero(hole)
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    H, W = hole.shape
    sig = 1.2
    hp = img - cv2.GaussianBlur(img, (0, 0), sig)
    known = ~cv2.dilate(hole.astype(np.uint8), _pr_disk(3)).astype(bool)
    g = img.mean(axis=2)
    lo, den = _pr_lowpass(g, known, max(3.0, size / 6.0))
    sl = _pr_pushpull(lo, (known & (den > 0.15)).astype(np.float32))
    an = np.abs(g - sl)
    ring = cv2.dilate(hole.astype(np.uint8), _pr_disk(max(8, size // 3))).astype(bool) & known
    if ring.sum() < 60:
        return filled
    s_an = float(np.median(an[ring])) * 1.4826 + 0.5
    bad = (an > max(4.0 * s_an, 8.0)) | ~known
    hp_g = hp.mean(axis=2)
    ref_sd = float(np.std(hp_g[ring]))
    best, best_sc = None, 1e9
    for rr in (1.0, 1.4, 1.9):
        for k in range(14):
            ang = 2.0 * math.pi * k / 14.0 + 0.3 * rr
            dx, dy = int(round(rr * (size + 10) * math.cos(ang))), int(round(rr * (size + 10) * math.sin(ang)))
            y2, x2 = ys + dy, xs + dx
            if y2.min() < 0 or x2.min() < 0 or y2.max() >= H or x2.max() >= W:
                continue
            if float(bad[y2, x2].mean()) > 0.02:
                continue
            seg = hp_g[y2, x2]
            if float(np.abs(seg).max()) > 5.0 * max(ref_sd, 1.0):        # a crease, an edge, a hair: not plain texture
                continue
            sd = float(np.std(seg))
            sc = abs(sd - ref_sd) / max(ref_sd, 1e-3) + 0.05 * rr
            if sc < best_sc:
                best, best_sc = (dy, dx), sc
    if best is None:
        return filled
    dy, dx = best
    out = filled.copy()
    lo_f = cv2.GaussianBlur(filled, (0, 0), sig)
    cap = 4.0 * max(ref_sd, 1.0)
    out[ys, xs] = lo_f[ys, xs] + np.clip(hp[ys + dy, xs + dx], -cap, cap)
    return out


def fill_still(img, hole):
    """Remove the mark under ``hole`` (bool) in float32 BGR ``img`` (a region of interest).
    -> (out, alpha, hole used, filled content)."""
    c = PATCH_CFG
    hole = hole.astype(bool)
    if c["halo_px"] > 0:
        hole = cv2.dilate(hole.astype(np.uint8), _pr_disk(c["halo_px"])).astype(bool)
    if float(PATCH_CFG.get("grow_band", 0.0) or 0.0) > 0:
        hole = _pr_grow(img, hole)
    filled = _pr_model_fill(img, hole)
    if filled is None:
        filled = _pr_seams(_pr_exemplar(img, hole), hole)
    filled = _pr_texture_clone(img, filled, hole)
    # (no lift here: the fill itself leaves no darkness - measured within +/-0.5 L of clean skin - and a lift on it
    #  overshot to a light spot; the lift acts per frame in PatchTrack.apply, where the faint darkness builds up)
    a = _pr_alpha(hole)
    return img * (1.0 - a[..., None]) + filled * a[..., None], a, hole, filled


def _pr_roi(hole, shape, extra=1.0):
    ys, xs = np.nonzero(hole)
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    pad = int(max(40, extra * (1.5 * size + 30)))
    H, W = shape[:2]
    return max(0, ys.min() - pad), min(H, ys.max() + 1 + pad), max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)


def remove_patch_image(img_bgr, mask):
    """Preview: remove the painted mark(s) on a still frame. img_bgr uint8; mask: painted where > 127 (any size,
    resized to the image). -> (out uint8, info list)."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m.max(axis=2)
    m = m > 127 if m.dtype != bool else m
    if m.shape[:2] != img_bgr.shape[:2]:
        m = cv2.resize(m.astype(np.uint8) * 255, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    out = img_bgr.copy()
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    info = []
    for k in range(1, n):
        comp = lab == k
        if comp.sum() < 4:
            continue
        y0, y1, x0, x1 = _pr_roi(comp, img_bgr.shape)
        roi = out[y0:y1, x0:x1].astype(np.float32)
        res, a, used, _ = fill_still(roi, comp[y0:y1, x0:x1])
        out[y0:y1, x0:x1] = np.clip(res + 0.5, 0, 255).astype(np.uint8)
        info.append({"area_px": int(used.sum()), "box": [int(x0), int(y0), int(x1), int(y1)]})
    return out, info


# --------------------------------------------------------------------------------------------------- tracking
def _pr_gray(f):
    return f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def _pr_ncc(a, b, m=None):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if m is not None:
        a, b = a[m], b[m]
    if a.size < 16:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum())) + 1e-6
    return float((a * b).sum()) / d


def _pr_inv(A):
    return cv2.invertAffineTransform(np.asarray(A, np.float64))


def _pr_compose(A, B):
    """A after B (both 2x3)."""
    return (np.vstack([A, [0, 0, 1]]) @ np.vstack([B, [0, 0, 1]]))[:2]


def _pr_pose_ok(A, lo=0.25, hi=4.0):
    """v11.2.105: a usable pose? Finite, and a scale between lo and hi. A prediction compounded from a wrong velocity can shrink to 0
    or blow up to inf / nan, and cv2.warpAffine with an infinite matrix never returns (a job hung for over an hour on one)."""
    try:
        A = np.asarray(A, np.float64)
        if A.shape != (2, 3) or not np.isfinite(A).all():
            return False
        sc = math.sqrt(abs(float(np.linalg.det(A[:, :2]))))
        return lo <= sc <= hi and abs(float(A[0, 2])) < 1e6 and abs(float(A[1, 2])) < 1e6
    except Exception:
        return False


class PatchTrack:
    """One mark: its anchor frame, its hole and fill (anchor coordinates); follows the area through the frames."""

    def __init__(self, anchor_idx, anchor_bgr, hole, make_fill=True):
        t0 = cv2.getTickCount()
        self.anchor_idx = int(anchor_idx)
        Hf, Wf = hole.shape
        fy0, fy1, fx0, fx1 = _pr_roi(hole, hole.shape)
        roi = anchor_bgr[fy0:fy1, fx0:fx1].astype(np.float32)
        if make_fill:
            _, _, used, filled = fill_still(roi, hole[fy0:fy1, fx0:fx1])
        else:                                                  # (a finder only: no fill needed)
            used = hole[fy0:fy1, fx0:fx1].copy()
            filled = roi
        self.off = (fx0, fy0)
        self.hole_roi = used                                   # bool, fill-ROI coords
        self.fill_roi = filled.astype(np.float32)              # anchor content with the mark removed
        self.orig_roi = roi                                    # anchor as it was (object gate, lighting)
        # the mark's core: pixels the removal really changes (the stroke around it is plain skin)
        chg = np.abs(cv2.GaussianBlur(roi, (0, 0), 1.5) - cv2.GaussianBlur(filled, (0, 0), 1.5)).max(axis=2)   # (not the texture)
        self.core_roi = used & (chg > max(10.0, float(np.percentile(chg[used], 60)) if used.any() else 10.0))
        # what the mark looks like against the skin around it (v11.2.102): polarity and strength, in grey levels
        self.core_area = float(self.core_roi.sum())
        self.mark_pol, self.mark_con = 1.0, 20.0
        self.mark_bgr = None
        self.skin_hair = False
        if self.core_area >= 6:
            dd = (filled.mean(axis=2) - roi.mean(axis=2))[self.core_roi]
            self.mark_pol = 1.0 if float(np.median(dd)) >= 0 else -1.0
            # how strongly the mark's CENTRE differs from the skin (the core's outer part is the soft edge): a high percentile
            self.mark_con = float(max(8.0, np.percentile(self.mark_pol * dd, 80)))
            strong = self.core_roi & ((self.mark_pol * (filled.mean(axis=2) - roi.mean(axis=2))) >= 0.6 * self.mark_con)
            self.mark_bgr = roi[strong].mean(axis=0) if int(strong.sum()) >= 4 else roi[self.core_roi].mean(axis=0)
        # Dark texture on lighter skin (body hair, a dark patch). The stroke itself must not drive the motion:
        # it is what we are removing, and locking onto it makes the paint flicker in place.
        if bool(PATCH_CFG.get("body_lock", True)) and self.core_area >= 12 and self.mark_pol > 0:
            try:
                g_roi = roi.mean(axis=2)
                ring = cv2.dilate(self.hole_roi.astype(np.uint8), _pr_disk(8)).astype(bool) & ~self.hole_roi
                if int(ring.sum()) >= 40:
                    skin_l = float(np.median(g_roi[ring]))
                    hair_l = float(np.median(g_roi[self.core_roi]))
                    if skin_l - hair_l >= 10.0:
                        self.skin_hair = True
                        extra = int(PATCH_CFG.get("hair_halo", 5) or 0)
                        if extra > 0:
                            grown = cv2.dilate(self.hole_roi.astype(np.uint8), _pr_disk(extra)).astype(bool)
                            _, _, used2, filled2 = fill_still(roi, grown)
                            self.hole_roi = used2
                            self.fill_roi = filled2.astype(np.float32)
                            used = used2
            except Exception:
                self.skin_hair = False
        self.weak = set()                                      # frames whose pose is a bridge / hold / rescue: paints only where the mark is found
        self.skin_anchor = None
        self.point_bound = set()
        self.skin_bind_n = 0
        self.skin_maps = {}
        self._skin_last = None
        self._skin_miss = 0
        self._oe_hat = None
        self._oe_d = None
        yh_, xh_ = np.nonzero(used)
        self._hbox = (int(xh_.min()), int(yh_.min()), int(xh_.max()), int(yh_.max())) if len(yh_) else (0, 0, 1, 1)
        full = np.zeros((Hf, Wf), bool)
        full[fy0:fy1, fx0:fx1] = used
        ys, xs = np.nonzero(full)
        self.size = int(max(np.ptp(ys), np.ptp(xs))) + 1
        pad = int(max(30, 1.0 * self.size + 24))               # tracking box: the mark and the skin around it
        self.box = (max(0, ys.min() - pad), min(Hf, ys.max() + 1 + pad), max(0, xs.min() - pad), min(Wf, xs.max() + 1 + pad))
        y0, y1, x0, x1 = self.box
        self.anchor_gray = _pr_gray(anchor_bgr)
        self.tmpl = cv2.GaussianBlur(self.anchor_gray[y0:y1, x0:x1], (5, 5), 1.0).astype(np.float32)
        self._det = bool(PATCH_CFG.get("detail_track", False))
        self.tmpl_c = _pr_detail(self.tmpl) if self._det else self.tmpl     # what _check / _ecc compare against
        tol = int(PATCH_CFG.get("detail_tol", 0) or 0) if self._det else 0
        self._tlo = cv2.erode(self.tmpl_c, np.ones((2 * tol + 1,) * 2, np.uint8)) if tol > 0 else None
        self._thi = cv2.dilate(self.tmpl_c, np.ones((2 * tol + 1,) * 2, np.uint8)) if tol > 0 else None
        gx, gy = cv2.Sobel(self.tmpl_c, cv2.CV_32F, 1, 0), cv2.Sobel(self.tmpl_c, cv2.CV_32F, 0, 1)
        self._tgrad = gx * gx + gy * gy                        # where the patch's structure is (the mark's edges)
        gx, gy = cv2.Sobel(self.tmpl, cv2.CV_32F, 1, 0), cv2.Sobel(self.tmpl, cv2.CV_32F, 0, 1)
        self._tgrad_g = gx * gx + gy * gy                      # (the same in the plain view)
        self.A = {}
        self.local_A = {}          # mark-centred pose; global A remains the reacquisition/validation pose
        self.conf = {}
        self.orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=6, edgeThreshold=15, patchSize=21)
        m = np.zeros(self.anchor_gray.shape, np.uint8)
        m[y0:y1, x0:x1] = 255
        self.kp, self.des = self.orb.detectAndCompute(self.anchor_gray, m)
        PATCH_STATS["fill_ms"] += (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()

    # -- alignment helpers --------------------------------------------------------------------------------------
    def _points(self, gray, A):
        y0, y1, x0, x1 = self.box
        H, W = gray.shape[:2]
        quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
        q = (quad @ A[:, :2].T + A[:, 2]).astype(np.int32)

        # Restrict mask allocation and corner detection to the clipped tracking
        # polygon ROI. This avoids a full-frame uint8 mask per track refresh.
        rx0 = max(0, int(q[:, 0].min()))
        ry0 = max(0, int(q[:, 1].min()))
        rx1 = min(W, int(q[:, 0].max()) + 1)
        ry1 = min(H, int(q[:, 1].max()) + 1)
        p = None
        if rx1 - rx0 >= 8 and ry1 - ry0 >= 8:
            local_q = q - np.array([rx0, ry0], np.int32)
            local_mask = np.zeros((ry1 - ry0, rx1 - rx0), np.uint8)
            cv2.fillConvexPoly(local_mask, local_q, 255)
            gray_roi = gray[ry0:ry1, rx0:rx1]
            p = cv2.goodFeaturesToTrack(
                gray_roi, maxCorners=260, qualityLevel=0.0025,
                minDistance=3, mask=local_mask, blockSize=5
            )
            if p is not None:
                p[:, 0, 0] += rx0
                p[:, 0, 1] += ry0
        if p is None or len(p) < 16:
            gy, gx = np.mgrid[y0:y1:7, x0:x1:7]
            g = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
            p = (g @ A[:, :2].T + A[:, 2]).astype(np.float32)
            inside = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < W) & (p[:, 1] < H)
            p = p[inside]
        return p.reshape(-1, 1, 2).astype(np.float32)

    def _boxwarp(self, gray, A, border=cv2.BORDER_REFLECT):
        y0, y1, x0, x1 = self.box
        M = _pr_compose(A, np.array([[1, 0, x0], [0, 1, y0]], np.float64))     # box coords -> frame
        cur = cv2.warpAffine(gray, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=border)
        return cur, M

    def _ecc(self, gray, A0, inl=None):
        """Refine the anchor->frame affine by aligning the anchor's patch to the frame (anchored: no drift).
        inl: pixels of the box known to match (an object in front is left out of the fit)."""
        y0, y1, x0, x1 = self.box
        cur, M = self._boxwarp(gray, A0)
        cur = cv2.GaussianBlur(cur, (5, 5), 1.0).astype(np.float32)
        if self._det:
            cur = _pr_detail(cur)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            msk = None if inl is None else inl.astype(np.uint8)
            cc, warp = cv2.findTransformECC(self.tmpl_c, cur, warp, cv2.MOTION_AFFINE,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4), msk, 3)
        except cv2.error:
            return A0, -1.0
        w, h = x1 - x0, y1 - y0
        if not np.isfinite(warp).all() or abs(warp[0, 2]) > 0.3 * w or abs(warp[1, 2]) > 0.3 * h:
            return A0, -1.0
        T = np.array([[1, 0, x0], [0, 1, y0]], np.float64)
        return _pr_compose(_pr_compose(M, warp.astype(np.float64)), _pr_inv(T)), float(cc)

    def _check(self, gray, A, plain=False):
        """Does the frame, aligned by A, show the anchor's patch? Robust to an object over part of it and to the
        area being partly out of the picture: only pixels that match count, but enough of them must match.
        plain: compare the plain brightness even when the lighting-free view is on (v11.2.99).
        -> (score, matching pixels of the box)."""
        det = self._det and not plain
        y0, y1, x0, x1 = self.box
        cur, M = self._boxwarp(gray, A, cv2.BORDER_CONSTANT)
        inside = cv2.warpAffine(np.full(gray.shape, 255, np.uint8), M, (x1 - x0, y1 - y0),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
        inside = cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if inside.mean() < 0.12:
            return -1.0, None
        c = cv2.GaussianBlur(cur, (5, 5), 1.0).astype(np.float32)
        t = self.tmpl_c if det else self.tmpl
        if det:                                               # v11.2.99: lighting-free view (a shadow cancels)
            c = _pr_detail(c, cv2.dilate(inside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        ci, ti = c[inside], t[inside]
        st = float(ti.std()) + 1e-3
        tmin = float(PATCH_CFG.get("detail_thr", 4.0)) if det else 6.0
        # brightness (contrast) of the frame vs the anchor, robustly (an object over part of the area must not skew it)
        if det:
            k = 1.0
        else:
            k = float(np.median(ci / np.maximum(ti, 1.0)))
            k = float(np.clip(k, 0.5, 2.0))
        thr = max(tmin, 0.45 * st * k)
        r0 = np.abs(ci - k * ti) < thr
        if r0.sum() > 50:
            k = float(np.clip((ci[r0] * ti[r0]).sum() / max(1e-6, float((ti[r0] * ti[r0]).sum())), 0.5, 2.0))
        if det and self._tlo is not None:                     # v11.2.99: a little spatial tolerance (bending skin)
            res = np.maximum(k * self._tlo - c, 0.0) + np.maximum(c - k * self._thi, 0.0)
        else:
            res = np.abs(c - k * t)
        thr = max(tmin, 0.45 * st * k)
        inl = inside & (cv2.dilate((res < thr).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (res < 2.5 * thr)
        frac = float(inl.sum()) / max(1.0, float(inside.sum()))
        self._last_frac = frac
        if frac < 0.40 or inl.sum() < 150:
            return -1.0, None
        ncc = _pr_ncc(c, t, inl)
        self._last_all = _pr_ncc(c, t, inside)                  # the plain score over everything in the picture
        G = self._tgrad if det else self._tgrad_g
        self._last_cov = float(G[inl].sum()) / max(1e-6, float(G[inside].sum()))   # share of the patch's structure matched
        # an object in front makes ONE compact region of disagreement; a wrong place makes it everywhere, scattered
        bad = (inside & ~inl).astype(np.uint8)
        nb = int(bad.sum())
        if nb < 30:
            self._last_compact = 1.0
        else:
            bad = cv2.morphologyEx(bad, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n_, lab_, st_, _ = cv2.connectedComponentsWithStats(bad, connectivity=8)
            self._last_compact = float(st_[1:, cv2.CC_STAT_AREA].max()) / max(1.0, float(st_[1:, cv2.CC_STAT_AREA].sum())) if n_ > 1 else 1.0
        return ncc * min(1.0, frac / 0.6), inl

    def _redetect_tm(self, gray, around=None, radius=160.0):
        """Find the anchor's patch by normalised template matching at half size, a few scales and turns (works on
        smooth skin where there are too few corners for ORB).
        around (v11.2.103): a pose; the search is then limited to ``radius`` px around where it puts the patch, and the turns and
        scales tried are centred on its own (a local search: cheap, and a repeat of the skin's texture elsewhere cannot win)."""
        y0, y1, x0, x1 = self.box
        t = cv2.resize(self.tmpl, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ox0 = oy0 = 0
        gsrc = gray
        a0, s0 = 0.0, 1.0
        if around is not None and not _pr_pose_ok(around):
            around = None                                      # (v11.2.105: search the whole frame rather than around a broken pose)
        if around is not None:
            cx_a, cy_a = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            pc = np.asarray(around, np.float64)[:, :2] @ np.array([cx_a, cy_a]) + np.asarray(around, np.float64)[:, 2]
            half = int(radius + 0.5 * max(x1 - x0, y1 - y0))
            gx0, gy0 = int(max(0, pc[0] - half)), int(max(0, pc[1] - half))
            gx1, gy1 = int(min(gray.shape[1], pc[0] + half)), int(min(gray.shape[0], pc[1] + half))
            if gx1 - gx0 < 24 or gy1 - gy0 < 24:
                return None
            gsrc = gray[gy0:gy1, gx0:gx1]
            ox0, oy0 = gx0, gy0
            a0 = math.degrees(math.atan2(float(around[0][1]), float(around[0][0])))      # (cv2.getRotationMatrix2D's sign)
            s0 = math.sqrt(abs(float(np.linalg.det(np.asarray(around, np.float64)[:, :2]))))
        g = cv2.GaussianBlur(cv2.resize(gsrc, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), (3, 3), 0.8).astype(np.float32)
        th, tw = t.shape
        best = (-2.0, None)
        scales = (0.9, 1.0, 1.1) if around is None else (s0 * 0.92, s0, s0 * 1.08)
        angles = (-8.0, 0.0, 8.0) if around is None else (a0 - 8.0, a0, a0 + 8.0)
        for scl in scales:
            for ang in angles:
                R = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), ang, scl)
                tt = cv2.warpAffine(t, R, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                cw, ch = int(tw * 0.7), int(th * 0.7)                      # the middle (no reflected border)
                ox, oy = (tw - cw) // 2, (th - ch) // 2
                core = tt[oy:oy + ch, ox:ox + cw]
                if g.shape[0] < ch or g.shape[1] < cw or float(core.std()) < 1.0:
                    continue
                r = cv2.matchTemplate(g, core, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                if mx > best[0]:
                    # pose: anchor box centre -> found centre, with this turn and scale
                    cx_a, cy_a = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                    cx_f, cy_f = (loc[0] + cw / 2.0) * 2.0 + ox0, (loc[1] + ch / 2.0) * 2.0 + oy0
                    A = cv2.getRotationMatrix2D((cx_a, cy_a), ang, scl)
                    A[0, 2] += cx_f - cx_a
                    A[1, 2] += cy_f - cy_a
                    best = (mx, A.astype(np.float64))
        return best[1] if best[0] >= 0.5 else None

    def _redetect(self, gray):
        if self.des is None or len(self.kp) < 8:
            return None
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 8:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.des, des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) < 8:
            return None
        src = np.float32([self.kp[g.queryIdx].pt for g in good])
        dst = np.float32([kp[g.trainIdx].pt for g in good])
        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if A is None or inl is None or int(inl.sum()) < 8:
            return None
        return A.astype(np.float64)

    def _ring_pose(self, gray, A0):
        """Affine of the surface around the stroke, with the stroke itself excluded.

        Matching the dark texture inside the paint makes the removal sit still while the body moves, or
        jump between strands. The ring (skin, cloth, creases beside the stroke) is what moves with the body.
        """
        if not bool(PATCH_CFG.get("body_lock", True)) or A0 is None or not _pr_pose_ok(A0):
            return A0
        try:
            y0, y1, x0, x1 = self.box
            h, w = y1 - y0, x1 - x0
            if h < 16 or w < 16:
                return A0
            hole = np.zeros((h, w), np.uint8)
            fx0, fy0 = self.off
            hh, hw = self.hole_roi.shape
            oy, ox = fy0 - y0, fx0 - x0
            yA, yB = max(0, oy), min(h, oy + hh)
            xA, xB = max(0, ox), min(w, ox + hw)
            if yB > yA and xB > xA:
                sy0, sx0 = yA - oy, xA - ox
                src = self.hole_roi[sy0:sy0 + (yB - yA), sx0:sx0 + (xB - xA)]
                hole[yA:yB, xA:xB] = src.astype(np.uint8)
            excl = cv2.dilate(hole, _pr_disk(max(4, int(0.18 * self.size))))
            ring = np.full((h, w), 255, np.uint8)
            ring[excl > 0] = 0
            m = max(2, int(0.04 * min(h, w)))
            ring[:m, :] = 0
            ring[-m:, :] = 0
            ring[:, :m] = 0
            ring[:, -m:] = 0
            if int((ring > 0).sum()) < 800:
                return A0
            cur, M = self._boxwarp(gray, A0)
            cur = cv2.GaussianBlur(cur, (5, 5), 1.0).astype(np.float32)
            tmpl = self.tmpl
            if self._det:
                cur = _pr_detail(cur, cv2.dilate(ring, np.ones((5, 5), np.uint8)) > 0)
                tmpl = self.tmpl_c
            warp = np.eye(2, 3, dtype=np.float32)
            cc, warp = cv2.findTransformECC(
                tmpl, cur, warp, cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4), ring, 3)
            if not np.isfinite(warp).all() or abs(float(warp[0, 2])) > 0.18 * w or abs(float(warp[1, 2])) > 0.18 * h:
                return A0
            if float(cc) < 0.30:
                return A0
            T = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0]], np.float64)
            refined = _pr_compose(_pr_compose(M, warp.astype(np.float64)), _pr_inv(T))
            return refined if _pr_pose_ok(refined) else A0
        except Exception:
            return A0

    def _init_skin_points(self, gray):
        """Corners in the skin around the stroke, taken once on the painted frame."""
        self.skin_anchor = None
        self._skin_last = np.eye(2, 3, dtype=np.float64)
        self._skin_miss = 0
        self._oe_hat = None
        self._oe_d = None
        try:
            H, W = gray.shape[:2]
            fx0, fy0 = self.off
            hh, hw = self.hole_roi.shape
            hole = np.zeros((H, W), np.uint8)
            hole[fy0:fy0 + hh, fx0:fx0 + hw] = self.hole_roi.astype(np.uint8)
            if int(hole.sum()) < 8:
                return
            r_in = 2
            r_out = max(12, int(0.20 * self.size))
            inner = cv2.dilate(hole, _pr_disk(r_in))
            outer = cv2.dilate(hole, _pr_disk(r_out))
            ring = ((outer > 0) & (inner == 0)).astype(np.uint8) * 255
            if int((ring > 0).sum()) < 40:
                return
            pts = cv2.goodFeaturesToTrack(gray, maxCorners=70, qualityLevel=0.008,
                                          minDistance=5, mask=ring, blockSize=5)
            got = [] if pts is None else [p.reshape(2) for p in pts]
            if len(got) < 24:
                ys, xs = np.nonzero(ring)
                step = max(1, len(ys) // 48)
                for x, y in zip(xs[::step], ys[::step]):
                    got.append(np.array([x, y], np.float32))
                    if len(got) >= 64:
                        break
            if len(got) < 8:
                return
            self.skin_anchor = np.asarray(got[:70], np.float32).reshape(-1, 2)
            self.skin_bind_n = int(len(self.skin_anchor))
        except Exception:
            self.skin_anchor = None

    def _skin_point_pose(self, gray, prev, filt=True):
        """Bind the cover to the nearby skin, measured against the painted frame
        every time so it cannot drift. Points are smoothed with a One Euro filter:
        still skin is steady, fast skin is followed. The cover then bends with
        those points instead of freezing when the body moves."""
        src0 = getattr(self, "skin_anchor", None)
        if src0 is None or len(src0) < 8 or gray is None:
            return None
        try:
            guess = self._oe_hat if self._oe_hat is not None else src0
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                self.anchor_gray, gray, src0.reshape(-1, 1, 2), guess.reshape(-1, 1, 2).astype(np.float32),
                winSize=(41, 41), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.03),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(
                gray, self.anchor_gray, nxt, src0.reshape(-1, 1, 2),
                winSize=(41, 41), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.03),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            fb = np.linalg.norm(back.reshape(-1, 2) - src0, axis=1)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 2.0) & np.isfinite(nxt.reshape(-1, 2)).all(axis=1)
            if int(ok.sum()) < 8:
                self._skin_miss += 1
                return None if self._skin_miss > 4 or self._skin_last is None else self._skin_last.copy()
            src = src0[ok]
            dst = nxt.reshape(-1, 2)[ok]
            A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.99)
            if A is None or inl is None or int(inl.sum()) < 8 or not _pr_pose_ok(A, 0.5, 2.0):
                self._skin_miss += 1
                return None if self._skin_miss > 4 or self._skin_last is None else self._skin_last.copy()
            A = np.asarray(A, np.float64)
            rigid = cv2.transform(src0.reshape(-1, 1, 2), A).reshape(-1, 2)
            measured = rigid.copy()
            measured[ok] = dst
            if not filt:
                self.skin_maps[self._skin_idx] = (src0.astype(np.float64), measured.astype(np.float64))
                return A
            dt = 1.0 / 30.0
            if self._oe_hat is None or len(self._oe_hat) != len(measured):
                self._oe_hat = measured.copy()
                self._oe_d = np.zeros_like(measured)
            else:
                dx = (measured - self._oe_hat) / dt
                ad = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * 1.0)) / dt)
                self._oe_d = ad * dx + (1.0 - ad) * self._oe_d
                speed = np.linalg.norm(self._oe_d, axis=1)
                cutoff = 0.8 + 0.04 * speed
                tau = 1.0 / (2.0 * math.pi * cutoff)
                a = 1.0 / (1.0 + tau / dt)
                self._oe_hat = a[:, None] * measured + (1.0 - a[:, None]) * self._oe_hat
            self.skin_maps[self._skin_idx] = (src0.astype(np.float64), self._oe_hat.astype(np.float64))
            self._skin_last = A
            self._skin_miss = 0
            return A
        except Exception:
            return None

    def _local_pose(self, anchor_pts, current_pts, global_A):
        """Fit a mark-centred affine from LK correspondences.

        The global tracker deliberately uses the whole surrounding patch for robust identity/reacquisition.
        Skin and soft tissue can deform locally, however, so the transform used to place the removal mask
        is refined from correspondences close to the painted region. Reject weak or implausible fits and
        retain the validated global transform as the safe fallback.
        """
        if not PATCH_CFG.get("local_deform", True) or anchor_pts is None or current_pts is None:
            return global_A
        try:
            ap = np.asarray(anchor_pts, np.float32).reshape(-1, 2)
            cp = np.asarray(current_pts, np.float32).reshape(-1, 2)
            if len(ap) != len(cp) or len(ap) < int(PATCH_CFG.get("local_min_points", 6)):
                return global_A
            x0, y0 = self.off
            hx0, hy0, hx1, hy1 = self._hbox
            cx, cy = x0 + 0.5 * (hx0 + hx1), y0 + 0.5 * (hy0 + hy1)
            radius = max(16.0, float(PATCH_CFG.get("local_fit_radius", 1.25)) * self.size)
            sel = (np.abs(ap[:, 0] - cx) <= radius) & (np.abs(ap[:, 1] - cy) <= radius)
            if int(sel.sum()) < int(PATCH_CFG.get("local_min_points", 6)):
                return global_A
            local, inl = cv2.estimateAffinePartial2D(ap[sel], cp[sel], method=cv2.RANSAC,
                                                      ransacReprojThreshold=3.0, maxIters=900,
                                                      confidence=0.997, refineIters=15)
            need = int(PATCH_CFG.get("local_min_points", 6))
            if local is None or inl is None or int(inl.sum()) < need or not _pr_pose_ok(local):
                return global_A
            # Local and global fits must describe the same neighbourhood, not a different object.
            p = np.array([cx, cy, 1.0], np.float64)
            if np.linalg.norm(local @ p - global_A @ p) > max(8.0, 0.22 * self.size):
                return global_A
            sl = math.sqrt(abs(float(np.linalg.det(local[:, :2]))))
            sg = math.sqrt(abs(float(np.linalg.det(global_A[:, :2]))))
            if sg <= 1e-6 or not (0.78 <= sl / sg <= 1.28):
                return global_A
            return local.astype(np.float64)
        except Exception:
            return global_A

    def _place_pose(self, gray, A, apts, pts):
        """Pose that places the paint. Dark-on-skin strokes follow the surface around the stroke."""
        if getattr(self, "skin_hair", False):
            return self._ring_pose(gray, A)
        if pts is not None and apts is not None:
            return self._local_pose(apts, pts, A)
        return np.array(A, np.float64, copy=True)

    def _dense_body_correspondences(self, prev, gray, lastA):
        """Dense, local body-surface motion samples around the selected patch.

        Sparse corners can disappear on smooth skin or fabric.  This supplements
        LK with forward/backward-checked Farneback flow restricted to the painted
        region and its immediate context.  Coordinates are returned in anchor
        space, so the local pose remains attached to the selected body surface.
        """
        try:
            if prev is None or lastA is None or not _pr_pose_ok(lastA):
                return None, None
            H, W = gray.shape[:2]
            # Patch bounds in the previous frame, expanded enough to include
            # nearby texture that deforms with the same local body segment.
            hx0, hy0, hx1, hy1 = self._hbox
            x0, y0 = self.off
            corners = np.array([[x0+hx0, y0+hy0], [x0+hx1, y0+hy0],
                                [x0+hx1, y0+hy1], [x0+hx0, y0+hy1]], np.float64)
            q = corners @ lastA[:, :2].T + lastA[:, 2]
            pad = int(max(14, min(64, 0.55 * self.size)))
            xlo = max(1, int(np.floor(q[:,0].min()))-pad); xhi = min(W-2, int(np.ceil(q[:,0].max()))+pad)
            ylo = max(1, int(np.floor(q[:,1].min()))-pad); yhi = min(H-2, int(np.ceil(q[:,1].max()))+pad)
            if xhi-xlo < 16 or yhi-ylo < 16:
                return None, None
            fwd = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 17, 3, 5, 1.1, 0)
            bwd = cv2.calcOpticalFlowFarneback(gray, prev, None, 0.5, 3, 17, 3, 5, 1.1, 0)
            step = max(4, min(8, int(self.size / 12)))
            gy, gx = np.mgrid[ylo:yhi:step, xlo:xhi:step]
            p0 = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
            ix = np.clip(np.rint(p0[:,0]).astype(int), 0, W-1)
            iy = np.clip(np.rint(p0[:,1]).astype(int), 0, H-1)
            d = fwd[iy, ix]
            p1 = p0 + d
            jx = np.clip(np.rint(p1[:,0]).astype(int), 0, W-1)
            jy = np.clip(np.rint(p1[:,1]).astype(int), 0, H-1)
            back = bwd[jy, jx]
            fb = np.linalg.norm(d + back, axis=1)
            valid = (fb < 1.35) & (p1[:,0] >= 1) & (p1[:,0] < W-1) & (p1[:,1] >= 1) & (p1[:,1] < H-1)
            # Avoid flat/occluded samples: require local image gradient in either frame.
            gmag0 = cv2.magnitude(cv2.Sobel(prev, cv2.CV_32F, 1, 0), cv2.Sobel(prev, cv2.CV_32F, 0, 1))
            gmag1 = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
            valid &= np.maximum(gmag0[iy,ix], gmag1[jy,jx]) > 2.0
            if int(valid.sum()) < 8:
                return None, None
            anchor_pts = cv2.transform(p0[valid].reshape(-1,1,2), _pr_inv(lastA)).reshape(-1,2)
            current_pts = p1[valid].reshape(-1,1,2)
            return current_pts.astype(np.float32), anchor_pts.reshape(-1,1,2).astype(np.float32)
        except Exception:
            return None, None

    def _step(self, gray, prev, pts, apts, lastA, velA, lost, strict_all=False):
        """One frame. Candidates: flow (LK points -> RANSAC affine, points that disagree with the motion model left
        out - an object crossing the area must not drag the pose), the motion model's prediction, and (when lost)
        ORB re-detection. Each is aligned by ECC and scored; the best acceptable one wins.
        -> (A or None, score, pts, apts)."""
        pred = None
        if lastA is not None:
            pred = lastA if velA is None else _pr_compose(velA, lastA)
            for _ in range(min(lost, 12)):
                if velA is not None:
                    pred = _pr_compose(velA, pred)
            if not _pr_pose_ok(pred):                          # (v11.2.105: a runaway prediction - hold the last pose instead)
                pred = lastA if _pr_pose_ok(lastA) else None
        flow, fpts, fapts = None, None, None
        if prev is not None and pts is not None and len(pts) >= 6:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, winSize=(21, 21), maxLevel=3)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, nxt, None, winSize=(21, 21), maxLevel=3)
            fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
            if pred is not None and ok.sum() >= 6:
                exp = cv2.transform(apts, pred).reshape(-1, 2)
                dev = np.linalg.norm(nxt.reshape(-1, 2) - exp, axis=1)
                tol = max(6.0, 0.15 * self.size) * (1 + min(lost, 4))
                ok &= dev < tol
            if ok.sum() >= 6:
                An, inl = cv2.estimateAffine2D(apts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2),
                                               method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if An is not None and inl is not None and int(inl.sum()) >= 6:
                    flow = An.astype(np.float64)
                    sel = np.nonzero(ok)[0][inl.ravel() > 0]
                    fpts, fapts = nxt[sel], apts[sel]
        # v11.2.111: reinforce sparse LK with dense, forward/backward-checked
        # flow local to the selected body region. This improves attachment when
        # the subject bends, turns or moves over low-texture skin/fabric.
        if prev is not None and lastA is not None:
            dpts, dapts = self._dense_body_correspondences(prev, gray, lastA)
            if dpts is not None and dapts is not None:
                if fpts is None or fapts is None:
                    fpts, fapts = dpts, dapts
                else:
                    fpts = np.concatenate([fpts.reshape(-1,1,2), dpts.reshape(-1,1,2)], axis=0)
                    fapts = np.concatenate([fapts.reshape(-1,1,2), dapts.reshape(-1,1,2)], axis=0)
        cands = []
        if flow is not None:
            cands.append(("flow", flow))
        if pred is not None:
            cands.append(("pred", pred))
        best = self._pick(gray, cands, lost, strict_all)
        # v11.2.103: when the followed pose and the prediction do not fit (a fast move: the body part is further than the
        # constant-velocity guess, and blurred), look for the patch NOW - near where it was first, then in the whole frame -
        # instead of waiting two lost frames (the mark showed through the paint on every one of them)
        if best is None and PATCH_CFG.get("redetect_now", True):
            near_ = self._redetect_tm(gray, around=(lastA if pred is None else pred), radius=float(PATCH_CFG.get("tm_radius", 160)))
            if near_ is not None:
                best = self._pick(gray, [("tm", near_)], lost, strict_all)
        if best is None and (not cands or lost >= 2 or PATCH_CFG.get("redetect_now", True)):
            far = [("orb", r) for r in (self._redetect(gray), self._redetect_tm(gray)) if r is not None]
            if far:
                best = self._pick(gray, far, lost, strict_all)
        if best is None:
            return None, -1.0, None, None
        q, A1, kind = best
        if kind == "flow":
            return A1, q, fpts, fapts
        return A1, q, None, None

    def _pick(self, gray, cands, lost, strict_all=False):
        """Align each candidate pose by ECC, score it, return the best acceptable (q, A, kind) or None."""
        best = None
        for kind, A0 in cands:
            if not _pr_pose_ok(A0):
                continue
            sc0, inl0 = self._check(gray, A0)
            if inl0 is None and self._det:
                sc0, inl0 = self._check(gray, A0, plain=True)
            if inl0 is None and kind != "orb":
                continue
            Ar, cc = self._ecc(gray, A0, inl0 if (inl0 is not None and inl0.mean() < 0.9) else None)
            for A1 in ((Ar, A0) if cc > 0.5 else (A0,)):
                # v11.2.99: the lighting-free view first (a shadow moving over the mark), then the plain view (an
                # object over part of it: in the lighting-free view only its two edges disagree, not one region)
                q = None
                for plain in ((False, True) if self._det else (False,)):
                    sc, inl = self._check(gray, A1, plain=plain)
                    if inl is None:
                        continue
                    al, cov = self._last_all, self._last_cov
                    # re-detected anywhere in the frame, or predicted across a gap: it must match the patch tightly
                    strict = strict_all or kind in ("orb", "tm") or (kind == "pred" and lost >= 1) or (lost >= 3 and kind != "flow")
                    whole = (al >= 0.8 and cov >= 0.7) if strict else (al >= PATCH_CFG["track_ncc_min"] and cov >= 0.5)
                    if not whole and strict and not strict_all and kind in ("orb", "tm") and PATCH_CFG.get("blur_ok", True):
                        # motion blur lowers the plain correlation of a sharp anchor against a smeared frame, but nearly all of its
                        # structure still lines up (coverage) and the lighting-free score stays high
                        whole = al >= 0.58 and cov >= 0.85 and sc >= 0.72
                    part = (not strict_all and kind != "orb" and sc >= 0.85 and cov >= 0.3 and lost <= 10
                            and self._last_compact >= 0.6)                         # partly covered by one object
                    if whole or part:
                        q = sc * (0.5 + 0.5 * cov)
                        break
                if q is None:
                    continue
                if best is None or q > best[0]:
                    best = (q, A1, kind)
                break
        return best

    def track(self, gray_at, idxs, tick=None):
        """Follow the area over the frame indices idxs (moving away from the anchor, the anchor first). After a
        gap, the area found again is also followed back into the gap. tick() is called once per frame; if it returns
        True the following stops (progress reporting and cancelling, v11.2.101)."""
        t0 = cv2.getTickCount()
        prev, pts, apts, lastA, velA = None, None, None, None, None
        lost = 0
        for p, idx in enumerate(idxs):
            if tick is not None and tick():
                break
            gray = gray_at(idx)
            if idx == self.anchor_idx:
                A = np.eye(2, 3, dtype=np.float64)
                self.A[idx], self.local_A[idx], self.conf[idx] = A, A.copy(), 1.0
                self._init_skin_points(gray)
                self.point_bound.add(idx)
                pts = self._points(gray, A)
                apts = pts.copy()
                prev, lastA = gray, A
                continue
            if idx in self.A and lost == 0:                    # (already followed by a back-fill)
                newA = self.A[idx]
                velA = _pr_compose(newA, _pr_inv(lastA)) if lastA is not None else None
                lastA = newA
                pts = self._points(gray, newA)
                apts = cv2.transform(pts, _pr_inv(newA)).astype(np.float32)
                prev = gray
                continue
            A, sc, pts, apts = self._step(gray, prev, pts, apts, lastA, velA, lost)
            if A is None and lost >= 2 and lost % 2 == 0:
                r = self._redetect(gray)
                if r is not None:
                    A, sc, pts, apts = self._step(gray, None, None, None, r, None, 0)
            if A is None and lastA is not None and lost <= 45 and PATCH_CFG.get("rescue", False):
                pred = lastA if velA is None else _pr_compose(velA, lastA)
                for _ in range(min(lost, 12)):
                    if velA is not None:
                        pred = _pr_compose(velA, pred)
                if not _pr_pose_ok(pred):
                    pred = lastA
                r = self._rescue(gray, pred) if _pr_pose_ok(pred) else None
                if r is not None:                                   # followed by the mark itself: weak, paints only where it is seen
                    A, sc, pts, apts = r, 0.45, None, None
                    self.weak.add(idx)
                    PATCH_STATS["frames_rescued"] += 1
            if A is not None:
                self.A[idx], self.conf[idx] = A, sc
                self._skin_idx = idx
                skinA = self._skin_point_pose(gray, prev) if self.skin_anchor is not None else None
                if skinA is not None:
                    self.local_A[idx] = skinA
                    self.point_bound.add(idx)
                else:
                    self.local_A[idx] = self._place_pose(gray, A, apts, pts)
                PATCH_STATS["frames_tracked"] += 1
                if lastA is not None and lost == 0:
                    velA = _pr_compose(A, _pr_inv(lastA))
                else:
                    velA = None
                if lost > 0:                                   # found again: walk back into the gap
                    self._backfill(gray_at, idxs[max(0, p - lost):p][::-1], gray, A)
                lastA = A
                lost = 0
                if pts is None or len(pts) < 40:
                    pts = self._points(gray, A)
                    apts = cv2.transform(pts, _pr_inv(A)).astype(np.float32)
            else:
                lost += 1
                PATCH_STATS["frames_lost"] += 1
                pts = None
                if lost > 45:
                    lastA, velA = None, None
            prev = gray
        PATCH_STATS["track_ms"] += (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()

    def _backfill(self, gray_at, gap, gray0, A0):
        prev, lastA, velA = gray0, A0, None
        pts = self._points(gray0, A0)
        apts = cv2.transform(pts, _pr_inv(A0)).astype(np.float32)
        for idx in gap:
            if idx in self.A:
                break
            g = gray_at(idx)
            A, sc, pts, apts = self._step(g, prev, pts, apts, lastA, velA, 0, strict_all=True)
            if A is None:
                break
            self.A[idx], self.conf[idx] = A, sc
            self._skin_idx = idx
            skinA = self._skin_point_pose(g, prev, filt=False) if self.skin_anchor is not None else None
            if skinA is not None:
                self.local_A[idx] = skinA
                self.point_bound.add(idx)
            else:
                self.local_A[idx] = self._place_pose(g, A, apts, pts)
            PATCH_STATS["frames_tracked"] += 1
            velA = _pr_compose(A, _pr_inv(lastA))
            lastA = A
            if pts is None or len(pts) < 40:
                pts = self._points(g, A)
                apts = cv2.transform(pts, _pr_inv(A)).astype(np.float32)
            prev = g

    def bridge(self):
        """Fill gaps of up to bridge_max frames between two followed frames whose poses agree, and hold the pose of a run's
        last (first) frame for hold_max frames into a longer gap or past the ends. These poses are WEAK (v11.2.102): the paint
        is applied on them only where the mark itself is found (find_mark), so a bridge across a blurred or turning stretch
        keeps the mark covered and a wrong guess paints nothing."""
        gmax = int(PATCH_CFG.get("bridge_max", 0) or 0)
        hmax = int(PATCH_CFG.get("hold_max", 0) or 0)
        if (gmax <= 0 and hmax <= 0) or not self.A:
            return
        ks = sorted(self.A)
        cx, cy = (self.box[2] + self.box[3]) / 2.0, (self.box[0] + self.box[1]) / 2.0
        c = np.array([cx, cy, 1.0])
        tol = 0.25 * self.size + 10.0
        new = {}
        strong_skip = set()
        for a, b in zip(ks[:-1], ks[1:]):
            g = b - a - 1
            if g < 1:
                continue
            Aa, Ab = self.A[a], self.A[b]
            if gmax > 0 and g <= gmax and np.linalg.norm(Aa @ c - Ab @ c) <= tol + 40.0 * (g + 1):
                ends_weak = a in self.weak or b in self.weak     # (a rescued end is a guess: what lies between is too)
                for j in range(1, g + 1):
                    w = j / float(g + 1)
                    new[a + j] = ((1.0 - w) * Aa + w * Ab, 0.55 if g <= 6 else 0.4)
                    if g > 6 or ends_weak:                       # (a short bridge between two followed frames is as good as one)
                        strong_skip.add(a + j)
                    PATCH_STATS["frames_bridged"] += 1
            elif hmax > 0:                                       # a longer gap: hold both ends' poses into it
                for j in range(1, min(hmax, g // 2 + 1) + 1):
                    new.setdefault(a + j, (Aa, 0.3))
                    new.setdefault(b - j, (Ab, 0.3))
        if hmax > 0:                                             # past the ends of the followed frames
            for j in range(1, hmax + 1):
                new.setdefault(ks[-1] + j, (self.A[ks[-1]], 0.3))
                if ks[0] - j >= 0:
                    new.setdefault(ks[0] - j, (self.A[ks[0]], 0.3))
        for k, (Ak, cf) in new.items():
            if k not in self.A:
                self.A[k], self.conf[k] = Ak, cf
                self.local_A[k] = Ak.copy()
                if k in strong_skip or cf < 0.5:
                    self.weak.add(k)

    def _rescue(self, gray, pred):
        """The patch's texture could not be matched on this frame (blur, a turn, a change of light), but the MARK may still be
        seen near where the motion puts it: find it (as find_mark does) and place the pose on it. -> pose or None."""
        if not PATCH_CFG.get("rescue", False) or self.core_area < 6:
            return None
        Hg, Wg = gray.shape[:2]
        h0, w0 = self.hole_roi.shape
        M = _pr_compose(pred, np.array([[1, 0, self.off[0]], [0, 1, self.off[1]]], np.float64))
        ys, xs = np.nonzero(self.hole_roi)
        corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], np.float64)
        q = corners @ M[:, :2].T + M[:, 2]
        scale = math.sqrt(abs(float(np.linalg.det(pred[:, :2]))))
        reach = int(min(70, np.clip(0.5 * self.size * scale + 6, 6, 45) * 1.8))
        px0, py0 = int(max(0, np.floor(q[:, 0].min()) - reach - 10)), int(max(0, np.floor(q[:, 1].min()) - reach - 10))
        px1, py1 = int(min(Wg, np.ceil(q[:, 0].max()) + reach + 10)), int(min(Hg, np.ceil(q[:, 1].max()) + reach + 10))
        if px1 - px0 < 16 or py1 - py0 < 16:
            return None
        Mc = M.copy()
        Mc[:, 2] -= (px0, py0)
        sz = (px1 - px0, py1 - py0)
        hole_c = cv2.warpAffine(self.hole_roi.astype(np.uint8) * 255, Mc, sz, flags=cv2.INTER_LINEAR) > 127
        core_c = cv2.warpAffine(self.core_roi.astype(np.uint8) * 255, Mc, sz, flags=cv2.INTER_LINEAR) > 127
        if hole_c.sum() < 4 or core_c.sum() < 4:
            return None
        r = _pr_find_mark(gray[py0:py1, px0:px1].astype(np.float32), hole_c, core_c, self.mark_pol, self.mark_con,
                          self.core_area * scale * scale, reach)
        if r is None:
            return None
        _, cen, area, _s = r
        if not (0.5 <= area / max(1.0, float(core_c.sum())) <= 2.0):
            return None
        yy, xx = np.nonzero(core_c)
        A2 = pred.copy()
        A2[:, 2] += (cen[0] - float(xx.mean()), cen[1] - float(yy.mean()))
        return A2

    def smooth(self):
        """Centred smoothing of the transforms (removes jitter). Symmetric only: near the end of a run the window
        shrinks on both sides, so the smoothing never lags behind a moving mark."""
        self.bridge()
        k = np.array(PATCH_CFG["smooth"], np.float64)
        r = len(k) // 2
        out = {}
        tol = 0.25 * self.size + 10.0                         # a neighbour this far off is not averaged in
        cx, cy = (self.box[2] + self.box[3]) / 2.0, (self.box[0] + self.box[1]) / 2.0
        ctr = {i: A @ np.array([cx, cy, 1.0]) for i, A in self.A.items()}
        # v11.2.103: the window also shrinks where the motion CURVES (a fast move that turns round): smoothing a curve with a window of
        # variance v moves it by about half its acceleration x v - 16 px at the turn of a 70 px/frame swing, and the paint sat off the mark
        curv = {}
        for i in self.A:
            if (i - 1) in self.A and (i + 1) in self.A:
                curv[i] = float(np.linalg.norm(ctr[i - 1][:2] - 2.0 * ctr[i][:2] + ctr[i + 1][:2]))
        budget = float(PATCH_CFG.get("smooth_err", 1.0))

        def _var(rr_):
            w_ = k[r - rr_:r + rr_ + 1]
            return float((w_ * (np.arange(-rr_, rr_ + 1) ** 2)).sum() / w_.sum())
        for i in self.A:
            rr = 0
            while (rr < r and (i - rr - 1) in self.A and (i + rr + 1) in self.A
                   and np.linalg.norm(ctr[i - rr - 1] - ctr[i]) < tol + 30 * (rr + 1)
                   and np.linalg.norm(ctr[i + rr + 1] - ctr[i]) < tol + 30 * (rr + 1)):
                rr += 1
            while rr > 0:                                         # ...down to the widest window whose bending of the path stays within budget
                a_loc = max([curv.get(i + j, 0.0) for j in range(-rr, rr + 1)] + [0.0])
                if 0.5 * a_loc * _var(rr) <= budget:
                    break
                rr -= 1
            acc, ws = np.zeros((2, 3)), 0.0
            for j in range(-rr, rr + 1):
                acc += k[j + r] * self.A[i + j]
                ws += k[j + r]
            out[i] = acc / ws
        self.A = out
        for i, Ai in out.items():
            if i not in self.local_A:
                self.local_A[i] = Ai.copy()
        # The pose that places the paint was not smoothed before, so a noisy local fit
        # flickered even when the body track was steady. Average it on the same path.
        bk = PATCH_CFG.get("body_smooth") if getattr(self, "skin_hair", False) else PATCH_CFG["smooth"]
        kb = np.array(bk, np.float64)
        rb = len(kb) // 2
        loc_out = {}
        for i, Ai in self.local_A.items():
            if i not in self.A:
                loc_out[i] = Ai
                continue
            acc, ws = np.zeros((2, 3)), 0.0
            for j in range(-rb, rb + 1):
                src = self.local_A.get(i + j)
                if src is None:
                    continue
                # Drop a neighbour whose paint centre has jumped off this frame's body.
                if j != 0 and i + j in ctr and i in ctr and np.linalg.norm(ctr[i + j] - ctr[i]) > tol + 30 * abs(j):
                    continue
                acc += kb[j + rb] * src
                ws += kb[j + rb]
            loc_out[i] = (acc / ws) if ws > 0 else Ai
        self.local_A = loc_out

    def apply(self, frame, idx):
        A = self.local_A.get(idx, self.A.get(idx))
        if A is None:
            return frame
        t0 = cv2.getTickCount()
        H, W = frame.shape[:2]
        h0, w0 = self.hole_roi.shape
        M = _pr_compose(A, np.array([[1, 0, self.off[0]], [0, 1, self.off[1]]], np.float64))   # fill ROI -> frame
        quad = np.array([[0, 0], [w0, 0], [w0, h0], [0, h0]], np.float64) @ M[:, :2].T + M[:, 2]
        x0, y0 = int(max(0, np.floor(quad[:, 0].min()))), int(max(0, np.floor(quad[:, 1].min())))
        x1, y1 = int(min(W, np.ceil(quad[:, 0].max()))), int(min(H, np.ceil(quad[:, 1].max())))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame
        # v11.2.102: work on the painted area and the skin around it, not the whole (much larger) fill window
        hb = self._hbox
        qh = np.array([[hb[0], hb[1]], [hb[2], hb[1]], [hb[2], hb[3]], [hb[0], hb[3]]], np.float64) @ M[:, :2].T + M[:, 2]
        sc0 = math.sqrt(abs(float(np.linalg.det(A[:, :2]))))
        pad2 = int(max(36, 0.9 * self.size * sc0 + 16))
        x0, y0 = max(x0, int(np.floor(qh[:, 0].min())) - pad2), max(y0, int(np.floor(qh[:, 1].min())) - pad2)
        x1, y1 = min(x1, int(np.ceil(qh[:, 0].max())) + pad2), min(y1, int(np.ceil(qh[:, 1].max())) + pad2)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame
        Mr = M.copy()
        Mr[:, 2] -= (x0, y0)                                   # fill ROI -> this ROI
        sz = (x1 - x0, y1 - y0)
        roi = frame[y0:y1, x0:x1].astype(np.float32)

        def _warp_all(Mw):
            ins = cv2.warpAffine(np.full((h0, w0), 255, np.uint8), Mw, sz, flags=cv2.INTER_NEAREST) > 0
            hl = (cv2.warpAffine(self.hole_roi.astype(np.uint8) * 255, Mw, sz, flags=cv2.INTER_LINEAR) > 127) & ins
            cr = (cv2.warpAffine(self.core_roi.astype(np.uint8) * 255, Mw, sz, flags=cv2.INTER_LINEAR) > 127) & ins
            return ins, hl, cr

        inside, hole, core = _warp_all(Mr)
        maps = getattr(self, "skin_maps", {}).get(idx)
        self._mls_xy = None
        if maps is not None and len(maps[0]) >= 8:
            try:
                anc, cur = maps
                step = 3
                xs = np.arange(0, sz[0], step)
                ys = np.arange(0, sz[1], step)
                ggx, ggy = np.meshgrid(xs, ys)
                v = np.stack([ggx.ravel() + x0, ggy.ravel() + y0], 1)
                src = _mls_to_anchor(v, cur, anc)
                sx = (src[:, 0] - float(self.off[0])).reshape(ggy.shape).astype(np.float32)
                sy = (src[:, 1] - float(self.off[1])).reshape(ggy.shape).astype(np.float32)
                self._mls_xy = (cv2.resize(sx, sz, interpolation=cv2.INTER_LINEAR),
                                cv2.resize(sy, sz, interpolation=cv2.INTER_LINEAR))
                hole = cv2.remap(self.hole_roi.astype(np.uint8) * 255, self._mls_xy[0], self._mls_xy[1], cv2.INTER_LINEAR) > 127
                core = cv2.remap(self.core_roi.astype(np.uint8) * 255, self._mls_xy[0], self._mls_xy[1], cv2.INTER_LINEAR) > 127
                inside = cv2.remap(np.full((h0, w0), 255, np.uint8), self._mls_xy[0], self._mls_xy[1], cv2.INTER_NEAREST) > 0
            except Exception:
                self._mls_xy = None
        if hole.sum() < 4:
            return frame
        # A single affine shrinks a narrow waist in the stroke under the warp threshold, so on a
        # few frames the paint opens down the middle. Close that gap. Only for a dark-on-skin
        # stroke: a face mark must not have two nearby strokes joined.
        if getattr(self, "skin_hair", False):
            hole = _seal_split(hole, inside)
        # v11.2.102: look for the MARK ITSELF in this frame, near where the pose puts it. It is found by what it is (it differs
        # from the skin around it, with the anchor's polarity and about its size), not by matching the anchor's texture - so a turn,
        # blur or a change of light that spoils the match does not hide it. The paint is centred on what is found and covers all of it.
        found, unsure = None, False
        scale = math.sqrt(abs(float(np.linalg.det(A[:, :2]))))
        # v11.2.118: a dark-on-skin (locked) mark keeps the mark finder when it is a painted mark. Without it a fast-moving mole or
        # tattoo lost its hold on frames the finder would have caught (the one-go belly positions keep the v11.2.117 behaviour).
        keep_f = getattr(self, "keep_finder", True)
        if PATCH_CFG.get("find_mark", False) and self._mls_xy is None and self.core_area >= 6 and (not getattr(self, "skin_hair", False) or (PATCH_CFG.get("lock_finder", False) and keep_f)):
            reach = int(np.clip(0.5 * self.size * scale + 6, 6, 45))
            if idx in self.weak:
                reach = int(min(70, reach * 1.8))
            gcur_ = roi.mean(axis=2)
            found = _pr_find_mark(gcur_, hole, core, self.mark_pol, self.mark_con, self.core_area * scale * scale, reach)
            evid0, wide, unsure = _PR_EVID["v"], False, False
            if found is None and idx not in self.weak and getattr(self, "group", None) != "belly":
                # the mark is not where the pose says: look further (the pose may be off), and where it is not found at all and
                # the predicted core does not look like the mark, paint NOTHING - a patch of skin on the wrong place is worse
                found = _pr_find_mark(gcur_, hole, core, self.mark_pol, self.mark_con, self.core_area * scale * scale,
                                      int(min(80, 2 * reach + 10)), zone_r=int(reach))
                wide = found is not None
                unsure = found is None and evid0 < 0.12
            if found is not None:
                mnow, cen, area, mstrong = found
                yy_, xx_ = np.nonzero(core)
                if len(yy_) >= 4:
                    sh = np.array([cen[0] - float(xx_.mean()), cen[1] - float(yy_.mean())])
                    ar = area / max(1.0, float(core.sum()))
                    lim = (0.6 if (idx in self.weak or wide) else 0.3) * self.size * scale + 4.0   # (a pose error, not a different feature)
                    if getattr(self, "group", None) == "belly":
                        lim = 0.18 * self.size * scale + 4.0
                    if wide:
                        lim = 1.3 * self.size * scale + 4.0
                    if 0.35 <= ar <= 2.0 and 1.0 <= float(np.hypot(*sh)) <= lim:
                        Mr = Mr.copy()
                        Mr[:, 2] += sh                          # centre the pose on the mark that was found
                        inside, hole, core = _warp_all(Mr)
                        PATCH_STATS["frames_shifted"] += 1
                PATCH_STATS["frames_found"] += 1
                # ...all of it: the mark's own strong pixels (not a faint shadow joined to it), a little margin, and never further
                # than 4 px from where the pose puts the painted area
                near = cv2.dilate(core.astype(np.uint8), _pr_disk(max(4, int(0.2 * self.size * scale)))).astype(bool)
                # ...and only pixels of the mark's own colour: on the line from the skin around to the mark's ink / pigment (blur and
                # edges mix the two), not a hand, a cup or a dark object of another hue passing over it
                lm = np.ones(near.shape, bool)
                if self.mark_bgr is not None:
                    ringc = cv2.dilate(hole.astype(np.uint8), _pr_disk(6)).astype(bool) & ~hole & inside
                    if int(ringc.sum()) >= 30:
                        skin_c = np.median(roi[ringc], axis=0)
                        u = np.asarray(self.mark_bgr, np.float32) - skin_c
                        nu = float(np.dot(u, u))
                        if nu > 400.0:
                            nyx = np.nonzero(near)                    # (only where it matters: the pixels near the mark)
                            dv = roi[nyx] - skin_c
                            t = (dv @ u) / nu
                            perp = np.sqrt(np.maximum(0.0, np.sum(dv * dv, axis=1) - t * t * nu))
                            lm = np.zeros(near.shape, bool)
                            lm[nyx] = (t > 0.1) & (t < 1.3) & (perp < 30.0)
                if (idx in self.weak or wide) and self.mark_bgr is not None:
                    n_all, n_ok = int((mstrong & near).sum()), int((mstrong & near & lm).sum())
                    if n_all >= 4 and n_ok < 0.6 * n_all:       # not the mark's colour: structure of the scene, not the mark
                        PATCH_STATS["frames_weak_skipped"] += 1
                        return frame
                near = near & lm
                found = (found[0] & near, found[1], found[2], mstrong & near)        # only what is at the mark itself
                hole = hole | (cv2.dilate(mstrong.astype(np.uint8) & near.astype(np.uint8), _pr_disk(2)).astype(bool)
                               & cv2.dilate(hole.astype(np.uint8), _pr_disk(4)).astype(bool) & inside)
            elif idx in self.weak:                              # a bridged / held pose paints only where the mark is seen
                PATCH_STATS["frames_weak_skipped"] += 1
                return frame
        if getattr(self, "_mls_xy", None) is not None:
            mx, my = self._mls_xy
            fill = cv2.remap(self.fill_roi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            orig = cv2.remap(self.orig_roi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            fill = cv2.warpAffine(self.fill_roi, Mr, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            orig = cv2.warpAffine(self.orig_roi, Mr, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        hole, core = _off_face(frame, hole, core, x0, y0)
        if hole is None:
            return frame
        # object gate: is this still the anchor's mark / skin? (2 px tolerance; lighting compensated)
        ring = cv2.dilate(hole.astype(np.uint8), _pr_disk(7)).astype(bool) & ~hole & inside
        if ring.sum() > 20:
            gain = (roi[ring].mean(0) + 1.0) / (orig[ring].mean(0) + 1.0)
        else:
            gain = np.ones(3, np.float32)
        ref = orig * gain
        lo = cv2.erode(ref, np.ones((5, 5), np.uint8))
        hi = cv2.dilate(ref, np.ones((5, 5), np.uint8))
        d = (np.maximum(lo - roi, 0) + np.maximum(roi - hi, 0)).max(axis=2)
        d = cv2.GaussianBlur(d, (0, 0), 1.5)
        thr = PATCH_CFG["gate_thr"]
        keep = np.clip((d - thr) / (0.6 * thr), 0.0, 1.0)
        if PATCH_CFG.get("opaque", False):
            # v11.2.102: the paste is OPAQUE. What the mark itself looks like is never an "object" (a turned or blurred mark
            # no longer lines up with the anchor's picture of it), and only strong evidence of something else in front -
            # a hand, a cup - holds the paint back at all; a half-way verdict no longer leaves the mark showing through.
            if found is not None:
                keep = keep * (1.0 - cv2.GaussianBlur(cv2.dilate(found[0].astype(np.uint8), _pr_disk(2)).astype(np.float32), (0, 0), 1.0))
            keep = np.clip((keep - 0.55) / 0.35, 0.0, 1.0)
        if getattr(self, "skin_hair", False) and hole.any():
            # A crease or the mark itself down the middle is not a hand. Drop any "object"
            # that does not also cover a solid patch of the skin outside the stroke.
            strong = (keep > 0.4).astype(np.uint8)
            nlab, lab, stats, _ = cv2.connectedComponentsWithStats(strong, 8)
            outside = inside & ~hole
            drop = np.zeros(hole.shape, np.bool_)
            need = max(24, int(0.04 * float(hole.sum())))
            for i in range(1, nlab):
                comp = lab == i
                if int((comp & outside).sum()) < need or int(stats[i, cv2.CC_STAT_AREA]) < need:
                    drop |= comp
            keep[drop] = 0.0
        pb = int(min(8, max(3, round(float(PATCH_CFG.get("peek_band", 0.0)) * self.size * scale))))
        zone = cv2.dilate(hole.astype(np.uint8), _pr_disk(max(4, pb + 2))).astype(bool)   # (the edge repair's band too)
        keep[~zone] = 0.0
        if float(keep[hole].mean()) > 0.02:
            PATCH_STATS["frames_gated"] += 1
        # nothing like the mark was found where the pose puts it, the core does not look like it, and no object explains that:
        # the pose is wrong here - paint NOTHING (a patch of skin on the wrong place is worse than a frame without the patch)
        if unsure and (PATCH_CFG.get("opaque", False) or float(keep[core].mean() if core.any() else 0.0) < 0.3):
            PATCH_STATS["frames_refused"] += 1
            return frame
        # last safeguard: the mark's own core must still look like the mark. If it does not (covered by an object,
        # or the pose is off and the core lands on plain skin), nothing of this patch is painted on this frame -
        # the skin part of the stroke would otherwise copy the anchor's skin onto some other part of the body.
        want = float(self.core_roi.sum()) * abs(float(np.linalg.det(A[:, :2])))
        if want >= 12 and float(core.sum()) < 0.3 * want:          # the mark is (mostly) outside the picture
            PATCH_STATS["frames_refused"] += 1
            return frame
        if core.sum() >= 12 and float((keep[core] < 0.5).mean()) < (0.15 if PATCH_CFG.get("opaque", False) else 0.3):
            PATCH_STATS["frames_refused"] += 1                  # (v11.2.102: only when the object covers nearly all of the core)
            return frame
        # lighting: (current - anchor) on the clean ring, spread smoothly over the hole
        valid = ~hole & inside & (keep < 0.5)
        delta = roi - orig
        if valid.sum() > 20:                                    # misaligned edges give big local deltas: not light
            dm = delta.mean(axis=2)
            med = float(np.median(dm[valid]))
            mad = float(np.median(np.abs(dm[valid] - med))) * 1.4826 + 0.5
            valid &= np.abs(dm - med) < max(8.0, 3.0 * mad)
        q = 4 if min(sz) >= 64 else 1                           # light is smooth: solve it at 1/4 size
        if q > 1:
            sm = (max(1, sz[0] // q), max(1, sz[1] // q))
            dv = cv2.resize(valid.astype(np.float32), sm, interpolation=cv2.INTER_AREA)
            dd = cv2.resize(delta * valid[..., None], sm, interpolation=cv2.INTER_AREA) / np.maximum(dv, 1e-3)[..., None]
            low, den = _pr_lowpass(dd, dv > 0.5, PATCH_CFG["light_sigma"] / q + 0.5)
            dls = _pr_pushpull(low, ((dv > 0.5) & (den > 0.2)).astype(np.float32))
            dl = cv2.resize(dls, sz, interpolation=cv2.INTER_LINEAR)
        else:
            low, den = _pr_lowpass(delta, valid, PATCH_CFG["light_sigma"])
            dl = _pr_pushpull(low, (valid & (den > 0.2)).astype(np.float32))
        # blur: soften the anchor's content when this frame is softer than the anchor (motion blur, focus)
        band = cv2.dilate(hole.astype(np.uint8), _pr_disk(10)).astype(bool) & valid
        g = 1.0
        if band.sum() > 30:
            hp_c = float((roi - cv2.GaussianBlur(roi, (0, 0), 1.5))[band].std())
            hp_a = float((orig - cv2.GaussianBlur(orig, (0, 0), 1.5))[band].std())
            g = float(np.clip(hp_c / max(hp_a, 1e-3), 0.15, 1.15))
        if g < 0.95:
            lf = cv2.GaussianBlur(fill, (0, 0), 1.5)
            fill = lf + (fill - lf) * g
        res = fill + dl
        core_now = core & hole
        res = _pr_cover(roi, res, hole & (keep < 0.5), core_now, keep < 0.5)
        res = _pr_lift(np.where(valid[..., None], roi, res), res, hole & (keep < 0.5), core_now & (keep < 0.5))
        a = _pr_alpha(hole) * (1.0 - keep)
        # edge repair (v11.2.97): dark mark pixels just outside the hole (skin bent a little, a pixel of tracking
        # error) are covered with the relit skin of the painted frame - the "opening" at the mark's edge
        if float(PATCH_CFG.get("peek_band", 0.0)) > 0:
            band = cv2.dilate(hole.astype(np.uint8), _pr_disk(pb)).astype(bool) & ~hole & inside & (keep < 0.5)
            if band.sum() > 10:
                gcur = roi.mean(axis=2)
                gres = res.mean(axis=2)
                dd = gcur - gres
                med = float(np.median(dd[band]))
                mad = float(np.median(np.abs(dd[band] - med))) * 1.4826 + 0.5
                peek = band & (dd < med - max(8.0, 3.5 * mad))
                if peek.sum() >= 3:
                    peek = cv2.dilate(peek.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & band
                    ap = cv2.GaussianBlur(peek.astype(np.float32), (0, 0), 0.8)
                    a = np.maximum(a, np.clip(ap * 1.6, 0.0, 1.0) * (1.0 - keep))
                    PATCH_STATS["frames_peek"] += 1
        out = frame.copy()
        out[y0:y1, x0:x1] = np.clip(roi * (1.0 - a[..., None]) + res * a[..., None] + 0.5, 0, 255).astype(np.uint8)
        PATCH_STATS["frames_filled"] += 1
        PATCH_STATS["fill_ms"] += (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()
        return out


# --------------------------------------------------------------------------------------------------- planning
def _pr_fit_ref(ref, rmask, size_wh):
    """The app's frame (any size; possibly not rotated like the decoded video) -> candidates at the video's size."""
    W, H = size_wh
    out = []
    for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        r = ref if rot is None else cv2.rotate(ref, rot)
        m = rmask if rot is None else cv2.rotate(rmask, rot)
        if abs(r.shape[1] / float(r.shape[0]) - W / float(H)) > 0.04 * (W / float(H)):
            continue
        out.append((cv2.resize(r, (W, H), interpolation=cv2.INTER_AREA if r.shape[1] > W else cv2.INTER_LINEAR),
                    cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)))
    return out


def _pr_thumb(gray):
    w = 160
    h = max(8, int(round(gray.shape[0] * w / float(gray.shape[1]))))
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


_PR_LOC = {}          # v11.2.104: why the last anchor search succeeded or failed (for the job log): numbers only


def _place_painted_body(ref, rmask, n, bgr_at, hint):
    """The painted picture did not match a decoded frame. The hole is stored on
    the body in that picture, then drawn on the video frame where the same body
    point lands. Returns (frame, hole, score) or None."""
    try:
        import belly_clear as bc
    except Exception:
        return None
    if bgr_at is None or n <= 0:
        return None
    mask = np.asarray(rmask) > 127
    if int(mask.sum()) < 8:
        return None
    ys, xs = np.nonzero(mask)
    step = max(1, int(len(xs) / 400))
    pts = np.stack([xs[::step], ys[::step]], 1).astype(np.float64)
    ref_bgr = ref if getattr(ref, "ndim", 0) == 3 else cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    pose = bc.MediaPipePose(still=True)
    try:
        lm0 = pose.detect(ref_bgr)
    finally:
        pose.close()
    if lm0 is None:
        return None
    ut = bc.to_body(pts, lm0)
    if ut is None:
        return None
    h = int(np.clip(int(hint or 0), 0, n - 1))
    # Only the frames next to the one the user painted. Searching the whole
    # clip dropped every mark onto the last frame that happened to have a body.
    order = [h]
    for k in range(1, 9):
        if h - k >= 0:
            order.append(h - k)
        if h + k < n:
            order.append(h + k)
    cu, ct = float(np.median(ut[:, 0])), float(np.median(ut[:, 1]))
    if not (-0.55 <= ct <= 1.05 and abs(cu) <= 1.05):
        return None
    video = bc.MediaPipePose(still=False)
    seen = set()
    try:
        for i in order:
            if i in seen:
                continue
            seen.add(i)
            fr = bgr_at(i)
            if fr is None:
                continue
            lm = video.detect(fr, t_ms=int(i) * 40)
            if lm is None:
                continue
            xy = bc.from_body(ut, lm)
            if xy is None:
                continue
            H, W = fr.shape[:2]
            poly = np.round(xy).astype(np.int32)
            if len(poly) < 3:
                continue
            inside = ((poly[:, 0] >= 0) & (poly[:, 0] < W) & (poly[:, 1] >= 0) & (poly[:, 1] < H))
            if float(inside.mean()) < 0.55:
                continue
            hole = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(hole, cv2.convexHull(poly), 1)
            if int(hole.sum()) < 8:
                continue
            return i, hole.astype(bool), 0.75
    finally:
        video.close()
    return None


def locate_anchor(ref, rmask, size_wh, thumbs, gray_at, hint, window):
    """Find the frame the user painted on, and the painted mask in that frame's pixels.
    thumbs: per-frame _pr_thumb; gray_at(i): full gray frame; hint: expected index. -> (idx, hole bool, score) or None."""
    best = (-2.0, None, None)
    n = len(thumbs)
    for r, m in _pr_fit_ref(ref, rmask, size_wh):
        t = _pr_thumb(_pr_gray(r))
        order = list(range(max(0, hint - window), min(n, hint + window + 1)))
        rest = [i for i in range(n) if i not in set(order)]
        for pass_ in (order, rest):
            for i in pass_:
                sc = _pr_ncc(t, thumbs[i])
                if sc > best[0]:
                    best = (sc, i, (r, m))
            if best[0] >= 0.85:
                break
    _PR_LOC.clear()
    _PR_LOC["frame_ncc"], _PR_LOC["frame_idx"] = float(best[0]), best[1]
    _PR_LOC["ref_wh"] = tuple(int(v) for v in ref.shape[1::-1])
    _PR_LOC["video_wh"] = tuple(int(v) for v in size_wh)
    _PR_LOC["candidates"] = len(_pr_fit_ref(ref, rmask, size_wh))
    if best[1] is None:
        return None
    sc, i, (r, m) = best
    g = gray_at(i)
    rg = cv2.GaussianBlur(_pr_gray(r), (5, 5), 1.0).astype(np.float32)
    fg = cv2.GaussianBlur(g, (5, 5), 1.0).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:                                        # the app's frame vs the decoded frame: small shifts / scaling
        half = (rg.shape[1] // 2, rg.shape[0] // 2)
        _, warp = cv2.findTransformECC(cv2.resize(rg, half), cv2.resize(fg, half), warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 3)
        warp[:, 2] *= 2.0
    except cv2.error:
        warp = np.eye(2, 3, dtype=np.float32)
    hole = cv2.warpAffine((m > 127).astype(np.uint8) * 255, warp.astype(np.float64), (g.shape[1], g.shape[0]),
                          flags=cv2.INTER_LINEAR) > 127
    fine = _pr_ncc(cv2.warpAffine(rg, warp, (g.shape[1], g.shape[0])), fg)
    _PR_LOC["fine"] = float(fine)
    if not hole.any() or fine < 0.6:
        return None
    return i, hole, fine


def find_patch(ref, rmask, size_wh, n, gray_at, stride=3, budget_s=None, stop=None):
    """The painted frame is not in this video (another part of a split job, or the clip was cut differently): look
    for the painted area itself in the frames (ORB + template matching against the app's frame, strictly checked).
    -> (idx, hole bool, score) or None."""
    best = (0.0, None, None)
    t_start = time.time()
    seen, near_best, timed_out = 0, 0.0, False
    for r, m in _pr_fit_ref(ref, rmask, size_wh):
        hole = m > 127
        if not hole.any():
            continue
        fnd = PatchTrack(-1, r, hole, make_fill=False)
        for i in range(0, n, stride):
            if (budget_s is not None and time.time() - t_start > budget_s) or (stop is not None and stop()):
                timed_out = True
                break
            seen += 1
            g = gray_at(i)
            for A0 in (fnd._redetect(g), fnd._redetect_tm(g)):
                if A0 is None:
                    continue
                Ar, cc = fnd._ecc(g, A0)
                A1 = Ar if cc > 0.5 else A0
                sc, inl = fnd._check(g, A1)
                if inl is None:
                    continue
                al, cov = fnd._last_all, fnd._last_cov
                near_best = max(near_best, min(al, cov))
                if al >= 0.8 and cov >= 0.7 and al * cov > best[0]:
                    best = (al * cov, i, (A1, hole))
            if best[0] >= 0.95:
                break
        if timed_out:
            break
    _PR_LOC["area_samples"], _PR_LOC["area_best"] = seen, float(near_best)
    _PR_LOC["area_seconds"], _PR_LOC["area_timed_out"] = float(time.time() - t_start), bool(timed_out)
    if best[1] is None:
        return None
    A1, hole = best[2]
    g = gray_at(best[1])
    h = cv2.warpAffine(hole.astype(np.uint8) * 255, A1, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR) > 127
    return (best[1], h, best[0]) if h.any() else None


class PatchGroup:
    """One mark painted at one or more positions (v11.2.97). Each position is followed by its own PatchTrack; on every
    frame the view nearest in time that is followed there is used (a view whose confidence is clearly lower gives
    way), and if that view refuses the frame, the next one is tried."""

    def __init__(self, tracks):
        self.tracks = list(tracks)

    @property
    def A(self):
        keys = set()
        for t in self.tracks:
            keys.update(t.A.keys())
        return {k: None for k in keys}

    def apply(self, frame, idx):
        ev = PATCH_EVENTS.setdefault(int(getattr(self, "mark_no", 0)), {})
        cands = [t for t in self.tracks if idx in t.A]
        if not cands:
            ev[idx] = "L"
            return frame
        best = max(float(t.conf.get(idx, 0.0)) for t in cands)
        near = sorted((t for t in cands if float(t.conf.get(idx, 0.0)) >= best - 0.15),
                      key=lambda t: abs(idx - t.anchor_idx))
        rest = [t for t in cands if t not in near]
        code = "X"
        for t in near + rest:
            b_ref, b_weak, b_gate = PATCH_STATS["frames_refused"], PATCH_STATS["frames_weak_skipped"], PATCH_STATS["frames_gated"]
            out = t.apply(frame, idx)
            if out is not frame:
                ev[idx] = "o" if PATCH_STATS["frames_gated"] > b_gate else "P"
                return out
            if PATCH_STATS["frames_refused"] > b_ref:
                code = "R"
            elif PATCH_STATS["frames_weak_skipped"] > b_weak:
                code = "W"
        ev[idx] = code
        return frame


# v11.2.142 BellyAnchor (SDOS-142): the belly patch is placed by the BODY (shoulders and hips), refined by the image.
BODY_ANCHOR = {
    # Measured (SDOS-142): the image tracker is the more exact of the two - on a fast-moving test body it was right while the
    # pose model was off by 11-25 px on 42 of 90 frames, and overriding the image there left 6.6 % of the marks showing (worst
    # frame 77 %) against 0.8 % with the image alone. So the body only steps in where the image has clearly FAILED:
    "tol_px": 24.0,      # the image and the body disagree by more than this (px) ...
    "tol_rel": 0.08,     # ...or this share of the torso's length, whichever is larger ...
    "persist": 4,        # ...for at least this many frames in a row (the image held still while the body moved: "hanging")
    "gap": 30,           # an image correction is carried at most this many frames to a frame the body places
    "low_px": 8.0,       # hysteresis: a confirmed failure is widened over the frames around it where the image is off by more
    "low_noise": 2.0,    #   than max(low_px, low_noise x the median image-vs-body distance on this track: the pose's own noise) - a
                         #   tracker that starts to hang drifts off slowly, and those first frames would otherwise stay with the image
    "ecc_min": 0.70,     # a frame the body places is then fitted to the image from there (ECC); kept when it correlates this well
    "search": 1.5,       # ...first coarse to fine (1/4, 1/2 size: ECC at full size converges from ~6 px only, the body's place is
                         #   3-28 px off on the stress clips); the fit may land at most this x tol from the body's place
}
BODY_STATS = {"tracks": 0, "frames": 0, "image_kept": 0, "body_placed": 0, "added": 0, "no_body": 0, "refined": 0}


def _aff3(M):
    return np.vstack([np.asarray(M, np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])


def _body_fit(tr, gray, F, levels=None):
    """Fit the anchor's patch to this frame starting from the body's place F, coarse to fine: the same alignment the tracker uses
    (ECC, affine) first on 1/8, 1/4 and 1/2 size images (those that leave the box 24 px or more), where the body's 10-30 px error is a
    few px (the full-size fit converges from ~6 px only - measured on the stress clips).
    -> (the fitted anchor -> frame affine, correlation at the finest level), or (F, -1.0)."""
    y0, y1, x0, x1 = tr.box
    if levels is None:
        levels = [s for s in (8, 4, 2) if min(y1 - y0, x1 - x0) / float(s) >= 24.0]
    cur, M = tr._boxwarp(gray, F)
    cur = cur.astype(np.float32)
    tm = np.asarray(tr.tmpl, np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    cc = -1.0
    for s in levels:
        h_, w_ = int(round(tm.shape[0] / float(s))), int(round(tm.shape[1] / float(s)))
        a = cv2.resize(tm, (w_, h_), interpolation=cv2.INTER_AREA)
        b = cv2.GaussianBlur(cv2.resize(cur, (w_, h_), interpolation=cv2.INTER_AREA), (5, 5), 1.0)
        if float(a.std()) < 1.0:
            return F, -1.0
        ws = warp.copy()
        ws[:, 2] /= float(s)
        try:
            cc, ws = cv2.findTransformECC(a, b, ws, cv2.MOTION_AFFINE,
                                          (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4), None, 3)
        except cv2.error:
            return F, -1.0
        if not np.isfinite(ws).all():
            return F, -1.0
        ws[:, 2] *= float(s)
        warp = ws
    if cc < 0:
        return F, -1.0
    T = np.array([[1, 0, x0], [0, 1, y0]], np.float64)
    return _pr_compose(_pr_compose(M, warp.astype(np.float64)), _pr_inv(T)), float(cc)


def anchor_to_body(tr, body_aff, log=None, gray_at=None):
    """Re-place one PatchTrack by the body. body_aff(i) -> 2x3 affine (u, t, 1) -> (x, y) on frame i (belly_clear.BodyPath),
    or None where the body is not found.
    On every frame of the track's stretch the body gives B_i = body_i o body_anchor^-1 (anchor frame -> frame i). Where the image
    tracker's own position lies within tol of it, the image is kept (it is exact to the pixel; the pose model is not); where it
    lies farther, or the tracker lost the patch, the body places it, with the image correction of the frames around carried over
    (no jump where the two meet). The paint can therefore not stay behind or drift off when the body moves, and does not wobble
    with the pose model's noise where the image holds."""
    Ma = body_aff(tr.anchor_idx)
    if Ma is None:
        if log:
            log("patch %s: body anchor off - no body on the anchor frame %d" % (getattr(tr, "group", "?"), tr.anchor_idx))
        return
    Ma3i = np.linalg.inv(_aff3(Ma))
    cx, cy = (tr.box[2] + tr.box[3]) / 2.0, (tr.box[0] + tr.box[1]) / 2.0
    c0 = np.array([cx, cy, 1.0])
    lo = int(getattr(tr, "lo", min(tr.A) if tr.A else tr.anchor_idx))
    hi = int(getattr(tr, "hi", max(tr.A) if tr.A else tr.anchor_idx))
    B, tol, d, agree, Mbody = {}, {}, {}, {}, {}
    for i in range(lo, hi + 1):
        Mi = body_aff(i)
        if Mi is None:
            BODY_STATS["no_body"] += 1
            continue
        Mbody[i] = Mi
        Bi = (_aff3(Mi) @ Ma3i)[:2]
        B[i] = Bi
        tl = float(np.linalg.norm(np.asarray(Mi)[:, 1]))               # the torso's length on this frame (px)
        tol[i] = max(float(BODY_ANCHOR["tol_px"]), float(BODY_ANCHOR["tol_rel"]) * tl)
        if i in tr.A:
            d[i] = np.asarray(tr.A[i]) @ c0 - Bi @ c0
    if not B:
        return
    # (a steady offset between the pose model and the skin - the pose's own bias on this body - is not a disagreement: the
    #  image is judged against the track's median correction, and that median is what a frame with no agreeing neighbour gets)
    d_ref = np.median(np.array(list(d.values())), axis=0) if d else np.zeros(2)
    dev = {i: float(np.linalg.norm(r - d_ref)) for i, r in d.items()}
    bad = {i: dev[i] > tol[i] for i in dev}
    # only a RUN of persist frames or more counts as the image having failed; a shorter one is the pose model's noise
    for i in dev:
        agree[i] = True
    run = []
    for i in sorted(dev) + [None]:
        if i is not None and bad[i] and (not run or i == run[-1] + 1):
            run.append(i)
            continue
        if len(run) >= int(BODY_ANCHOR["persist"]):
            for k in run:
                agree[k] = False
        run = [i] if (i is not None and bad[i]) else []
    # hysteresis: widen each confirmed failure over its neighbours while they are off by more than the pose's noise on this track
    ok_dev = [dev[i] for i in dev if agree[i]]
    noise = float(np.median(ok_dev)) if len(ok_dev) >= 5 else float(BODY_ANCHOR["tol_px"])     # (the median: a slow drift does not lift it)
    low = max(float(BODY_ANCHOR["low_px"]), float(BODY_ANCHOR["low_noise"]) * noise)
    for i in sorted(dev):
        if agree[i]:
            continue
        for step in (-1, 1):
            k = i + step
            while k in dev and agree[k] and dev[k] > min(low, tol[k]):
                agree[k] = False
                k += step
    keys = sorted(i for i in B if agree.get(i))
    import bisect

    def corr(i):
        """the image correction at frame i: its own where the image agrees, else interpolated from the agreeing frames around"""
        if agree.get(i):
            return d[i]
        j = bisect.bisect_left(keys, i)
        a = keys[j - 1] if j > 0 else None
        b = keys[j] if j < len(keys) else None
        gap = int(BODY_ANCHOR["gap"])
        if a is not None and b is not None and i - a <= gap and b - i <= gap:
            w = (i - a) / float(b - a)
            return (1 - w) * d[a] + w * d[b]
        if a is not None and i - a <= gap:
            return d[a]
        if b is not None and b - i <= gap:
            return d[b]
        return d_ref
    newA, newL = {}, {}
    refined = set()
    for i, Bi in B.items():
        if agree.get(i):
            newA[i] = np.asarray(tr.A[i], np.float64)
            newL[i] = np.asarray(tr.local_A.get(i, tr.A[i]), np.float64)
            BODY_STATS["image_kept"] += 1
            continue
        r = corr(i)
        F = Bi.copy()
        F[:, 2] += r                                                    # the body's pose, moved by the image's correction
        # ...then found in this frame's image FROM there: the body says where to look, the image how the patch lies exactly. The
        # tracker's own fit (ECC) from the body's place, coarse to fine, then at full size on the tracker's view. Kept when it
        # correlates well and lands within reach of the body's place; else the body's place stands.
        if gray_at is not None:
            try:
                g_ = gray_at(i)
                if g_ is not None:
                    reach = float(BODY_ANCHOR["search"]) * tol[i]
                    C_, _cc0 = _body_fit(tr, g_, F)
                    got = None
                    for F0 in ([C_] if _cc0 > 0 else []) + [F]:
                        A1, cc = tr._ecc(g_, F0)
                        if cc >= float(BODY_ANCHOR["ecc_min"]) and float(np.linalg.norm(np.asarray(A1) @ c0 - F @ c0)) <= reach:
                            got = np.asarray(A1, np.float64)
                            break
                    if got is not None:
                        F = got
                        BODY_STATS["refined"] += 1
                        refined.add(i)
            except Exception:
                pass
        newA[i] = F
        newL[i] = F.copy()
        if i in tr.A:
            BODY_STATS["body_placed"] += 1
        else:
            BODY_STATS["added"] += 1
            tr.conf[i] = max(float(tr.conf.get(i, 0.0)), 0.6)
        tr.weak.discard(i)
    # A belly patch may only sit on the low abdomen. Face skin matches the fill,
    # so a drift onto the face used to be accepted. A frame whose centre is above
    # the chest, off the hips, or a long way from the body's own point is not painted.
    if getattr(tr, "group", None) == "belly":
        drop = []
        for i, F in list(newA.items()):
            Mi = Mbody.get(i)
            if Mi is None or i not in B:
                drop.append(i)
                continue
            paint = np.asarray(F) @ c0
            body = np.asarray(B[i]) @ c0
            tl = float(np.linalg.norm(np.asarray(Mi)[:, 1]))
            if float(np.linalg.norm(paint[:2] - body[:2])) > 0.22 * max(tl, 1.0):
                drop.append(i)
                continue
            try:
                inv = cv2.invertAffineTransform(np.asarray(Mi, np.float64).reshape(2, 3))
                ut = np.array([paint[0], paint[1]], np.float64) @ inv[:, :2].T + inv[:, 2]
                u, t = float(ut[0]), float(ut[1])
            except Exception:
                drop.append(i)
                continue
            if not (-1.05 <= u <= 1.05 and -0.25 <= t <= 0.58):
                drop.append(i)
        for i in drop:
            newA.pop(i, None)
            newL.pop(i, None)
            tr.conf.pop(i, None)
            tr.weak.discard(i)
        if log and drop:
            log("patch %s: %d frame(s) not painted - the patch would have left the belly or landed on the face"
                % (getattr(tr, "group", "?"), len(drop)))
    BODY_STATS["frames"] += len(B)
    BODY_STATS["tracks"] += 1
    # frames of the stretch with no body keep the image's own position,
    # except a belly patch: with no body there is nothing to keep it off the face
    for i in list(tr.A):
        if i not in newA:
            if getattr(tr, "group", None) == "belly":
                continue
            newA[i] = tr.A[i]
            newL[i] = tr.local_A.get(i, tr.A[i])
    tr.A, tr.local_A = newA, newL
    if log:
        n_img = sum(1 for i in B if agree.get(i))
        dv = np.array(list(dev.values())) if dev else np.zeros(1)
        log("patch %s: body anchor - %d frames placed by the body (%d where the image had lost the patch; %d of them found in the image "
            "around the body's place), %d kept by the image; "
            "image vs body: median %.1f px, 90th %.1f px, worst %.1f px (limit %.0f px for %d frames); %d frames without a body"
            % (getattr(tr, "group", "?"), len(B) - n_img, sum(1 for i in B if i not in d), len(refined), n_img, float(np.median(dv)),
               float(np.percentile(dv, 90)), float(dv.max()), float(np.median(list(tol.values()))), int(BODY_ANCHOR["persist"]),
               (hi - lo + 1) - len(B)))


def plan_tracks(specs, size_wh, n, thumbs, gray_at, bgr_at, log=None, progress=None, stop=None):
    """specs: [(ref_bgr, mask_uint8, hint_index, window[, group])] - several entries with the same group are views of
    one mark at different positions. -> list of PatchGroup ready to apply (by frame index).
    progress(text): called about once a second while following (v11.2.101: the job no longer sits silent);
    stop(): True cancels (the result is then an empty list)."""
    groups = {}
    order = []
    PATCH_EVENTS.clear()
    dropped = max(0, len(specs) - int(PATCH_CFG.get("max_views", 20)))
    for sp in specs[:int(PATCH_CFG.get("max_views", 20))]:
        gk = sp[4] if len(sp) > 4 and sp[4] is not None else ("_v%d" % len(order))
        if gk not in groups:
            if len(groups) >= PATCH_CFG["max_patches"]:
                dropped += 1
                continue
            groups[gk] = []
            order.append(gk)
        groups[gk].append(sp[:4])
    if dropped and log:                                      # never silent: the job log says what was left out
        log("WARNING: %d painted position(s) were left out - the engine takes at most %d marks and %d positions per job"
            % (dropped, int(PATCH_CFG["max_patches"]), int(PATCH_CFG.get("max_views", 20))))
    # 1) find every painted position in this video (cheap), 2) follow each through its own stretch
    located = []
    deadline = time.time() + float(PATCH_CFG.get("find_total_s", 240.0))      # (searches for the painted AREA share this time)
    for m, gk in enumerate(order):
        if stop is not None and stop():
            return []
        fl_ = _locate_views(groups[gk], m + 1, size_wh, n, thumbs, gray_at, bgr_at, log, stop, deadline)
        for f_ in fl_:                                           # v11.2.118: which mark (group) each position belongs to
            f_["group"] = gk
        located.append(fl_)
    total = max(1, sum(f["hi"] - f["lo"] + 1 for fl in located for f in fl))
    st = {"done": 0, "t0": time.time(), "last": 0.0, "mark": 0, "marks": len(order)}

    def tick():
        if stop is not None and stop():
            return True
        st["done"] += 1
        now = time.time()
        if progress is not None and now - st["last"] >= 1.0:
            st["last"] = now
            frac = min(1.0, st["done"] / float(total))
            eta = (now - st["t0"]) * (1.0 - frac) / max(frac, 1e-3)
            progress("Patch removal: following mark %d of %d · %d%% · about %s left"
                     % (st["mark"], st["marks"], int(100 * frac), ("%d min" % int(eta / 60 + 0.5)) if eta >= 90 else ("%d s" % int(eta))))
        return False

    out = []
    for m, gk in enumerate(order):
        st["mark"] = m + 1
        PATCH_STATS["patches"] += 1
        tracks = _follow_views(located[m], m + 1, n, gray_at, bgr_at, log, tick, stop)
        if stop is not None and stop():
            return []
        if tracks:
            grp_ = PatchGroup(tracks)
            grp_.mark_no = m + 1
            out.append(grp_)
            if log and len(tracks) > 1:
                cov = set()
                for t in tracks:
                    cov.update(t.A.keys())
                log("patch %d: %d positions, together followed on %d of %d frames" % (m + 1, len(tracks), len(cov), n))
    return out


def _locate_views(views, mark_no, size_wh, n, thumbs, gray_at, bgr_at, log, stop=None, deadline=None):
    """Where is each painted position of this mark in this video? -> [{anchor, hole, score, tag, lo, hi}], by anchor frame,
    with the stretch each one is to be followed over: from the previous painted position (and follow_margin frames
    before it) to the next one (and follow_margin frames past it); the first goes back to frame 0, the last to the end."""
    found = []
    for v, (ref, rmask, hint, window) in enumerate(views):
        if stop is not None and stop():
            return []
        tag = "%d" % mark_no if len(views) == 1 else "%d.%d" % (mark_no, v + 1)
        PATCH_STATS["views"] += 1
        hint = int(np.clip(hint, 0, max(0, n - 1)))
        loc = locate_anchor(ref, rmask, size_wh, thumbs, gray_at, hint, window)
        d0 = dict(_PR_LOC)
        if loc is None:
            try:
                loc = _place_painted_body(ref, rmask, n, bgr_at, hint)
            except Exception:
                loc = None
            if loc is not None and log:
                log("patch %s: the painted picture did not match a frame; the mark was placed on the body at frame %d"
                    % (tag, loc[0]))
        if loc is None:
            left_s = None if deadline is None else deadline - time.time()
            budget = float(PATCH_CFG.get("find_budget_s", 90.0))
            if left_s is not None:
                budget = min(budget, max(0.0, left_s))
            if budget >= 3.0:
                loc = find_patch(ref, rmask, size_wh, n, gray_at, budget_s=budget, stop=stop)
            elif log:
                log("patch %s: the time for searching the painted area is used up - not searched" % tag)
            if loc is not None and log:
                log("patch %s: the painted frame is not in this video; the painted area was found on frame %d" % (tag, loc[0]))
        if loc is None:
            if log:
                # numbers only: how close the best frame came (needs 0.85 on the 160 px picture and 0.60 after alignment), and how the
                # search for the area itself went (needs 0.8 and 0.7)
                d1 = dict(_PR_LOC)
                log("patch %s: neither the painted frame nor the painted area was found in this video - skipped "
                    "(app frame %s px, video %s px, %s size match(es); best whole-frame match %.2f at frame %s, after alignment %s; "
                    "area search: %d frames tried in %.0f s%s, best %.2f)"
                    % (tag, "x".join(str(v) for v in d0.get("ref_wh", ("?",))), "x".join(str(v) for v in d0.get("video_wh", ("?",))),
                       d0.get("candidates", "?"), float(d0.get("frame_ncc", -1.0)), d0.get("frame_idx", "-"),
                       ("%.2f" % d0["fine"]) if "fine" in d0 else "n/a",
                       int(d1.get("area_samples", 0)) if "area_samples" in d1 else 0, float(d1.get("area_seconds", 0.0)),
                       " (stopped: time budget)" if d1.get("area_timed_out") else "", float(d1.get("area_best", 0.0))))
            continue
        i, hole, sc = loc
        if any(f["anchor"] == i for f in found):                # (two positions painted on the same frame)
            continue
        found.append({"anchor": i, "hole": hole, "score": sc, "tag": tag})
    found.sort(key=lambda f: f["anchor"])
    mg = int(PATCH_CFG.get("follow_margin", 45))
    for j, f in enumerate(found):
        f["lo"] = 0 if j == 0 else max(0, found[j - 1]["anchor"] - mg)
        f["hi"] = n - 1 if j == len(found) - 1 else min(n - 1, found[j + 1]["anchor"] + mg)
    return found


def _double_bind(tr, bgr_at, log):
    """Skin lock stays. On a move, the cover is also tied to the hips, the bikini
    line, the chest and the shoulders. Only the motion the skin lock missed is
    added, so a wrong pose cannot carry the paint to another place."""
    if bgr_at is None or not tr.A:
        return
    ys, xs = np.nonzero(tr.hole_roi)
    if len(ys) < 8:
        return
    cx = float(xs.mean()) + float(tr.off[0])
    cy = float(ys.mean()) + float(tr.off[1])
    c0 = np.array([cx, cy, 1.0])
    try:
        import belly_clear as BC
    except Exception:
        return
    pose = BC.MediaPipePose(still=False)
    lms = {}
    try:
        for i in sorted(set(tr.A) | {int(tr.anchor_idx)}):
            try:
                lm = pose.detect(bgr_at(i), t_ms=int(i) * 40)
            except Exception:
                lm = None
            if lm is None:
                continue
            try:
                if min(float(lm[k][2]) for k in (BC.L_SH, BC.R_SH, BC.L_HIP, BC.R_HIP)) < 0.35:
                    continue
            except Exception:
                continue
            lms[i] = lm
    finally:
        pose.close()
    lm0 = lms.get(int(tr.anchor_idx))
    ut = None if lm0 is None else BC.to_body(np.array([[cx, cy]]), lm0)
    Ma = None if lm0 is None else BC.body_affine(lm0)
    if ut is None or Ma is None:
        if log:
            log("patch: double bind off, the shoulders and hips were not on the painted frame")
        return
    u, tv = float(ut[0, 0]), float(ut[0, 1])
    # The painted mark sits below the navel, often a little under the hip
    # line. The old band stopped at the bikini line, so that mark got no
    # body hold and the cover shook on the skin tracker alone.
    if not (-0.55 <= tv <= 1.05 and abs(u) <= 1.05):
        if log:
            log("patch: double bind off, the mark is not on the torso")
        return
    torso0 = float(np.linalg.norm(np.asarray(Ma)[:, 1]))
    if torso0 < 16.0:
        return

    def pred_of(lm):
        xy = BC.from_body(ut, lm)
        M = None if lm is None else BC.body_affine(lm)
        if xy is None or M is None:
            return None
        tor = float(np.linalg.norm(np.asarray(M)[:, 1]))
        if tor < 0.65 * torso0 or tor > 1.45 * torso0:
            return None
        return np.asarray(xy, np.float64).reshape(-1)[:2]

    rows = []
    for i in sorted(k for k in tr.A if k in lms):
        pred = pred_of(lms[i])
        if pred is None:
            continue
        skin = np.asarray(tr.A[i], np.float64) @ c0
        rows.append((i, pred, skin[:2]))
    if len(rows) < 3:
        if log:
            log("patch: double bind off, the body was not seen on enough frames")
        return
    bias = np.median(np.stack([p - s for _, p, s in rows]), axis=0)
    prev_pred, n_fast = None, 0
    cap = 0.28 * torso0
    for i, pred, skin in rows:
        body_delta = None if prev_pred is None else pred - prev_pred
        prev_pred = pred
        if body_delta is None:
            continue
        if i in getattr(tr, "point_bound", ()):
            continue
        speed = float(np.linalg.norm(body_delta))
        if not (2.5 < speed <= cap):
            continue
        residual = (pred - skin) - bias
        along = float(np.dot(residual, body_delta)) / speed
        if along < 2.0:
            continue
        gain = 0.45 if speed < 8.0 else (0.70 if speed < 18.0 else 0.90)
        shift = (body_delta / speed) * (along * gain)
        lim = min(0.08 * torso0, speed)
        nrm = float(np.linalg.norm(shift))
        if nrm > lim:
            shift *= lim / nrm
        if nrm < 1.0:
            continue
        A = np.asarray(tr.A[i], np.float64).copy()
        A[:, 2] += shift
        tr.A[i] = A
        L = np.asarray(tr.local_A.get(i, A), np.float64).copy()
        L[:, 2] += shift
        tr.local_A[i] = L
        n_fast += 1
    if log:
        where = "on the bikini line" if tv < 0.18 else ("on the chest" if tv > 0.55 else "between the bikini line and the chest")
        log("patch: double bind %s; skin lock kept, and the hips, bikini line, chest and shoulders pulled the cover on %d fast frames"
            % (where, n_fast))


def _follow_views(found, mark_no, n, gray_at, bgr_at, log, tick, stop=None):
    tracks = []
    for f in found:                                              # ascending anchors: one pass over the video serves bgr_at
        if stop is not None and stop():
            break
        i = f["anchor"]
        PATCH_STATS["anchored"] += 1
        tr = PatchTrack(i, bgr_at(i), f["hole"])
        tr.keep_finder = f.get("group") != "belly"               # v11.2.118: painted marks keep the mark finder; one-go belly does not
        tr.group, tr.lo, tr.hi = f.get("group"), int(f["lo"]), int(f["hi"])     # v11.2.142: for the body anchor
        tr.track(gray_at, list(range(i, f["hi"] + 1)), tick)
        tr.track(gray_at, list(range(i, f["lo"] - 1, -1)), tick)
        tr.smooth()
        _double_bind(tr, bgr_at, log)
        if log and getattr(tr, "point_bound", None):
            log("patch %s: bound to %d nearby skin points on %d frames; the cover bends with that skin"
                % (f["tag"], int(getattr(tr, "skin_bind_n", 0)), len(tr.point_bound)))
        if log:
            log("patch %s: anchored on frame %d (match %.2f), %d px, followed on %d of %d frames"
                % (f["tag"], i, f["score"], int(tr.hole_roi.sum()), len(tr.A), f["hi"] - f["lo"] + 1 if len(found) > 1 else n))
        tracks.append(tr)
    return tracks

def specs_from(items):
    """Painted marks only. Anything marked belly or auto is dropped."""
    out = []
    for k, p in enumerate(items or []):
        try:
            if not isinstance(p, dict):
                continue
            grp = p.get("group")
            if str(grp or "").lower() in ("belly", "auto", "clean_belly"):
                logging.info("PATCH-REMOVAL skipped an automatic mark")
                continue
            ref = cv2.imread(str(p.get("image")))
            m = cv2.imread(str(p.get("mask")), cv2.IMREAD_GRAYSCALE)
            if ref is None or m is None or not (m > 127).any():
                logging.warning("patch %d: unreadable or empty - skipped", k + 1)
                continue
            out.append((ref, m, max(0.0, float(p.get("t") or 0.0)), None if grp in (None, "") else str(grp)))
        except Exception as e:
            logging.warning("patch %d: %s - skipped", k + 1, e)
    return out


def prepare(vp, specs, start_frame, end_frame, step, lim, ow, oh, upright, interp, src_fps, out_fps, u, cxl):
    """Read the job the way the render will, then follow each painted mark. No automatic belly."""
    set_patch_mode("classic")
    t0 = time.perf_counter()
    logging.info("PATCH-REMOVAL mode: manual (automatic clean is off)")
    q = 93 if lim <= 6000 else 85
    win = max(3, int(round(float(PATCH_CFG["anchor_window_s"]) * float(out_fps))))
    hints = [int(round((sp[2] * float(src_fps) - start_frame) / float(max(1, step)))) for sp in specs]
    grays, thumbs, keep_bgr = [], [], {}

    def _frames():
        cap = cv2.VideoCapture(vp)
        try:
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fi, src, n = 0, start_frame, 0
            while n < lim and src <= end_frame:
                if fi % step:
                    if not cap.grab():
                        break
                    fi += 1
                    src += 1
                    continue
                ok, f = cap.read()
                if not ok or f is None:
                    break
                f = upright(f)
                if f.shape[1] != ow or f.shape[0] != oh:
                    f = cv2.resize(f, (ow, oh), interpolation=interp)
                yield n, f
                n += 1
                fi += 1
                src += 1
        finally:
            cap.release()

    for i, f in _frames():
        if cxl():
            return []
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        grays.append(cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, q])[1])
        thumbs.append(_pr_thumb(g))
        if any(abs(i - h) <= win for h in hints):
            keep_bgr[i] = cv2.imencode(".png", f, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1]
        if i % 200 == 0:
            u(10, "Patch removal: reading the video %d/%d…" % (i, lim))
    n = len(grays)
    if n == 0:
        return []
    cache = OrderedDict()
    cache_bytes = 0
    limit = 48 * 1024 * 1024

    def gray_at(i):
        nonlocal cache_bytes
        i = int(i)
        hit = cache.get(i)
        if hit is not None:
            cache.move_to_end(i)
            return hit
        decoded = cv2.imdecode(grays[i], cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            return None
        size = int(decoded.nbytes)
        if size <= limit:
            while cache and cache_bytes + size > limit:
                _, ev = cache.popitem(last=False)
                cache_bytes -= int(ev.nbytes)
            cache[i] = decoded
            cache_bytes += size
        return decoded

    scan = {"gen": None, "pos": -1}

    def bgr_at(i):
        if i in keep_bgr:
            return cv2.imdecode(keep_bgr[i], cv2.IMREAD_COLOR)
        if scan["gen"] is None or i <= scan["pos"]:
            if scan["gen"] is not None:
                scan["gen"].close()
            scan["gen"], scan["pos"] = _frames(), -1
        for j, f in scan["gen"]:
            scan["pos"] = j
            if j == i:
                return f
            if j > i:
                break
        return None

    u(11, "Patch removal: following %d painted mark(s)…" % len(specs))
    msgs = []
    try:
        tracks = plan_tracks(
            [(sp[0], sp[1], h, win, sp[3]) for sp, h in zip(specs, hints)],
            (ow, oh), n, thumbs, gray_at, bgr_at,
            log=msgs.append, progress=lambda text: u(11, text), stop=cxl)
    finally:
        if scan["gen"] is not None:
            scan["gen"].close()
    for m in msgs:
        logging.info("PATCH-REMOVAL %s", m)
    logging.info("PATCH-REMOVAL planned in %.1fs (%d frames read)", time.perf_counter() - t0, n)
    return tracks or []
