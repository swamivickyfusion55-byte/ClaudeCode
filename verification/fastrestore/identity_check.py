"""v11.2.141: who is painted on whom - from the OUTPUT video, so it does not depend on the engine's internals.
usage: identity_check.py OUT.mp4 CLIP.mp4 GT.npz SRC_A.jpg SRC_B.jpg [MASK_DUMP.npz]   (make_kiss.py / make_cross.py clips; synthetic, public photo)
For every frame and each person whose face is at least 45 % visible: the face is cut from the output at the person's TRUE 5 points
(insightface's 112-px alignment), ONLY this person's own visible pixels kept (the other person's face and the background grey), embedded
with ArcFace (w600k_r50), and compared with both replacement photos and with that person's own original face (the same cut from the
input frame). The replacement photos are embedded whole, so a half-hidden face scores lower against them than a whole one.
  own      the replacement this person got on the first 10 frames (faces apart): majority of the better match
  identity (frames with at least 70 % of the face visible - ArcFace on a sliver of face is not meaningful):
    correct  own >= other + 0.05 and own >= original
    MIX-UP   other > own + 0.05            (the other person's replacement is on this face)
    unclear  anything else
  painting (frames with at least 30 % visible): the share of the person's visible FACE that the output changed by more than 10
  levels - the face outline (106 points + forehead) found on the source photo (insightface's t1.jpg, faces f2 = A, f3 = B, as in
  make_kiss.py) and moved into each frame by the exact similarity the clip used (fitted from the true 5 points), intersected with
  the visible mask, eroded 4 px - against the same share on that person's fully visible frames (90 % or more; the median):
    REVERT   under 40 % of it              (the original face shows)
    weak     40-70 % of it                 (the replacement is faded)
  With MASK_DUMP (bench_pipeline.py MASK_DUMP=1: the engine's own paste mask per slot per frame) the painting share is the mean of
  that mask over the same region instead - the paint actually laid on the face, which a colour test cannot tell apart from a
  replacement whose skin tone matches the original.
Also: flicker = mean |frame-to-frame change| inside the face in the output, divided by the input's own (1.0 = as steady as the input)."""
import sys, cv2, numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
OUT, CLIP, GT, SA, SB = sys.argv[1:6]
DUMP = sys.argv[6] if len(sys.argv) > 6 else None
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition", "landmark_2d_106"])
fa.prepare(ctx_id=-1, det_size=(640, 640))
# the two clip people's face outlines on the source photo (make_kiss.py: A = f2, B = f3)
_PH = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg"); _PF = fa.get(_PH); PHOTO = []
for (x1, y1, x2, y2) in ((466, 268, 573, 415), (745, 338, 845, 479)):
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2]); f = min(_PF, key=lambda q: float(np.linalg.norm((q.bbox[:2] + q.bbox[2:]) / 2 - c)))
    L = f.landmark_2d_106.astype(np.float32); top = L[:, 1].min(); h = L[:, 1].max() - top
    fh = L[L[:, 1] < top + 0.35 * h].copy(); fh[:, 1] -= 0.30 * h                      # a forehead estimate above the brows
    PHOTO.append((f.kps.astype(np.float32), np.vstack([L, fh])))
def face_region(t, p, shape):
    if REGION is not None:                    # v11.2.144: the clip's own face outline per frame (make_pkiss.py)
        m = np.zeros(shape[:2], np.uint8); cv2.fillConvexPoly(m, cv2.convexHull(np.round(REGION[t, p]).astype(np.int32)), 1); return m > 0
    k0, L = PHOTO[p]; M, _ = cv2.estimateAffinePartial2D(k0, np.asarray(kps[t, p], np.float32))
    if M is None: return None
    P = L @ M[:, :2].T + M[:, 2]; m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(P).astype(np.int32)), 1); return m > 0
rec = fa.models["recognition"]
def emb_at(img, kps, vmask=None):
    """ArcFace of the face at its true 5 points; with vmask, only this person's own visible pixels count (the rest grey, 127),
    so a crop half filled by the other person's face is not scored as that face."""
    M = face_align.estimate_norm(np.asarray(kps, np.float32), 112)
    crop = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    if vmask is not None:
        vm = cv2.warpAffine(vmask, M, (112, 112), flags=cv2.INTER_LINEAR) > 127
        crop = np.where(vm[..., None], crop, 127).astype(np.uint8)
    e = rec.get_feat(crop).ravel(); return e / (np.linalg.norm(e) + 1e-9)
def src_emb(p):
    im = cv2.imread(p); f = max(fa.get(im), key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1])); return f.normed_embedding
SRC = [src_emb(SA), src_emb(SB)]
def frames(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(f)
    return fr
out, inp = frames(OUT), frames(CLIP); g = np.load(GT); vis, kps = g["a"], g["kps"]
REGION = g["region"] if "region" in g.files else None
n = min(len(out), len(inp), len(vis)); full = [float((vis[:, p] > 127).sum(axis=(1, 2)).max()) for p in range(2)]
lab = {0: [], 1: []}; sims = {0: [], 1: []}
for t in range(n):
    for p in range(2):
        share = float((vis[t, p] > 127).sum()) / max(1.0, full[p])
        if share < 0.70:
            lab[p].append("hidden"); sims[p].append(None); continue
        e_out = emb_at(out[t], kps[t, p], vis[t, p]); e_in = emb_at(inp[t], kps[t, p], vis[t, p])
        s = (float(e_out @ SRC[0]), float(e_out @ SRC[1]), float(e_out @ e_in))
        sims[p].append(s); lab[p].append(None)
own = {}
for p in range(2):
    early = [s for s in sims[p][:10] if s is not None]
    own[p] = int(np.argmax(np.bincount([int(s[1] > s[0]) for s in early], minlength=2))) if early else p
for p in range(2):
    for t in range(n):
        s = sims[p][t]
        if s is None: continue
        o, x, orig = s[own[p]], s[1 - own[p]], s[2]
        lab[p][t] = "correct" if (o >= x + 0.05 and o >= orig) else "MIXUP" if x > o + 0.05 else "unclear"
# flicker inside each face (the visible mask, eroded), output vs input
def flick(seq, p):
    v = []
    for t in range(1, n):
        m = cv2.erode(((vis[t, p] > 127) & (vis[t - 1, p] > 127)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if m.sum() < 500: continue
        a = cv2.cvtColor(seq[t], cv2.COLOR_BGR2GRAY).astype(np.float32); b = cv2.cvtColor(seq[t - 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        v.append(float(np.abs(a - b)[m].mean()))
    return np.array(v)
_dump = None
if DUMP:
    _z = np.load(DUMP); _sl = _z["slot"]; _idx = {int(s): np.nonzero(_sl == s)[0] for s in np.unique(_sl)}
    # v11.2.144: dumps that carry the frame number ("g", not all 0) are looked up by frame: a frame without paint (a revert) has no
    # entry, and counting entries would shift every later frame onto the wrong one
    _byg = None
    if "g" in _z.files and int(np.max(_z["g"])) > 0:
        _byg = {}
        for _k in range(len(_sl)):
            _byg.setdefault(int(_sl[_k]), {})[int(_z["g"][_k])] = _k
    def _centre(k):
        M = _z["M"][k]; size = int(_z["size"][k]); IM = cv2.invertAffineTransform(M); return IM[:, :2] @ np.array([size / 2, size / 2]) + IM[:, 2]
    _slot_of = {}
    for p_ in range(2):
        _slot_of[p_] = min(_idx, key=lambda s: float(np.linalg.norm(_centre(_idx[s][0]) - kps[0, p_].mean(0))))
    _dump = (_z, _idx, _slot_of)
def _paint_map(t, p, shape):
    z, idx, slot_of = _dump
    if _byg is not None:
        k = _byg.get(slot_of[p], {}).get(int(t))
        if k is None: return np.zeros(shape[:2], np.float32)          # (not painted on this frame)
    else:
        ks = idx.get(slot_of[p]);
        if ks is None or t >= len(ks): return None
        k = ks[t]
    size = int(z["size"][k]); m = cv2.resize(z["m"][k].astype(np.float32), (size, size))
    return cv2.warpAffine(m, cv2.invertAffineTransform(z["M"][k]), (shape[1], shape[0]))
def painted(p):
    sh, vv = [], []
    for t in range(n):
        v = vis[t, p] > 127; share = float(v.sum()) / max(1.0, full[p])
        fr_ = face_region(t, p, v.shape)
        m = cv2.erode((v & fr_).astype(np.uint8) if fr_ is not None else v.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if m.sum() < 400: sh.append(None); vv.append(share); continue
        if _dump is not None:
            pm = _paint_map(t, p, v.shape)
            sh.append(float(pm[m].mean()) if pm is not None else 0.0); vv.append(share); continue
        d = np.abs(out[t].astype(np.int16) - inp[t].astype(np.int16)).max(axis=2) > 10
        sh.append(float((d & m).sum()) / float(m.sum())); vv.append(share)
    base = np.median([s for s, v in zip(sh, vv) if s is not None and v >= 0.9] or [s for s in sh if s is not None] or [1.0])
    rev = [t for t in range(n) if sh[t] is not None and vv[t] >= 0.30 and sh[t] < 0.40 * base]
    weak = [t for t in range(n) if sh[t] is not None and vv[t] >= 0.30 and 0.40 * base <= sh[t] < 0.70 * base]
    return base, rev, weak, sum(1 for t in range(n) if sh[t] is not None and vv[t] >= 0.30)
for p, nm in ((0, "A (left at start)"), (1, "B (right at start, in front)")):
    L = [x for x in lab[p] if x != "hidden"]
    cnt = {k: L.count(k) for k in ("correct", "MIXUP", "unclear")}
    bad = [t for t, x in enumerate(lab[p]) if x == "MIXUP"]
    base, rev, weak, npt = painted(p)
    so = [s[own[p]] for s in sims[p] if s is not None]; sx = [s[1 - own[p]] for s in sims[p] if s is not None]
    fo, fi = flick(out, p), flick(inp, p)
    print("%s: own replacement = photo %s | identity on %d frames (under 70 %% visible: %d): correct %d, MIX-UP %d, unclear %d | painting on %d frames (base %.0f %% %s): REVERT %d %s, weak %d %s | own sim median %.2f (min %.2f), other's %.2f | flicker x%.2f%s" % (
        nm, "AB"[own[p]], len(L), lab[p].count("hidden"), cnt["correct"], cnt["MIXUP"], cnt["unclear"], npt, 100 * base, "paint over the face" if _dump is not None else "of the face changed", len(rev), rev[:10], len(weak), weak[:10],
        float(np.median(so)) if so else -1, float(np.min(so)) if so else -1, float(np.median(sx)) if sx else -1,
        float(fo.mean() / max(1e-6, fi.mean())) if len(fo) else -1, (" | MIX-UP frames " + str(bad[:12])) if bad else ""))
print("same replacement on both people:", own[0] == own[1])
