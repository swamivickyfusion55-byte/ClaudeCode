"""SDOS-205 test scenes: two people cut out of insightface's public sample photo (t1.jpg; NOT user media), layered on a soft
backdrop so that the front one covers part of the other. Every pixel's true owner is known (the layers).
Writes OUT/pair_<name>.npz (frames, owner map, skin map, face kps/bboxes per frame) and an mp4 to look at.
usage: make_pair.py PACKDIR OUTDIR"""
import sys, os
PACK, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, PACK)
import numpy as np, cv2, skin_tone as ST
from insightface.app import FaceAnalysis
os.makedirs(OUT, exist_ok=True)
src = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg"); SH, SW = src.shape[:2]
fa = FaceAnalysis(name="buffalo_l", root=os.environ["IFROOT"], allowed_modules=["detection"], providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
faces = sorted(fa.get(src), key=lambda f: f.bbox[0])
rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
# skin of the source photo, full frame (multiclass), and each person's cut-out (person model at their face)
sq = cv2.copyMakeBorder(src, 0, SW - SH, 0, 0, cv2.BORDER_CONSTANT)
P, _ = ST._skin_prob(cv2.cvtColor(cv2.resize(sq, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
skin_src = cv2.resize(P, (SW, SW))[:SH] > 0.5
def cutout(i):
    c = ((faces[i].bbox[0] + faces[i].bbox[2]) / 2, (faces[i].bbox[1] + faces[i].bbox[3]) / 2)
    m = ST._person_prob(np.ascontiguousarray(rgb), c)
    m = cv2.GaussianBlur((m > 0.5).astype(np.float32), (0, 0), 1.2)
    return m
lay = {i: cutout(i) for i in (0, 4, 5)}          # the left woman (#1), the standing man (#5), the right woman (#6)
yy, xx = np.mgrid[0:720, 0:1280].astype(np.float32)
bg = np.dstack([60 + 40 * xx / 1280, 95 + 50 * yy / 720, 150 + 40 * (1 - yy / 720)]).astype(np.float32)   # a plain warm wall
bg += np.random.default_rng(3).normal(0, 2.0, bg.shape).astype(np.float32)
def place(img, M, size=(1280, 720), border=0):
    return cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
def tr(pts, M):
    return (np.c_[pts, np.ones(len(pts))] @ M.T).astype(np.float32)
SCENES = {
    # name: (back person, front person, back placement (dx, dy, scale), front path start -> end, frames)
    "manwoman": (4, 5, (-560, 40, 1.0), ((-700, 80), (-420, 80)), 48),
    "twowomen": (0, 5, (380, -60, 1.0), ((-560, 0), (-260, 0)), 48),
}
for name, (ib, ifr, (bx, by, bs), ((fx0, fy0), (fx1, fy1)), n) in SCENES.items():
    frames, owner, skin, recs = [], [], [], []
    Mb = np.float32([[bs, 0, bx], [0, bs, by]])
    for k in range(n):
        t = k / max(1, n - 1); t = t * t * (3 - 2 * t)
        Mf = np.float32([[1, 0, fx0 + (fx1 - fx0) * t], [0, 1, fy0 + (fy1 - fy0) * t]])
        ab, af = place(lay[ib], Mb), place(lay[ifr], Mf)
        img = bg.astype(np.float32)
        img = img * (1 - ab[..., None]) + place(src, Mb).astype(np.float32) * ab[..., None]
        img = img * (1 - af[..., None]) + place(src, Mf).astype(np.float32) * af[..., None]
        own = np.where(af > 0.5, 2, np.where(ab > 0.5, 1, 0)).astype(np.uint8)       # 1 = back person, 2 = front person
        sk = np.where(own == 2, place(skin_src.astype(np.uint8), Mf), np.where(own == 1, place(skin_src.astype(np.uint8), Mb), 0))
        frames.append(np.clip(img + 0.5, 0, 255).astype(np.uint8)); owner.append(own); skin.append(sk.astype(np.uint8))
        r = {}
        for slot, (i, M) in enumerate(((ib, Mb), (ifr, Mf))):
            f = faces[i]; b = tr(f.bbox.reshape(2, 2), M).reshape(4)
            r[slot] = dict(kps=tr(f.kps, M), bbox=b)
        recs.append(r)
    np.savez_compressed(os.path.join(OUT, "pair_%s.npz" % name), frames=np.array(frames), owner=np.array(owner), skin=np.array(skin),
                        kps=np.array([[r[s]["kps"] for s in (0, 1)] for r in recs]), bbox=np.array([[r[s]["bbox"] for s in (0, 1)] for r in recs]))
    vw = cv2.VideoWriter(os.path.join(OUT, "pair_%s.mp4" % name), cv2.VideoWriter_fourcc(*"mp4v"), 24, (1280, 720))
    for f in frames: vw.write(f)
    vw.release()
    cv2.imwrite(os.path.join(OUT, "pair_%s_mid.jpg" % name), cv2.resize(np.hstack([frames[n // 2], cv2.cvtColor(owner[n // 2] * 120, cv2.COLOR_GRAY2BGR)]), None, fx=0.4, fy=0.4))
    print(name, n, "frames; the front person covers the back one on", int(((np.array(owner) == 2) & (place(lay[ib], Mb)[None] > 0.5)).sum(axis=(1, 2)).max()), "px at most")
