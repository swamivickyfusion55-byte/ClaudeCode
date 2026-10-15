"""Image-domain test of the compositor's object edge, without the swapper: real detector on a clip frame, real TrackState /
build_mask / paste_back, a fixed 'other identity' face as the swap. Reports the profile of |out-in| across the object's outline
(normalised by the painted face), the same quantity as bench/profile.py gets from full pipeline jobs.
usage: python3 imgharness.py PACK SPEC [size] ; OVR="k=v,..." overrides tunables"""
import sys, os, numpy as np, cv2
PACK = sys.argv[1]; SPEC = sys.argv[2]; SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 128
sys.path.insert(0, PACK)
import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
for kv in filter(None, os.environ.get('OVR', '').split(',')):
    k_, v_ = kv.split('='); E._P[k_] = float(v_)
BENCH = '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/'
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name='buffalo_l', root='/root/.insightface', allowed_modules=['detection', 'landmark_2d_106'], providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=-1, det_size=(640, 640))
photo = cv2.imread('/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg')
sf = sorted(fa.get(photo), key=lambda f: f.bbox[0])[0]
Ms = E.estimate_norm(sf.kps, 128); src128 = cv2.warpAffine(photo, Ms, (128, 128), borderMode=cv2.BORDER_REPLICATE)
fake = src128 if SIZE == 128 else cv2.resize(src128, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
cap = cv2.VideoCapture(BENCH + f'food_{SPEC}.mp4'); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
gt_all = np.load(BENCH + f'food_{SPEC}_gt.npz')['a']
POSD = {"L": (0.372, 0.579), "R": (0.621, 0.579), "M": (0.50, 0.667)}
_cx0, _cy0 = POSD[SPEC.split('_')[1]]
_yy, _xx = np.mgrid[0:720, 0:1280].astype(np.float32)
def scoop_mask(i):
    cx = _cx0 + 0.008 * np.sin((i - 35) / 3.0); cy = _cy0 + 0.006 * np.cos((i - 35) / 4.0)
    return ((((_xx - (cx * 720 + 280)) / (0.134 * 720)) ** 2 + ((_yy - cy * 720) / (0.134 * 0.92 * 720)) ** 2) < 1.0)
class _GT:
    def __getitem__(self, i): return (scoop_mask(i) * 255).astype(np.uint8)
gt = _GT()
cone_all = lambda i: (gt_all[i] > 127) & ~cv2.dilate(scoop_mask(i).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
comp = E.AlignedCompositor(None); tr = E.TrackState(0); outs = {}; lastM = None
comp2 = E.AlignedCompositor(None); tr2 = E.TrackState(0); alph = {}
white = np.full(fake.shape, 255, np.uint8)
saved = {k: E._P.get(k) for k in ('grain_strength', 'mblur_strength')}
CACHE = BENCH + f'det_{SPEC}.npz'
dets = None
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True); dets = list(z['d'])
else:
    dets = []
    for i, img in enumerate(frames[:82]):
        fs = [f for f in fa.get(img) if abs((f.bbox[0] + f.bbox[2]) / 2 - 640) < 160]
        dets.append((fs[0].kps, fs[0].landmark_2d_106) if fs else None)
    np.savez(CACHE, d=np.array(dets, dtype=object))
import types
for i, img in enumerate(frames[:82]):
    d = dets[i]
    if d is None and lastM is None: continue
    if d is not None:
        face = types.SimpleNamespace(kps=d[0], landmark_2d_106=d[1]); M = E.estimate_norm(d[0], SIZE); lastM = (M, face)
    else:
        M, face = lastM
    outs[i] = comp._composite(img.copy(), img, face, fake, M, track=tr, alpha=1.0, colour_strength=1.0, occlusion_guard=0.0, rivals=None)
    E._P['grain_strength'] = 0.0; E._P['mblur_strength'] = 0.0
    alph[i] = comp2._composite(np.zeros_like(img), img, face, white, M, track=tr2, alpha=1.0, colour_strength=0.0, occlusion_guard=0.0, rivals=None)[:, :, 0].astype(np.float32) / 255.0
    for k_, v_ in saved.items():
        if v_ is not None: E._P[k_] = v_
BX = (477, 144, 798, 585)
sup = np.zeros((720, 1280), bool); sup[BX[1]:BX[3], BX[0]:BX[2]] = True; rows = np.arange(720)[:, None] > 300
P = {d: [] for d in range(-14, 17)}; interior = []; plate = []
for i in range(48, 80):
    if i not in outs: continue
    g = gt[i] > 127
    if g.sum() < 500: continue
    D = np.abs(outs[i].astype(np.float32) - frames[i].astype(np.float32)).mean(axis=2)
    dout = cv2.distanceTransform((~g).astype(np.uint8), cv2.DIST_L2, 5); din = cv2.distanceTransform(g.astype(np.uint8), cv2.DIST_L2, 5); sd = np.where(g, -din, dout)
    for d in P:
        m = sup & rows & (sd >= d - 0.5) & (sd < d + 0.5) & ~cv2.dilate(cone_all(i).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        if m.sum() > 20: P[d].append(float(D[m].mean()))
pl = np.mean([np.mean(P[d]) for d in range(12, 17) if P[d]])
base = np.mean([np.mean(P[d]) for d in range(-14, -8) if P[d]])
prof = {d: (np.mean(v) - base) / (pl - base) for d, v in P.items() if v}      # 0 = untouched object, 1 = fully painted face
def cross(level):
    ds = sorted(prof)
    for a, b in zip(ds[:-1], ds[1:]):
        if prof[a] < level <= prof[b]: return a + (level - prof[a]) / (prof[b] - prof[a]) * (b - a)
    return float('nan')
base_a = np.mean([alph[i] for i in range(15, 25) if i in alph], axis=0)       # the face mask with no object (same photo): what the alpha would be anyway
# ---- exact alpha profile (white fake on black): painted-face alpha vs signed distance to the object's outline
PA = {d: [] for d in range(-14, 17)}
for i in range(48, 80):
    if i not in alph: continue
    g = gt[i] > 127
    if g.sum() < 500: continue
    dout = cv2.distanceTransform((~g).astype(np.uint8), cv2.DIST_L2, 5); din = cv2.distanceTransform(g.astype(np.uint8), cv2.DIST_L2, 5); sd = np.where(g, -din, dout)
    ref_a = alph[i] / np.maximum(base_a, 1e-3)
    for d in PA:
        m = sup & rows & (sd >= d - 0.5) & (sd < d + 0.5) & (base_a > 0.5) & ~cv2.dilate(cone_all(i).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        if m.sum() > 20: PA[d].append(float(ref_a[m].mean()))
pa = {d: float(np.mean(v)) for d, v in PA.items() if v}
# the face-mask value far from the object (the most the face can be painted there) is the plateau
plat = np.mean([pa[d] for d in range(12, 17) if d in pa]); pan = {d: v / plat for d, v in pa.items()}
def crossa(level):
    ds = sorted(pan)
    for a, b in zip(ds[:-1], ds[1:]):
        if pan[a] < level <= pan[b]: return a + (level - pan[a]) / (pan[b] - pan[a]) * (b - a)
    return float('nan')
# edge position error in frame pixels (area of disagreement per unit outline length) and ring flicker
errs = []; flick = []; prev_a = None
for i in range(48, 80):
    if i not in alph: continue
    g = gt[i] > 127
    if g.sum() < 500: continue
    ratio = np.clip(alph[i] / np.maximum(base_a, 1e-3), 0, 1)
    pred = ratio < 0.5
    ring_ = cv2.dilate(g.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & sup & rows & (base_a > 0.5) & ~cv2.dilate(cone_all(i).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    edge_ = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool) & sup
    L_ = float(edge_.sum()) / 2.0 + 1e-6
    errs.append(float(((pred ^ g) & ring_).sum()) / L_)
    if prev_a is not None: flick.append(float(np.abs(ratio - prev_a)[ring_].mean()))
    prev_a = ratio
perf = []
for i in range(48, 80):
    if i not in alph: continue
    g = gt[i] > 127
    if g.sum() < 500: continue
    din = cv2.distanceTransform(g.astype(np.uint8), cv2.DIST_L2, 5)
    inner = sup & (din >= 2) & (din <= 6) & (base_a > 0.5) & ~cv2.dilate(cone_all(i).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    perf.append((i, float((alph[i] / np.maximum(base_a, 1e-3))[inner].mean())))
print("  per-frame inside leak:", " ".join(f"{i}:{v:.2f}" for i, v in perf))
print(f"{SPEC} crop {SIZE} OVR={os.environ.get('OVR','')}")
print("  EDGE position error %.2f px | edge flicker (mean |d alpha| per frame in the ring) %.3f" % (np.mean(errs), np.mean(flick)))
print("  ALPHA profile (face alpha / far-face alpha):", " ".join(f"{d:+d}:{pan[d]:.2f}" for d in range(-8, 13, 2) if d in pan))
print("  ALPHA 10-90%% width %.1f px | 50%% point %+.1f px | alpha inside at -2..-6 px: %.3f | alpha at +2 px: %.2f, +6 px: %.2f" % (crossa(0.9) - crossa(0.1), crossa(0.5), np.mean([pan[-2], pan[-4], pan[-6]]), pan[2], pan[6]))
print("  profile (0 = object untouched, 1 = face fully painted):", " ".join(f"{d:+d}:{prof[d]:.2f}" for d in range(-8, 13, 2) if d in prof))
print("  10-90%% width %.1f px | 50%% point %+.1f px (negative = face stops short of the object edge) | leak at -2..-4 px: %.2f | face at +4 px: %.2f" % (cross(0.9) - cross(0.1), cross(0.5), np.mean([prof[-2], prof[-3], prof[-4]]), prof[4]))
g60 = (gt[60] > 127).astype(np.uint8); cnt = cv2.findContours(g60, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
am = cv2.cvtColor((np.clip(alph[60] / np.maximum(base_a, 1e-3), 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR); cv2.drawContours(am, cnt, -1, (0, 0, 255), 1)
cv2.imwrite(os.environ.get('IMG', '/tmp/claude-0/imgh.png').replace('.png', '_alpha.png'), am[200:640, 380:900])
cv2.imwrite(os.environ.get('IMG', '/tmp/claude-0/imgh.png'), np.hstack([frames[60][300:620, 380:900], outs[60][300:620, 380:900]]))
