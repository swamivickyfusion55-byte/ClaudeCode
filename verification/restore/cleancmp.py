"""Clean faces (no object): how much does each version's composited frame differ from v11.2.83's, and how much does the occluder gate cut?
Real compositor (_composite), real detections of the bench clips, a fixed other identity as the swap, grain off (random)."""
import sys, os, types, numpy as np, cv2
BASE = '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/'
PACKS = os.environ.get('PACKS', 'pack83,pack84,pack85,pack86,pack87').split(',')
OVR = {p.split(':')[0]: dict(kv.split('=') for kv in p.split(':')[1].split(';')) for p in filter(None, os.environ.get('POVR', '').split('|'))}
def load(path, name):
    sys.path.insert(0, path)
    for m in ('swap_engine', 'config'): sys.modules.pop(m, None)
    import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
    sys.path.remove(path); sys.modules[name] = E; return E
Es = {}
for p in PACKS:
    pk = p.split('@')[0]; Es[p] = load(BASE + pk, 'e_' + p.replace('@', '_')); Es[p]._P['grain_strength'] = 0.0
    for k, v in OVR.get(p, {}).items(): Es[p]._P[k] = float(v)
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name='buffalo_l', root='/root/.insightface', allowed_modules=['detection', 'landmark_2d_106'], providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=-1, det_size=(640, 640))
photo = cv2.imread('/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg')
sf = sorted(fa.get(photo), key=lambda f: f.bbox[0])[-1]
E0 = Es[PACKS[0]]
fake = cv2.flip(cv2.warpAffine(photo, E0.estimate_norm(sf.kps, 128), (128, 128), borderMode=cv2.BORDER_REPLICATE), 1)
def frames(path, lim=None):
    c = cv2.VideoCapture(path); out = []
    while True:
        ok, f = c.read()
        if not ok or (lim and len(out) >= lim): break
        out.append(f)
    return out
def dets_for(clip, fr):
    cache = BASE + f'lip88/det_{os.path.basename(clip)}.npz'
    if os.path.exists(cache): return list(np.load(cache, allow_pickle=True)['d'])
    D = []
    for img in fr:
        D.append([(f.bbox, f.kps, f.landmark_2d_106) for f in fa.get(img) if f.det_score > 0.4])
    np.savez(cache, d=np.array(D, dtype=object)); return D
clips = sys.argv[1:] or ['kiss_cheek.mp4', 'kiss_hide.mp4', 'cheek_swell.mp4', 'clip90.mp4']
for clip in clips:
    fr = frames(BASE + 'bench/' + clip, 120); D = dets_for(clip, fr)
    # associate faces over time by the nearest centre
    tracks = []; assign = []
    for t, ds in enumerate(D):
        cur = {}
        for d in ds:
            c = np.array([(d[0][0] + d[0][2]) / 2, (d[0][1] + d[0][3]) / 2])
            best = None
            for ti, tc in enumerate(tracks):
                if ti in cur: continue
                dist = np.linalg.norm(c - tc)
                if dist < 120 and (best is None or dist < best[1]): best = (ti, dist)
            if best is None: tracks.append(c); best = (len(tracks) - 1, 0)
            tracks[best[0]] = c; cur[best[0]] = d
        assign.append(cur)
    res = {p: {'diff': [], 'vis': [], 'cut': []} for p in PACKS}
    comps = {p: Es[p].AlignedCompositor(None) for p in PACKS}; trs = {p: {} for p in PACKS}
    for t, img in enumerate(fr):
        outs = {}
        for p in PACKS:
            E = Es[p]; out = img.copy(); cutf = 0.0
            for ti, d in sorted(assign[t].items()):
                tr = trs[p].setdefault(ti, E.TrackState(ti))
                face = types.SimpleNamespace(kps=d[1], landmark_2d_106=d[2]); M = E.estimate_norm(d[1], 128)
                out = comps[p]._composite(out, img, face, fake, M, track=tr, alpha=1.0, colour_strength=1.0, occlusion_guard=0.0, rivals=None)
                k = getattr(tr, 'occ_keep', None)
            outs[p] = out
        # face region: union of the boxes
        reg = np.zeros(img.shape[:2], bool)
        for d in assign[t].values():
            b = d[0].astype(int); reg[max(0, b[1]):b[3], max(0, b[0]):b[2]] = True
        if reg.sum() < 100: continue
        o0 = outs[PACKS[0]].astype(np.int16)
        for p in PACKS:
            dd = np.abs(outs[p].astype(np.int16) - o0).max(axis=2)
            res[p]['diff'].append(float(dd[reg].mean())); res[p]['vis'].append(float((dd[reg] > 12).mean() * 100))
    print(f"== {clip}  ({len(res[PACKS[0]]['diff'])} frames)  difference from {PACKS[0]} inside the face boxes")
    for p in PACKS:
        v = np.array(res[p]['vis']); d = np.array(res[p]['diff'])
        print(f"   {p:10s} mean |diff| {d.mean():5.2f}  | % of face-box px changed >12 levels: mean {v.mean():5.2f}  worst frame {v.max():5.2f}  | frames with >0.5% changed: {int((v > 0.5).sum())}")
