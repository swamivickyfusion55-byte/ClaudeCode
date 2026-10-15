"""With occ_img_refine=0, depth_order=0, contact_scale_lock=0 the new pack must give the SAME composited frame as v11.2.86, bit for bit.
Real compositor (_composite: build_mask + paste_back), cached detections of the food clips, a fixed 'other identity' face."""
import sys, os, types, numpy as np, cv2
BASE = '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/'
def load(path, name):
    sys.path.insert(0, path)
    for m in ('swap_engine', 'config'): sys.modules.pop(m, None)
    import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
    sys.path.remove(path); sys.modules[name] = E; return E
E86 = load(BASE + 'pack86', 'e86'); E87 = load(BASE + 'pack87', 'e87')
E87._P.update({'occ_img_refine': 0.0, 'depth_order': 0.0, 'contact_scale_lock': 0.0})
for E in (E86, E87): E._P['grain_strength'] = 0.0
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name='buffalo_l', root='/root/.insightface', allowed_modules=['detection', 'landmark_2d_106'], providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=-1, det_size=(640, 640))
photo = cv2.imread('/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg')
sf = sorted(fa.get(photo), key=lambda f: f.bbox[0])[0]
Ms = E86.estimate_norm(sf.kps, 128); fake = cv2.warpAffine(photo, Ms, (128, 128), borderMode=cv2.BORDER_REPLICATE)
worst = 0.0; n = 0; moved_default = 0.0
for spec in sys.argv[1:] or ['vanilla_L', 'mint_R', 'strawberry_L']:
    cap = cv2.VideoCapture(BASE + f'bench/food_{spec}.mp4'); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    dets = list(np.load(BASE + f'bench/det_{spec}.npz', allow_pickle=True)['d'])
    c6, t6 = E86.AlignedCompositor(None), E86.TrackState(0); c7, t7 = E87.AlignedCompositor(None), E87.TrackState(0)
    last = None
    for i, img in enumerate(frames[:82]):
        d = dets[i]
        if d is None and last is None: continue
        if d is not None:
            face = types.SimpleNamespace(kps=d[0], landmark_2d_106=d[1]); M = E86.estimate_norm(d[0], 128); last = (M, face)
        else: M, face = last
        o6 = c6._composite(img.copy(), img, face, fake, M, track=t6, alpha=1.0, colour_strength=1.0, occlusion_guard=0.0, rivals=None)
        o7 = c7._composite(img.copy(), img, face, fake, M, track=t7, alpha=1.0, colour_strength=1.0, occlusion_guard=0.0, rivals=None)
        worst = max(worst, float(np.abs(o6.astype(np.int16) - o7.astype(np.int16)).max())); n += 1
print("frames compared:", n, "| largest pixel difference v11.2.86 vs v11.2.87 with occ_img_refine / depth_order / contact_scale_lock at 0:", worst)
