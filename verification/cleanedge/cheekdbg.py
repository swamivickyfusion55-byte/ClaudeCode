"""Run the real detector + inswapper + AlignedCompositor over the first frames of the cheek clip and dump the aligned pieces."""
import sys, os, numpy as np, cv2, types
PACK = sys.argv[1]; CLIP = sys.argv[2]; FR = [int(x) for x in sys.argv[3].split(',')]; OUT = sys.argv[4]
sys.path.insert(0, PACK)
import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
for kv in filter(None, os.environ.get('OVR', '').split(',')):
    k_, v_ = kv.split('='); E._P[k_] = float(v_)
import insightface
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name='buffalo_l', root='/root/.insightface', allowed_modules=['detection', 'landmark_2d_106', 'recognition'], providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=-1, det_size=(640, 640))
sw = insightface.model_zoo.get_model('/tmp/claude-0/swapmodel/inswapper_128.onnx', providers=['CPUExecutionProvider'])
photo = cv2.imread('/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg')
srcs = sorted(fa.get(photo), key=lambda f: f.bbox[0]); src = srcs[0]            # face 0 as the identity
cap = cv2.VideoCapture(CLIP); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
comp = E.AlignedCompositor(sw); tr = E.TrackState(0); out = {}
for i in range(0, max(FR) + 1):
    img = frames[i]
    faces = fa.get(img)
    faces = [f for f in faces if abs((f.bbox[0] + f.bbox[2]) / 2 - 520) < 90 and f.bbox[1] > 120]
    if not faces: continue
    face = faces[0]
    fake, M = comp._raw_swap(img, face, src)
    size = fake.shape[0]
    aligned = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
    tr.cache_fake(fake, E.aligned_correction(face.kps, M, size), i)
    mask = comp.build_mask(face, M, size, track=tr, aligned_target=aligned)
    toned = comp.colour_match(fake, aligned, mask, track=tr)
    res = comp.paste_back(img.copy(), toned, mask, M, orig=img, track=tr)
    if i in FR:
        out[i] = (aligned, fake, toned, (mask * 255).astype(np.uint8), res[140:340, 430:610])
    print(i, end=' ', flush=True)
np.savez_compressed(OUT, **{f"{k}_{n}": v[j] for k, v in out.items() for j, n in enumerate(("aligned", "fake", "toned", "mask", "res"))})
