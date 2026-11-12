"""v11.2.206 (SDOS-206): no false cuts. Faces with nothing in front of them (insightface's public sample photos: six people
at a table, with hair along their faces, some turned; NOT user media), each moved a little frame after frame (a slow drift and a
small shake, 16 frames). The mask of v11.2.206 must not cut more of the face than v11.2.205's does.
usage: nofalse206.py PACK205 PACK206 XSEG.onnx IMAGE..."""
import sys, os, types, importlib.util
P205, P206, XSEG = sys.argv[1:4]
IMGS = sys.argv[4:]
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
sys.path.insert(0, P206)
import numpy as np, cv2


def load(name, pack):
    spec = importlib.util.spec_from_file_location(name, os.path.join(pack, "swap_engine.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


E5, E6 = load("se205", P205), load("se206", P206)
for E in (E5, E6):
    assert E.occ_net_load(XSEG, threads=2)
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_2d_106"], providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(640, 640))
fails = 0


def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1


def run(E, img, face, size=256, n=16):
    comp = E.AlignedCompositor.__new__(E.AlignedCompositor)
    tr = E.TrackState(slot=0)
    out = []
    for t in range(n):
        dx, dy = 0.6 * t + 1.5 * np.sin(t * 1.7), 0.3 * t + 1.5 * np.cos(t * 2.3)       # drift and shake
        T = np.float32([[1, 0, dx], [0, 1, dy]])
        fr = cv2.warpAffine(img, T, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
        kps = face.kps + np.float32([dx, dy]); lm = face.landmark_2d_106 + np.float32([dx, dy])
        f = types.SimpleNamespace(kps=kps, landmark_2d_106=lm, bbox=face.bbox + np.float32([dx, dy, dx, dy]),
                                  det_score=face.det_score, pose=getattr(face, "pose", None))
        M = E.estimate_norm(kps, size)
        at = cv2.warpAffine(fr, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
        out.append(comp.build_mask(f, M, size, track=tr, aligned_target=at, frame_wh=(fr.shape[1], fr.shape[0])))
    return out


worst, n_faces = 0.0, 0
for p in IMGS:
    img = cv2.imread(p)
    faces = [f for f in fa.get(img) if (f.bbox[2] - f.bbox[0]) >= 40]
    for i, f in enumerate(faces):
        m5, m6 = run(E5, img, f), run(E6, img, f)
        face_area = np.mean([float((a > 0.05).sum()) for a in m5])
        more = [float(((b < a - 0.10) & (a > 0.05)).sum()) / max(1.0, face_area) for a, b in zip(m5, m6)]
        less = [float(((b > a + 0.10) & (a > 0.05)).sum()) / max(1.0, face_area) for a, b in zip(m5, m6)]
        print("  %s face %d (%d px wide): v11.2.206 cuts more than v11.2.205 on %.2f%% of the face (worst frame), less on %.2f%%"
              % (os.path.basename(p), i, int(f.bbox[2] - f.bbox[0]), 100 * max(more), 100 * max(less)), flush=True)
        worst = max(worst, max(more)); n_faces += 1
check("%d faces, nothing in front: v11.2.206 cuts no more of any face than v11.2.205 (at most 1 %% of it, on its worst frame)" % n_faces,
      n_faces > 0 and worst <= 0.01, "worst %.2f%% of a face" % (100 * worst))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
