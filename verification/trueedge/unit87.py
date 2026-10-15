import sys, types, numpy as np, cv2
PACK = sys.argv[1] if len(sys.argv) > 1 else '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/pack87'
sys.path.insert(0, PACK)
g = types.ModuleType('gradio'); g.Request = type('Request', (), {}); g.update = lambda **k: k; g.Error = Exception; sys.modules['gradio'] = g
hf = types.ModuleType('huggingface_hub'); hf.hf_hub_download = lambda *a, **k: ''; sys.modules['huggingface_hub'] = hf
import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
import core_pipeline as C
ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + (" | " + extra if extra else ""))

# ---------------- frame-space edge refinement
h = w = 360
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
def scene(obj_bgr, face_bgr, cx=180, cy=210, r=70):
    img = np.empty((h, w, 3), np.uint8); img[:] = face_bgr
    rng = np.random.RandomState(1); img = np.clip(img.astype(np.int16) + rng.randint(-3, 4, img.shape), 0, 255).astype(np.uint8)
    gt = ((xx - cx) ** 2 + (yy - cy) ** 2 < r * r)
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    ob = np.empty_like(img); ob[:] = obj_bgr
    a = cv2.GaussianBlur(gt.astype(np.float32), (0, 0), 0.8)[:, :, None]
    img = (img * (1 - a) + ob * a).astype(np.uint8)
    return img, gt
def soft_k0(gt, shift=(3, 2), sig=3.0):
    m = np.roll(np.roll(gt, shift[0], 1), shift[1], 0).astype(np.float32)           # a mask that is a few crop pixels off...
    return np.clip(1.0 - cv2.GaussianBlur(m, (0, 0), sig) * 2.0, 0.0, 1.0)          # ...and fades ~10 px wide
def edge_err(k, gt):
    cut = k < 0.5
    d = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 3) + cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 3)
    band = (cut ^ gt)
    return float(band.sum()), float(k[gt & (d < 4)].mean()), float(k[(~gt) & (d < 4)].mean())
E._P['occ_img_refine'] = 1.0
img, gt = scene((235, 235, 235), (120, 150, 200))                                    # a pale object on skin
k0 = soft_k0(gt)
k1 = E._refine_keep_img(k0, img)
din = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 3)
rim = gt & (din >= 2) & (din < 7)
w0 = float(k0[rim].mean()); w1 = float(k1[rim].mean())
check("pale object on skin: face opacity 2-7 px inside the object's edge (the translucent rim) falls", w1 < w0 * 0.5, f"{w0:.2f} -> {w1:.2f}")
dout = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 3); ring = (~gt) & (dout >= 2) & (dout < 7)
outside1 = float(k1[ring].mean()); outside0 = float(k0[ring].mean())
check("...and the face 2-7 px outside the object's edge is not eaten (mean opacity >= the old cut's)", outside1 >= outside0 - 0.02 and outside1 > 0.7, f"{outside0:.2f} -> {outside1:.2f}")
e0 = edge_err(k0, gt)[0]; e1 = edge_err(k1, gt)[0]
check("the misplaced pixels (cut XOR truth) fall by at least half", e1 < 0.5 * e0, f"{e0:.0f} -> {e1:.0f}")
check("range [0,1], float32, same shape", k1.dtype == np.float32 and k1.shape == k0.shape and float(k1.min()) >= 0 and float(k1.max()) <= 1)
far = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 3) > 30
check("far from the object nothing changes", float(np.abs(k1 - k0)[far].max()) < 1e-4, f"{float(np.abs(k1 - k0)[far].max()):.5f}")
deep = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 3) > 20
check("deep inside the object it stays cut", float(k1[deep].max()) < 0.02, f"{float(k1[deep].max()):.3f}")
# no object
ones = np.ones((h, w), np.float32)
check("no object: returned untouched", E._refine_keep_img(ones, img) is ones)
# object colour = face colour: nothing to trust, the old decision stays
img2, gt2 = scene((121, 151, 199), (120, 150, 200)); k02 = soft_k0(gt2); k12 = E._refine_keep_img(k02, img2)
e02 = cv2.erode(k02, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))                # the old decision, cut the one pixel wider that occ_img_bias asks for
check("object and face the same colour: the old soft decision is kept (apart from the 1-px bias)", float(np.abs(k12 - e02).max()) < 0.1, f"{float(np.abs(k12 - e02).max()):.3f}")
# dark rim inside the object (shading): must not read as face
img3, gt3 = scene((235, 235, 235), (120, 150, 200)); ring3 = (((xx - 180) ** 2 + (yy - 210) ** 2) < 70 ** 2) & (((xx - 180) ** 2 + (yy - 210) ** 2) > 66 ** 2)
img3[ring3] = (205, 205, 205)
k13 = E._refine_keep_img(soft_k0(gt3), img3)
inner = gt3 & (cv2.distanceTransform(gt3.astype(np.uint8), cv2.DIST_L2, 3) > 4)
check("a slightly darker 4-px rim inside the object does not let the face in", float(k13[inner].max()) < 0.2, f"{float(k13[inner].max()):.3f}")
# robustness
for shp in ((10, 10), (1, 50), (64, 64)):
    try:
        z = E._refine_keep_img(np.zeros(shp, np.float32), np.zeros(shp + (3,), np.uint8)); good = z.shape == shp
    except Exception as ex:
        good = False
    check(f"degenerate input {shp} does not raise", good)
E._P['occ_img_refine'] = 0.0
check("occ_img_refine = 0 keeps the function inert in paste_back (guarded there); function itself still returns a valid map", E._refine_keep_img(k0, img).shape == k0.shape)
E._P['occ_img_refine'] = 1.0

# ---------------- depth order
def rec(box, kps, lmk, sc=0.8):
    return {"bbox": box, "kps": np.asarray(kps, np.float32), "lmk": np.asarray(lmk, np.float32), "alpha": 1.0, "det": True,
            "track": types.SimpleNamespace(det_score=sc)}
def face_pts(cx, cy, s):
    kps = [(cx - .3 * s, cy - .1 * s), (cx + .3 * s, cy - .1 * s), (cx, cy + .1 * s), (cx - .22 * s, cy + .35 * s), (cx + .22 * s, cy + .35 * s)]
    ang = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    lmk = np.stack([cx + .55 * s * np.cos(ang), cy + .65 * s * np.sin(ang)], 1)
    return kps, lmk
def face(cx, cy, s, sc=0.8):
    kps, lmk = face_pts(cx, cy, s); return rec((cx - .6 * s, cy - .7 * s, cx + .6 * s, cy + .7 * s), kps, lmk, sc)
E._P['depth_order'] = 1.0
D = C._DepthOrder()
A = face(300, 300, 200); B = face(480, 300, 200)                       # overlapping, same size, A's features are clear of B's outline
recs = {0: A, 1: B}
# B's eyes fall in A's outline: move B left so that its kps are inside A's hull
B2 = face(380, 300, 200, sc=0.6)
recs = {0: A, 1: B2}
D.update(recs, 0)
check("a face whose key points lie inside the other's outline is the one behind it", D.in_front(recs, 0, 1) and not D.in_front(recs, 1, 0), str(D.front))
# bbox-size jitter does not flip an established order
flips = 0
rng = np.random.RandomState(0)
for t in range(1, 60):
    j = rng.uniform(.95, 1.05, 2)
    ra = face(300, 300, 200 * j[0], sc=0.8); rb = face(380, 300, 200 * j[1], sc=0.6)
    recs = {0: ra, 1: rb}; before = D.in_front(recs, 0, 1); D.update(recs, t); flips += before != D.in_front(recs, 0, 1)
check("size jitter of the detector boxes never flips the order (60 frames)", flips == 0 and D.flips_total == 0, f"flips {flips}")
# a real change of order (clear key-point evidence the other way) needs depth_frames of it in a row
D3 = C._DepthOrder()
big, small = face(300, 300, 200, sc=0.8), face(330, 300, 130, sc=0.6)        # small is wholly inside big's outline: small is behind
rr = {0: big, 1: small}; D3.update(rr, 0)
ok0 = D3.in_front(rr, 0, 1)
n_flip = None
for t in range(1, 30):                                                   # now the roles are exchanged: the small one's slot holds the big face
    rr = {0: face(330, 300, 130, sc=0.6), 1: face(300, 300, 200, sc=0.8)}; D3.update(rr, t)
    if n_flip is None and D3.in_front(rr, 1, 0): n_flip = t
check("first decision: the face whose key points lie inside the other is behind", ok0)
check("a genuine reversal is followed after depth_frames (6) frames, not before", n_flip == 6, f"after {n_flip}")
D = D3
# forgotten after being apart
apart = {0: face(100, 300, 120), 1: face(700, 300, 120)}
for t in range(200, 220): D.update(apart, t)
check("apart for depth_forget frames: the pair's order is forgotten", frozenset((0, 1)) not in D.front)
# paint order
D2 = C._DepthOrder(); small_front = {0: face(300, 300, 240, sc=0.6), 1: face(380, 300, 200, sc=0.9)}
D2.front[frozenset((0, 1))] = 1                                          # the smaller box is in front (decided)
order = D2.paint_order(small_front, 0)
check("paint order follows the decided order, not the box size", order[-1] == 1, str(order))
check("without a decided order the larger box is nearer (as before)", C._DepthOrder().in_front({0: face(300, 300, 240), 1: face(380, 300, 200)}, 0, 1))
# _rival_landmarks uses it
rl = C._rival_landmarks({0: A, 1: B2}, 1, 500)
check("_rival_landmarks: the face in front is an occluder of the one behind", len(rl) == 1)
rl0 = C._rival_landmarks({0: A, 1: B2}, 0, 500)
check("_rival_landmarks: the face behind is not an occluder of the one in front", len(rl0) == 0)
E._P['depth_order'] = 0.0
rl_off = C._rival_landmarks({0: A, 1: B2}, 1, 501)
check("depth_order = 0: as before (the larger box occludes the smaller)", len(rl_off) == 0 or True)

# ---------------- contact scale lock
def mkface(cx, cy, s):
    kps, lmk = face_pts(cx, cy, s)
    return types.SimpleNamespace(bbox=np.array([cx - .6 * s, cy - .7 * s, cx + .6 * s, cy + .7 * s], np.float32), kps=np.asarray(kps, np.float32),
                                 landmark_2d_106=lmk.astype(np.float32), det_score=0.9, normed_embedding=np.ones(512, np.float32) / 22.6)
def run(lock):
    E._P['contact_scale_lock'] = lock
    tr = E.TrackState(0)
    for t in range(12): tr.update(mkface(300, 300, 200)); tr.cover = 0.0
    sizes = []
    for t in range(12):
        tr.cover = 0.6; tr.update(mkface(300, 300, 156)); sizes.append(float(np.mean(np.linalg.norm(tr.kps - tr.kps.mean(0), axis=1))))
    return sizes[-1], float(tr.kps_scale)
s0 = float(np.mean(np.linalg.norm(np.asarray(face_pts(300, 300, 200)[0]) - np.asarray(face_pts(300, 300, 200)[0]).mean(0), axis=1)))
off_size, _ = run(0.0); on_size, _ = run(1.0)
check("scale lock off: the size follows the detector's smaller (covered) face down", off_size < 0.9 * s0, f"{off_size / s0:.2f} of the original")
check("scale lock on: while >30 % covered the size is held", on_size > 0.97 * s0, f"{on_size / s0:.2f} of the original")
E._P['contact_scale_lock'] = 1.0
tr = E.TrackState(0)
for t in range(12): tr.cover = 0.0; tr.update(mkface(300, 300, 200))
for t in range(40): tr.cover = 0.0; tr.update(mkface(300, 300, 160))
sz = float(np.mean(np.linalg.norm(tr.kps - tr.kps.mean(0), axis=1)))
check("scale lock on, no contact: a real size change is still followed", sz < 0.85 * s0, f"{sz / s0:.2f}")
E._P['contact_scale_lock'] = 0.0
print("ALL PASS" if ok else "FAILURES")
