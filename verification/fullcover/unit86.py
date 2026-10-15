import sys, numpy as np, cv2
sys.path.insert(0, '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/pack86')
import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + (" | " + extra if extra else ""))
h = w = 192
# --- _ext_floor
base = np.zeros((h, w), bool); base[int(.46 * h):int(.80 * h), 60:130] = True
check("ext floor: a little above the found part (0.46-0.15=0.31, never above occ_ext_min 0.20)", abs(E._ext_floor(base, h) - 0.31) < 0.01, f"{E._ext_floor(base, h):.3f}")
base2 = np.zeros((h, w), bool); base2[int(.25 * h):int(.80 * h), 60:130] = True
check("ext floor: never above occ_ext_min", abs(E._ext_floor(base2, h) - 0.20) < 1e-6, f"{E._ext_floor(base2, h):.3f}")
E._P['occ_ext_min'] = 0.0; check("ext floor off -> 0 (old limit)", E._ext_floor(base, h) == 0.0); E._P['occ_ext_min'] = 0.2
check("ext floor: no object -> 0", E._ext_floor(np.zeros((h, w), bool), h) == 0.0)
# --- nearest-colour novelty vs per-channel range
ref = np.zeros((h, w, 3), np.float32)
ref[:, :, 0] = 0.0; ref[:, :, 1] = 0.0; ref[:, :, 2] = 0.0
ref[:, 90:94, 0] = 8.0                         # a bright (luma +8) stripe of the face, chroma 0
ref[:, 96:100, 1] = 8.0                        # a reddish (chroma +8) stripe of the face, luma 0
nrm = np.zeros((h, w, 3), np.float32); nrm[:, :, 0] = 8.0; nrm[:, :, 1] = 8.0   # luma +8 AND chroma +8 together: never seen
nov = E._novel_nn(nrm, ref, 5)
tol = 5; kk = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tol + 1, 2 * tol + 1)); lo = cv2.erode(ref, kk); hi = cv2.dilate(ref, kk)
dist = np.maximum(np.maximum(lo - nrm, nrm - hi), 0.0); env_nov = (np.sqrt(dist[:, :, 1] ** 2 + dist[:, :, 2] ** 2) > 1.5) | (dist[:, :, 0] > 1.5)
check("between two stripes: the per-channel range calls luma+8/chroma+8 'seen', the nearest-colour test calls it new", (~env_nov[:, 94:96]).all() and nov[:, 94:96].all(),
      f"range novel {env_nov[:, 94:96].mean():.2f}, nearest novel {nov[:, 94:96].mean():.2f}")
nrm2 = np.zeros((h, w, 3), np.float32); nrm2[:, :, 0] = 8.0
check("a colour that IS in the learned face within the tolerance is not new", not E._novel_nn(nrm2, ref, 5)[:, 92].any())
# --- held statistics
img = np.full((h, w, 3), (120, 150, 200), np.uint8)
st = E._face_stats(img); check("face stats return the 6 numbers", st is not None and st[3].shape == (6,))
held = st[3].copy(); held[0] += 20
nrm_h = E._face_stats(img, held=held)[0]
check("held statistics are used (a uniform image reads 20/sigma below the held luma median)", float(nrm_h[:, :, 0].mean()) < -0.1, f"{float(nrm_h[:, :, 0].mean()):.2f}")
check("held=None -> identical to before", np.allclose(E._face_stats(img)[0], E._face_stats(img, held=None)[0]))
# --- quarantine
tr = E.TrackState(0); nov = np.zeros((h, w), bool); nov[100:140, 60:120] = True; inface = np.ones((h, w), bool)
q = E.AlignedCompositor._quarantine(tr, {"novel": nov, "inface": inface})
check("quarantine: new pixels are not learned at all (rate 0), the rest at the normal one", q is not None and float(q[120, 90]) == 0.0 and float(q[10, 10]) == 1.0)
big = np.ones((h, w), bool)
check("quarantine: not when most of the face is new at once and no object is being cut (head turn / light)", E.AlignedCompositor._quarantine(tr, {"novel": big, "inface": inface}) is None)
check("...but yes when an object is being cut (a large object is the likelier reason)", E.AlignedCompositor._quarantine(E.TrackState(0), {"novel": big, "inface": inface, "present": True}) is not None)
tr2 = E.TrackState(0)
for _ in range(95): q = E.AlignedCompositor._quarantine(tr2, {"novel": nov, "inface": inface})
check("quarantine ends after occ_quar_age frames (amnesty)", float(q[120, 90]) == 1.0)
E._P['occ_quar_on'] = 0.0; check("occ_quar_on=0 -> off", E.AlignedCompositor._quarantine(E.TrackState(0), {"novel": nov, "inface": inface}) is None); E._P['occ_quar_on'] = 1.0
tr3 = E.TrackState(0); tr3.occ_age = np.ones((10, 10), np.uint8); tr3.occ_pres = 5; tr3.occ_stats = np.zeros(6, np.float32)
tr3.reset_for_cut(); check("reset_for_cut clears age map, presence counter and held statistics", tr3.occ_age is None and tr3.occ_pres == 0 and tr3.occ_stats is None)
# --- freeze: the model must not move while an object is being cut, and must follow again afterwards
sys.argv = ['x']; exec(open('/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip/realset.py').read())
sys.path.insert(0, '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip86'); from food import food
face0 = load_faces()[0]; rng = np.random.default_rng(3); trk = E.TrackState(0); refs = {}
for t in range(110):
    base = jitter(face0, rng)
    f = food(base, 0.5, 0.76, 0.19, (40, 55, 85), rng, ang=65)[0] if 40 <= t < 80 else base
    E.AlignedCompositor.occluder_keep_final(trk, np.clip(f, 0, 255).astype(np.uint8)); refs[t] = trk.occ_ref.copy()
frozen = max(float(np.abs(refs[t] - refs[50]).max()) for t in range(50, 78))
check("face model frozen while the object is cut (frames 50-77: no change at all)", frozen == 0.0, f"max change {frozen:.6f}")
moved = float(np.abs(refs[109] - refs[84]).max()); check("...and it follows the face again after the object is gone", moved > 0.0, f"change {moved:.4f}")
E._P['occ_freeze'] = 0.0; trk2 = E.TrackState(0); rng = np.random.default_rng(3); r2 = {}
for t in range(80):
    base = jitter(face0, rng)
    f = food(base, 0.5, 0.76, 0.19, (40, 55, 85), rng, ang=65)[0] if 40 <= t < 80 else base
    E.AlignedCompositor.occluder_keep_final(trk2, np.clip(f, 0, 255).astype(np.uint8)); r2[t] = trk2.occ_ref.copy()
check("occ_freeze=0 -> the model does move during the hold (old behaviour)", max(float(np.abs(r2[t] - r2[50]).max()) for t in range(50, 78)) > 0.0); E._P['occ_freeze'] = 1.0
# --- attach
ys = (np.arange(h, dtype=np.float32) + 0.5) / h
pic = np.full((h, w, 3), (140, 160, 200), np.uint8)
occ = np.zeros((h, w), np.uint8); cv2.circle(occ, (96, 110), 24, 255, -1)
cv2.circle(pic, (96, 110), 24, (240, 240, 240), -1)
cone = np.zeros((h, w), bool); cone[134:180, 84:108] = True                      # a cone below the scoop
pic[cone] = (60, 120, 175)                                                      # with a real outline
nov = cone | (occ > 0)
res = E._attach_novel(occ, nov, ys, h, w, pic)
check("an attached piece with a real outline is taken in", int((res > 0).sum()) > int((occ > 0).sum()) + 0.8 * cone.sum(), f"{int((occ>0).sum())} -> {int((res>0).sum())}")
pic2 = np.full((h, w, 3), (140, 160, 200), np.uint8); cv2.circle(pic2, (96, 110), 24, (240, 240, 240), -1)       # same 'new' pixels but no edge in the picture
res2 = E._attach_novel(occ, nov, ys, h, w, pic2)
check("the same new pixels with no edge around them are NOT taken in (a drift of skin tone has no outline)", int((res2 > 0).sum()) == int((occ > 0).sum()))
big = np.zeros((h, w), bool); big[100:190, 20:180] = True; pic3 = pic.copy(); pic3[big & ~(occ > 0)] = (60, 120, 175)
res3 = E._attach_novel(occ, big | (occ > 0), ys, h, w, pic3)
check("more than the cap -> nothing added", int((res3 > 0).sum()) == int((occ > 0).sum()))
E._P['occ_attach'] = 0.0; check("occ_attach=0 -> unchanged", np.array_equal(E._attach_novel(occ, nov, ys, h, w, pic), occ)); E._P['occ_attach'] = 0.1
check("above the eye line nothing is attached", int((E._attach_novel(occ, np.where(ys[:, None] < 0.3, True, False) | (occ > 0), ys, h, w, pic) > 0).sum()) == int((occ > 0).sum()))
print("ALL PASS" if ok else "SOME FAILED")
