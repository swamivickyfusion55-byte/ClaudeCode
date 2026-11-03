"""v11.2.104 unit checks (Patch Removal: why a mark was not found, and a time budget for searching).
usage: unit104.py PACKDIR CLIPDIR      Synthetic clips only; a stand-in inpainter (no model)."""
import sys, os, types, time, numpy as np, cv2
PACK, CLIPS = sys.argv[1], sys.argv[2]
sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: None; sys.modules["huggingface_hub"] = hf
os.environ.setdefault("PHOENIX_MODEL_DIR", "/tmp/claude-0/occmodels")
import core_pipeline as cp
E = cp._E
fails = 0


def check(name, ok, info=""):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))


E.PATCH_INPAINTER = lambda crop, mk: cv2.inpaint(crop, (mk > 0).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)


def frames(p, n):
    c = cv2.VideoCapture(p); o = []
    while len(o) < n:
        ok, f = c.read()
        if not ok:
            break
        o.append(f)
    return o


fr = frames(os.path.join(CLIPS, "pr_proc_mole.mp4"), 70)
ga = np.load(os.path.join(CLIPS, "pr_proc_mole_gt.npz"))
gb = np.load(os.path.join(CLIPS, "pr_hard_astro_gt.npz"))
grays = [E._pr_gray(f) for f in fr]; th = [E._pr_thumb(x) for x in grays]
wh = (fr[0].shape[1], fr[0].shape[0])


def plan(ref, mask, t, log):
    return E.plan_tracks([(ref, mask, t, 45, "m")], wh, len(fr), th, lambda i: grays[i], lambda i: fr[i], log=log)


# 1. the normal case still finds the painted frame
msgs = []
trk = plan(ga["ref"], ga["refmask"], int(round(float(ga["ref_t"]) * 30)), msgs.append)
check("normal case: the painted frame is found and the mark followed", len(trk) == 1 and len(trk[0].tracks) == 1, "; ".join(msgs)[:120])

# 2. a frame from ANOTHER video: skipped, with numbers, inside the time budget
E.PATCH_CFG["find_budget_s"], E.PATCH_CFG["find_total_s"] = 5.0, 8.0
msgs = []
t0 = time.time()
trk = plan(gb["ref"], gb["refmask"], 20, msgs.append)
el = time.time() - t0
check("another video's frame: skipped (no track)", len(trk) == 0)
check("...within the search budget (under 5 s of searching plus the one-off set-up)", el < 20.0, f"{el:.1f} s")
txt = " ".join(msgs)
check("...and the log line says how close the best frame came, the picture sizes and the area search", "skipped" in txt and "best whole-frame match" in txt and "area search" in txt and "px" in txt, txt[:300])

# 3. a frame of the wrong shape: no size match, and it says so
sq = cv2.resize(ga["ref"], (640, 640)); sqm = cv2.resize(ga["refmask"], (640, 640), interpolation=cv2.INTER_NEAREST)
msgs = []
trk = plan(sq, sqm, 20, msgs.append)
check("a frame of another shape (square): no size match is reported", len(trk) == 0 and "0 size match(es)" in " ".join(msgs), " ".join(msgs)[:200])

# 4. the total budget is shared: with it used up the search is not even started
E.PATCH_CFG["find_budget_s"], E.PATCH_CFG["find_total_s"] = 90.0, 0.0
msgs = []
t0 = time.time()
trk = plan(gb["ref"], gb["refmask"], 20, msgs.append)
check("total search time used up: not searched, and the log says so", len(trk) == 0 and "used up" in " ".join(msgs), f"{time.time() - t0:.1f} s")

# 5. the "one mouth while an object is at the lips" rule runs on frames that carry a real detection (it ran on none before)
class _T:
    fake = np.zeros((128, 128, 3), np.uint8); fake_corr = None; fake_size = 128
_seen = []
class _C:
    def reuse(self, work, orig, kps, face, **kw):
        _seen.append(bool(face.predicted)); return work, True
_oc = cp._compositor
cp._compositor = lambda: _C()
fr0 = np.zeros((64, 64, 3), np.uint8)
base_rec = dict(bbox=np.array([0, 0, 10, 10], np.float32), kps=np.zeros((5, 2), np.float32), lmk=None, track=_T(), alpha=1.0, guard=0.0)
cp._reuse_one(fr0, fr0, dict(base_rec, det=True), "Balanced")
cp._reuse_one(fr0, fr0, dict(base_rec, det=False), "Balanced")
cp._compositor = _oc
check("render path: a face on a real detection (or between two) is not 'predicted'; a held one is", _seen == [False, True], str(_seen))

print("\n%d check(s) failed" % fails if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
