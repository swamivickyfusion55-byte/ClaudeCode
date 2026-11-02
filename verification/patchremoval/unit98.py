"""v11.2.98 unit checks (no faint darkness under the mark; nothing outside the painted area changes).
usage: unit98.py PACKDIR CLIPDIR      Synthetic data only."""
import sys, os, types, numpy as np, cv2
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


# a painted area (disk r=24) on smooth skin; the mark's core (disk r=9) at its centre
H = W = 128
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
skin = np.dstack([150 + 0.2 * xx, 170 + 0.1 * yy, 210 + 0 * xx]).astype(np.float32)
hole = ((yy - 64) ** 2 + (xx - 64) ** 2) <= 24 ** 2
core = ((yy - 64) ** 2 + (xx - 64) ** 2) <= 9 ** 2
ghost = skin.copy(); ghost[core] -= 4.0                                  # a faint dark spot left by the fill
around = skin.copy(); around[core] -= 60.0                               # the frame: the mark itself under the core
out = E._pr_lift(around, ghost, hole, core)
lv = lambda im, m: float(im.mean(axis=2)[m].mean())
inner = cv2.erode(core.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
check("a faint dark spot under the mark is lifted to the level around it", abs(lv(out, inner) - lv(skin, inner)) < 1.2,
      f"{lv(ghost, inner) - lv(skin, inner):+.1f} -> {lv(out, inner) - lv(skin, inner):+.1f} L")
check("nothing outside the painted area changes", np.abs(out - ghost)[~hole].max() == 0.0)
flat = E._pr_lift(around, skin.copy(), hole, core)
check("no spot, no lift (nothing brightened)", float(np.abs(flat - skin)[hole].max()) < 0.5, f"max {float(np.abs(flat - skin)[hole].max()):.2f}")
deep = skin.copy(); deep[core] -= 30.0
dl = E._pr_lift(around, deep, hole, core)
check("the lift is capped (a real dark structure inside is not flattened)",
      float((dl - deep).mean(axis=2)[inner].max()) <= float(E.PATCH_CFG["lift_max"]) + 1e-3, f"max lift {float((dl - deep).mean(axis=2)[inner].max()):.1f}")
lightspot = skin.copy(); lightspot[core] += 4.0
ll = E._pr_lift(around, lightspot, hole, core)
check("a lighter spot is not darkened (the lift only brightens)", float(np.abs(ll - lightspot).max()) < 0.5)
big = np.zeros((400, 400, 3), np.float32) + 180; bh = np.zeros((400, 400), bool); cv2.circle(bh.view(np.uint8), (200, 200), 90, 1, -1)
bc = np.zeros((400, 400), bool); cv2.circle(bc.view(np.uint8), (200, 200), 40, 1, -1)
bg = big.copy(); bg[bc] -= 4.0
bo = E._pr_lift(big.copy(), bg, bh.view(bool), bc.view(bool))
ib = cv2.erode(bc.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
check("a big mark (solved at reduced size) is lifted too, and nothing outside changes",
      abs(lv(bo, ib) - 180.0) < 1.2 and np.abs(bo - bg)[~bh.view(bool)].max() == 0.0, f"{lv(bg, ib) - 180:+.1f} -> {lv(bo, ib) - 180:+.1f}")

# --- the fill's alpha: soft edge inside, nothing outside --------------------------------------------------------------
a = E._pr_alpha(hole)
check("fill alpha: 0 everywhere outside the painted area", float(a[~hole].max()) == 0.0)
check("fill alpha: 1 inside (away from the edge)", float(a[cv2.erode(hole.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)].min()) > 0.99)

# --- the switches: no edge repair, no rim extension outside the painted area -------------------------------------------
check("edge repair outside the painted area is off", float(E.PATCH_CFG["peek_band"]) == 0.0)
check("extension of the painted area over a missed rim is off", float(E.PATCH_CFG["grow_band"]) == 0.0)

# --- the preview on a real (synthetic) clip: at most the halo changes outside the painted area -------------------------
gt = np.load(os.path.join(CLIPS, "pr_proc_mole_gt.npz"))
ref, rm = gt["ref"], gt["refmask"]
pv, _ = E.remove_patch_image(ref, rm)
pa = cv2.dilate((rm > 127).astype(np.uint8), E._pr_disk(int(E.PATCH_CFG["halo_px"]))).astype(bool)
chg = int((np.abs(pv.astype(np.int16) - ref.astype(np.int16)).max(axis=2)[~pa] > 0).sum())
check("preview: nothing changes outside the painted area (+ the 2 px halo)", chg == 0, f"{chg} px")
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
