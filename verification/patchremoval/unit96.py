"""Patch Removal (v11.2.96) unit checks against the pack.  usage: unit96.py PACKDIR CLIPDIR
Synthetic media and public photos only (make_patchclip.py)."""
import sys, os, types, json, base64, tempfile, numpy as np, cv2
PACK, CLIPS = sys.argv[1], sys.argv[2]
sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: None; sys.modules["huggingface_hub"] = hf
os.environ.setdefault("PHOENIX_MODEL_DIR", "/tmp/claude-0/occmodels")
import core_pipeline as cp
import phoenix_api_adapter as ad
E = cp._E
fails = 0


def check(name, ok, info=""):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))


L = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
gt = np.load(os.path.join(CLIPS, "pr_proc_mole_gt.npz"))
ref, rmask, mark = gt["ref"], gt["refmask"], gt["mark"][20]

# --- the still fill (preview) ------------------------------------------------------------------------------------------
out, info = E.remove_patch_image(ref, rmask)
con = float((cv2.GaussianBlur(L(ref), (0, 0), 6)[mark] - L(ref)[mark]).mean())
rem = float((cv2.GaussianBlur(L(out), (0, 0), 6)[mark] - L(out)[mark]).mean())
check("preview: the mark is removed (its darkness against the skin around it)", rem < 0.1 * con, f"{con:.1f} -> {rem:.1f} L")
far = ~cv2.dilate((rmask > 127).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
check("preview: nothing changes away from the painted area", int((np.abs(out.astype(int) - ref.astype(int)).max(axis=2)[far] > 0).sum()) == 0)
check("preview: LaMa is used (model in PHOENIX_MODEL_DIR, checksum ok)", cp._LAMA.get("sess") is not None, str(cp._LAMA.get("error")))
small = cv2.resize(rmask, (rmask.shape[1] // 3, rmask.shape[0] // 3), interpolation=cv2.INTER_NEAREST)
out2, _ = E.remove_patch_image(ref, small)
check("preview: a mask of another size is fitted to the frame", float((L(out2) - L(ref))[mark].mean()) > 0.6 * (L(ref)[mark].mean() * 0 + con * 0.6))
E_inp = E.PATCH_INPAINTER
E.PATCH_INPAINTER = None
out3, _ = E.remove_patch_image(ref, rmask)
E.PATCH_INPAINTER = E_inp
rem3 = float((cv2.GaussianBlur(L(out3), (0, 0), 6)[mark] - L(out3)[mark]).mean())
check("fallback without LaMa (exemplar fill) also removes the mark", rem3 < 0.15 * con, f"{rem3:.1f} L")
E.PATCH_INPAINTER = lambda c, m: (_ for _ in ()).throw(RuntimeError("model crashed"))
out4, _ = E.remove_patch_image(ref, rmask)
E.PATCH_INPAINTER = E_inp
check("an inpainter that crashes falls back instead of failing", out4.shape == ref.shape)

# --- the adapter ---------------------------------------------------------------------------------------------------------
b = lambda im, ext: base64.b64encode(cv2.imencode(ext, im)[1].tobytes()).decode()
good = {"patches": [{"image_b64": b(ref, ".jpg"), "mask_b64": b(rmask, ".png"), "t": 0.667}]}
pl, err = ad._load_patches(good)
check("settings.patches: frame and mask decoded and written for the job", err is None and len(pl) == 1 and os.path.isfile(pl[0]["image"]) and os.path.isfile(pl[0]["mask"]), str(err))
mm = cv2.imread(pl[0]["mask"], cv2.IMREAD_GRAYSCALE)
check("settings.patches: the mask is kept exactly (0/255)", mm is not None and set(np.unique(mm)) <= {0, 255} and int((mm > 127).sum()) == int((rmask > 127).sum()))
rgba = np.zeros(rmask.shape + (4,), np.uint8); rgba[..., 3] = rmask; rgba[..., :3] = 255
pl2, err2 = ad._load_patches({"patches": [{"image_b64": b(ref, ".jpg"), "mask_b64": b(rgba, ".png"), "t": 1}]})
mm2 = cv2.imread(pl2[0]["mask"], cv2.IMREAD_GRAYSCALE) if pl2 else None
check("settings.patches: an RGBA mask (painted = alpha) is read from its alpha", mm2 is not None and int((mm2 > 127).sum()) == int((rmask > 127).sum()))
opaque = np.zeros(rmask.shape + (4,), np.uint8); opaque[..., 3] = 255; opaque[rmask > 127, :3] = 255   # the app's PNG
pl3, _ = ad._load_patches({"patches": [{"image_b64": b(ref, ".jpg"), "mask_b64": b(opaque, ".png"), "t": 2.5}]})
mm3 = cv2.imread(pl3[0]["mask"], cv2.IMREAD_GRAYSCALE) if pl3 else None
check("settings.patches: the app's mask (opaque ARGB PNG, white strokes on black) is read as painted = white",
      mm3 is not None and int((mm3 > 127).sum()) == int((rmask > 127).sum()) and abs(pl3[0]["t"] - 2.5) < 1e-9)
_, e1 = ad._load_patches({"patches": [{"image_b64": b(ref, ".jpg"), "mask_b64": b(np.zeros_like(rmask), ".png")}]})
_, e2 = ad._load_patches({"patches": [{"image_b64": "not base64 !!", "mask_b64": b(rmask, ".png")}]})
_, e3 = ad._load_patches({"patches": [good["patches"][0]] * 5})
_, e4 = ad._load_patches({"patches": "{bad json"})
check("settings.patches: empty mask / unreadable image / too many / bad JSON are refused with a reason",
      all(e is not None and not e.get("ok", True) for e in (e1, e2, e3, e4)),
      " | ".join(str((e or {}).get("error", {}).get("code") if isinstance((e or {}).get("error"), dict) else (e or {}).get("error")) for e in (e1, e2, e3, e4)))
check("settings without patches: unchanged (no patches)", ad._load_patches({}) == ([], None))

# --- the preview endpoint ------------------------------------------------------------------------------------------------
td = tempfile.mkdtemp()
fp, mp = os.path.join(td, "f.jpg"), os.path.join(td, "m.png")
cv2.imwrite(fp, ref, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(mp, rmask)
path, js = ad.api_patch_preview(fp, mp)
j = json.loads(js)
pv = cv2.imread(path) if path else None
check("phoenix_patch_preview: returns the erased frame and ok JSON", pv is not None and j.get("ok") and j.get("marks") == 1 and j.get("inpainter") == "LaMa", js)
if pv is not None:
    rp = float((cv2.GaussianBlur(L(pv), (0, 0), 6)[mark] - L(pv)[mark]).mean())
    check("phoenix_patch_preview: the mark is gone in the returned frame", rp < 0.15 * con, f"{rp:.1f} L")
path2, js2 = ad.api_patch_preview(fp, os.path.join(td, "missing.png"))
check("phoenix_patch_preview: a missing mask gives ok=false, no crash", path2 is None and not json.loads(js2).get("ok"), js2)

# --- the job config ------------------------------------------------------------------------------------------------------
import inspect
check("submit_video takes patches (and keeps the old call shape)", "patches" in inspect.signature(cp.submit_video).parameters
      and inspect.signature(cp.submit_video).parameters["patches"].default is None)
sp = cp._patch_specs({"patches": pl})
check("job: patch files -> frame, mask and time", len(sp) == 1 and sp[0][0].shape == ref.shape and abs(sp[0][2] - 0.667) < 1e-6)
check("job: a missing patch file is skipped, not fatal", cp._patch_specs({"patches": [{"image": "/nope.jpg", "mask": "/nope.png", "t": 0}]}) == [])
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
