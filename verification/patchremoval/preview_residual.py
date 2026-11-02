"""usage: preview_residual.py PACKDIR CLIPDIR name...   The preview (phoenix_patch_preview's fill, LaMa) on the painted frame:
how much darker than the clean skin the result still is inside the mark (eroded 2 px), and how many pixels changed
outside the painted area (+1 px)."""
import sys, os, types, numpy as np, cv2
P, C = sys.argv[1], sys.argv[2]; sys.path.insert(0, P)
g = types.ModuleType("gradio"); g.Request = type("R", (), {}); g.update = lambda **k: k; g.Error = Exception; sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: None; sys.modules["huggingface_hub"] = hf
os.environ.setdefault("PHOENIX_MODEL_DIR", "/tmp/claude-0/occmodels")
import core_pipeline as cp
L = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
for name in sys.argv[3:]:
    gt = np.load(f"{C}/pr_{name}_gt.npz"); i = int(round(float(gt["ref_t"]) * 30))
    c = cv2.VideoCapture(f"{C}/pr_{name}_clean.mp4"); [c.read() for _ in range(i)]; _, cl = c.read()
    ref, rm = gt["ref"], gt["refmask"]
    out, _ = cp._E.remove_patch_image(ref, rm)
    inner = cv2.erode(gt["mark"][i].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    pa = cv2.dilate((rm > 127).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ch = int((np.abs(out.astype(np.int16) - ref.astype(np.int16)).max(axis=2)[~pa] > 3).sum())
    print(f"{name}: inside the mark {float((L(cl) - L(out))[inner].mean()):+.2f} L darker than clean skin; changed outside the painted area {ch} px")
