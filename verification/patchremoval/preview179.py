"""v11.2.179: the Patch Removal preview and the job's patch loading with masks in the APP's format: PatchFrames.maskBitmap is an
ARGB_8888 bitmap, black background, white paint, saved as PNG - opaque (alpha 255) everywhere. Synthetic frame, no user media.
usage: preview179.py PACKDIR OUTDIR"""
import sys, types, json, os, base64
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception; sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
sys.path.insert(0, sys.argv[1]); out = sys.argv[2]
import numpy as np, cv2, phoenix_api_adapter as A
rng = np.random.default_rng(4)
H, W = 720, 1280
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
skin = np.stack([120 + 30 * xx / W, 150 + 20 * yy / H, 200 + 15 * np.sin(xx / 90)], 2)
skin += rng.normal(0, 4, skin.shape); skin = cv2.GaussianBlur(skin, (0, 0), 1.2)
frame = np.clip(skin, 0, 255).astype(np.uint8)
mark = np.zeros((H, W), np.uint8); cv2.ellipse(mark, (700, 420), (60, 35), 20, 0, 360, 255, -1)
frame[mark > 0] = (40, 45, 60)
paint = cv2.dilate(mark, np.ones((15, 15), np.uint8))
app_mask = np.zeros((H, W, 4), np.uint8); app_mask[..., 3] = 255                    # opaque black ...
app_mask[paint > 0, :3] = 255                                                       # ... white paint
cv2.imwrite(out + "/frame.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(out + "/app_mask.png", app_mask)
assert cv2.imread(out + "/app_mask.png", cv2.IMREAD_UNCHANGED).shape[2] == 4
fails = 0
def check(name, cond, extra=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if cond else 1
# 1) the mask as the Space reads it
m = A._mask_from_image(cv2.imread(out + "/app_mask.png", cv2.IMREAD_UNCHANGED)) > 127
iou = float((m & (paint > 0)).sum()) / float((m | (paint > 0)).sum())
check("the app's opaque mask is read as the painted strokes only", iou > 0.99, "IoU %.3f with the paint, %.1f %% of the frame" % (iou, 100 * m.mean()))
# 2) the preview
p, js = A.api_patch_preview(out + "/frame.jpg", out + "/app_mask.png"); info = json.loads(js)
res = cv2.imread(p) if p else None
check("the preview returns a cleaned frame", res is not None and info.get("ok"), js[:120])
if res is not None:
    mk = mark > 0; ring = (cv2.dilate(paint, np.ones((41, 41), np.uint8)) > 0) & ~(paint > 0)
    check("the mark is removed", abs(float(res[mk].mean()) - float(frame[ring].mean())) < 12,
          "mark mean %.0f -> %.0f, skin around %.0f" % (float(frame[mk].mean()), float(res[mk].mean()), float(frame[ring].mean())))
    outside = ~(cv2.dilate(paint, np.ones((61, 61), np.uint8)) > 0)
    d = float(np.abs(res.astype(np.int16) - frame.astype(np.int16))[outside].mean())
    check("the rest of the frame is not changed (no darkening)", d < 2.0, "mean change outside %.2f levels" % d)
    cv2.imwrite(out + "/compare179.jpg", cv2.resize(np.hstack([frame, res]), (1280, 360)))
# 3) the job's patch loading (what the app sends with a job: image_b64 + mask_b64)
b64 = lambda path: base64.b64encode(open(path, "rb").read()).decode()
items, err = A._load_patches({"patches": [{"image_b64": b64(out + "/frame.jpg"), "mask_b64": b64(out + "/app_mask.png"), "t": 0.7}]})
ok3 = err is None and items and os.path.isfile(items[0]["mask"])
jm = cv2.imread(items[0]["mask"], cv2.IMREAD_GRAYSCALE) > 127 if ok3 else None
iou3 = float((jm & (paint > 0)).sum()) / float((jm | (paint > 0)).sum()) if ok3 else 0.0
check("a job's painted mark (mask_b64 from the app) is saved as the strokes only", ok3 and iou3 > 0.99,
      "IoU %.3f, %.1f %% of the frame" % (iou3, 100 * jm.mean()) if ok3 else str(err))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
