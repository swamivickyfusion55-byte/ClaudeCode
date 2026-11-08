"""v11.2.178: the Patch Removal preview endpoint on a synthetic frame (a skin-like gradient with texture, a dark painted mark).
usage: test_preview.py PACKDIR OUTDIR"""
import sys, types, json, os
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
dark = frame.copy(); dark[mark > 0] = (40, 45, 60); frame = cv2.addWeighted(frame, 1, dark, 0, 0); frame[mark > 0] = dark[mark > 0]
paint = cv2.dilate(mark, np.ones((15, 15), np.uint8))                  # the finger paint, a little bigger than the mark
rgba = np.zeros((H, W, 4), np.uint8); rgba[..., 3] = paint; rgba[..., 2] = 255
cv2.imwrite(out + "/frame.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(out + "/mask.png", rgba)
p, js = A.api_patch_preview(out + "/frame.jpg", out + "/mask.png")
info = json.loads(js); print("info", info)
ok = p is not None and os.path.isfile(p) and info.get("ok")
res = cv2.imread(p) if ok else None
fails = 0
def check(name, cond, extra=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("  [" + extra + "]" if extra else "")); fails += 0 if cond else 1
check("the endpoint returns a cleaned frame file and JSON ok", bool(ok), str(p))
if ok:
    m = mark > 0; ring = (cv2.dilate(paint, np.ones((41, 41), np.uint8)) > 0) & ~(paint > 0)
    before = float(frame[m].mean()); after = float(res[m].mean()); around = float(frame[ring].mean())
    check("the dark mark is gone (its pixels now match the skin around it)", abs(after - around) < 12 and abs(before - around) > 60,
          "mark mean %.0f -> %.0f, skin around %.0f" % (before, after, around))
    outside = ~(cv2.dilate(paint, np.ones((61, 61), np.uint8)) > 0)
    d = float(np.abs(res.astype(np.int16) - frame.astype(np.int16))[outside].mean())
    check("the frame away from the mark is unchanged (JPEG only)", d < 2.0, "mean change %.2f levels" % d)
    cv2.imwrite(out + "/compare.jpg", np.hstack([frame[300:540, 540:860], res[300:540, 540:860]]))
p2, js2 = A.api_patch_preview(out + "/frame.jpg", None)
check("no mask: a clear error, no crash", p2 is None and not json.loads(js2).get("ok"), js2)
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
