"""App 1.2.61: the Space's own phoenix_detect_video_frame code (phoenix_api_adapter.api_detect_video_frame), given the VIDEO at
n % (as up to 1.2.60) and given only THAT FRAME as a JPEG at 0 % (as 1.2.61 sends it). Same faces, same crops?
Synthetic clip (insightface's public sample photo; NOT user media). usage: detect_frame_check.py PACK CLIP.mp4 PCT"""
import sys, os, types
PACK, CLIP, PCT = sys.argv[1], sys.argv[2], int(sys.argv[3])
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: "/tmp/claude-0/swapmodel/inswapper_128.onnx"; sys.modules["huggingface_hub"] = hf   # as bench_pipeline.py
sys.path.insert(0, PACK)
import numpy as np, cv2
import phoenix_api_adapter as pa, core_pipeline as cp
cap = cv2.VideoCapture(CLIP); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idx = int(round(PCT / 100.0 * (total - 1))); cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, frm = cap.read()
jpg = os.path.join(os.path.dirname(os.path.abspath(CLIP)), "frame_check.jpg"); cv2.imwrite(jpg, frm, [cv2.IMWRITE_JPEG_QUALITY, 95])
a = pa.api_detect_video_frame(CLIP, PCT)
b = pa.api_detect_video_frame(jpg, 0)
print("video at %d %%:" % PCT, a[1]); print("the frame alone :", b[1])
def crop(p):
    return None if p is None else cv2.imread(p if isinstance(p, str) else getattr(p, "name", str(p)))
na = sum(x is not None for x in a[2:]); nb = sum(x is not None for x in b[2:])
print("faces: %d vs %d" % (na, nb))
for k in range(2, 6):
    ca, cb = crop(a[k]), crop(b[k])
    if ca is None or cb is None:
        continue
    cb = cv2.resize(cb, (ca.shape[1], ca.shape[0]))
    print("  face %d crop: %dx%d, mean difference %.1f grey levels (the JPEG step)" % (k - 1, ca.shape[1], ca.shape[0], float(np.abs(ca.astype(int) - cb.astype(int)).mean())))
print("PASS" if na == nb and na > 0 else "FAIL")

# the skin-tone Preview (phoenix_skin_preview) the same way: the video at n % vs only that frame at 0 %
import json
sa = pa.api_skin_preview(CLIP, json.dumps({"position_pct": PCT, "skin_tone": [0.5, 0, 0, 0]}))
sb = pa.api_skin_preview(jpg, json.dumps({"position_pct": 0, "skin_tone": [0.5, 0, 0, 0]}))
ia, ib = json.loads(sa[2]), json.loads(sb[2])
print("skin preview, video at %d %%: ok=%s people=%s" % (PCT, ia.get("ok"), [(p.get("slot"), p.get("amount")) for p in ia.get("people", [])]))
print("skin preview, frame alone : ok=%s people=%s" % (ib.get("ok"), [(p.get("slot"), p.get("amount")) for p in ib.get("people", [])]))
A, B = crop(sa[0]), crop(sb[0])
if A is not None and B is not None:
    print("  toned frames differ by %.1f grey levels on average (the JPEG step)" % float(np.abs(A.astype(int) - cv2.resize(B, (A.shape[1], A.shape[0])).astype(int)).mean()))
print("PASS skin" if ia.get("ok") and ib.get("ok") and len(ia.get("people", [])) == len(ib.get("people", [])) else "FAIL skin")
