"""SDOS-205: one picture, one person chosen at a time. insightface's public sample photo (t1.jpg), NOT user media.
usage: try_photo.py PACKDIR OUTDIR [amount]"""
import sys, os, time, json
PACK, OUT = sys.argv[1], sys.argv[2]; AMT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
sys.path.insert(0, PACK)
import numpy as np, cv2, skin_tone as ST
from insightface.app import FaceAnalysis
os.makedirs(OUT, exist_ok=True)
img = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
fa = FaceAnalysis(name="buffalo_l", root=os.environ.get("IFROOT", "/root/.insightface"), allowed_modules=["detection"],
                  providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
faces = sorted(fa.get(img), key=lambda f: f.bbox[0])
people = {i: dict(kps=f.kps, bbox=f.bbox) for i, f in enumerate(faces)}
print("faces", len(faces))
rows = []
for chosen in range(len(faces)):
    t = time.perf_counter()
    out, masks, info = ST.preview(img, people, {chosen: AMT})
    dt = (time.perf_counter() - t) * 1000
    d = np.abs(out.astype(np.int16) - img.astype(np.int16)).max(2)
    def box_change(f, shrink=0.25):
        x0, y0, x1, y1 = f.bbox; w, h = x1 - x0, y1 - y0
        x0, x1, y0, y1 = int(x0 + shrink * w), int(x1 - shrink * w), int(y0 + shrink * h), int(y1 - shrink * h)
        return float(d[y0:y1, x0:x1].mean())
    own = box_change(faces[chosen]); others = [round(box_change(f), 2) for i, f in enumerate(faces) if i != chosen]
    rows.append((chosen, round(own, 1), max(others), round(dt)))
    print("chosen #%d: own face change %.1f levels, other faces max %.2f %s, %.0f ms, %s" % (chosen + 1, own, max(others), others, dt, info["backend"]))
    vis = img.copy(); m = masks.get(chosen)
    if m is not None:
        g3 = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.6
        tint = np.zeros_like(g3); tint[:] = (40, 40, 255)
        vis = np.clip(g3 * (1 - m[:, :, None]) + (0.5 * g3 + 0.6 * tint) * m[:, :, None], 0, 255).astype(np.uint8)
    for i, f in enumerate(faces):
        x0, y0, x1, y1 = f.bbox.astype(int); cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0) if i == chosen else (255, 255, 255), 2)
        cv2.putText(vis, str(i + 1), (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(OUT, "mask_%d.jpg" % (chosen + 1)), cv2.resize(vis, None, fx=0.5, fy=0.5))
    cv2.imwrite(os.path.join(OUT, "after_%d.jpg" % (chosen + 1)), cv2.resize(np.hstack([img, out]), None, fx=0.4, fy=0.4))
json.dump(rows, open(os.path.join(OUT, "rows.json"), "w"))
