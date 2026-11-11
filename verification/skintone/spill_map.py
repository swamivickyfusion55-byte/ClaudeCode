"""SDOS-205: where the change lands on one frame of a layered scene. green = the chosen person's skin reached, red = the other
person's skin changed, blue = the chosen person's skin missed. usage: spill_map.py PACK PAIR.npz frame chosen OUT.jpg [still]"""
import sys, os
sys.path.insert(0, sys.argv[1])
import numpy as np, cv2, skin_tone as ST
d = np.load(sys.argv[2]); i, ch = int(sys.argv[3]), int(sys.argv[4])
fr, own, sk, kps, bb = d["frames"], d["owner"], d["skin"], d["kps"], d["bbox"]
temporal = len(sys.argv) <= 6 or sys.argv[6] != "still"
st = ST.SkinTone({ch: 0.6}, temporal=temporal); st.debug = {}
for j in (range(max(0, i - 6), i + 1) if temporal else [i]):
    out = st.apply(fr[j], fr[j], {s: dict(kps=kps[j, s], bbox=bb[j, s]) for s in (0, 1)})
L = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
ratio = np.abs(L(out) - L(fr[i])) / np.maximum(3.0, np.abs(L(ST.toned(fr[i], 0.6)) - L(fr[i])))
ratio = np.clip(ratio, 0, 1)
g = cv2.cvtColor(cv2.cvtColor(fr[i], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.5
me, oth, skin = own[i] == ch + 1, (own[i] > 0) & (own[i] != ch + 1), sk[i] > 0
vis = g.copy()
vis[me & skin] += np.array([0, 160, 0]) * ratio[me & skin][:, None]
vis[me & skin] += np.array([160, 0, 0]) * (1 - ratio[me & skin])[:, None]
vis[oth & skin] += np.array([0, 0, 255]) * ratio[oth & skin][:, None]
vis[~skin] += np.array([0, 120, 200]) * ratio[~skin][:, None]       # orange: not skin, changed
cv2.imwrite(sys.argv[5], cv2.resize(np.clip(vis, 0, 255).astype(np.uint8), None, fx=0.6, fy=0.6))
print("I keys", list(st.debug.get("I", {}).keys()), "G used", bool(st.debug.get("G")))
