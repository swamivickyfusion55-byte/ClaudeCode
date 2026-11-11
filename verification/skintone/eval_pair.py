"""SDOS-205: score the skin tone on the layered two-person scenes (make_pair.py). For each scene and each chosen person (+0.6):
how much of the full change reaches the chosen person's skin (want 1), the other person's skin (want 0), the chosen person's
clothes and hair (want ~0), the backdrop (want 0); how steady it is; the time. usage: eval_pair.py PACK PAIRDIR key=value ..."""
import sys, os, time, glob
sys.path.insert(0, sys.argv[1])
import numpy as np, cv2, skin_tone as ST
for kv in sys.argv[3:]:
    k, v = kv.split("="); setattr(ST, k, type(getattr(ST, k))(v) if not isinstance(getattr(ST, k), str) else v)
def lum(x): return cv2.cvtColor(x, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[2], "pair_*.npz"))):
    d = np.load(f); fr, own, sk, kps, bb = d["frames"], d["owner"], d["skin"], d["kps"], d["bbox"]
    for chosen in (0, 1):
        st = ST.SkinTone({chosen: 0.6}); acc = {k: [0.0, 0.0] for k in ("own", "other", "cloth", "back")}; per = []
        t = time.perf_counter()
        for i in range(len(fr)):
            recs = {s: dict(kps=kps[i, s], bbox=bb[i, s]) for s in (0, 1)}
            out = st.apply(fr[i], fr[i], recs)
            dl = np.abs(lum(out) - lum(fr[i])); full = np.abs(lum(ST.toned(fr[i], 0.6)) - lum(fr[i]))
            me, oth = own[i] == chosen + 1, (own[i] != chosen + 1) & (own[i] > 0)
            regs = {"own": me & (sk[i] > 0), "other": oth & (sk[i] > 0), "cloth": me & (sk[i] == 0), "back": own[i] == 0}
            for k, r in regs.items():
                acc[k][0] += float(dl[r].sum()); acc[k][1] += float(full[r].sum())
            r = regs["own"]; per.append(float(dl[r].sum()) / max(1.0, float(full[r].sum())))
        ms = (time.perf_counter() - t) * 1000 / len(fr)
        sc = {k: v[0] / max(1.0, v[1]) for k, v in acc.items()}
        jit = float(np.mean(np.abs(np.diff(per))))
        rows.append(sc)
        print("%-9s chosen %s: own skin %.2f | other's skin %.3f | own clothes/hair %.2f | backdrop %.3f | own-skin frame-to-frame %.3f | %.0f ms/frame" % (
            os.path.basename(f)[5:-4], "back " if chosen == 0 else "front", sc["own"], sc["other"], sc["cloth"], sc["back"], jit, ms))
print("MEAN own %.3f other %.4f cloth %.3f back %.4f | %s" % tuple([np.mean([r[k] for r in rows]) for k in ("own", "other", "cloth", "back")] + [" ".join(sys.argv[3:])]))
