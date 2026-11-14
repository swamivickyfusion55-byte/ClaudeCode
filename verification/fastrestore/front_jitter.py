"""v11.2.209 (SDOS-209): is the front / behind decision steady when the detector's boxes jitter? Two faces of the SAME size
overlapping for 300 frames (a long kiss), nothing really changing; each box's area jitters by a random few per cent from frame
to frame, as a real detector's does. The real core_pipeline._front_relations, 400 trials per case. A change of the face in
front here is a change on noise alone. Stand-in records, no models, no user media. usage: front_jitter.py PACKDIR"""
import sys, types
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
sys.path.insert(0, sys.argv[1])
import numpy as np, core_pipeline as cp
rng = np.random.default_rng(5)
def box(cx, cy, area):
    w = float(np.sqrt(area)); return [cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2]
def trial(sigma, trend, frames=300):
    cp.FRONT_TREND = trend
    cp._FRONT_MEM.clear(); st0 = dict(cp._FRONT_STATS)
    for g in range(frames):
        a0, a1 = 62500.0 * (1 + rng.normal(0, sigma)), 62500.0 * (1 + rng.normal(0, sigma))
        recs = {0: {"bbox": box(400, 400, a0), "det": True, "seen_near": True, "xid_near": None},
                1: {"bbox": box(560, 400, a1), "det": True, "seen_near": True, "xid_near": None}}
        cp._front_relations(recs, g)
    return cp._FRONT_STATS["changes"] - st0["changes"]
has_trend = hasattr(cp, "FRONT_TREND")
print("box-area jitter | trials where the face in front changed on noise alone (of 400), mean changes per trial")
for sigma in (0.01, 0.02, 0.03, 0.05):
    row = []
    for trend in ((False, True) if has_trend else (False,)):
        ch = [trial(sigma, trend) for _ in range(400)]
        row.append("%s: %3d trials (%.2f changes each)" % ("with the 11.2.207 rule" if trend else "11.2.176-206 rule    ", sum(c > 0 for c in ch), float(np.mean(ch))))
    print("  %2.0f %%          | %s" % (100 * sigma, " | ".join(row)), flush=True)
