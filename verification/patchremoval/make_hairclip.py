"""Synthetic body-hair clip for the BodyLock test: dark strands (1.5 px wide, 18-34 px long) on procedural skin, moving with the
skin (make_patchclip.render_hard, the same motion as the 'hard' and 'fast' clips). Ground truth: the strands.
usage: make_hair.py OUTDIR [fast=1.0]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import make_patchclip as M
OUT = sys.argv[1]; FAST = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
M.OUT = OUT; os.makedirs(OUT, exist_ok=True)

def hair_layer(seed=21):
    r = np.random.default_rng(seed)
    S = 120; m = np.zeros((S, S), np.uint8)
    for _ in range(64):
        while True:
            x, y = r.uniform(10, 110), r.uniform(10, 110)
            if ((x - 60) / 48.0) ** 2 + ((y - 60) / 36.0) ** 2 < 1.0:
                break
        ang = np.deg2rad(r.normal(62, 22)) ; L = r.uniform(18, 34)
        x2, y2 = x + L * np.cos(ang), y + L * np.sin(ang)
        cv2.line(m, (int(x), int(y)), (int(x2), int(y2)), 255, 2, cv2.LINE_AA)
    a = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 0.6) * 0.9
    col = np.array([45, 55, 70], np.float32)                          # dark brown hair (BGR)
    return np.dstack([np.full((S, S, 3), col), a]), (S // 2, S // 2)

ps = M.proc_skin()
layer, lc = hair_layer()
frames, clean, marks, occs = M.render_hard(ps, (900, 600), layer, lc, fast=FAST)
M.write("hair", frames, crf=24); M.write("hair_clean", clean, crf=24)
ri = 20
np.savez_compressed(os.path.join(OUT, "pr_hair_gt.npz"), mark=np.array(marks), occ=np.array(occs), ref=frames[ri],
                    refmask=M.finger_mask(marks[ri]), ref_t=ri / M.FPS)
print("hair clip written:", int(np.array(marks[ri]).sum()), "strand px in the reference frame", flush=True)
