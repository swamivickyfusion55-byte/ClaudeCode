"""v11.2.213 (SDOS-213): how much of the visible face each output changed, frame by frame, against the true turn of a make_turn.py
clip (its _gt.npz: the visible face mask 'a' at half resolution and the turn 'yaw'). Changed = differs from the input by more than
10 grey levels. Reported per band of the true turn, for each output given (e.g. 11.2.212 and 11.2.213).
usage: turn_change.py CLIP.mp4 LABEL=OUT.mp4 [LABEL=OUT.mp4 ...]   (synthetic test clips only)"""
import sys, numpy as np, cv2
CLIP = sys.argv[1]
OUTS = [a.split("=", 1) for a in sys.argv[2:]]
g = np.load(CLIP[:-4] + "_gt.npz"); Y = np.abs(g["yaw"]); A = g["a"]


def frames(p):
    c = cv2.VideoCapture(p); out = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return out


IN = frames(CLIP)
res = {}
for lab, p in OUTS:
    O = frames(p); ch = []
    for t in range(min(len(IN), len(O), len(Y))):
        m = cv2.resize(A[t], (IN[t].shape[1], IN[t].shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        if m.sum() < 200:
            ch.append(np.nan); continue
        d = np.abs(O[t].astype(np.int16) - IN[t].astype(np.int16)) > 10
        ch.append(float(d[m].mean()))
    res[lab] = np.array(ch)
bands = [(0, 45), (45, 60), (60, 70), (70, 76), (76, 80), (80, 85), (85, 90), (90, 100), (100, 181)]
print("share of the visible face changed by the swap, by the head's true turn (mean over the frames in the band; frames)")
print("turn       | " + " | ".join("%-14s" % lab for lab, _ in OUTS))
for lo, hi in bands:
    sel = (Y[:len(IN)] >= lo) & (Y[:len(IN)] < hi)
    if not sel.any():
        continue
    cells = []
    for lab, _ in OUTS:
        v = res[lab][:len(sel)][sel]; v = v[np.isfinite(v)]
        cells.append("%5.2f (%3d)    " % (float(v.mean()) if len(v) else float("nan"), len(v)))
    print("%3d-%3d deg | " % (lo, hi) + " | ".join(cells))
# frame-to-frame jumps of the changed share (a swap switching on and off shows as large jumps)
for lab, _ in OUTS:
    v = res[lab]; d = np.abs(np.diff(v)); d = d[np.isfinite(d)]
    print("%s: largest frame-to-frame change of the share %.2f; changes over 0.15: %d" % (lab, float(d.max()) if len(d) else 0, int((d > 0.15).sum())))
