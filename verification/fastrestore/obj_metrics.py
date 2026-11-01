"""usage: obj_metrics.py RUN:SPEC ...   (make_objects.py clips; synthetic, no user footage)
In the object's own area the output must equal the footage (the object stays solid): 'painted over' = share of the object
(eroded 3 px) changed > 12 grey levels; on the first frame it is fully in (frame 24), mean while it holds (24-57), and the
worst frame. The rest of the face must stay swapped: share of the face outside the object (dilated 15 px) still changed,
against the frames before the object came, and how many frames it dropped below 60 % of that (a revert)."""
import sys, glob, cv2, numpy as np
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
def frames(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, f = c.read()
        if not ok: return o
        o.append(f)
for arg in sys.argv[1:]:
    run, spec = arg.split(":")
    inp = frames(B + f"obj_{spec}.mp4"); out = frames(glob.glob(f"/tmp/claude-0/bench_work/{run}/out/*/final.mp4")[0])
    A = np.load(B + f"obj_{spec}_gt.npz")["a"]; n = min(len(inp), len(out))
    face = np.zeros((720, 1280), np.uint8); cv2.ellipse(face, (640, 400), (170, 210), 0, 0, 360, 1, -1); face = face.astype(bool)
    po, keep = {}, {}
    for t in range(n):
        d = cv2.GaussianBlur(np.abs(out[t].astype(np.float32) - inp[t].astype(np.float32)).mean(2), (0, 0), 1.0)
        a = A[t] > 127
        if a.sum() > 500:
            core = cv2.erode(a.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            po[t] = float((d[core] > 12).mean() * 100)
        rest = face & ~cv2.dilate(a.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        keep[t] = float((d[rest] > 12).mean() * 100)
    base = np.median([keep[t] for t in range(0, 20)])
    hold = [po[t] for t in range(24, 58) if t in po]
    drops = [t for t in range(20, 62) if keep[t] < 0.6 * base]
    print(f"{run} {spec}: object painted over - entering (frames 20-23) {np.mean([po[t] for t in range(20, 24) if t in po]):.1f}%, "
          f"first full frame {po.get(24, float('nan')):.1f}%, holding mean {np.mean(hold):.1f}% worst {np.max(hold):.1f}%; "
          f"rest of the face swapped {base:.0f}% before, {np.mean([keep[t] for t in range(24, 58)]):.0f}% during; reverted frames {len(drops)}")
