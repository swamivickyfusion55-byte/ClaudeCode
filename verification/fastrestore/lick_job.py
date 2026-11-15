"""v11.2.210 (SDOS-210): eating an ice cream, the whole job. From the job's OUTPUT video and the engine's own paint
(bench_pipeline.py MASK_DUMP=1) on a make_lick.py clip (synthetic, public photo, NOT user footage).
  lips    paint laid over the footage's own lips where the scoop does not cover them: 1 = the replacement's lips are shown,
          0 = the footage's lips. With the scoop ON the lips, and between licks (the scoop under the lips).
  change  the largest change of that paint from one frame to the next (the mouth changing between the two people)
  scoop   the scoop where it lies over the face (the face = the swap's area on frame 5), from the output video: the share of
          it that differs from the footage by more than 12 grey levels (painted over). Must be near 0.
usage: lick_job.py LABEL COLOUR      (run: bench_pipeline.py PACK LABEL clip=.../lick_COLOUR.mp4 one=1 face_mode="1 face")"""
import sys, glob, cv2, numpy as np
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
label, col = sys.argv[1], sys.argv[2]


def frames(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, f = c.read()
        if not ok:
            return o
        o.append(f)


inp = frames(B + f"lick_{col}.mp4"); out = frames(sorted(glob.glob(f"/tmp/claude-0/bench_work/{label}/out/*/final.mp4"))[-1])
g = np.load(B + f"lick_{col}_gt.npz"); A = g["a"]; LIPS = g["lips"]
z = np.load(f"/tmp/claude-0/bench_work/{label}/mask_dump.npz"); byg = {int(z["g"][k]): k for k in range(len(z["g"]))}
n = min(len(inp), len(out), len(A))
lipm = np.zeros((720, 1280), np.uint8); cv2.fillPoly(lipm, [np.round(LIPS).astype(np.int32)], 1); lipm = lipm > 0


def paint(t):
    k = byg.get(t)
    if k is None:
        return np.zeros((720, 1280), np.float32)
    size = int(z["size"][k]); m = cv2.resize(z["m"][k].astype(np.float32), (size, size))
    return cv2.warpAffine(m, cv2.invertAffineTransform(z["M"][k]), (1280, 720))


face = cv2.erode((paint(5) > 0.6).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0       # the swap's own area, nothing in front
rows = []
for t in range(n):
    P = paint(t); a = A[t].astype(np.float32) / 255.0
    free = lipm & (a < 0.05)
    obj = cv2.erode((a > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & face
    d = cv2.GaussianBlur(np.abs(out[t].astype(np.float32) - inp[t].astype(np.float32)).mean(2), (0, 0), 1.0)
    on_lips = float((lipm & (a > 0.5)).sum()) / float(lipm.sum())
    rows.append((t, float(P[free].mean()) if free.sum() > 50 else np.nan, float((d[obj] > 12).mean()) if obj.sum() > 200 else np.nan,
                 float(P[obj].mean()) if obj.sum() > 200 else np.nan, on_lips, int((a > 0.5).sum())))
R = np.array(rows, float)
at = R[:, 4] > 0.10; low = (R[:, 5] > 2000) & ~at
base = np.nanmean(R[:18, 1])
print("%s %s: lips shown as the replacement's - nothing there %.2f | scoop on the lips: mean %.2f, lowest %.2f | between licks: mean %.2f, lowest %.2f | "
      "largest change in one frame %.2f" % (label, col, base, np.nanmean(R[at, 1]), np.nanmin(R[at, 1]), np.nanmean(R[low, 1]), np.nanmin(R[low, 1]),
                                        np.nanmax(np.abs(np.diff(R[:, 1])))))
print("%s %s: the scoop over the face painted over (output differs from the footage) - on the lips: mean %.0f%%, worst frame %.0f%% | under the lips: "
      "mean %.0f%%, worst frame %.0f%% | the engine's paint on it: %.2f / %.2f" % (
          label, col, 100 * np.nanmean(R[at, 2]), 100 * np.nanmax(R[at, 2]), 100 * np.nanmean(R[low, 2]), 100 * np.nanmax(R[low, 2]),
          np.nanmean(R[at, 3]), np.nanmean(R[low, 3])))
