"""v11.2.140: the whole-belly test clip - make_bellyreal.py's body (a REAL pose, synthetic skin) with tattoos above, beside and
below the navel and a brown mole. usage: make_bellytattoo.py POSE_JPG OUTDIR [frames=90] [head=pixelate|keep] [skin=proc|flat] [navel_t=0.27]
Writes clip.mp4, clean.mp4, truth.npz (mask: all marks per frame; parts: id per pixel 1 upper tattoo, 2 beside the navel,
3 lower belly, 4 mole; navel per frame; lm per frame).

Based on:
usage: make_bellyreal.py POSE_JPG OUTDIR [frames=90] [head=pixelate|keep] [skin=proc|flat]
POSE_JPG = MediaPipe's own pose test picture (https://storage.googleapis.com/mediapipe-assets/pose.jpg, a person standing in a
yoga pose, clothed). The head is pixelated (no face), the tank top between the shoulders and the hips is replaced by procedural
skin (the arms' own tone) with a navel and a blue-black tattoo (strokes + a ring) below it; the shorts stay as clothing. The
picture then moves (shift, zoom, small turn). Writes clip.mp4, clean.mp4 (no tattoo) and truth.npz (the tattoo mask per frame)."""
import sys, os, numpy as np, cv2
src, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 90
HEAD = sys.argv[4] if len(sys.argv) > 4 else "pixelate"
NAVEL_T = float(sys.argv[6]) if len(sys.argv) > 6 else 0.27
WAIST = len(sys.argv) > 7 and sys.argv[7] == "waistband"   # a dark waistband across the belly (t 0.11-0.15) and a stroke running into it
SKIN = sys.argv[5] if len(sys.argv) > 5 else "proc"     # proc: make_patchclip.py's procedural skin texture (pores, mottling); flat: none
os.makedirs(out, exist_ok=True)
im = cv2.imread(src)
x0, y0, x1, y1 = 300, 220, 700, 560                      # crop around the person, then x2
im = cv2.resize(im[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
H, W = im.shape[:2]
P = lambda x, y: (int(round((x - x0) * 2)), int(round((y - y0) * 2)))
hx0, hy0 = P(430, 230); hx1, hy1 = P(530, 315)            # pixelate the head
roi = im[hy0:hy1, hx0:hx1]
if HEAD == "pixelate": im[hy0:hy1, hx0:hx1] = cv2.resize(cv2.resize(roi, (5, 5), interpolation=cv2.INTER_AREA), (hx1 - hx0, hy1 - hy0), interpolation=cv2.INTER_CUBIC)
# torso: the tank top's area between the shoulders and the top of the shorts
sh_l, sh_r, hp_l, hp_r = P(450, 318), P(548, 318), P(462, 462), P(526, 466)
yy, xx = np.mgrid[0:H, 0:W]
poly = np.array([P(448, 305), P(552, 305), P(546, 400), P(540, 462), P(460, 458), P(456, 400)], np.int32)
tor = np.zeros((H, W), np.uint8); cv2.fillPoly(tor, [poly], 255)
b, g, r = [im[..., k].astype(int) for k in range(3)]
bg = (b > r + 25)                                          # sky / sea behind
tor = (tor > 0) & ~bg
ax, ay = P(380, 318)
arm = im[ay - 5: ay + 5, ax - 20: ax + 20].reshape(-1, 3).astype(np.float32)
tone = np.median(arm, axis=0)
rng = np.random.default_rng(7)
noise = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 2.0) * 3.0 + cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 9.0) * 4.0
shade = 1.0 - 0.18 * np.clip(np.abs(xx - 0.5 * (sh_l[0] + sh_r[0])) / (0.5 * (sh_r[0] - sh_l[0]) + 1e-6), 0, 1) ** 2
skin = np.clip(tone[None, None, :] * shade[..., None] + noise[..., None], 0, 255)
if SKIN == "proc":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_patchclip import proc_skin
    tex = proc_skin(seed=11, w=W, h=H).astype(np.float32) / np.array([150, 170, 215], np.float32)
    skin = np.clip(tone[None, None, :] * shade[..., None] * tex, 0, 255)
clean = im.copy()
feather = cv2.GaussianBlur(tor.astype(np.float32), (0, 0), 1.2)[..., None]
clean = (clean.astype(np.float32) * (1 - feather) + skin * feather).astype(np.uint8)
# the body's frame: t = 0 at the hip joints, 1 at the shoulders; x across, in px
hy = 0.5 * (hp_l[1] + hp_r[1]); shy = 0.5 * (sh_l[1] + sh_r[1]); mx = int(0.25 * (hp_l[0] + hp_r[0] + sh_l[0] + sh_r[0]))
hipw = abs(hp_r[0] - hp_l[0])
Y = lambda t: int(round(hy - t * (hy - shy)))
nav = (mx, Y(NAVEL_T))
cv2.ellipse(clean, nav, (5, 7), 0, 0, 360, tuple(int(v) for v in tone * 0.55), -1, cv2.LINE_AA)
parts = np.zeros((H, W), np.uint8)
def put(mask, k):
    parts[(mask > 100) & (parts == 0)] = k
# 1: upper belly, a small rose-like design (a ring, petals, a stem) around t = 0.47, a little left of the middle
m = np.zeros((H, W), np.uint8); c1 = (mx - 18, Y(0.47))
cv2.circle(m, c1, 12, 255, 3, cv2.LINE_AA); cv2.ellipse(m, (c1[0] + 4, c1[1] - 3), (6, 4), 30, 0, 300, 255, 2, cv2.LINE_AA)
cv2.line(m, (c1[0] + 8, c1[1] + 11), (c1[0] + 26, Y(0.40)), 255, 3, cv2.LINE_AA)
cv2.ellipse(m, (c1[0] + 22, Y(0.42)), (7, 3), -35, 0, 360, 255, -1, cv2.LINE_AA)
put(m, 1)
# 2: beside the navel, a script-like stroke and three dots, 0.42 of the hip width to the right
m = np.zeros((H, W), np.uint8); x2 = mx + int(0.42 * hipw)
cv2.polylines(m, [np.array([[x2 - 14, Y(0.31)], [x2 - 6, Y(0.26)], [x2 + 2, Y(0.30)], [x2 + 10, Y(0.25)], [x2 + 16, Y(0.29)]], np.int32)],
              False, 255, 3, cv2.LINE_AA)
for d in (-8, 0, 8):
    cv2.circle(m, (x2 + d, Y(0.22)), 3, 255, -1, cv2.LINE_AA)
put(m, 2)
# 3: lower belly, above the floor (t 0.12-0.20): a wave and a ring   (waistband variant: 5, a stroke running down into the band)
m = np.zeros((H, W), np.uint8)
if WAIST:
    cv2.line(m, (mx + int(0.15 * hipw), Y(0.25)), (mx + int(0.15 * hipw) - 6, Y(0.12)), 255, 4, cv2.LINE_AA)   # (its end runs under the band)
    put(m, 5)
    m = np.zeros((H, W), np.uint8)
else:
  cv2.polylines(m, [np.array([[mx - 46, Y(0.14)], [mx - 24, Y(0.20)], [mx, Y(0.13)], [mx + 24, Y(0.20)], [mx + 44, Y(0.14)]], np.int32)],
              False, 255, 4, cv2.LINE_AA)
  cv2.circle(m, (mx - 64, Y(0.17)), 8, 255, 3, cv2.LINE_AA)
  put(m, 3)
ink = np.array([70, 45, 35], np.float32)
a = cv2.GaussianBlur((parts > 0).astype(np.float32), (0, 0), 0.8)[..., None] * 0.85
marked = (clean.astype(np.float32) * (1 - a) + ink * a)
# 4: a brown mole (a mark, not ink), round, left of the middle at t = 0.37
m = np.zeros((H, W), np.uint8); cv2.circle(m, (mx - int(0.33 * hipw), Y(0.37)), 5, 255, -1, cv2.LINE_AA)
put(m, 4)
am = cv2.GaussianBlur((m > 100).astype(np.float32), (0, 0), 1.0)[..., None] * 0.80
mole = tone * np.array([0.45, 0.55, 0.70], np.float32)
marked = (marked * (1 - am) + mole * am).astype(np.uint8)
if WAIST:
    wb = np.zeros((H, W), np.uint8)
    cv2.rectangle(wb, (0, Y(0.15)), (W - 1, Y(0.11)), 255, -1)
    wb = (wb > 0) & tor                                     # on the body only (the background shows beside it)
    cloth = np.array([38, 34, 32], np.float32) + rng.normal(0, 3, (H, W, 3)).astype(np.float32)
    marked = np.where(wb[..., None], np.clip(cloth, 0, 255), marked).astype(np.uint8)
    clean = np.where(wb[..., None], np.clip(cloth, 0, 255), clean).astype(np.uint8)
    parts[wb] = 0
tat = (parts > 0).astype(np.uint8) * 255
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw, vc = cv2.VideoWriter(out + "/clip.mp4", fourcc, 30.0, (W, H)), cv2.VideoWriter(out + "/clean.mp4", fourcc, 30.0, (W, H))
masks, lms, pmasks, navs, wbs = [], [], [], [], []
REF = {11: (546, 320), 12: (449, 324), 23: (522, 472), 24: (464, 466)}     # MediaPipe on the untouched picture
ref = np.array([P(*REF[k]) for k in (11, 12, 23, 24)], np.float32)
for t in range(N):
    ph = 2 * np.pi * t / N
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 3.0 * np.sin(ph), 1.0 + 0.05 * np.sin(2 * ph))
    M[0, 2] += 24 * np.sin(ph); M[1, 2] += 10 * np.sin(2 * ph + 0.5)
    vw.write(cv2.warpAffine(marked, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    vc.write(cv2.warpAffine(clean, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    masks.append(cv2.warpAffine(tat, M, (W, H), flags=cv2.INTER_NEAREST) > 127)
    lms.append(ref @ M[:, :2].T + M[:, 2])
    pmasks.append(cv2.warpAffine(parts, M, (W, H), flags=cv2.INTER_NEAREST))
    wbs.append(cv2.warpAffine((wb if WAIST else np.zeros((H, W), bool)).astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0)
    navs.append(np.array(nav, np.float32) @ M[:, :2].T + M[:, 2])
vw.release(); vc.release()
np.savez_compressed(out + "/truth.npz", mask=np.array(masks), parts=np.array(pmasks), ref=marked, clean=clean, navel=np.array(navs), lm=np.array(lms),
                    navel_t=np.float64(NAVEL_T), waist=np.array(wbs))
cv2.imwrite(out + "/frame0.jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, 95])
print("clip", W, "x", H, N, "frames; marks", {k: int((parts == k).sum()) for k in (1, 2, 3, 4, 5)}, "px; navel", nav, "t", NAVEL_T, "waistband" if WAIST else "")
