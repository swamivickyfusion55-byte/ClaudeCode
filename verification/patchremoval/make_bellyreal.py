"""v11.2.112: a moving clip for belly clean with a REAL body pose and synthetic skin.
usage: make_bellyreal.py POSE_JPG OUTDIR [frames=90] [head=pixelate|keep] [skin=proc|flat]
POSE_JPG = MediaPipe's own pose test picture (https://storage.googleapis.com/mediapipe-assets/pose.jpg, a person standing in a
yoga pose, clothed). The head is pixelated (no face), the tank top between the shoulders and the hips is replaced by procedural
skin (the arms' own tone) with a navel and a blue-black tattoo (strokes + a ring) below it; the shorts stay as clothing. The
picture then moves (shift, zoom, small turn). Writes clip.mp4, clean.mp4 (no tattoo) and truth.npz (the tattoo mask per frame)."""
import sys, os, numpy as np, cv2
src, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 90
HEAD = sys.argv[4] if len(sys.argv) > 4 else "pixelate"
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
# navel ~0.27 of the way up from the hips to the shoulders, at the body's middle
hy = 0.5 * (hp_l[1] + hp_r[1]); shy = 0.5 * (sh_l[1] + sh_r[1]); mx = int(0.25 * (hp_l[0] + hp_r[0] + sh_l[0] + sh_r[0]))
nav = (mx, int(hy - 0.27 * (hy - shy)))
cv2.ellipse(clean, nav, (5, 7), 0, 0, 360, tuple(int(v) for v in tone * 0.55), -1, cv2.LINE_AA)
# the tattoo: below the navel, above the shorts
tat = np.zeros((H, W), np.uint8)
top, bot = int(hy - 0.17 * (hy - shy)), int(hy - 0.03 * (hy - shy))
cv2.polylines(tat, [np.array([[mx - 50, bot - 4], [mx - 25, top], [mx, bot - 10], [mx + 25, top + 2], [mx + 48, bot - 6]], np.int32)], False, 255, 4, cv2.LINE_AA)
cv2.circle(tat, (mx + 70, int(0.5 * (top + bot))), 9, 255, 3, cv2.LINE_AA)
cv2.line(tat, (mx - 70, top + 4), (mx - 58, bot - 2), 255, 3, cv2.LINE_AA)
tat = (tat > 100).astype(np.uint8) * 255
a = cv2.GaussianBlur(tat.astype(np.float32) / 255.0, (0, 0), 0.8)[..., None] * 0.85
ink = np.array([70, 45, 35], np.float32)
marked = (clean.astype(np.float32) * (1 - a) + ink * a).astype(np.uint8)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw, vc = cv2.VideoWriter(out + "/clip.mp4", fourcc, 30.0, (W, H)), cv2.VideoWriter(out + "/clean.mp4", fourcc, 30.0, (W, H))
masks, lms = [], []
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
vw.release(); vc.release()
np.savez_compressed(out + "/truth.npz", mask=np.array(masks), ref=marked, clean=clean, navel=np.array(nav), lm=np.array(lms))
cv2.imwrite(out + "/frame0.jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, 95])
print("clip", W, "x", H, N, "frames; tattoo", int((tat > 0).sum()), "px; navel", nav, "band rows ~", top, bot)
