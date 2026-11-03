"""Synthetic clips for Patch Removal (NOT user footage): skin with a mark that moves, turns, scales, changes brightness,
leaves the picture and comes back, and has an object pass over it. Writes for each clip:
  pr_<name>.mp4          the clip with the mark
  pr_<name>_clean.mp4    the same clip without the mark (same noise, same encoder)
  pr_<name>_gt.npz       ground truth: 'mark' (mark
                         support per frame, bool), 'occ' (object in front, bool), 'ref' (the app's frame), 'refmask'
                         (a finger-painted mask on it: the mark grown by a wobbly stroke), 'ref_t' (its time, s)
Skins: 'proc' (procedural skin: shading, mottling, pores, fine noise) and 'astro' (skimage's public-domain NASA
astronaut portrait, 2x: a mole on the cheek, and one on the jaw line where skin meets the shadow/collar - an edge case). Marks: 'mole' (dark brown, soft edge, ~16 px), 'tattoo' (blue-black ink
strokes, ~150x90 px), 'birth' (irregular light-brown birthmark, ~60 px)."""
import sys, os, subprocess, cv2, numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else "."      # usage: make_patchclip.py OUTDIR [name,name...] | make_patchclip.py OUTDIR cut | make_patchclip.py OUTDIR warp | make_patchclip.py OUTDIR jit | make_patchclip.py OUTDIR multi | make_patchclip.py OUTDIR hard
FF = "/tmp/claude-0/bin/ffmpeg"
W, H, N, FPS = 1280, 720, 150, 30


def proc_skin(seed=3, w=1800, h=1200):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.array([150, 170, 215], np.float32)                     # BGR light skin
    shade = 1.0 + 0.10 * np.sin(xx / 260.0) * np.cos(yy / 310.0) + 0.06 * (yy / h - 0.5)
    mott = cv2.GaussianBlur(r.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 18) * 9.0
    pores = (r.random((h, w)) < 0.010).astype(np.float32)
    pores = cv2.GaussianBlur(pores, (0, 0), 0.9) * -55.0
    fine = cv2.GaussianBlur(r.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.7) * 3.2
    img = base[None, None, :] * shade[..., None] + (mott + pores + fine)[..., None] * np.array([0.9, 1.0, 0.8], np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def astro_skin():
    import skimage.data
    a = cv2.cvtColor(skimage.data.astronaut(), cv2.COLOR_RGB2BGR)       # 512x512, public domain (NASA)
    a = cv2.resize(a, (1024, 1024), interpolation=cv2.INTER_CUBIC)     # face ~140 px wide
    big = cv2.copyMakeBorder(a, 88, 88, 388, 388, cv2.BORDER_REFLECT)   # 1200 x 1800
    return big, {"cheek": (388 + 390, 88 + 256), "jaw": (388 + 432, 88 + 300)}   # left cheek; the jaw / collar edge


def mark_layer(kind, seed=5):
    """Returns (rgba float layer HxWx4 in a small canvas, centre) - colour and opacity of the mark."""
    r = np.random.default_rng(seed)
    if kind == "mole":
        S = 41; c = S // 2
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        d = np.hypot((xx - c) / 8.0, (yy - c) / 7.0)
        a = np.clip((1.25 - d) / 0.45, 0, 1) * 0.92
        col = np.array([45, 60, 95], np.float32)
        return np.dstack([np.full((S, S, 3), col), a]), (c, c)
    if kind == "tattoo":
        S0, S1 = 120, 190
        m = np.zeros((S0, S1), np.uint8)
        cv2.ellipse(m, (95, 60), (70, 38), 0, 20, 330, 255, 5, cv2.LINE_AA)
        cv2.line(m, (40, 30), (150, 92), 255, 4, cv2.LINE_AA)
        for k in range(5):
            cv2.circle(m, (60 + 18 * k, 60 + int(10 * np.sin(k))), 4, 255, -1, cv2.LINE_AA)
        cv2.putText(m, "M", (78, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3, cv2.LINE_AA)
        a = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 0.8) * 0.85
        col = np.array([70, 45, 30], np.float32)                     # blue-black ink (BGR)
        return np.dstack([np.full((S0, S1, 3), col), a]), (S1 // 2, S0 // 2)
    if kind == "birth":
        S = 90; c = S // 2
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        ang = np.arctan2(yy - c, xx - c)
        rad = 26 + 6 * np.sin(3 * ang + 1.0) + 4 * np.cos(5 * ang)
        d = np.hypot(xx - c, yy - c) / rad
        a = np.clip((1.0 - d) / 0.25, 0, 1) * 0.55
        a *= 0.8 + 0.2 * cv2.GaussianBlur(r.random((S, S)).astype(np.float32), (0, 0), 3)
        col = np.array([70, 105, 150], np.float32)
        return np.dstack([np.full((S, S, 3), col), a]), (c, c)
    raise ValueError(kind)


def motion(i, kind):
    """Affine of the skin at frame i: translate, rotate, scale; leaves the picture at 95-110 and returns."""
    t = i / FPS
    tx = 160 * np.sin(2 * np.pi * t / 3.2)
    ty = 60 * np.sin(2 * np.pi * t / 2.3 + 0.7)
    if 92 <= i < 112:                                             # swings out of the picture and back
        tx += 900 * np.sin(np.pi * (i - 92) / 20.0)
    ang = 9.0 * np.sin(2 * np.pi * t / 4.1)
    sc = 1.0 + 0.10 * np.sin(2 * np.pi * t / 3.7 + 0.3)
    return tx, ty, ang, sc


def render(skin, centre, layer, lc, kind):
    frames, clean, marks, occs = [], [], [], []
    r = np.random.default_rng(11)
    lay = np.zeros(skin.shape[:2] + (4,), np.float32)
    lh, lw = layer.shape[:2]
    y0, x0 = centre[1] - lc[1], centre[0] - lc[0]
    lay[y0:y0 + lh, x0:x0 + lw] = layer
    marked = skin.astype(np.float32) * (1 - lay[..., 3:]) + lay[..., :3] * lay[..., 3:]
    for i in range(N):
        tx, ty, ang, sc = motion(i, kind)
        M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), ang, sc)
        M[0, 2] += W / 2 - centre[0] + tx
        M[1, 2] += H / 2 - centre[1] + ty
        bright = 1.0 + 0.12 * np.sin(2 * np.pi * i / 70.0)
        f_m = cv2.warpAffine(marked, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * bright
        f_c = cv2.warpAffine(skin.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * bright
        mk = cv2.warpAffine(lay[..., 3], M, (W, H), flags=cv2.INTER_LINEAR) > 0.03
        # object passing over the mark (a dark cylinder, e.g. a pen / cup edge) at frames 40-60
        oc = np.zeros((H, W), bool)
        if 40 <= i < 60:
            p = M @ np.array([centre[0], centre[1], 1.0])
            ox = int(p[0] - 120 + 12 * (i - 40))
            cv2.rectangle(oc.view(np.uint8), (ox, 0), (ox + 46, H), 1, -1)
            oc = oc.view(bool)
            obj = np.array([40, 90, 160], np.float32) * (0.8 + 0.2 * np.linspace(0, 1, W)[None, :, None])
            f_m = np.where(oc[..., None], obj, f_m)
            f_c = np.where(oc[..., None], obj, f_c)
        noise = r.normal(0, 1.5, (H, W, 3)).astype(np.float32)
        frames.append(np.clip(f_m + noise, 0, 255).astype(np.uint8))
        clean.append(np.clip(f_c + noise, 0, 255).astype(np.uint8))
        marks.append(mk & ~oc)
        occs.append(oc)
    return frames, clean, marks, occs


def finger_mask(markmask, seed=2):
    """What a finger paints over the mark: the mark grown by 5-9 px with a wobbly edge."""
    r = np.random.default_rng(seed)
    m = cv2.dilate(markmask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    ys, xs = np.nonzero(m)
    for _ in range(25):
        k = r.integers(len(ys))
        cv2.circle(m, (int(xs[k]), int(ys[k])), int(r.integers(5, 10)), 1, -1)
    return m.astype(np.uint8) * 255


def write(name, frames, crf=14):
    tmp = os.path.join(OUT, f"pr_{name}.avi")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    os.path.join(OUT, f"pr_{name}.mp4")], check=True)
    os.remove(tmp)


def render_cut(skin, centre, layer, lc):
    """v11.2.97: the person's position changes - a shot change at frame 75 to the same skin turned 35 deg, 1.7x larger,
    darker, moving on. One painted position cannot know what the mark looks like after the cut; a second one can."""
    frames, clean, marks, occs = [], [], [], []
    r = np.random.default_rng(13)
    lay = np.zeros(skin.shape[:2] + (4,), np.float32)
    lh, lw = layer.shape[:2]
    y0, x0 = centre[1] - lc[1], centre[0] - lc[0]
    lay[y0:y0 + lh, x0:x0 + lw] = layer
    marked = skin.astype(np.float32) * (1 - lay[..., 3:]) + lay[..., :3] * lay[..., 3:]
    for i in range(N):
        t = i / FPS
        if i < 75:
            ang, sc, br = 5.0 * np.sin(2 * np.pi * t / 3.0), 1.0, 1.0
            tx, ty = 120 * np.sin(2 * np.pi * t / 2.7), 40 * np.sin(2 * np.pi * t / 2.1)
        else:
            ang, sc, br = 35.0 + 6.0 * np.sin(2 * np.pi * t / 2.5), 1.7 + 0.08 * np.sin(2 * np.pi * t / 1.9), 0.82
            tx, ty = -180 + 90 * np.sin(2 * np.pi * t / 2.3), 60 + 30 * np.sin(2 * np.pi * t / 1.7)
        M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), ang, sc)
        M[0, 2] += W / 2 - centre[0] + tx
        M[1, 2] += H / 2 - centre[1] + ty
        f_m = cv2.warpAffine(marked, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * br
        f_c = cv2.warpAffine(skin.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * br
        mk = cv2.warpAffine(lay[..., 3], M, (W, H), flags=cv2.INTER_LINEAR) > 0.03
        noise = r.normal(0, 1.5, (H, W, 3)).astype(np.float32)
        frames.append(np.clip(f_m + noise, 0, 255).astype(np.uint8))
        clean.append(np.clip(f_c + noise, 0, 255).astype(np.uint8))
        marks.append(mk)
        occs.append(np.zeros((H, W), bool))
    return frames, clean, marks, occs


def render_warp(skin, centre, layer, lc, crf_note=None):
    """v11.2.99: closer to real footage - the skin also BENDS (a smooth non-rigid displacement, up to ~3 px, changing
    over time: an affine track cannot follow it exactly), a soft shadow edge sweeps across the mark, and the clip is
    encoded at phone quality (CRF 24). No object, no out-of-picture swing."""
    frames, clean, marks, occs = [], [], [], []
    r = np.random.default_rng(17)
    lay = np.zeros(skin.shape[:2] + (4,), np.float32)
    lh, lw = layer.shape[:2]
    y0, x0 = centre[1] - lc[1], centre[0] - lc[0]
    lay[y0:y0 + lh, x0:x0 + lw] = layer
    marked = skin.astype(np.float32) * (1 - lay[..., 3:]) + lay[..., :3] * lay[..., 3:]
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    for i in range(N):
        t = i / FPS
        tx, ty = 140 * np.sin(2 * np.pi * t / 3.2), 50 * np.sin(2 * np.pi * t / 2.3 + 0.7)
        ang, sc = 8.0 * np.sin(2 * np.pi * t / 4.1), 1.0 + 0.08 * np.sin(2 * np.pi * t / 3.7 + 0.3)
        M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), ang, sc)
        M[0, 2] += W / 2 - centre[0] + tx
        M[1, 2] += H / 2 - centre[1] + ty
        amp = 3.0 * np.sin(2 * np.pi * t / 2.9 + 0.4)
        mx = gx + amp * np.sin(2 * np.pi * (gy / 210.0 + t / 3.1))
        my = (gy + amp * np.cos(2 * np.pi * (gx / 190.0 + t / 2.6))).astype(np.float32)
        mx = mx.astype(np.float32)
        p = M @ np.array([centre[0], centre[1], 1.0])
        sweep = p[0] - 160 + 320 * ((t / 2.5) % 1.0)
        shade = 1.0 - 0.16 / (1.0 + np.exp(-((gx - sweep) * 0.8 + (gy - p[1]) * 0.6) / 25.0))
        def go(img, interp=cv2.INTER_LINEAR):
            a = cv2.warpAffine(img, M, (W, H), flags=interp, borderMode=cv2.BORDER_REFLECT)
            return cv2.remap(a, mx, my, interp, borderMode=cv2.BORDER_REFLECT)
        f_m = go(marked) * shade[..., None]
        f_c = go(skin.astype(np.float32)) * shade[..., None]
        mk = go(lay[..., 3]) > 0.03
        noise = r.normal(0, 1.5, (H, W, 3)).astype(np.float32)
        frames.append(np.clip(f_m + noise, 0, 255).astype(np.uint8))
        clean.append(np.clip(f_c + noise, 0, 255).astype(np.uint8))
        marks.append(mk)
        occs.append(np.zeros((H, W), bool))
    return frames, clean, marks, occs


def render_jitter(skin, centre, layer, lc, amp=3.0):
    """v11.2.100: the mark is not glued to the skin the tracker follows - it sits a little off its anchored place on
    every frame (relief, skin sliding over it, tracking error of 1-3 px): the mark layer is displaced against the skin
    by a smooth random-walk (up to ~amp px, with a slight turn), phone-quality encoding, no object."""
    frames, clean, marks, occs = [], [], [], []
    r = np.random.default_rng(23)
    lay = np.zeros(skin.shape[:2] + (4,), np.float32)
    lh, lw = layer.shape[:2]
    y0, x0 = centre[1] - lc[1], centre[0] - lc[0]
    lay[y0:y0 + lh, x0:x0 + lw] = layer
    ph = r.uniform(0, 6.28, 6)
    for i in range(N):
        t = i / FPS
        tx, ty = 130 * np.sin(2 * np.pi * t / 3.2), 45 * np.sin(2 * np.pi * t / 2.3 + 0.7)
        ang, sc = 7.0 * np.sin(2 * np.pi * t / 4.1), 1.0 + 0.07 * np.sin(2 * np.pi * t / 3.7 + 0.3)
        M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), ang, sc)
        M[0, 2] += W / 2 - centre[0] + tx
        M[1, 2] += H / 2 - centre[1] + ty
        jx = amp * (0.6 * np.sin(2 * np.pi * t / 1.3 + ph[0]) + 0.4 * np.sin(2 * np.pi * t / 0.47 + ph[1]))
        jy = amp * (0.6 * np.sin(2 * np.pi * t / 1.1 + ph[2]) + 0.4 * np.sin(2 * np.pi * t / 0.53 + ph[3]))
        Mm = M.copy(); Mm[0, 2] += jx; Mm[1, 2] += jy
        S = cv2.warpAffine(skin.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        Lw = cv2.warpAffine(lay, Mm, (W, H), flags=cv2.INTER_LINEAR)
        a = Lw[..., 3:]
        bright = 1.0 + 0.08 * np.sin(2 * np.pi * i / 70.0)
        f_m = (S * (1 - a) + Lw[..., :3] * a) * bright
        f_c = S * bright
        noise = r.normal(0, 1.5, (H, W, 3)).astype(np.float32)
        frames.append(np.clip(f_m + noise, 0, 255).astype(np.uint8))
        clean.append(np.clip(f_c + noise, 0, 255).astype(np.uint8))
        marks.append(Lw[..., 3] > 0.03)
        occs.append(np.zeros((H, W), bool))
    return frames, clean, marks, occs


def render_multi(skin, centre, layers, offs):
    """v11.2.100: several marks on the same skin (up to 10): layers[i] is placed at centre + offs[i]. Same motion as the
    plain clips (the whole skin moves, turns, changes brightness, an object passes at frames 40-59)."""
    frames, clean, marks, occs = [], [], [], []
    r = np.random.default_rng(11)
    lay = np.zeros(skin.shape[:2] + (4,), np.float32)
    per = []
    for (layer, lc), (dx, dy) in zip(layers, offs):
        one = np.zeros(skin.shape[:2] + (4,), np.float32)
        lh, lw = layer.shape[:2]
        y0, x0 = centre[1] + dy - lc[1], centre[0] + dx - lc[0]
        one[y0:y0 + lh, x0:x0 + lw] = layer
        per.append(one)
        a = one[..., 3:]
        lay[..., :3] = lay[..., :3] * (1 - a) + one[..., :3] * a
        lay[..., 3] = np.maximum(lay[..., 3], one[..., 3])
    marked = skin.astype(np.float32) * (1 - lay[..., 3:]) + lay[..., :3] * lay[..., 3:]
    for i in range(N):
        tx, ty, ang, sc = motion(i, "multi")
        ang, sc = ang * 0.7, 1.0 + (sc - 1.0) * 0.7
        M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), ang, sc)
        M[0, 2] += W / 2 - centre[0] + tx * 0.35
        M[1, 2] += H / 2 - centre[1] + ty * 0.35
        bright = 1.0 + 0.10 * np.sin(2 * np.pi * i / 70.0)
        f_m = cv2.warpAffine(marked, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * bright
        f_c = cv2.warpAffine(skin.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * bright
        mk = [cv2.warpAffine(o[..., 3], M, (W, H), flags=cv2.INTER_LINEAR) > 0.03 for o in per]
        noise = r.normal(0, 1.5, (H, W, 3)).astype(np.float32)
        frames.append(np.clip(f_m + noise, 0, 255).astype(np.uint8))
        clean.append(np.clip(f_c + noise, 0, 255).astype(np.uint8))
        marks.append(mk)
        occs.append(np.zeros((H, W), bool))
    return frames, clean, marks, occs


def render_hard(skin, centre, layer, lc, amp=4.0, blur=True, fast=1.0):
    """v11.2.102: footage that is hard to follow - the body part turns up to +-32 deg, scales 0.75-1.35, changes perspective,
    moves up to ~250 px (fast: motion blur along the motion), the skin bends, a shadow sweeps over it, the mark sits up to
    ~amp px off where the skin pattern puts it, phone-quality encoding. No object."""
    frames, clean, marks, occs = [], [], [], []
    r = np.random.default_rng(31)
    lay = np.zeros(skin.shape[:2] + (4,), np.float32)
    lh, lw = layer.shape[:2]
    y0, x0 = centre[1] - lc[1], centre[0] - lc[0]
    lay[y0:y0 + lh, x0:x0 + lw] = layer
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    ph = r.uniform(0, 6.28, 4)
    prev_c = None
    for i in range(N):
        t = i / FPS
        tx, ty = 230 * np.sin(2 * np.pi * t * fast / 2.6), 70 * np.sin(2 * np.pi * t * fast / 1.9 + 0.7)     # (fast: x times the speed)
        ang = 32.0 * np.sin(2 * np.pi * t * (1 + 0.5 * (fast - 1)) / 3.4)
        sc = 1.05 + 0.30 * np.sin(2 * np.pi * t / 2.9 + 0.3)
        M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), ang, sc)
        M[0, 2] += W / 2 - centre[0] + tx
        M[1, 2] += H / 2 - centre[1] + ty
        P = np.vstack([M, [0, 0, 1]]).astype(np.float64)
        P[2, 0] = 2.2e-4 * np.sin(2 * np.pi * t / 2.2 + ph[0])          # perspective (keystone)
        P[2, 1] = 1.6e-4 * np.sin(2 * np.pi * t / 2.7 + ph[1])
        jx = amp * (0.6 * np.sin(2 * np.pi * t / 1.2 + ph[2]) + 0.4 * np.sin(2 * np.pi * t / 0.45 + ph[3]))
        jy = amp * (0.6 * np.sin(2 * np.pi * t / 1.0 + ph[3]) + 0.4 * np.sin(2 * np.pi * t / 0.5 + ph[2]))
        Pm = P.copy(); Pm[0, 2] += jx; Pm[1, 2] += jy
        bend = 3.0 * np.sin(2 * np.pi * t / 2.9 + 0.4)
        mx = (gx + bend * np.sin(2 * np.pi * (gy / 210.0 + t / 3.1))).astype(np.float32)
        my = (gy + bend * np.cos(2 * np.pi * (gx / 190.0 + t / 2.6))).astype(np.float32)
        S = cv2.remap(cv2.warpPerspective(skin.astype(np.float32), P, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        Lw = cv2.remap(cv2.warpPerspective(lay, Pm, (W, H), flags=cv2.INTER_LINEAR), mx, my, cv2.INTER_LINEAR)
        a = Lw[..., 3:]
        p0 = P @ np.array([centre[0], centre[1], 1.0]); p0 = p0[:2] / p0[2]
        sweep = p0[0] - 160 + 320 * ((t / 2.5) % 1.0)
        shade = 1.0 - 0.14 / (1.0 + np.exp(-((gx - sweep) * 0.8 + (gy - p0[1]) * 0.6) / 25.0))
        f_m = (S * (1 - a) + Lw[..., :3] * a) * shade[..., None]
        f_c = S * shade[..., None]
        if blur and prev_c is not None:                                   # motion blur along the motion of this frame
            d = p0 - prev_c; sp = float(np.hypot(*d))
            L_ = int(min(17, round(0.7 * sp)))
            if L_ >= 3:
                k = np.zeros((L_, L_), np.float32)
                cv2.line(k, (0, 0), (L_ - 1, L_ - 1), 1.0, 1)
                th = np.degrees(np.arctan2(d[1], d[0]))
                Rk = cv2.getRotationMatrix2D(((L_ - 1) / 2.0, (L_ - 1) / 2.0), th - 45.0, 1.0)
                k = cv2.warpAffine(k, Rk, (L_, L_)); k /= max(float(k.sum()), 1e-6)
                f_m = cv2.filter2D(f_m, -1, k); f_c = cv2.filter2D(f_c, -1, k)
        prev_c = p0
        noise = r.normal(0, 1.5, (H, W, 3)).astype(np.float32)
        frames.append(np.clip(f_m + noise, 0, 255).astype(np.uint8))
        clean.append(np.clip(f_c + noise, 0, 255).astype(np.uint8))
        marks.append(Lw[..., 3] > 0.03)
        occs.append(np.zeros((H, W), bool))
    return frames, clean, marks, occs


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2].startswith("hard"):
    os.makedirs(OUT, exist_ok=True)
    ps = proc_skin()
    asx, acen = astro_skin()
    for name, skin, cen, kind in [("hard_mole", ps, (900, 600), "mole"), ("hard_birth", ps, (900, 600), "birth"),
                                  ("hard_tattoo", ps, (900, 600), "tattoo"), ("hard_astro", asx, acen["cheek"], "mole")]:
        layer, lc = mark_layer(kind)
        frames, clean, marks, occs = render_hard(skin, cen, layer, lc)
        write(name, frames, crf=24)
        write(name + "_clean", clean, crf=24)
        ri = 20
        np.savez_compressed(os.path.join(OUT, f"pr_{name}_gt.npz"), mark=np.array(marks), occ=np.array(occs),
                            ref=frames[ri], refmask=finger_mask(marks[ri]), ref_t=ri / FPS)
        print(name, "ok")
elif __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2].startswith("fast"):
    # v11.2.103: the hard clip moving 2.5 / 4 / 6 times as fast (peak about 45 / 72 / 108 px per frame) with motion blur
    os.makedirs(OUT, exist_ok=True)
    ps = proc_skin()
    for fk in (2.5, 4.0, 6.0):
        for kind in ("mole", "tattoo"):
            name = f"fast{fk:g}_{kind}"
            layer, lc = mark_layer(kind)
            frames, clean, marks, occs = render_hard(ps, (900, 600), layer, lc, fast=fk)
            write(name, frames, crf=24)
            write(name + "_clean", clean, crf=24)
            ri = 20
            np.savez_compressed(os.path.join(OUT, f"pr_{name}_gt.npz"), mark=np.array(marks), occ=np.array(occs),
                                ref=frames[ri], refmask=finger_mask(marks[ri]), ref_t=ri / FPS)
            print(name, "ok")
elif __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2].startswith("multi"):
    os.makedirs(OUT, exist_ok=True)
    ps = proc_skin()
    kinds = ["mole", "birth", "mole", "mole", "birth", "mole", "mole", "birth", "mole", "mole"]
    layers = [mark_layer(k, seed=5 + i) for i, k in enumerate(kinds)]
    offs = [(-300, -150), (-120, -170), (60, -160), (240, -150), (330, -20), (-330, 20), (-170, 60), (40, 70), (190, 60), (-40, 190)]
    frames, clean, marks, occs = render_multi(ps, (900, 600), layers, offs)
    write("multi", frames, crf=24)
    write("multi_clean", clean, crf=24)
    ri = 20
    d = dict(mark=np.array([np.any(m, axis=0) for m in marks]), occ=np.array(occs), ref=frames[ri],
             refmask=finger_mask(marks[ri][0]), ref_t=ri / FPS, nmarks=len(offs), marks_all=np.array([[m[k] for m in marks] for k in range(len(offs))]))
    for k in range(len(offs)):
        d[f"refmask_{k}"] = finger_mask(marks[ri][k], seed=2 + k)
    np.savez_compressed(os.path.join(OUT, "pr_multi_gt.npz"), **d)
    print("multi ok")
elif __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2].startswith("jit"):
    os.makedirs(OUT, exist_ok=True)
    ps = proc_skin()
    asx, acen = astro_skin()
    for name, skin, cen, kind in [("jit_mole", ps, (900, 600), "mole"), ("jit_tattoo", ps, (900, 600), "tattoo"),
                                  ("jit_astro", asx, acen["cheek"], "mole")]:
        layer, lc = mark_layer(kind)
        frames, clean, marks, occs = render_jitter(skin, cen, layer, lc)
        write(name, frames, crf=24)
        write(name + "_clean", clean, crf=24)
        np.savez_compressed(os.path.join(OUT, f"pr_{name}_gt.npz"), mark=np.array(marks), occ=np.array(occs),
                            ref=frames[20], refmask=finger_mask(marks[20]), ref_t=20 / FPS)
        print(name, "ok")
elif __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2].startswith("warp"):
    os.makedirs(OUT, exist_ok=True)
    ps = proc_skin()
    asx, acen = astro_skin()
    for name, skin, cen, kind in [("warp_mole", ps, (900, 600), "mole"), ("warp_birth", ps, (900, 600), "birth"),
                                  ("warp_tattoo", ps, (900, 600), "tattoo"), ("warp_astro", asx, acen["cheek"], "mole")]:
        layer, lc = mark_layer(kind)
        frames, clean, marks, occs = render_warp(skin, cen, layer, lc)
        write(name, frames, crf=24)
        write(name + "_clean", clean, crf=24)
        np.savez_compressed(os.path.join(OUT, f"pr_{name}_gt.npz"), mark=np.array(marks), occ=np.array(occs),
                            ref=frames[20], refmask=finger_mask(marks[20]), ref_t=20 / FPS)
        print(name, "ok")
elif __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2].startswith("cut"):
    os.makedirs(OUT, exist_ok=True)
    ps = proc_skin()
    asx, acen = astro_skin()
    for name, skin, cen, kind in [("cut_tattoo", ps, (900, 600), "tattoo"), ("cut_mole", asx, acen["cheek"], "mole")]:
        layer, lc = mark_layer(kind)
        frames, clean, marks, occs = render_cut(skin, cen, layer, lc)
        write(name, frames)
        write(name + "_clean", clean)
        np.savez_compressed(os.path.join(OUT, f"pr_{name}_gt.npz"), mark=np.array(marks), occ=np.array(occs),
                            ref=frames[20], refmask=finger_mask(marks[20]), ref_t=20 / FPS,
                            ref2=frames[110], refmask2=finger_mask(marks[110], seed=4), ref_t2=110 / FPS)
        print(name, "ok")
elif __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ps = proc_skin()
    asx, acen = astro_skin()
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    for name, skin, cen, kind in [("proc_mole", ps, (900, 600), "mole"), ("proc_tattoo", ps, (900, 600), "tattoo"),
                                  ("proc_birth", ps, (900, 600), "birth"), ("astro_mole", asx, acen["cheek"], "mole"),
                                  ("astro_edge", asx, acen["jaw"], "mole")]:
        if only and name not in only:
            continue
        layer, lc = mark_layer(kind)
        frames, clean, marks, occs = render(skin, cen, layer, lc, kind)
        write(name, frames)
        ri = 20                                                     # the app's frame: t = 0.667 s
        write(name + "_clean", clean)
        np.savez_compressed(os.path.join(OUT, f"pr_{name}_gt.npz"), mark=np.array(marks),
                            occ=np.array(occs), ref=frames[ri], refmask=finger_mask(marks[ri]), ref_t=ri / FPS)
        print(name, "ok")
