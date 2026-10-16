"""Does the occluder gate cut the FACE (false cut) while the head turns fast / sits in a side pose?
Aligned real-face crops (12 = 6 + mirrors). Sequence: 24 frontal frames (model warm-up), a fast turn to yaw Y over `turn` frames with horizontal
motion blur, hold 24 frames, fast turn back, 10 frames. The turned face is the crop squeezed horizontally about the near cheek; what the
turn uncovers (far cheek side) is filled with hair-like texture taken from the crop's own corner. Final keep map via TrackState.
usage: turns.py PACK  (env YAWS=0.2,0.35,0.5 TURN=4)"""
import sys, os, numpy as np, cv2
PACK = sys.argv[1]; sys.path.insert(0, PACK)
import swap_engine as E, config, types; E.configure(**config.ENGINE_TUNABLES)
for kv in filter(None, os.environ.get('OVR', '').split(',')):
    k_, v_ = kv.split('='); E._P[k_] = float(v_)
sys.path.insert(0, '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip')
from realset import load_faces, jitter, SIZE
faces = load_faces(); faces = faces + [f[:, ::-1].copy() for f in faces]
YAWS = [float(x) for x in os.environ.get('YAWS', '0.2,0.35,0.5').split(',')]; TURN = int(os.environ.get('TURN', '4'))
FILL = os.environ.get('FILL', 'hair')
FILLS = {'wall_bright': (235, 232, 228), 'wall_dark': (28, 30, 36), 'green': (60, 140, 70), 'blue': (190, 120, 70), 'skin': (140, 165, 205), 'red': (50, 50, 190)}
def hair_fill(crop, rng):
    if FILL in FILLS:
        base = np.array(FILLS[FILL], np.float32)[None, None, :] + np.zeros((SIZE, SIZE, 3), np.float32)
        return base + rng.standard_normal((SIZE, SIZE, 3)).astype(np.float32) * 4.0
    h = crop[:40, :60].astype(np.float32).mean(axis=(0, 1))                  # a hair / background tone from the top-left of the crop
    tex = np.tile(crop[:40, :60], (7, 5, 1))[:SIZE, :SIZE].astype(np.float32)
    return 0.5 * tex + 0.5 * h
def turned(crop, y, side, rng, blur):
    """y: 0..1 squeeze amount; side: +1 turns toward the right edge"""
    if y <= 0.001: out = crop.astype(np.float32)
    else:
        S = SIZE; c = 1.0 - y
        cx = S * (0.5 + 0.12 * side * y / max(y, 1e-3) * min(1.0, y * 2.0))       # the near cheek moves toward the edge a little
        M = np.array([[c, 0, S * 0.5 * (1 - c) + side * S * 0.10 * y], [0, 1, 0]], np.float32)
        warp = cv2.warpAffine(crop, M, (S, S), borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        a = cv2.warpAffine(np.ones((S, S), np.float32), M, (S, S), borderMode=cv2.BORDER_CONSTANT, borderValue=0)[:, :, None]
        a = cv2.GaussianBlur(a, (0, 0), 1.5)[:, :, None] if a.ndim == 2 else cv2.GaussianBlur(a[:, :, 0], (0, 0), 1.5)[:, :, None]
        out = warp * a + hair_fill(crop, rng) * (1 - a)
    if blur > 1:
        k = np.zeros((blur, blur), np.float32); k[blur // 2, :] = 1.0 / blur
        out = cv2.filter2D(out, -1, k)
    out = out + rng.standard_normal(out.shape).astype(np.float32) * 1.8
    return out
res = {}
_face = types.SimpleNamespace(landmark_2d_106=None, kps=None); _M = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
_tmpl = E.canonical_template(SIZE) > 0.5; _comp = E.AlignedCompositor(None)
def gate_cut(tr, f):
    m = _comp.build_mask(_face, _M, SIZE, track=tr, aligned_target=f)
    return 100.0 * float((_tmpl & (m < 0.5)).sum()) / float(_tmpl.sum())
for Y in YAWS:
    worst, mean_turn, mean_hold = [], [], []
    for fi, crop in enumerate(faces):
        rng = np.random.default_rng(100 + fi); tr = E.TrackState(0); side = 1 if fi % 2 == 0 else -1
        ys = [0.0] * 24 + [Y * (i + 1) / TURN for i in range(TURN)] + [Y] * 24 + [Y * (1 - (i + 1) / TURN) for i in range(TURN)] + [0.0] * 10
        cuts = []
        for t, y in enumerate(ys):
            moving = (24 <= t < 24 + TURN) or (24 + TURN + 24 <= t < 24 + 2 * TURN + 24)
            f = np.clip(turned(jitter(crop, rng, 0.6), y, side, rng, 9 if moving else 1), 0, 255).astype(np.uint8)
            cuts.append(gate_cut(tr, f))
        cuts = np.array(cuts)
        worst.append(cuts[24:].max()); mean_turn.append(cuts[24:24 + TURN + 6].mean()); mean_hold.append(cuts[24 + TURN + 6:24 + TURN + 24].mean())
    print(f"yaw {int(Y*100):2d}%: share of the face area cut - worst frame mean {np.mean(worst):5.2f}% max {np.max(worst):5.2f}% | during turn {np.mean(mean_turn):5.2f}% | held side pose {np.mean(mean_hold):5.2f}% | faces with >3% in some frame: {int((np.array(worst) > 3).sum())}/12", flush=True)
