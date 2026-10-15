"""With occ_gc=0, occ_cont=0, occ_open_near=0, occ_bias=0 and cheek_ema=1 the new pack must give the SAME mask as v11.2.84, bit for bit."""
import sys, importlib.util, numpy as np, cv2, types
sys.path.insert(0, '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip86')
def load(path, name):
    sys.path.insert(0, path)
    for m in ('swap_engine', 'config'): sys.modules.pop(m, None)
    import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
    sys.path.remove(path); sys.modules[name] = E; return E
BASE = '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/'
E84 = load(BASE + 'pack85s', 'e84'); E85 = load(BASE + 'pack86', 'e85')
E85._P.update({'occ_ext_min': 0, 'occ_stat_hold': 0, 'occ_nn': 0, 'occ_quar_on': 0, 'occ_attach': 0, 'occ_freeze': 0})
sys.argv = ['x']
exec(open('/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip/realset.py').read())
from food import food
import os
faces = load_faces()[:6]
face = types.SimpleNamespace(landmark_2d_106=None, kps=None)
worst = 0.0; n = 0
for fi, crop in enumerate(faces):
    rng = np.random.default_rng(5 + fi); t84 = E84.TrackState(0); t85 = E85.TrackState(0)
    for t in range(80):
        base = jitter(crop, rng)
        pos = [(0.5, 0.76, 65), (0.27, 0.60, 125), (0.73, 0.60, 55)][fi % 3]
        if t >= 40: f, a = food(base, pos[0], pos[1], 0.19, [(235, 240, 248), (40, 55, 85), (150, 215, 170)][fi % 3], rng, ang=pos[2])
        else: f = base
        f = np.clip(f, 0, 255).astype(np.uint8)
        M = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        m84 = E84.AlignedCompositor(None).build_mask(face, M, 256, track=t84, aligned_target=f)
        m85 = E85.AlignedCompositor(None).build_mask(face, M, 256, track=t85, aligned_target=f)
        worst = max(worst, float(np.abs(m84 - m85).max())); n += 1
print("frames compared:", n, "| largest mask difference v11.2.85 vs v11.2.86 with the new dials at their off values:", worst)
