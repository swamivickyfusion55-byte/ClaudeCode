"""With occ_gc=0, occ_cont=0, occ_open_near=0, occ_bias=0 and cheek_ema=1 the new pack must give the SAME mask as v11.2.84, bit for bit."""
import sys, importlib.util, numpy as np, cv2, types
sys.path.insert(0, '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip85')
def load(path, name):
    sys.path.insert(0, path)
    for m in ('swap_engine', 'config'): sys.modules.pop(m, None)
    import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
    sys.path.remove(path); sys.modules[name] = E; return E
BASE = '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/'
E84 = load(BASE + 'pack84', 'e84'); E85 = load(BASE + 'pack85', 'e85')
E85._P.update({'occ_gc': 0, 'occ_cont': 0, 'occ_open_near': 0, 'occ_bias': 0, 'cheek_ema': 1.0})
sys.argv = ['x']
exec(open('/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip/realset.py').read())
from food import food
faces = load_faces()[:3]
face = types.SimpleNamespace(landmark_2d_106=None, kps=None)
worst = 0.0; n = 0
for fi, crop in enumerate(faces):
    rng = np.random.default_rng(5 + fi); t84 = E84.TrackState(0); t85 = E85.TrackState(0)
    for t in range(80):
        base = jitter(crop, rng)
        if t >= 40: f, a = food(base, 0.5, 0.76, 0.19, (235, 240, 248), rng)
        else: f = base
        f = np.clip(f, 0, 255).astype(np.uint8)
        M = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        m84 = E84.AlignedCompositor(None).build_mask(face, M, 256, track=t84, aligned_target=f)
        m85 = E85.AlignedCompositor(None).build_mask(face, M, 256, track=t85, aligned_target=f)
        worst = max(worst, float(np.abs(m84 - m85).max())); n += 1
print("frames compared:", n, "| largest mask difference v11.2.84 vs v11.2.85 with the new dials at their off values:", worst)
