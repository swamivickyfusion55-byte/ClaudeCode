import sys, types, time, numpy as np, cv2
g = types.ModuleType('gradio'); g.Request = type('R', (), {}); g.update = lambda **k: k; g.Error = Exception; sys.modules['gradio'] = g
hf = types.ModuleType('huggingface_hub'); hf.hf_hub_download = lambda *a, **k: ''; sys.modules['huggingface_hub'] = hf
def load(path):
    for m in ('core_pipeline', 'swap_engine', 'config'): sys.modules.pop(m, None)
    sys.path.insert(0, path); import core_pipeline as C; sys.path.remove(path); return C
old = load('/home/user/ClaudeCode/v11.2.90_MotionKeys'); new = load('pack91')
sys.path.insert(0, 'lip'); from realset import load_faces
faces = load_faces()
for S in (128, 256, 512):
    crop = cv2.resize(faces[0], (S, S), interpolation=cv2.INTER_CUBIC)
    for nm, C in (('old', old), ('new', new)):
        C._soft_polish(crop, "Cinematic (clarity + smooth)")
        t = time.perf_counter()
        for _ in range(20): C._soft_polish(crop, "Cinematic (clarity + smooth)")
        print(f"{S}px {nm}: {1000 * (time.perf_counter() - t) / 20:.1f} ms")
def stats(a, b):
    la, lb = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32), cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
    ca = np.hypot(la[:, :, 1] - 128, la[:, :, 2] - 128); cb = np.hypot(lb[:, :, 1] - 128, lb[:, :, 2] - 128)
    lap = lambda x: cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
    return (f"mean change {np.abs(a.astype(np.float32) - b.astype(np.float32)).mean():4.1f} | luma mean {la[:,:,0].mean()*100/255:4.1f}->{lb[:,:,0].mean()*100/255:4.1f}"
            f" | luma p5/p95 {np.percentile(la[:,:,0],5)*100/255:4.1f}/{np.percentile(la[:,:,0],95)*100/255:4.1f} -> {np.percentile(lb[:,:,0],5)*100/255:4.1f}/{np.percentile(lb[:,:,0],95)*100/255:4.1f}"
            f" | chroma {ca.mean():4.1f}->{cb.mean():4.1f} | fine detail x{lap(b) / lap(a):.2f}")
rows = []
for i in range(4):
    crop = cv2.resize(faces[i], (256, 256), interpolation=cv2.INTER_CUBIC)
    o = old._soft_polish(crop, "Cinematic (clarity + smooth)"); n = new._soft_polish(crop, "Cinematic (clarity + smooth)")
    print(f"face {i} old: " + stats(crop, o)); print(f"face {i} new: " + stats(crop, n))
    rows.append(np.hstack([crop, o, n]))
cv2.imwrite('lip91/enh_cmp.png', np.vstack(rows[:3]))
