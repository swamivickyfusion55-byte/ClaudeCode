import sys, time, numpy as np, cv2
PACK=sys.argv[1]; sys.path.insert(0,PACK); sys.argv=['x']
import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
exec(open('/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip/realset.py').read())
from food import food
faces=load_faces()[:3]; rng=np.random.default_rng(5)
NEW=dict(occ_gc=3,occ_cont=0.3,occ_open_near=0.07,occ_bias=0.004); OLD=dict(occ_gc=0,occ_cont=0,occ_open_near=0,occ_bias=0)
for S in (256,512):
    for tag,over in (("v11.2.84-equivalent",OLD),("v11.2.85",NEW)):
        E._P.update(over)
        for withobj in (False,True):
            ts=[]
            for crop0 in faces:
                crop=cv2.resize(crop0,(S,S),interpolation=cv2.INTER_CUBIC) if S!=256 else crop0
                tr=E.TrackState(0)
                for t in range(44):
                    f=jitter(crop,rng)
                    if withobj and t>=20: f,_=food(f,0.5,0.76,0.19,(235,240,248),rng,ss=2)
                    f=np.clip(f,0,255).astype(np.uint8)
                    t0=time.perf_counter(); E.AlignedCompositor.occluder_keep_final(tr,f); dt=(time.perf_counter()-t0)*1000
                    if t>=30: ts.append(dt)
            print(f"crop {S}px  {tag:<22} {'object' if withobj else 'no object':<10} {np.median(ts):6.1f} ms/call (median)")
    # cheek smoothing
    E._P['cheek_ema']=0.4; fk=np.random.default_rng(1).integers(0,255,(S,S,3),dtype=np.uint8); tr=E.TrackState(0); E.AlignedCompositor.cheek_smooth(tr,fk); ts=[]
    for i in range(30):
        t0=time.perf_counter(); E.AlignedCompositor.cheek_smooth(tr,fk); ts.append((time.perf_counter()-t0)*1000)
    print(f"crop {S}px  cheek smoothing {np.median(ts):.2f} ms/call")
