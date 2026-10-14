import sys, time, importlib
PACK=sys.argv[1]; sys.path.insert(0,PACK); sys.argv=['x',PACK]
import numpy as np, cv2
import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
exec(open('realset.py').read())
faces=load_faces(); rng=np.random.default_rng(5)
res={}
for S in (128,256,512):
    for tag,withobj in (("no object",False),("object",True)):
        ts=[]
        for crop0 in faces[:4]:
            crop=cv2.resize(crop0,(S,S),interpolation=cv2.INTER_AREA if S<crop0.shape[0] else cv2.INTER_CUBIC)
            ref=None; nref=0; acc=None
            for t in range(36):
                f=jitter(crop,rng)
                if withobj and t>=18:
                    f,_=scoop(f,0.5,0.80,0.18,0.15,COLS['vanilla'] if 'vanilla' in COLS else list(COLS.values())[0],rng,t-18)
                f=np.clip(f,0,255).astype(np.uint8)
                t0=time.perf_counter(); keep=E.occluder_keep(f,ref=ref,nref=nref); dt=time.perf_counter()-t0
                if acc is None or acc.shape!=keep.shape: acc=np.ones_like(keep)
                acc=np.minimum(acc,keep) if withobj else np.ones_like(keep)
                ref,nref=E.update_face_ref(ref,nref,f,acc)
                if t>=24: ts.append(dt*1000)
        res[(S,tag)]=np.median(ts)
print(PACK.split('/')[-1], "  ".join(f"{S}px {tag}: {v:.1f} ms" for (S,tag),v in res.items()))
