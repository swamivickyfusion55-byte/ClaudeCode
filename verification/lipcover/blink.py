import sys
PACK=sys.argv[1]
ns={'__name__':'lib'}; sys.argv=['x',PACK]
exec(open('lip/moving.py').read(), ns); globals().update({k:v for k,v in ns.items() if not k.startswith('__')})
exec(open('lip/moving_sweep.py').read().split("if __name__")[0].split("import itertools")[0].replace("PACK=sys.argv[1]","").replace("ns={'__name__':'lib'}; sys.argv=['x',PACK]","").replace("exec(open('lip/moving.py').read(), ns); globals().update({k:v for k,v in ns.items() if not k.startswith('__')})",""), globals()) if False else None
def blink(over, n_obj):
    saved={k:E._P[k] for k in over}; E._P.update(over); out=[]
    try:
        for sd in (3,17):
            F=lambda **k: face(seed=sd,**k)
            hist=[F(mouth_open=((i//3)%2==0)) for i in range(60)]
            seq=hist[:]; objs=None
            for t in range(n_obj):
                f,a=aa_scoop(sd,0.5,0.78,0.18,0.15,(40,55,85)); seq.append(f); objs=a
            for t in range(10): seq.append(F())
            masks=run_mask(seq)[60:]; ref=run_mask(hist+[F() for _ in range(n_obj+10)])[60:]
            ob=(objs>0.5)&(ref[0]>0.5)
            cut=[float((1-np.clip(masks[k]/np.maximum(ref[k],1e-3),0,1))[ob].mean()) for k in range(n_obj+10)]
            out.append(cut)
    finally: E._P.update(saved)
    c=np.mean(out,axis=0); return c
for name,over in (("v82-like (open .45, no steep, wide feather)",{"occ_open":0.45,"occ_open_first":0.45,"occ_steep":1.0,"occ_snap":0.0,"occ_feather":0.05,"occ_grow":0.0}),
                  ("v83 defaults (open .85, first .45, steep 3)",{}),
                  ("v83 with first-sighting .65",{"occ_open_first":0.65}),
                  ("v83 with first-sighting .80",{"occ_open_first":0.80})):
    for nobj in (1,2):
        c=blink(over,nobj)
        print(f"{name:<48} object visible for {nobj} frame(s): peak hole {c.max()*100:3.0f}% | frames with hole>25%: {int((c>0.25).sum())} | total hole-frames {c.sum():.2f}")
