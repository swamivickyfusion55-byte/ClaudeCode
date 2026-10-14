import sys
PACK=sys.argv[1]; sys.path.insert(0,PACK); sys.argv=['x',PACK]
import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
exec(open('realset.py').read())
import collections
def run_gate(seq):
    ref=None; nref=0; acc=None; cuts=[]
    for f in seq:
        f=np.clip(f,0,255).astype(np.uint8)
        keep=E.occluder_keep(f,ref=ref,nref=nref)
        if acc is None or acc.shape!=keep.shape: acc=np.ones_like(keep)
        o,c=E._P['occ_open'],E._P['occ_close']; a=np.where(keep<acc,o,c).astype(np.float32); acc=(acc*(1-a)+keep*a).astype(np.float32)
        ref,nref=E.update_face_ref(ref,nref,f,acc); cuts.append(acc.copy())
    return cuts
if __name__=="__main__":
    faces=load_faces(); print(len(faces),"real faces")
    rng=np.random.default_rng(7)
    det=collections.defaultdict(list); fp=[]
    for fi,crop in enumerate(faces):
        clean=[jitter(crop,rng) for _ in range(45)]
        # clean with light changes
        lt=[clean[i]*(1.0 if i<20 else (0.7 if i<35 else 1.25)) for i in range(45)]
        for name,seqc in (("clean+jitter",clean),("clean+light",lt)):
            cuts=run_gate(seqc); fp.append((fi,name,max(float((1-c).mean())*100 for c in cuts[15:])))
        for cname,col in COLS.items():
            seq=[jitter(crop,rng) for _ in range(40)]; a_last=None
            for t in range(20):
                f,a=scoop(jitter(crop,rng),0.5,0.80,0.18,0.15,col,rng,t); seq.append(f); a_last=a
            cuts=run_gate(seq); k=cuts[-1]; ob=a_last>0.5
            det[cname].append(float((1-k[ob]).mean())*100)
    print("object colour       detection (part of the object cut, mean over 6 real faces) | per face")
    for c,v in det.items(): print(f"  {c:<14} {np.mean(v):5.0f}%   | "+" ".join(f"{x:4.0f}" for x in v))
    print("false cut on clean real-face sequences (peak % of the crop):", {f"{n}": round(max(x for (i,nn,x) in fp if nn==n),2) for n in ("clean+jitter","clean+light")})
