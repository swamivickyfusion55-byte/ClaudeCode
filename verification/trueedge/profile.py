"""Profile of |out-in| (normalised by the face plateau) across the object's outline, image domain, hold frames."""
import sys, cv2, numpy as np
exec(open('foodmetrics.py').read().split('def analyse')[0])
def prof(label, spec):
    W=f"/tmp/claude-0/bench_work/{label}"; inp=frames(f"food_{spec}.mp4"); out=frames(f"{W}/out/BENCHFD_/final.mp4"); gt=np.load(f"food_{spec}_gt.npz")["a"]
    P={d:[] for d in range(-14,17)}; sup=np.zeros((720,1280),bool); sup[BX[1]:BX[3],BX[0]:BX[2]]=True; rows=np.arange(720)[:,None]>300
    for i in range(48,80):
        g=gt[i]>127
        if g.sum()<500: continue
        D=np.abs(out[i].astype(np.float32)-inp[i].astype(np.float32)).mean(axis=2)
        dout=cv2.distanceTransform((~g).astype(np.uint8),cv2.DIST_L2,5); din=cv2.distanceTransform(g.astype(np.uint8),cv2.DIST_L2,5); sd=np.where(g,-din,dout)
        for d in P:
            m=sup&rows&(sd>=d-0.5)&(sd<d+0.5)
            if m.sum()>20: P[d].append(float(D[m].mean()))
    pl=np.mean([np.mean(P[d]) for d in range(12,17) if P[d]])
    return {d:np.mean(v)/pl for d,v in P.items() if v}
if __name__=="__main__":
    for lab in sys.argv[1:]:
        spec="_".join(lab.split("_")[-2:]); p=prof(lab,spec)
        print(lab, " ".join(f"{d:+d}:{p[d]:.2f}" for d in range(-12,13,2) if d in p))
