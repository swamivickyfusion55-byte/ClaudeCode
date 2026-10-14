import sys
PACK=sys.argv[1]; sys.path.insert(0,PACK); sys.argv=['x',PACK]
exec(open('realdetect.py').read().split('if __name__')[0])
S=SIZE
def noise(f,rng): return f.astype(np.float32)+rng.standard_normal(f.shape).astype(np.float32)*1.8
def warp(crop,M,rng): return noise(cv2.warpAffine(crop,M,(S,S),borderMode=cv2.BORDER_REPLICATE),rng)
def talking(crop,rng,n=60):
    out=[]; yy,xx=np.mgrid[0:S,0:S].astype(np.float32)
    for i in range(n):
        a=0.5+0.5*np.sin(i/2.2)
        g=np.exp(-(((xx-0.5*S)/(0.22*S))**2+((yy-0.78*S)/(0.16*S))**2))   # lower face around the mouth and chin
        dy=a*10.0*g; dx=np.sin(i/3.1)*1.5*g
        out.append(noise(cv2.remap(crop,(xx-dx).astype(np.float32),(yy-dy).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE),rng))
    return out
def sequences(crop,rng):
    clean=[noise(cv2.warpAffine(crop,np.float32([[1,0,rng.normal(0,1.5)],[0,1,rng.normal(0,1.5)]]),(S,S),borderMode=cv2.BORDER_REPLICATE),rng) for _ in range(60)]
    yield "jitter",clean
    yield "light (-50%/+50%)",[clean[i]*(1.0 if i<20 else (0.7 if i<40 else 1.25)) for i in range(60)]
    yield "light gradient",[np.clip(clean[i]*(1.0+0.25*np.linspace(-1,1,S)[None,:,None]*min(1,max(0,(i-25)/15.0))),0,255) for i in range(60)]
    mo=[]
    for i in range(60):
        f=clean[i].copy()
        if i>=35:
            cv2.ellipse(f,(S//2,int(0.745*S)),(int(0.11*S),int(0.075*S)),0,0,360,(25,25,35),-1); cv2.rectangle(f,(int(0.42*S),int(0.675*S)),(int(0.58*S),int(0.705*S)),(225,230,235),-1)
        mo.append(f)
    yield "wide-open mouth appears",mo
    yield "TALKING (jaw+lips move 10px)",talking(crop,rng)
    for kind in ("align-noise 4px","yaw 14%","pitch 12%","rigid 5px/3deg"):
        out=[]
        for i in range(60):
            ph=i/9.0; M=cv2.getRotationMatrix2D((S/2,S/2),0,1.0)
            if kind.startswith("align"): M=cv2.getRotationMatrix2D((S/2,S/2),rng.normal(0,1.5),1+rng.normal(0,0.01)); M[:,2]+=rng.normal(0,4.0,2)
            elif kind.startswith("yaw"): M[0,:2]*=1-0.14*abs(np.sin(ph*0.5)); M[0,2]+=S*0.05*np.sin(ph*0.5)
            elif kind.startswith("pitch"): M[1,:2]*=1-0.12*abs(np.sin(ph*0.6)); M[1,2]+=S*0.04*np.sin(ph*0.6)
            else: M=cv2.getRotationMatrix2D((S/2,S/2),3*np.sin(ph),1.0); M[:,2]+=(5*np.sin(ph*1.1),3.5*np.sin(ph*0.8+1))
            out.append(warp(crop,M,rng))
        yield kind,out
def battery(over, faces, seeds=(7,8)):
    saved={k:E._P[k] for k in over}; E._P.update(over)
    fp=collections.defaultdict(float); det=collections.defaultdict(list)
    try:
        for sd in seeds:
            rng=np.random.default_rng(sd)
            allf=faces+[c[:,::-1].copy() for c in faces]
            for crop in allf:
                for name,seq in sequences(crop,rng):
                    cuts=run_gate(seq); fp[name]=max(fp[name],max(float((1-c).mean())*100 for c in cuts[20:]))
                for cname,col in COLS.items():
                    seq=[noise(crop,rng) for _ in range(40)]; a_last=None
                    for t in range(20):
                        f,a=scoop(noise(crop,rng),0.5,0.80,0.18,0.15,col,rng,t); seq.append(f); a_last=a
                    cuts=run_gate(seq); det[cname].append(float((1-cuts[-1][a_last>0.5]).mean())*100)
    finally: E._P.update(saved)
    return {c:float(np.mean(v)) for c,v in det.items()}, dict(fp)
if __name__=="__main__":
    faces=load_faces()
    for sz in (256,):
        d,fp=battery({},faces,seeds=(21,22,23))
        print("== FINAL defaults, harsh battery, 3 new seeds x 12 real faces\n   detection: "+"  ".join(f"{k[:6]} {v:.0f}%" for k,v in d.items())+f"   (mean {np.mean(list(d.values())):.0f}%)\n   false cuts, worst frame (% of crop): "+"  ".join(f"{k} {v:.1f}" for k,v in fp.items()),flush=True)
