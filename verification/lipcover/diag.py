import sys, numpy as np, cv2
PACK=sys.argv[1] if len(sys.argv)>1 else '/home/user/ClaudeCode/v11.2.81_Brisk'
sys.path.insert(0,PACK); import swap_engine as E, config
E.configure(**config.ENGINE_TUNABLES)
exec(open('face_syn.py').read())
S=512
def ice_mask(ice_y):
    m=np.zeros((S,S),np.uint8)
    cv2.ellipse(m,(S//2,int(ice_y*S)),(int(0.20*S),int(0.17*S)),0,0,360,255,-1)
    cv2.rectangle(m,(int(0.40*S),int((ice_y+0.06)*S)),(int(0.60*S),S),255,-1)
    return m>0
def lip_region():
    m=np.zeros((S,S),np.uint8)
    cv2.ellipse(m,(S//2,int(0.72*S)),(int(0.17*S),int(0.065*S)),0,0,360,255,-1)   # closed-mouth lips ellipse
    return m>0
def lower_lip():
    m=lip_region(); m[:int(0.72*S)]=False; return m
def run(seq):
    ref=None; nref=0; acc=None; outs=[]
    for frm in seq:
        keep=E.occluder_keep(frm, ref=ref, nref=nref)
        if acc is None or acc.shape!=keep.shape: acc=np.ones_like(keep)
        o,c=E._P['occ_open'],E._P['occ_close']
        a=np.where(keep<acc,o,c).astype(np.float32)
        acc=(acc*(1.0-a)+keep*a).astype(np.float32)
        outs.append(acc.copy())
        ref,nref=E.update_face_ref(ref,nref,frm,acc)
    return outs
def up(k): return cv2.resize(k,(S,S),interpolation=cv2.INTER_LINEAR)
def report(name, seq, ice_y, last=6):
    outs=run(seq)
    k=up(outs[-1]); im=ice_mask(ice_y); ll=lower_lip()&im; lp=lip_region()&im; chin=im&~lip_region()
    def cut(m): return float((1-k[m]).mean()) if m.sum()>50 else float('nan')
    print(f"{name:<44} ice-area cut {cut(im)*100:5.1f}% | on lips {cut(lp)*100:5.1f}% (px {int(lp.sum())}) | on LOWER lip {cut(ll)*100:5.1f}% (px {int(ll.sum())}) | ice off-lips {cut(chin)*100:5.1f}%")
    return outs
if __name__=="__main__":
    W=int(E._P['occ_ref_warm'])
    for sd in (3,17,42):
        print(f"--- seed {sd}")
        F=lambda **k: face(seed=sd,**k)
        for iy in (0.74,0.78,0.80,0.86):
            for label,hist in (("closed-mouth history",[F() for _ in range(16)]),
                               ("eating history (mouth open/close)",[F(mouth_open=(i//2)%2==0) for i in range(16)])):
                seq=hist+[F(ice=True,ice_y=iy,mouth_open=False) for _ in range(12)]
                report(f"ice_y={iy} {label}", seq, iy)
