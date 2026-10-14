import sys
PACK=sys.argv[1]
ns={'__name__':'lib'}; sys.argv=['x',PACK]
exec(open('lip/moving.py').read(), ns)
globals().update({k:v for k,v in ns.items() if not k.startswith('__')})
sd=17; col=(40,55,85)
F=lambda **k: face(seed=sd,**k)
hist=[F(mouth_open=((i//3)%2==0)) for i in range(60)]
path=[(0.5+0.30*(1-t/14.0), 0.78+0.12*(1-t/14.0)) for t in range(15)]+[(0.5,0.78)]*14
seq=hist[:]; objs=[]
for (cx,cy) in path:
    f,a=aa_scoop(sd,cx,cy,0.18,0.15,col); seq.append(f); objs.append(a)
masks=run_mask(seq)[60:]; ref=run_mask(hist+[F() for _ in range(len(path))])[60:]
k=24; al=np.clip(masks[k]/np.maximum(ref[k],1e-3),0,1)
vis=seq[60+k].copy()
heat=cv2.applyColorMap((al*255).astype(np.uint8),cv2.COLORMAP_JET)
blend=(0.5*vis+0.5*heat).astype(np.uint8)
cnts,_=cv2.findContours((objs[k]>0.5).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(blend,cnts,-1,(255,255,255),1)
cv2.imwrite('lip/moving_alpha.png',np.hstack([vis,blend]))
core=(cv2.erode((objs[k]>0.5).astype(np.uint8),np.ones((9,9),np.uint8))>0)
print("fraction of object interior with face alpha > 0.5:", round(float((al[core]>0.5).mean()),3), " > 0.9:", round(float((al[core]>0.9).mean()),3), " mean:", round(float(al[core].mean()),3))
