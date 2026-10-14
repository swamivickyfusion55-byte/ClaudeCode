import sys
PACK=sys.argv[1]; sys.argv=['x',PACK]
exec(open('lip/diag3.py').read().split('if __name__')[0])
def blobface(sd, ellipses, mouth_open, col=(40,55,85)):
    img=face(seed=sd,mouth_open=mouth_open).astype(np.float32)
    m=np.zeros((S,S),np.uint8)
    for (cx,cy,rx,ry) in ellipses: cv2.ellipse(m,(int(cx*S),int(cy*S)),(int(rx*S),int(ry*S)),0,0,360,255,-1)
    img[m>0]=np.array(col,np.float32); return np.clip(img,0,255).astype(np.uint8), m>0
CASES={
 "cheek object (reaches face edge), talking": [(0.10,0.60,0.16,0.20)],
 "object under chin with a gap, talking":     [(0.50,1.00,0.17,0.10)],
 "object beside the mouth corner, talking":   [(0.80,0.72,0.12,0.12)],
 "scoop below the lip, not touching it":      [(0.50,0.93,0.15,0.09)],
}
worst=0
for sd in (3,17,42):
    for name,els in CASES.items():
        F=lambda **k: face(seed=sd,**k)
        seq=[F(mouth_open=((i//3)%2==0)) for i in range(60)]
        for t in range(24):
            f,m=blobface(sd,els,mouth_open=((t//3)%2==0)); seq.append(f)
        outs=run(seq)
        lips=lip_region()&~m
        cuts=[float((1-up(o)[lips]).mean())*100 for o in outs[60:]]
        obj=float((1-up(outs[-1])[m]).mean())*100
        worst=max(worst,max(cuts))
        print(f"seed {sd:>2} | {name:<44} | object cut {obj:5.1f}% | max lip cut over 24 frames {max(cuts):5.2f}%")
print("WORST lip cut with an unrelated object present: %.2f%%"%worst)
