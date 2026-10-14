import sys
PACK=sys.argv[1] if len(sys.argv)>1 else '/home/user/ClaudeCode/v11.2.81_Brisk'
sys.argv=['x',PACK]
exec(open('lip/diag.py').read().split('if __name__')[0])
def face_scoop(sd, cx, cy, rx, ry, col, mouth_open=False, cone=False):
    img=face(seed=sd,mouth_open=mouth_open).astype(np.float32)
    m=np.zeros((S,S),np.uint8)
    cv2.ellipse(m,(int(cx*S),int(cy*S)),(int(rx*S),int(ry*S)),0,0,360,255,-1)
    if cone:
        pts=np.array([[int((cx-rx*0.8)*S),int((cy+ry*0.6)*S)],[int((cx+rx*0.8)*S),int((cy+ry*0.6)*S)],[int(cx*S),int(min(0.99,cy+ry*3.2)*S)]],np.int32)
        cv2.fillPoly(m,[pts],255)
    # shaded sphere so it is not flat
    yy,xx=np.mgrid[0:S,0:S].astype(np.float32)
    sh=1.0-0.25*np.clip(((xx-cx*S)/(rx*S))*0.6+((yy-cy*S)/(ry*S))*0.6,-1,1)
    colf=np.array(col,np.float32)[None,None,:]*sh[:,:,None]
    img[m>0]=colf[m>0]
    img+=np.random.default_rng(sd+5).standard_normal((S,S,3)).astype(np.float32)*2.0
    return np.clip(img,0,255).astype(np.uint8), (m>0)
COLS={"vanilla (235,240,248)":(235,240,248),"chocolate (40,55,85)":(40,55,85),"strawberry (150,150,235)":(150,150,235),"butterscotch (70,150,215)":(70,150,215),"mango/orange (60,150,240)":(60,150,240)}
rows=[]
for sd in (3,17):
    F=lambda **k: face(seed=sd,**k)
    hist=[F(mouth_open=(i//3)%2==0) for i in range(60)]
    for sname,(cx,cy,rx,ry,cone) in {"small scoop @lower lip (1.8%)":(0.5,0.77,0.08,0.07,False),
                                      "medium scoop @lower lip (3.8%)":(0.5,0.78,0.12,0.10,False),
                                      "small scoop + cone below":(0.5,0.77,0.08,0.07,True)}.items():
        for cname,col in COLS.items():
            seq=hist[:]; last=None
            for t in range(12):
                f,m=face_scoop(sd,cx,cy,rx,ry,col,cone=cone); seq.append(f); last=m
            outs=run(seq); k=up(outs[-1])
            ll=lower_lip()&last; lp=lip_region()&last
            def cut(mm): return float((1-k[mm]).mean())*100 if mm.sum()>40 else float('nan')
            print(f"seed {sd:>2} | {sname:<30} | {cname:<26} | scoop cut {cut(last):5.1f}% | on lips {cut(lp):5.1f}% (px {int(lp.sum()):5d}) | lower lip {cut(ll):5.1f}% (px {int(ll.sum()):5d})")
