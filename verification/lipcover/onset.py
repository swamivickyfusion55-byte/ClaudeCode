import sys
PACK=sys.argv[1]; sys.argv=['x',PACK]
exec(open('lip/diag3.py').read().split('if __name__')[0])
res=[]
for sd in (3,17,42):
    F=lambda **k: face(seed=sd,**k)
    hist=[F(mouth_open=((i//3)%2==0)) for i in range(60)]
    for cname,col in (("vanilla",(235,240,248)),("chocolate",(40,55,85)),("butterscotch",(70,150,215))):
        for grow_after in (6,12,20):
            seq=hist[:]
            for t in range(grow_after):
                f,m=face_scoop(sd,0.5,0.77,0.08,0.07,col); seq.append(f)           # small scoop at the lips: not cut
            for t in range(14):
                f,m=face_scoop(sd,0.5,0.78,0.20,0.17,col); seq.append(f); last=m   # then the whole scoop arrives
            outs=run(seq); k=up(outs[-1])
            ll=lower_lip()&last
            cut=float((1-k[ll]).mean())*100; off=float((1-k[last&~lip_region()]).mean())*100
            res.append(cut)
            print(f"seed {sd:>2} | {cname:<13} | small scoop sits {grow_after:>2} frames, then the whole scoop | scoop away from lips cut {off:5.1f}% | LOWER lip cut {cut:5.1f}%")
print("mean lower-lip cut: %.1f%%"%np.mean(res))
