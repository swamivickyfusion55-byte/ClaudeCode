import sys
PACK=sys.argv[1] if len(sys.argv)>1 else '/home/user/ClaudeCode/v11.2.81_Brisk'
sys.argv=['x',PACK]
exec(open('lip/diag2.py').read().split('rows=[]')[0])
def case(sd, label, col, cx, cy, rx, ry, hist_open=True, cone=False, n=12):
    F=lambda **k: face(seed=sd,**k)
    hist=[F(mouth_open=((i//3)%2==0) if hist_open else False) for i in range(60)]
    seq=hist[:]; last=None
    for t in range(n):
        f,m=face_scoop(sd,cx,cy,rx,ry,col,cone=cone); seq.append(f); last=m
    outs=run(seq); k=up(outs[-1])
    ll=lower_lip()&last; lp=lip_region()&last; off=last&~lip_region()
    def cut(mm): return float((1-k[mm]).mean())*100 if mm.sum()>40 else float('nan')
    print(f"seed {sd:>2} | {label:<40} | scoop cut {cut(last):5.1f}% | off-lips {cut(off):5.1f}% | lips {cut(lp):5.1f}% | LOWER lip {cut(ll):5.1f}% (px {int(ll.sum())})")
    return outs
if __name__=="__main__":
    for sd in (3,17):
        for cname,col in COLS.items():
            case(sd, f"BIG scoop @mouth, eating hist, {cname}", col, 0.5,0.78,0.20,0.17)
        case(sd, "BIG chocolate scoop, CLOSED-mouth hist", (40,55,85), 0.5,0.78,0.20,0.17, hist_open=False)
