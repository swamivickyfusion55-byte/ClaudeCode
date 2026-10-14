import sys
PACK=sys.argv[1]; MODE=sys.argv[2] if len(sys.argv)>2 else "default"
ns={'__name__':'lib'}; sys.argv=['x',PACK]
exec(open('lip/moving.py').read(), ns); globals().update({k:v for k,v in ns.items() if not k.startswith('__')})
if MODE=="v83":
    E._P.update({"occ_grow":1.0,"occ_open":0.85,"occ_snap":0.02,"occ_snap_eps":0.0005,"occ_feather":0.012,"occ_steep":3.0})
COLS={"vanilla":(235,240,248),"chocolate":(40,55,85),"strawberry":(150,150,235),"butterscotch":(70,150,215),"mango":(60,150,240),"SKIN-LIKE (150,178,205)":(150,178,205),"hair-dark (25,25,30)":(25,25,30)}
SHAPES={"big @mouth":(0.5,0.78,0.20,0.17),"medium @mouth":(0.5,0.78,0.12,0.10),"big, low (chin)":(0.5,0.88,0.20,0.15)}
worst_out=0; rows=[]
for sd in (3,17):
    F=lambda **k: face(seed=sd,**k)
    hist=[F(mouth_open=((i//3)%2==0)) for i in range(60)]
    for sname,(cx,cy,rx,ry) in SHAPES.items():
        for cname,col in COLS.items():
            seq=hist[:]; objs=None
            for t in range(16):
                f,a=aa_scoop(sd,cx,cy,rx,ry,col); seq.append(f); objs=a
            masks=run_mask(seq); ref=run_mask(hist+[F() for _ in range(16)])
            al=np.clip(masks[-1]/np.maximum(ref[-1],1e-3),0,1); ob=(objs>0.5).astype(np.uint8)
            near=cv2.dilate(ob,np.ones((25,25),np.uint8))>0
            valid=(ref[-1]>0.5)
            outside=valid&~near
            core=(cv2.erode(ob,np.ones((9,9),np.uint8))>0)&valid
            ring=(near&~(ob>0))&valid
            rows.append((sd,sname,cname,float(1-al[core].mean()) if core.sum()>30 else np.nan, float(1-al[outside].mean()), float(1-al[ring].mean())))
            worst_out=max(worst_out,rows[-1][4])
print(f"{'object':<26}{'colour':<26}{'object cut':>11}{'ring (12px) cut':>17}{'face far away cut':>19}")
import collections
agg=collections.defaultdict(list)
for sd,sn,cn,c,o,r in rows: agg[(sn,cn)].append((c,r,o))
for (sn,cn),v in agg.items():
    v=np.array(v); print(f"{sn:<26}{cn:<26}{np.nanmean(v[:,0])*100:10.0f}%{np.mean(v[:,1])*100:16.1f}%{np.max(v[:,2])*100:18.2f}%")
print("WORST cut of the face far from the object: %.2f%%"%(worst_out*100))
