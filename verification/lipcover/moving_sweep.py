import sys
PACK=sys.argv[1]
ns={'__name__':'lib'}; sys.argv=['x',PACK]
exec(open('lip/moving.py').read(), ns); globals().update({k:v for k,v in ns.items() if not k.startswith('__')})
import itertools, io, contextlib
def evaluate(over, seeds=(3,17)):
    saved={k:E._P[k] for k in over}; E._P.update(over)
    res=[]
    try:
        for sd in seeds:
            F=lambda **k: face(seed=sd,**k)
            hist=[F(mouth_open=((i//3)%2==0)) for i in range(60)]
            path=[(0.5+0.30*(1-t/14.0), 0.78+0.12*(1-t/14.0)) for t in range(15)]+[(0.5,0.78)]*14+[(0.5+0.30*(t/10.0),0.78+0.12*(t/10.0)) for t in range(1,11)]
            seq=hist[:]; objs=[]
            for (cx,cy) in path:
                f,a=aa_scoop(sd,cx,cy,0.18,0.15,(40,55,85)); seq.append(f); objs.append(a)
            masks=run_mask(seq)[60:]; ref=run_mask(hist+[F() for _ in range(len(path))])[60:]
            rows=[]
            for k in range(len(path)):
                ob=(objs[k]>0.5).astype(np.uint8)
                core_in=(cv2.erode(ob,np.ones((9,9),np.uint8))>0)&(ref[k]>0.5)
                edge=(ob>0)&~(cv2.erode(ob,np.ones((9,9),np.uint8))>0)&(ref[k]>0.5)
                ring=(cv2.dilate(ob,np.ones((9,9),np.uint8))>0)&(objs[k]<0.02)&(ref[k]>0.5)
                if core_in.sum()>30 and edge.sum()>30 and ring.sum()>30:
                    rows.append((masks[k][core_in].mean()/ref[k][core_in].mean(), masks[k][edge].mean()/ref[k][edge].mean(), 1-masks[k][ring].mean()/ref[k][ring].mean()))
            res.append(np.mean(rows,axis=0))
    finally:
        E._P.update(saved)
    return np.mean(res,axis=0)
if __name__=="__main__":
    print(f"{'setting':<56}{'interior':>9}{'edge':>7}{'spill':>7}   (interior/edge: face alpha left over the object, 0 is ideal; spill: face next to it wrongly cut)")
    base={}
    G={"occ_grow":1.0,"occ_open":0.85}
    for name,over in (("grow+open0.85, feather .03, no snap",dict(G,occ_snap=0.0,occ_feather=0.03)),
                      ("feather .015, snap r.02 eps.001",dict(G,occ_snap=0.02,occ_snap_eps=0.001,occ_feather=0.015)),
                      ("feather .015, snap r.03 eps.001",dict(G,occ_snap=0.03,occ_snap_eps=0.001,occ_feather=0.015)),
                      ("feather .015, snap r.02 eps.0003",dict(G,occ_snap=0.02,occ_snap_eps=0.0003,occ_feather=0.015)),
                      ("feather .015, snap r.02 eps.001, steep 2",dict(G,occ_snap=0.02,occ_snap_eps=0.001,occ_feather=0.015,occ_steep=2.0)),
                      ("feather .015, snap r.02 eps.001, steep 3",dict(G,occ_snap=0.02,occ_snap_eps=0.001,occ_feather=0.015,occ_steep=3.0)),
                      ("feather .010, snap r.02 eps.0003, steep 3",dict(G,occ_snap=0.02,occ_snap_eps=0.0003,occ_feather=0.010,occ_steep=3.0)),
                      ("feather .010, snap r.03 eps.0003, steep 4",dict(G,occ_snap=0.03,occ_snap_eps=0.0003,occ_feather=0.010,occ_steep=4.0))):
        m=evaluate(over); print(f"{name:<56}{m[0]:9.2f}{m[1]:7.2f}{m[2]:7.2f}")
