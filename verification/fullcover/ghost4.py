import sys
PACK=sys.argv[1]; name=sys.argv[2]; fi=int(sys.argv[3])
sys.argv=[sys.argv[0],PACK,'256']
exec(open('harness.py').read().split('if __name__')[0])
faces=load_faces(); S=256
rng=np.random.default_rng(31+fi); tr=E.TrackState(0); snap={}
for t in range(40+30+25):
    base=jitter256(faces[fi],rng,S)
    if 40<=t<70: f,a=food(base,HOLD[0],HOLD[1],0.19,FOODS[name],rng,ang=HOLD[2])
    else: f=base
    f=np.clip(f,0,255).astype(np.uint8)
    if t in (39,94):
        small=E._occ_work(f); st=E._face_stats(small); snap[t]=(st[0],st[3] if len(st)>3 else None, tr.occ_ref.copy() if tr.occ_ref is not None else None, small)
    if t==94:
        raw=E.occluder_keep(f,ref=tr.occ_ref,nref=tr.occ_nref,prev=None)
        cutm=cv2.resize(1-raw,(192,192))>0.5
        ys,xs=np.where(cutm); print("residual cut bbox x %.2f-%.2f y %.2f-%.2f  area %.1f%%"%(xs.min()/192,xs.max()/192,ys.min()/192,ys.max()/192,100*cutm.mean()))
        n39,s39,r39,sm39=snap[39]; n94,s94,r94,sm94=snap[94]
        print("stats at 39:",None if s39 is None else np.round(s39,1)," at 94:",None if s94 is None else np.round(s94,1))
        for nm,ch in (("luma",0),("chroma-r",1),("chroma-b",2)):
            print("  %-8s in residual region: nrm@39 %.2f | nrm@94 %.2f | ref@94 %.2f | ref@39 %.2f"%(nm,n39[...,ch][cutm].mean(),n94[...,ch][cutm].mean(),tr.occ_ref[...,ch][cutm].mean(),r39[...,ch][cutm].mean() if r39 is not None else float('nan')))
        cv2.imwrite(sys.argv[0].replace('.py','.png'),np.hstack([sm94,cv2.cvtColor((cutm*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)]))
    E.AlignedCompositor.occluder_keep_final(tr,f)
