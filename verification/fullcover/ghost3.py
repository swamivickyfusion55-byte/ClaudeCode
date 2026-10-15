import sys
PACK=sys.argv[1]; name=sys.argv[2]; fi=int(sys.argv[3])
sys.argv=[sys.argv[0],PACK,'256']
exec(open('harness.py').read().split('if __name__')[0])
faces=load_faces(); S=256
rng=np.random.default_rng(31+fi); tr=E.TrackState(0); last=None
for t in range(40+30+25):
    base=jitter256(faces[fi],rng,S)
    if 40<=t<70: f,a=food(base,HOLD[0],HOLD[1],0.19,FOODS[name],rng,ang=HOLD[2])
    else: f=base
    f=np.clip(f,0,255).astype(np.uint8)
    if t==94:
        r_prev=E.occluder_keep(f,ref=tr.occ_ref,nref=tr.occ_nref,prev=tr.occ_keep)
        r_noprev=E.occluder_keep(f,ref=tr.occ_ref,nref=tr.occ_nref,prev=None)
        print("raw cut with continuity %.1f%%  | without continuity %.1f%%"%(100*(1-r_prev).mean(),100*(1-r_noprev).mean()))
        for k in ('occ_cont','occ_gc','occ_attach','occ_ext_min','occ_quar','occ_nn'):
            if k in E._P:
                old=E._P[k]; E._P[k]=0; r=E.occluder_keep(f,ref=tr.occ_ref,nref=tr.occ_nref,prev=tr.occ_keep); E._P[k]=old
                print("   with %s=0: raw cut %.1f%%"%(k,100*(1-r).mean()))
    E.AlignedCompositor.occluder_keep_final(tr,f)
