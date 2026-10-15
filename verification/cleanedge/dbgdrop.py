import sys
PACK=sys.argv[1]
sys.argv=[sys.argv[0],PACK,'256']
exec(open('harness.py').read().split('if __name__')[0])
faces=load_faces(); S=256
# replicate run() but record raw occluder_keep too
crop=faces[1]; rng=np.random.default_rng(11); tr=E.TrackState(0)
for t,p in enumerate(seq_positions()):
    base=jitter256(crop,rng,S)
    if p is None: f=base
    else: f,a=food(base,p[0],p[1],0.19,FOODS['vanilla'],rng)
    f=np.clip(f,0,255).astype(np.uint8)
    raw=E.occluder_keep(f,ref=tr.occ_ref,nref=tr.occ_nref)
    keep=E.AlignedCompositor.occluder_keep_final(tr,f)
    if p is not None and 40+20<=t<=40+40: print(t-40, "raw cut %.1f%%"%(100*(1-raw).mean()), "final cut %.1f%%"%(100*(1-keep).mean()))
