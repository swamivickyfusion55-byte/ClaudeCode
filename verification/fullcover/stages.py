"""Per-stage coverage of the object at hold frame: notskin, novel, both, after opening, final keep."""
import sys
PACK=sys.argv[1]; foodn=sys.argv[2]; fis=[int(x) for x in sys.argv[3].split(',')]; tt=int(sys.argv[4])
sys.argv=[sys.argv[0],PACK,'256']
exec(open('harness.py').read().split('if __name__')[0])
faces=load_faces(); S=256
for fi in fis:
    rng=np.random.default_rng(11); crop=faces[fi]; tr=E.TrackState(0)
    for t,p in enumerate(seq_positions()[:tt+1]):
        base=jitter256(crop,rng,S)
        if p is None: f=base; a=np.zeros((S,S),np.float32)
        else: f,a=food(base,p[0],p[1],0.19,FOODS[foodn],rng,ang=HOLD[2])
        f=np.clip(f,0,255).astype(np.uint8)
        if t==tt:
            small=E._occ_work(f); h,w=small.shape[:2]
            excl=None
            nrm,inface,ys=E._face_stats(small)[:3]
            if E._P.get('occ_plane',0)>0 and tr.occ_ref is not None:
                nrm=nrm-E._illum_plane(nrm,tr.occ_ref,inface)
            notskin=((np.sqrt(nrm[:,:,1]**2+nrm[:,:,2]**2)>E._P['occ_chroma'])|(np.abs(nrm[:,:,0])>E._P['occ_luma']))
            ref=tr.occ_ref; tol=int(E._P['occ_tol'])
            kk=cv2.getStructuringElement(cv2.MORPH_RECT,(2*tol+1,2*tol+1)); lo=cv2.erode(ref,kk); hi=cv2.dilate(ref,kk)
            dist=np.maximum(np.maximum(lo-nrm,nrm-hi),0.0); novel=((np.sqrt(dist[:,:,1]**2+dist[:,:,2]**2)>E._P['occ_ref_chroma'])|(dist[:,:,0]>E._P['occ_ref_luma']))
            ag=cv2.resize(a,(w,h))>0.5; sc=ag&inface&(ys>=0.46)[:,None].repeat(w,1)
            # scoop = alpha region whose rows < cone start: approximate by circle
            yy,xx=np.mgrid[0:h,0:w]; cx,cy=HOLD[0]*w,HOLD[1]*h; scoop=(((xx-cx)/(0.19*w))**2+((yy-cy)/(0.19*0.92*h))**2<1)&inface
            n=max(1,scoop.sum())
            print(f"face {fi} {foodn}: scoop px {int(n)} | notskin {100*(notskin&scoop).sum()/n:.0f}%  novel {100*(novel&scoop).sum()/n:.0f}%  both {100*(notskin&novel&scoop).sum()/n:.0f}% | skin sigma-capped stats median-shift vs clean? ")
        keep=E.AlignedCompositor.occluder_keep_final(tr,f)
        if t==tt:
            kw=cv2.resize(keep,(w,h)); print("   final cut of scoop %.0f%%"%(100*(kw[scoop]<0.5).mean() if scoop.sum() else 0))
