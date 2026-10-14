import sys, cv2, numpy as np
PACK=sys.argv[1]; CLIP=sys.argv[2]; SIZE=256
sys.path.insert(0,PACK)
import swap_engine as E, config
E.configure(**config.ENGINE_TUNABLES); P=E._P
from insightface.app import FaceAnalysis
fa=FaceAnalysis(name='buffalo_l',root='/root/.insightface',allowed_modules=['detection','landmark_2d_106'],providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=-1,det_size=(640,640))
c=cv2.VideoCapture(CLIP); ref=None; nref=0
for i in range(70):
    ok,f=c.read()
    faces=[x for x in fa.get(f) if 440<(x.bbox[0]+x.bbox[2])/2<600]
    kps=faces[0].kps; M=E.estimate_norm(kps,SIZE); al=cv2.warpAffine(f,M,(SIZE,SIZE),borderMode=cv2.BORDER_REPLICATE)
    keep=E.occluder_keep(al,ref=ref,nref=nref); ref,nref=E.update_face_ref(ref,nref,al,keep)
    if i==65:
        small=E._occ_work(al); h=small.shape[0]
        nrm,inface,ys=E._face_stats(small)
        ch=np.sqrt(nrm[:,:,1]**2+nrm[:,:,2]**2); lu=np.abs(nrm[:,:,0])
        # scoop pixels: where the crop is the flat reddish colour
        b,g,r=[small[:,:,k].astype(int) for k in range(3)]
        sc=(abs(b-70*0.9)<28)&(abs(g-80*0.9)<28)&(abs(r-140*0.9)<35)&(ys>0.55)
        print("scoop px:",int(sc.sum()),"in-face area:",int(inface.sum()))
        print("scoop chroma z (vs skin): median %.2f  p10 %.2f | luma z: median %.2f"%(np.median(ch[sc]),np.percentile(ch[sc],10),np.median(lu[sc])))
        d_ch=np.sqrt((nrm[:,:,1]-ref[:,:,1])**2+(nrm[:,:,2]-ref[:,:,2])**2); d_lu=np.abs(nrm[:,:,0]-ref[:,:,0])
        print("vs own history: chroma dist median %.2f  luma dist median %.2f"%(np.median(d_ch[sc]),np.median(d_lu[sc])))
        ycc=cv2.cvtColor(small,cv2.COLOR_BGR2YCrCb).astype(np.float32)
        zone=inface&(((ys>0.14)&(ys<0.28))); print("forehead zone Y std %.1f Cr std %.1f Cb std %.1f ; thresholds are %.1f sigma"%(ycc[:,:,0][zone].std(),ycc[:,:,1][zone].std(),ycc[:,:,2][zone].std(),P['occ_chroma']))
        print("scoop Y,Cr,Cb:",[round(float(ycc[:,:,k][sc].mean()),1) for k in range(3)], " skin zone median Y,Cr,Cb:",[round(float(np.median(ycc[:,:,k][zone])),1) for k in range(3)])
