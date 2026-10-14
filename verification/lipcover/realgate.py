import sys, cv2, numpy as np
PACK=sys.argv[1]; CLIP=sys.argv[2]; SIZE=int(sys.argv[3]) if len(sys.argv)>3 else 128
sys.path.insert(0,PACK)
import swap_engine as E, config
E.configure(**config.ENGINE_TUNABLES)
from insightface.app import FaceAnalysis
fa=FaceAnalysis(name='buffalo_l',root='/root/.insightface',allowed_modules=['detection','landmark_2d_106'],providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=-1,det_size=(640,640))
c=cv2.VideoCapture(CLIP); ref=None; nref=0; acc=None
i=0; rows=[]
while True:
    ok,f=c.read()
    if not ok: break
    faces=[x for x in fa.get(f) if 440<(x.bbox[0]+x.bbox[2])/2<600]   # person A only
    if not faces: i+=1; continue
    kps=faces[0].kps
    M=E.estimate_norm(kps,SIZE)
    al=cv2.warpAffine(f,M,(SIZE,SIZE),borderMode=cv2.BORDER_REPLICATE)
    keep=E.occluder_keep(al,ref=ref,nref=nref)
    if acc is None or acc.shape!=keep.shape: acc=np.ones_like(keep)
    o,cl=E._P['occ_open'],E._P['occ_close']; a=np.where(keep<acc,o,cl).astype(np.float32); acc=(acc*(1-a)+keep*a).astype(np.float32)
    ref,nref=E.update_face_ref(ref,nref,al,acc)
    rows.append((i,nref,round(float((1-keep).mean())*100,2),round(float((1-acc).mean())*100,2)))
    if i in (39,40,41,45,60,100): cv2.imwrite(f'realgate_{SIZE}_{i}.png',np.hstack([al,cv2.cvtColor((acc*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)]))
    i+=1
print("frame nref raw_cut% smoothed_cut%"); 
for r in rows[::10]+rows[-1:]: print(r)
