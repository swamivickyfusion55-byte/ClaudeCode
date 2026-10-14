import sys, cv2, numpy as np
PACK=sys.argv[1]; CLIP=sys.argv[2]; TH=float(sys.argv[3]); SIZE=256
sys.path.insert(0,PACK)
import swap_engine as E, config
E.configure(**config.ENGINE_TUNABLES); E._P['occ_chroma']=TH; E._P['occ_luma']=TH
from insightface.app import FaceAnalysis
fa=FaceAnalysis(name='buffalo_l',root='/root/.insightface',allowed_modules=['detection','landmark_2d_106'],providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=-1,det_size=(640,640))
c=cv2.VideoCapture(CLIP); ref=None; nref=0; acc=None; i=0; rows=[]
while True:
    ok,f=c.read()
    if not ok: break
    faces=[x for x in fa.get(f) if 440<(x.bbox[0]+x.bbox[2])/2<600]
    if not faces: i+=1; continue
    M=E.estimate_norm(faces[0].kps,SIZE); al=cv2.warpAffine(f,M,(SIZE,SIZE),borderMode=cv2.BORDER_REPLICATE)
    keep=E.occluder_keep(al,ref=ref,nref=nref)
    if acc is None or acc.shape!=keep.shape: acc=np.ones_like(keep)
    o,cl=E._P['occ_open'],E._P['occ_close']; a=np.where(keep<acc,o,cl).astype(np.float32); acc=(acc*(1-a)+keep*a).astype(np.float32)
    ref,nref=E.update_face_ref(ref,nref,al,acc); rows.append((i,float((1-acc).mean())*100)); i+=1
r=np.array(rows)
print(f"threshold {TH}: frames 0-39 (clean) max cut {r[r[:,0]<40,1].max():.2f}% | frames 45-109 (scoop) mean cut {r[r[:,0]>=45,1].mean():.1f}% max {r[r[:,0]>=45,1].max():.1f}%")
