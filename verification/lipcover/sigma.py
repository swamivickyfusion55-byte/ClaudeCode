import sys, cv2, numpy as np
sys.path.insert(0,'/home/user/ClaudeCode/v11.2.81_Brisk')
import swap_engine as E, config
E.configure(**config.ENGINE_TUNABLES)
from insightface.app import FaceAnalysis
fa=FaceAnalysis(name='buffalo_l',root='/root/.insightface',allowed_modules=['detection','landmark_2d_106'],providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=-1,det_size=(640,640))
img=cv2.imread('/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg')
print(img.shape)
def mad(x): return 1.4826*float(np.median(np.abs(x-np.median(x))))
for n,x in enumerate(sorted(fa.get(img),key=lambda f:f.bbox[0])):
    size=192
    M=E.estimate_norm(x.kps,size); al=cv2.warpAffine(img,M,(size,size),borderMode=cv2.BORDER_REPLICATE)
    ycc=cv2.cvtColor(al,cv2.COLOR_BGR2YCrCb).astype(np.float32)
    yy,cr,cb=ycc[:,:,0],ycc[:,:,1],ycc[:,:,2]
    ys,xs=np.mgrid[0:size,0:size].astype(np.float32); ys=(ys+.5)/size; xs=(xs+.5)/size
    inface=E.canonical_template(size)>0.5
    zone=inface&(((ys>0.14)&(ys<0.28))|((ys>0.44)&(ys<0.62)&((xs<0.33)|(xs>0.67))))
    my=float(np.median(yy[zone]))
    print(f"face {n} bbox x={int(x.bbox[0])}: zone n={int(zone.sum())} medianY={my:5.0f} | Y std {yy[zone].std():5.1f} MAD-sigma {mad(yy[zone]):5.1f} | Cr std {cr[zone].std():4.1f} MAD {mad(cr[zone]):4.1f} | Cb std {cb[zone].std():4.1f} MAD {mad(cb[zone]):4.1f} | floor term {1.4*max(2,0.06*my):4.1f}")
