"""Real-face calibration set: the 6 real faces of the insightface sample photo, aligned like the engine does,
clean sequences (jitter, light changes) and sequences with a synthetic object pasted at the mouth/chin."""
import sys, cv2, numpy as np
SIZE=256
def load_faces(cache='/tmp/claude-0/realfaces.npz'):
    import os
    if os.path.exists(cache):
        z=np.load(cache); return [z[k] for k in sorted(z.files)]
    sys.path.insert(0,'/home/user/ClaudeCode/v11.2.81_Brisk')
    import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
    from insightface.app import FaceAnalysis
    fa=FaceAnalysis(name='buffalo_l',root='/root/.insightface',allowed_modules=['detection','landmark_2d_106'],providers=['CPUExecutionProvider'])
    fa.prepare(ctx_id=-1,det_size=(640,640))
    img=cv2.imread('/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg')
    crops=[]
    for x in sorted(fa.get(img),key=lambda f:f.bbox[0]):
        M=E.estimate_norm(x.kps,SIZE); crops.append(cv2.warpAffine(img,M,(SIZE,SIZE),borderMode=cv2.BORDER_REPLICATE))
    np.savez(cache,**{f"f{i}":c for i,c in enumerate(crops)}); return crops
def jitter(crop,rng,amp=1.5):
    ang=rng.normal(0,0.7); sc=1+rng.normal(0,0.008); tx,ty=rng.normal(0,amp,2)
    M=cv2.getRotationMatrix2D((SIZE/2,SIZE/2),ang,sc); M[:,2]+=(tx,ty)
    out=cv2.warpAffine(crop,M,(SIZE,SIZE),borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out+=rng.standard_normal(out.shape).astype(np.float32)*1.8
    return out
def scoop(crop,cx,cy,rx,ry,col,rng,t=0):
    S=SIZE; yy,xx=np.mgrid[0:S,0:S].astype(np.float32)
    cx=cx+0.004*np.sin(t/3.0); cy=cy+0.004*np.cos(t/4.0)
    d=np.sqrt(((xx-cx*S)/(rx*S))**2+((yy-cy*S)/(ry*S))**2); a=np.clip((1.0-d)*(rx*S)/1.2+0.5,0,1)
    sh=1.0-0.2*np.clip(((xx-cx*S)/(rx*S))*0.6+((yy-cy*S)/(ry*S))*0.6,-1,1)
    colf=np.array(col,np.float32)[None,None,:]*sh[:,:,None]
    out=crop*(1-a[:,:,None])+colf*a[:,:,None]; return out,a
COLS={"vanilla":(235,240,248),"chocolate":(40,55,85),"strawberry":(150,150,235),"butterscotch":(70,150,215),"mango":(60,150,240),"pistachio":(120,200,150),"white-pink":(225,215,245)}
