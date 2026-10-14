import numpy as np, cv2
S=512
def _shade(seed):
    r=np.random.default_rng(seed)
    y,x=np.mgrid[0:S,0:S].astype(np.float32)
    g=1.0+0.18*((x/S)-0.5)-0.12*((y/S)-0.5)          # directional light
    for _ in range(5):                                # soft blobs (cheeks, nose)
        cx,cy,sg,am=r.uniform(.2,.8)*S,r.uniform(.2,.8)*S,r.uniform(.12,.3)*S,r.uniform(-.10,.10)
        g+=am*np.exp(-(((x-cx)**2+(y-cy)**2)/(2*sg*sg)))
    return g
def _tex(seed):
    r=np.random.default_rng(seed+7)
    t=cv2.GaussianBlur(r.standard_normal((S,S)).astype(np.float32),(0,0),1.3)*9.0
    t+=cv2.GaussianBlur(r.standard_normal((S,S)).astype(np.float32),(0,0),4.0)*7.0
    return t
def face(ice=False,ice_col=(235,240,248),spoon=False,beard=False,glasses=False,
         dark=False,shadow=False,ice_y=0.80,mouth_open=False,gain=1.0,seed=3):
    skin=np.array((70,84,106),np.float32) if dark else np.array((118,138,172),np.float32)
    img=np.ones((S,S,3),np.float32)*skin
    sh=_shade(seed)[:,:,None]; img*=sh
    img+=_tex(seed)[:,:,None]
    # Independent low-frequency variation PER CHANNEL: real skin is blotchy in
    # hue (redness over the cheeks and nose), not just in brightness, and a
    # single grey shading field gives an unrealistically low Cr/Cb spread -
    # which silently changes what any chroma threshold means.
    # Generated small and UPSCALED, not blurred: blurring standard normals at
    # sigma 28 collapses the amplitude by ~100x, so the intended variation was
    # not actually there (measured Cr std 2.1 instead of the 4-7 a real face
    # has). Low-resolution noise resized up keeps its amplitude.
    r2=np.random.default_rng(seed+21)
    for c,amp in enumerate((7.0, 5.5, 4.5)):
        lo=r2.standard_normal((12,12)).astype(np.float32)
        img[:,:,c]+=cv2.resize(lo,(S,S),interpolation=cv2.INTER_CUBIC)*amp
    for cx in (0.35,0.65):
        cv2.ellipse(img,(int(cx*S),int(0.35*S)),(int(0.07*S),int(0.035*S)),0,0,360,(58,60,68),-1)
    ry=int(0.12*S) if mouth_open else int(0.065*S)
    cv2.ellipse(img,(S//2,int(0.72*S)),(int(0.17*S),ry),0,0,360,
                (66,72,120) if dark else (94,102,176),-1)
    if mouth_open: cv2.ellipse(img,(S//2,int(0.74*S)),(int(0.10*S),int(0.06*S)),0,0,360,(28,28,38),-1)
    if beard:
        m=np.zeros((S,S),np.uint8)
        cv2.ellipse(m,(S//2,int(0.84*S)),(int(0.30*S),int(0.20*S)),0,0,360,255,-1)
        cv2.ellipse(m,(S//2,int(0.70*S)),(int(0.16*S),int(0.07*S)),0,0,360,0,-1)
        b=np.random.default_rng(11).standard_normal((S,S)).astype(np.float32)*7
        img[m>0]=np.stack([42+b,46+b,56+b],-1)[m>0]
    if glasses:
        for cx in (0.35,0.65):
            cv2.ellipse(img,(int(cx*S),int(0.35*S)),(int(0.12*S),int(0.08*S)),0,0,360,(28,30,34),int(0.015*S))
    if shadow: img[:,int(0.6*S):]*=0.60
    if ice:
        cv2.ellipse(img,(S//2,int(ice_y*S)),(int(0.20*S),int(0.17*S)),0,0,360,ice_col,-1)
        cv2.rectangle(img,(int(0.40*S),int((ice_y+0.06)*S)),(int(0.60*S),S),(150,178,205),-1)
    if spoon:
        cv2.ellipse(img,(S//2,int(0.74*S)),(int(0.10*S),int(0.055*S)),0,0,360,(200,200,205),-1)
        cv2.rectangle(img,(int(0.47*S),int(0.78*S)),(int(0.53*S),S),(190,190,195),-1)
    img*=gain
    img+=np.random.default_rng(seed+1).standard_normal((S,S,3)).astype(np.float32)*2.5
    return np.clip(img,0,255).astype(np.uint8)
