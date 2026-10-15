"""An object is cut for 30 frames, then vanishes at once: how many frames until the face is fully back? (no stuck cut)"""
import sys
PACK=sys.argv[1]; sys.argv=[sys.argv[0],PACK,'256']
exec(open('harness.py').read().split('if __name__')[0])
faces=load_faces()[:4]; S=256; worst=[]
for name in ("vanilla","chocolate","mint"):
  for fi,face in enumerate(faces):
    rng=np.random.default_rng(31+fi); tr=E.TrackState(0); cut=[]
    for t in range(40+30+25):
        base=jitter256(face,rng,S)
        if 40<=t<70: f,a=food(base,0.5,0.76,0.19,FOODS[name],rng)
        else: f=base
        f=np.clip(f,0,255).astype(np.uint8)
        k=E.AlignedCompositor.occluder_keep_final(tr,f); cut.append(float((1-k).mean())*100)
    after=cut[70:]; n_back=next((i for i,c in enumerate(after) if c<0.5),len(after))
    worst.append(n_back); print(name,fi,"cut at last object frame %.1f%%"%cut[69],"| frames until <0.5%% cut: %d"%n_back)
print("max frames until the face is back:",max(worst))
