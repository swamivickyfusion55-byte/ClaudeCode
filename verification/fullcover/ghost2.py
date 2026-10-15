import sys
PACK=sys.argv[1]; name=sys.argv[2]; fi=int(sys.argv[3]); outp=sys.argv[4]
sys.argv=[sys.argv[0],PACK,'256']
exec(open('harness.py').read().split('if __name__')[0])
faces=load_faces(); S=256
rng=np.random.default_rng(31+fi); tr=E.TrackState(0); cut=[]; keeps=[]
for t in range(40+30+25):
    base=jitter256(faces[fi],rng,S)
    if 40<=t<70: f,a=food(base,HOLD[0],HOLD[1],0.19,FOODS[name],rng,ang=HOLD[2])
    else: f=base
    f=np.clip(f,0,255).astype(np.uint8)
    k=E.AlignedCompositor.occluder_keep_final(tr,f); cut.append(round(float((1-k).mean())*100,1)); keeps.append((f,k))
print(name,fi,"cut% after the object vanished:",cut[68:95])
f,k=keeps[94]; 
cv2.imwrite(outp,np.hstack([f,cv2.cvtColor((k*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)]))
