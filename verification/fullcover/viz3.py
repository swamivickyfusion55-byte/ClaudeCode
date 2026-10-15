import sys
PACK=sys.argv[1]; foodn=sys.argv[2]; fis=[int(x) for x in sys.argv[3].split(',')]; out=sys.argv[4]; frames=[int(x) for x in sys.argv[5].split(',')]
sys.argv=[sys.argv[0],PACK,'256']
exec(open('harness.py').read().split('if __name__')[0])
faces=load_faces(); S=256
rowsimg=[]
for fi in fis:
    fr=run(faces[fi],S,FOODS[foodn],np.random.default_rng(11))
    tiles=[]
    for t in frames:
        _,p,keep,a,f=fr[t][:5]
        k=cv2.cvtColor((keep*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        cnt=cv2.findContours((fr[t][5]>0.5).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        k2=k.copy(); cv2.drawContours(k2,cnt,-1,(0,0,255),1)
        f2=f.copy(); cv2.drawContours(f2,cnt,-1,(0,255,0),1)
        tiles+=[f2,k2]
    rowsimg.append(np.hstack(tiles))
cv2.imwrite(out,np.vstack(rowsimg)); print(out)
