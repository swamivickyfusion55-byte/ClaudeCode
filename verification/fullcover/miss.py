import sys
PACK=sys.argv[1]; foodn=sys.argv[2]; fi=int(sys.argv[3]); out=sys.argv[4]; tt=int(sys.argv[5])
sys.argv=[sys.argv[0],PACK,'256']
exec(open('harness.py').read().split('if __name__')[0])
faces=load_faces(); S=256
tm=E.canonical_template(S); rows=(np.arange(S)[:,None]+0.5)/S; support=(tm>0.5)&(rows<0.84)
fr=run(faces[fi],S,FOODS[foodn],np.random.default_rng(11))
t,p,keep,a,f=fr[tt][:5]
g=(a>0.5)&support; miss=g&(keep>=0.5)
ys,xs=np.where(miss); print("object px in support",int(g.sum()),"missed",int(miss.sum()),"(%.1f%%)"%(100*miss.sum()/max(1,g.sum())))
if len(ys): print("missed bbox x %.2f-%.2f y %.2f-%.2f"%(xs.min()/S,xs.max()/S,ys.min()/S,ys.max()/S))
vis=f.copy(); vis[miss]=(0,0,255); vis[(a>0.5)&~support]=(vis[(a>0.5)&~support]*0.5).astype(np.uint8)
k3=cv2.cvtColor((keep*255).astype(np.uint8),cv2.COLOR_GRAY2BGR); 
cv2.imwrite(out,np.hstack([f,vis,k3]))
