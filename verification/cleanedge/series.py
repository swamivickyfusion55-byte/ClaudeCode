import sys
PACK=sys.argv[1]; foodn=sys.argv[2]; nf=int(sys.argv[3])
sys.argv=[sys.argv[0],PACK,'256']
exec(open('harness.py').read().split('if __name__')[0])
faces=load_faces(); S=256
tm=E.canonical_template(S); rows=(np.arange(S)[:,None]+0.5)/S; support=(tm>0.5)&(rows<0.84)
for fi in range(nf):
    fr=run(faces[fi],S,FOODS[foodn],np.random.default_rng(11))
    s=[]
    for t,p,keep,a,f in fr[40:]:
        g=(a>0.5)&support
        s.append(int(100*(keep[g]<0.5).mean()) if g.sum()>200 else -1)
    print(foodn,fi,s)
