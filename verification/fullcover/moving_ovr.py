import sys, types
PACK=sys.argv[1]; sys.argv=['x',PACK]
exec(open('lip/diag3.py').read().split('if __name__')[0])
import os
for kv in filter(None, os.environ.get('OVR','').split(',')):
    k_, v_ = kv.split('='); E._P[k_] = float(v_)
def aa_scoop(sd, cx, cy, rx, ry, col, mouth_open=False):
    """scoop with an antialiased, slightly soft edge (like a real object after video compression)."""
    img=face(seed=sd,mouth_open=mouth_open).astype(np.float32)
    yy,xx=np.mgrid[0:S,0:S].astype(np.float32)
    d=np.sqrt(((xx-cx*S)/(rx*S))**2+((yy-cy*S)/(ry*S))**2)
    a=np.clip((1.0-d)*(rx*S)/1.5+0.5,0,1)                     # ~1.5 px soft edge
    sh=1.0-0.22*np.clip(((xx-cx*S)/(rx*S))*0.6+((yy-cy*S)/(ry*S))*0.6,-1,1)
    colf=np.array(col,np.float32)[None,None,:]*sh[:,:,None]
    img=img*(1-a[:,:,None])+colf*a[:,:,None]
    img+=np.random.default_rng(sd+5).standard_normal((S,S,3)).astype(np.float32)*2.0
    return np.clip(img,0,255).astype(np.uint8), a
def run_mask(seq):
    comp=E.AlignedCompositor(object())
    tr=E.TrackState(0); face_=types.SimpleNamespace(landmark_2d_106=None)
    M=np.array([[1,0,0],[0,1,0]],np.float32); outs=[]
    for f in seq:
        outs.append(comp.build_mask(face_,M,S,track=tr,occlusion_guard=0.0,aligned_target=f))
    return outs
def metrics(alpha_face, a_obj):
    core=a_obj>0.98
    ring=(cv2.dilate((a_obj>0.5).astype(np.uint8),np.ones((13,13),np.uint8))>0)&(a_obj<0.02)&(cv2.dilate((a_obj>0.5).astype(np.uint8),np.ones((3,3),np.uint8))==0)
    ring_near=(cv2.dilate((a_obj>0.5).astype(np.uint8),np.ones((15,15),np.uint8))>0)&~(cv2.dilate((a_obj>0.5).astype(np.uint8),np.ones((5,5),np.uint8))>0)
    # face alpha is measured against the no-object mask (1 on the face region)
    return float(alpha_face[core].mean()) if core.sum()>30 else float('nan')
if __name__=="__main__":
    col=(40,55,85)
    for sd in (3,17):
        F=lambda **k: face(seed=sd,**k)
        hist=[F(mouth_open=((i//3)%2==0)) for i in range(60)]
        # scoop approaches the mouth from the lower right, holds, then leaves
        path=[(0.5+0.30*(1-t/14.0), 0.78+0.12*(1-t/14.0)) for t in range(15)]+[(0.5,0.78)]*14+[(0.5+0.30*(t/10.0),0.78+0.12*(t/10.0)) for t in range(1,11)]
        seq=hist[:]; objs=[]
        for (cx,cy) in path:
            f,a=aa_scoop(sd,cx,cy,0.18,0.15,col, mouth_open=False); seq.append(f); objs.append(a)
        masks=run_mask(seq)[60:]
        # reference: mask over the same pixels when the face is clean
        ref=run_mask(hist+[F() for _ in range(len(path))])[60:]
        print(f"seed {sd}: face-alpha left over the object's interior (0 = object fully in front, 1 = face painted over it)")
        rows=[]
        for k in range(len(path)):
            core=(objs[k]>0.98)&(ref[k]>0.5)
            ring=(cv2.dilate((objs[k]>0.5).astype(np.uint8),np.ones((9,9),np.uint8))>0)&(objs[k]<0.02)&(ref[k]>0.5)   # face pixels within ~4px of the object
            core_in=(cv2.erode((objs[k]>0.5).astype(np.uint8),np.ones((9,9),np.uint8))>0)&(ref[k]>0.5)
            edge_band=(objs[k]>0.5)&~(cv2.erode((objs[k]>0.5).astype(np.uint8),np.ones((9,9),np.uint8))>0)&(ref[k]>0.5)
            rows.append((k, float(masks[k][core_in].mean()/max(1e-6,ref[k][core_in].mean())) if core_in.sum()>30 else np.nan,
                            float(masks[k][edge_band].mean()/max(1e-6,ref[k][edge_band].mean())) if edge_band.sum()>30 else np.nan,
                            float(1-masks[k][ring].mean()/max(1e-6,ref[k][ring].mean())) if ring.sum()>30 else np.nan))
        for k,a,b,c in rows[::3]: print(f"   frame {k:2d}: object interior {a:4.2f} | object's edge band (4 px) {b:4.2f} | face next to the object wrongly cut {c:4.2f}")
        arr=np.array([[r[1],r[2],r[3]] for r in rows]); print(f"   MEAN over {len(rows)} frames: interior {np.nanmean(arr[:,0]):.2f}  edge band {np.nanmean(arr[:,1]):.2f}  spill {np.nanmean(arr[:,2]):.2f}   | frames with interior > 0.2: {int(np.nansum(arr[:,0]>0.2))}")
