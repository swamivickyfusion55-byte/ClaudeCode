import sys, time, numpy as np, cv2
PACK=sys.argv[1]; sys.path.insert(0,PACK); sys.argv=['x']
import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
exec(open('/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip/realset.py').read())
sys.path.insert(0,'/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip86'); from food import food
faces=load_faces()[:3]; rng=np.random.default_rng(5)
OFF=dict(occ_ext_min=0,occ_stat_hold=0,occ_nn=0,occ_quar_on=0,occ_attach=0,occ_freeze=0); ON=dict(occ_ext_min=0.2,occ_stat_hold=1,occ_nn=1,occ_quar_on=1,occ_attach=0.1,occ_freeze=1)
for S in (256,512):
    for tag,over in (("v11.2.85-equivalent",OFF),("v11.2.86",ON)):
        E._P.update(over)
        for pos,label in (((0.27,0.60,125),"object at the cheek"),(None,"no object")):
            ts=[]
            for crop0 in faces:
                crop=cv2.resize(crop0,(S,S),interpolation=cv2.INTER_CUBIC) if S!=256 else crop0
                tr=E.TrackState(0)
                for t in range(44):
                    f=jitter(crop,rng)
                    if pos and t>=20: f,_=food(f,pos[0],pos[1],0.19,(235,240,248),rng,ss=2,ang=pos[2])
                    f=np.clip(f,0,255).astype(np.uint8)
                    t0=time.perf_counter(); E.AlignedCompositor.occluder_keep_final(tr,f); dt=(time.perf_counter()-t0)*1000
                    if t>=30: ts.append(dt)
            print(f"crop {S}px  {tag:<22} {label:<20} {np.median(ts):6.1f} ms/call (median)")
