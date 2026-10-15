"""realbattery_final with the FINAL keep map (temporal + snap + steepen, via TrackState) instead of the raw detector."""
import sys, os
PACK=sys.argv[1]; sys.path.insert(0,PACK); sys.argv=['x',PACK]
os.chdir('/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip')
src=open('realbattery_final.py').read().rsplit('if __name__',1)[0]
exec(src)
for kv in filter(None, os.environ.get('OVR','').split(',')):
    k_, v_ = kv.split('='); E._P[k_] = float(v_)
def run_gate(seq):
    tr=E.TrackState(0); cuts=[]
    for f in seq:
        f=np.clip(f,0,255).astype(np.uint8)
        cuts.append(E.AlignedCompositor.occluder_keep_final(tr,f).copy())
    return cuts
import numpy as np
faces=load_faces()
d,fp=battery({},faces,seeds=tuple(int(x) for x in os.environ.get('SEEDS','21,22,23').split(',')))
print("== OVR=%s SEEDS=%s\n   detection: "%(os.environ.get('OVR',''),os.environ.get('SEEDS',''))+"  ".join(f"{k[:6]} {v:.0f}%" for k,v in d.items())+f"  (mean {np.mean(list(d.values())):.0f}%)\n   false cuts worst frame: "+"  ".join(f"{k} {v:.1f}" for k,v in fp.items()),flush=True)
