import sys
PACK=sys.argv[1]; sys.path.insert(0,PACK); sys.argv=['x',PACK]
src=open('realbattery_final.py').read().rsplit('if __name__',1)[0]
exec(src)
import numpy as np
faces=load_faces()
for label,over in (("tol 6 / blob .07",{'occ_tol':6,'occ_blob':0.07}),("tol 5 / first .25 / blob .07",{'occ_blob':0.07})):
    d,fp=battery(over,faces,seeds=(21,22,23))
    print("==",label,"\n   detection: "+"  ".join(f"{k[:6]} {v:.0f}%" for k,v in d.items())+f"  (mean {np.mean(list(d.values())):.0f}%)\n   false cuts worst: "+"  ".join(f"{k} {v:.1f}" for k,v in fp.items()),flush=True)
