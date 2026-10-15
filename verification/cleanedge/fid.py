import sys, cv2, numpy as np
exec(open('cheekmetrics.py').read().split('def metrics')[0])
ref = frames("/tmp/claude-0/bench_work/ck_p84_Ultra_swell/out/BENCHCK_/final.mp4")
for lab in sys.argv[1:]:
    o = frames(f"/tmp/claude-0/bench_work/{lab}/out/BENCHCK_/final.mp4"); n = min(len(o), len(ref))
    v = []
    for i in range(15, 81):
        v.append(np.mean([np.abs(o[i][b[1]:b[3], b[0]:b[2]].astype(np.float32) - ref[i][b[1]:b[3], b[0]:b[2]].astype(np.float32)).mean() for b in band_boxes()]))
    print(lab, "mean |out - Ultra out| in the cheek band: %.2f" % np.mean(v))
