"""SDOS-206: a picture sheet from the job outputs on the synthetic object clips (make_objects.py; NOT user media): the footage,
11.2.205 and 11.2.206 side by side on chosen frames, cropped to the face.
usage: sheet206.py OUT.png clip:frame[,frame...] ..."""
import sys, glob, cv2, numpy as np
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
def frames(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, f = c.read()
        if not ok: return o
        o.append(f)
rows = []
for arg in sys.argv[2:]:
    clip, ts = arg.split(":")
    src = frames(B + "obj_%s.mp4" % clip)
    a = frames(glob.glob("/tmp/claude-0/bench_work/o205_%s/out/*/final.mp4" % clip)[0])
    b = frames(glob.glob("/tmp/claude-0/bench_work/o206_%s/out/*/final.mp4" % clip)[0])
    for t in [int(x) for x in ts.split(",")]:
        crop = lambda f: cv2.resize(f[150:650, 400:880], (288, 300), interpolation=cv2.INTER_AREA)
        cells = [crop(src[t]), crop(a[t]), crop(b[t])]
        lab = ["footage", "v11.2.205", "v11.2.206"]
        for c, l in zip(cells, lab):
            cv2.rectangle(c, (0, 0), (288, 26), (255, 255, 255), -1)
            cv2.putText(c, "%s  %s f%d" % (l, clip, t), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
        rows.append(np.concatenate([np.pad(c, ((3, 3), (3, 3), (0, 0)), constant_values=235) for c in cells], 1))
cv2.imwrite(sys.argv[1], np.concatenate(rows, 0))
print("written", sys.argv[1])
