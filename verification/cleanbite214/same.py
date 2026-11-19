import sys, cv2, numpy as np
a=cv2.VideoCapture(sys.argv[1]); b=cv2.VideoCapture(sys.argv[2]); d=[]
while True:
    o1,x=a.read(); o2,y=b.read()
    if not (o1 and o2): break
    d.append(float(np.abs(x.astype(np.int16)-y.astype(np.int16)).mean()))
d=np.array(d); print("%d frames; frames differing %d; mean abs diff %.5f; max %.3f" % (len(d), int((d>0).sum()), d.mean(), d.max()))
