"""v11.2.207: the skin tone's two MediaPipe models under the MediaPipe the Space now installs. Runs skin_tone's own model
code on insightface's public sample photo (six people; NOT user media) and saves the masks, to compare two installs.
usage: mp_versions.py PACKDIR IMAGE OUT.npz"""
import sys, os
PACK, IMG, OUT = sys.argv[1:4]
sys.path.insert(0, PACK)
import numpy as np, cv2, mediapipe, skin_tone as st
img = cv2.imread(IMG)
rgb = cv2.cvtColor(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
sk = st._skin_prob(rgb)
h, w = img.shape[:2]
side = min(h, w)
sq = cv2.cvtColor(cv2.resize(img[:side, :side], (512, 512)), cv2.COLOR_BGR2RGB)
pp = st._person_prob(sq, (256, 200))
print("mediapipe", mediapipe.__version__, "|", st.backend())
print("skin model: %s | person model: %s" % ("ran, mean skin %.3f" % float(sk[0].mean()) if sk is not None else "NOT available",
                                             "ran, mean %.3f" % float(pp.mean()) if pp is not None else "NOT available"))
np.savez(OUT, skin=sk[0] if sk is not None else np.zeros(1), person=pp if pp is not None else np.zeros(1))
