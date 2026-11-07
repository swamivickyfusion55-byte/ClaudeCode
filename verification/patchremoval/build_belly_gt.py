"""Turn the one-go job's own positions into a video_test ground-truth set (the belly clip, the engine's real input)."""
import sys, os, types, shutil, json
# usage: build_belly_gt.py CLIPDIR OUTDIR [PACKDIR] [NAME] [pent=auto] [parts=6,..]   (v11.2.140: the pack and the set name can be given)
# v11.2.143: pent=auto - as the app does it: Preview belly on frame 0 (the automatic pentagon), then the job with the pentagon and the
# preview's marks it returned; parts=... - score only these marks of the clip (truth.npz parts ids), e.g. the soft patch (6)
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else "/home/user/ClaudeCode/v11.2.117_OneGoBelly")
import numpy as np, cv2
import belly_clear as BC, clean_belly as CB
CLIP, OUT = sys.argv[1], sys.argv[2]
NAME = sys.argv[4] if len(sys.argv) > 4 else "onego"
KW = dict(a.split("=", 1) for a in sys.argv[5:] if "=" in a)
if KW.get("pent") == "auto":
    f0 = cv2.VideoCapture(os.path.join(CLIP, "clip.mp4")).read()[1]
    lm0 = BC.still_pose().detect(f0)
    _, _, info0 = CB.preview_frame(f0, lm0)
    print("preview: pentagon %s, %s marks" % (info0.get("pentagon_source"), len(info0.get("marks_body") or [])), flush=True)
    pos = CB.build_clean_belly(os.path.join(CLIP, "clip.mp4"), pentagon_body=BC.parse_pentagon(info0.get("pentagon_body")),
                               marks_body=BC.parse_marks(info0.get("marks_body")))
else:
    pos = CB.build_clean_belly(os.path.join(CLIP, "clip.mp4"))
T = np.load(os.path.join(CLIP, "truth.npz"))
n = T["mask"].shape[0]
shutil.copy(os.path.join(CLIP, "clip.mp4"), os.path.join(OUT, f"pr_{NAME}.mp4"))
shutil.copy(os.path.join(CLIP, "clean.mp4"), os.path.join(OUT, f"pr_{NAME}_clean.mp4"))
p = pos[0]
ref = cv2.imread(p["image"]); rm = cv2.imread(p["mask"], cv2.IMREAD_GRAYSCALE)
mark = np.isin(T["parts"], [int(v) for v in KW["parts"].split(",")]) if KW.get("parts") else T["mask"]
np.savez_compressed(os.path.join(OUT, f"pr_{NAME}_gt.npz"), mark=mark, occ=np.zeros_like(T["mask"]), ref=ref, refmask=rm,
                    ref_t=np.float64(p["t"]))
print("RESULT positions", len(pos), "anchor t %.3f s, mask px %d" % (p["t"], int((rm > 127).sum())), flush=True)
os._exit(0)
