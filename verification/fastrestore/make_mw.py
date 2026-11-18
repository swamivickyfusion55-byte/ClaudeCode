"""v11.2.212 (SDOS-212): man-woman contact clips - the clips of make_kiss.py / make_cross.py with one woman and one man (the earlier
ones are two men). Faces of insightface's public sample photo (f0, f1, f5 women; f2, f3, f4 men - checked by eye and by its own
gender model), cut-outs as make_kiss.py, NOT user footage. Writes kiss_<name>.mp4 and kiss_<name>_gt.npz (run from the output folder).
  mw_kiss    woman (f0, left) and man (f2, right, in front): heads tilted together, faces pressed (as smooch)
  mw_kissb   man (f3, left) behind a woman (f5, right, in front), as smooch
  mw_cross   woman (f0) passes behind a man (f2) and both end on the other side
  mw_hug     woman (f1) and man (f3) cheek to cheek, the pair turns half a circle: they end with the sides swapped
usage: make_mw.py [names ...]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_kiss as K
import make_cross as C
K.FACE.update({"f0": (58, 258, 167, 396), "f5": (1132, 269, 1227, 397)})
ALL = {
    "mw_kiss": lambda: K.build("mw_kiss", a="f0", b="f2", gap_end=135.0, rot=18.0, pulse=0.02),
    "mw_kissb": lambda: K.build("mw_kissb", a="f3", b="f5", gap_end=135.0, rot=18.0, pulse=0.02),
    "mw_cross": lambda: C.build_path("mw_cross", C.cross, a="f0", b="f2"),
    "mw_hug": lambda: C.build_path("mw_hug", C.hug, a="f1", b="f3"),
}
if __name__ == "__main__":
    for nm in (sys.argv[1:] or list(ALL)):
        ALL[nm]()
