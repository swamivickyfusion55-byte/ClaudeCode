"""v11.2.111 unit checks: belly clean preview (phoenix_belly_preview) and the band's length / breadth.
usage: unit111.py PACKDIR      Synthetic skin only (the scene of unit110.py); the body's landmarks are given, the still fill is
cv2.inpaint (no model)."""
import sys, os, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "unit110.py")).read().split("# --- 1. the band")[0]
src = src.replace("sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))", "pass")
exec(compile(src, "unit110-scene", "exec"))          # -> PACK, cp, BC, E, draw(), LM, P(), band, W, H, check(), fails
import phoenix_api_adapter as A

# --- 1. length and breadth ---------------------------------------------------------------------------------------------------
p1 = BC.region_polygon(LM)
hgt = lambda p: float(0.5 * (p[2][1] + p[3][1]) - 0.5 * (p[0][1] + p[1][1]))
wid = lambda p: float(np.linalg.norm(0.5 * (p[0] + p[3]) - 0.5 * (p[1] + p[2])))
mid_y = lambda p: float(p[:, 1].mean())
p_long, p_short = BC.region_polygon(LM, length=1.3), BC.region_polygon(LM, length=0.6)
check("length x1.3: the band is taller (about its middle)", hgt(p_long) > 1.15 * hgt(p1), f"{hgt(p1):.0f} -> {hgt(p_long):.0f} px")
check("length x0.6: the band is shorter, about its middle", abs(hgt(p_short) - 0.6 * hgt(p1)) < 3.0 and abs(mid_y(p_short) - mid_y(p1)) < 3.0,
      f"{hgt(p1):.0f} -> {hgt(p_short):.0f} px")
p_max = BC.region_polygon(LM, length=1.5)
t_of = lambda y: (LM[23][1] - y) / (LM[23][1] - LM[11][1])          # how far up from the hips (0) to the shoulders (1)
check("length: never above 0.40 of the way to the shoulders nor below just under the hip line",
      t_of(float(p_max[:, 1].min())) <= BC.T_TOP_MAX + 1e-3 and t_of(float(p_max[:, 1].max())) >= BC.T_BOT_MIN - 1e-3,
      f"top {t_of(float(p_max[:, 1].min())):.3f}, bottom {t_of(float(p_max[:, 1].max())):.3f}")
p_wide, p_narrow = BC.region_polygon(LM, width=1.4), BC.region_polygon(LM, width=0.6)
check("breadth x1.4 / x0.6: wider / narrower about the body's middle line",
      abs(wid(p_wide) - 1.4 * wid(p1)) < 3.0 and abs(wid(p_narrow) - 0.6 * wid(p1)) < 3.0 and abs(float(p_narrow[:, 0].mean()) - float(p1[:, 0].mean())) < 2.0,
      f"{wid(p1):.0f} -> {wid(p_wide):.0f} / {wid(p_narrow):.0f} px")
check("scales out of range are brought to 0.5-1.5; junk means 1",
      BC.clamp_scale(9, BC.WIDTH_RANGE) == 1.5 and BC.clamp_scale(0.1, BC.WIDTH_RANGE) == 0.5 and BC.clamp_scale("x", BC.WIDTH_RANGE) == 1.0
      and BC.clamp_scale(float("nan"), BC.WIDTH_RANGE) == 1.0)
img, truth, wb, upm = draw()
mk_n, _ = BC.find_marks(img, LM, 1.0, 0.6)
bn = BC.region_mask(p_narrow, (H, W))
check("a narrower band finds marks only inside it", mk_n is not None and int((mk_n & ~bn).sum()) == 0 and int(mk_n.sum()) < int(BC.find_marks(img, LM)[0].sum()))

worst = 0.0; rec_min = 1.0
for Lg in (0.5, 0.8, 1.0, 1.2, 1.5):
    for Wd in (0.5, 0.8, 1.0, 1.2, 1.5):
        mk_, _ = BC.find_marks(img, LM, Lg, Wd)
        bd_ = BC.region_mask(BC.region_polygon(LM, Lg, Wd), (H, W))
        if mk_ is None:
            continue
        worst = max(worst, float((mk_ & wb).sum()) / max(1, float((wb & bd_).sum())))
        tin = truth & bd_ & ~cv2.dilate(((~bd_).astype(np.uint8)), np.ones((9, 9), np.uint8)).astype(bool)
        if tin.sum() > 50:
            rec_min = min(rec_min, float((mk_ & tin).sum()) / float(tin.sum()))
check("every length / breadth (x0.5-x1.5): the waistband is never taken for a mark (under 1 %)", worst < 0.01, f"worst {100 * worst:.2f} %")
check("every length / breadth: the marks inside the band are found (95 % or more)", rec_min >= 0.95, f"lowest {100 * rec_min:.1f} %")

# --- 2. the one-frame preview ---------------------------------------------------------------------------------------------------
still = lambda im, m: (cv2.inpaint(im, (m > 127).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA), [{"area_px": int((m > 127).sum())}])
clean, _, _, _ = draw(with_marks=False)
after, outlined, info = BC.preview(img, LM, 1.0, 1.0, 0.5, still)
Lf = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
con = float((Lf(clean) - Lf(img))[truth].mean()); rem = float((Lf(clean) - Lf(after))[truth].mean())
check("preview: the marks are removed on the frame (under 15 % of their contrast left)", rem / con < 0.15, f"{100 * rem / con:.1f} % left, {info.get('marks_px')} px found")
edge = np.zeros((H, W), np.uint8)
cv2.polylines(edge, [np.round(p1).astype(np.int32)], True, 255, 9)
diff = np.abs(outlined.astype(int) - after.astype(int)).max(axis=2) > 0
check("preview: the outlined picture differs from the clean one only on the band's outline", int((diff & ~(edge > 0)).sum()) == 0 and int(diff.sum()) > 200)
check("preview: the info gives the band (4 corners), the marks, length and breadth", info.get("band") is not None and len(info["band"]) == 4
      and info.get("marks_px", 0) > 0 and info.get("length") == 1.0 and info.get("width") == 1.0)
a0, o0, i0 = BC.preview(img, None, 1.0, 1.0, 0.5, still)
check("preview: no body found -> the frame unchanged, and the reason", a0 is img and int(np.abs(o0.astype(int) - img.astype(int)).max()) == 0 and i0.get("reason") == "no body found")

# --- 3. a preview never changes a running job's strength ---------------------------------------------------------------------------
bt = BC.BellyTrack({0: LM}, 10, 30.0, 1.0, 1.0, 0.5)
BC.preview(img, LM, 1.0, 1.0, 0.9, still)
check("a preview at another strength leaves a running job's even-skin strength alone", bt.smooth == 0.5 and BC.CFG["smooth"] == 0.5, f"{bt.smooth}, {BC.CFG['smooth']}")

# --- 4. the API route ----------------------------------------------------------------------------------------------------------------
class StubPose:
    error = None
    def __init__(self, still=False):
        self.still = still
    def detect(self, bgr):
        return LM
    def close(self):
        pass
BC.MediaPipePose = StubPose
E.remove_patch_image = still
tmp = os.path.join(os.environ.get("TMPDIR", "/tmp/claude-0"), "u111"); os.makedirs(tmp, exist_ok=True)
fp = os.path.join(tmp, "frame.jpg"); cv2.imwrite(fp, img, [cv2.IMWRITE_JPEG_QUALITY, 96])
os.environ["GRADIO_TEMP_DIR"] = tmp
o1, o2, js = A.api_belly_preview(fp, json.dumps({"belly_length": 1.2, "belly_width": "0.8"}))
r = json.loads(js)
check("API: returns the cleaned frame, the outlined frame and ok info", r.get("ok") and o1 and o2 and os.path.isfile(o1) and os.path.isfile(o2) and r.get("body_found"), js[:200])
check("API: the length / breadth sent are used", abs(r.get("length", 0) - 1.2) < 1e-6 and abs(r.get("width", 0) - 0.8) < 1e-6)
check("API: the waistband is reported as clothing, not as marks", r.get("clothing_px", 0) > 2000 and r.get("marks_px", 0) < 8000, f"marks {r.get('marks_px')}, clothing {r.get('clothing_px')}")
o1b, _, js2 = A.api_belly_preview(fp, json.dumps({"belly_length": 7, "belly_width": "abc"}))
r2 = json.loads(js2)
check("API: out-of-range / junk values are brought into range", r2.get("length") == 1.5 and r2.get("width") == 1.0, js2[:160])
_, _, js3 = A.api_belly_preview(os.path.join(tmp, "missing.jpg"), "{}")
check("API: an unreadable frame gives ok=false and a reason", json.loads(js3).get("ok") is False)
class NoPose(StubPose):
    error = "ModuleNotFoundError: No module named 'mediapipe'"
    def detect(self, bgr):
        return None
BC.MediaPipePose = NoPose
_, _, js4 = A.api_belly_preview(fp, "{}")
check("API: without the pose model: ok=false, says so", json.loads(js4).get("ok") is False and "pose" in json.loads(js4).get("error", ""), js4[:160])
class NoBody(StubPose):
    def detect(self, bgr):
        return None
BC.MediaPipePose = NoBody
o5, o6, js5 = A.api_belly_preview(fp, "{}")
r5 = json.loads(js5)
check("API: no body on the frame: ok, body_found false, the reason given", r5.get("ok") and r5.get("body_found") is False and r5.get("reason") == "no body found", js5[:160])

# --- 5. job settings ---------------------------------------------------------------------------------------------------------------
d = A._defaults_from(json.dumps({"belly_clear": True, "belly_length": "1.3", "belly_width": 9}))
check("job settings: belly_length / belly_width parsed and limited", d["belly_clear"] is True and d["belly_length"] == 1.3 and d["belly_width"] == 1.5, str({k: d[k] for k in ("belly_clear", "belly_length", "belly_width")}))
d0 = A._defaults_from("{}")
check("job settings: absent -> 1.0 / 1.0, belly clean off", d0["belly_length"] == 1.0 and d0["belly_width"] == 1.0 and d0["belly_clear"] is False)

print("\n%d check(s) failed" % fails if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
