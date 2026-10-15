import sys, numpy as np, cv2
sys.path.insert(0, '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/pack85')
import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + (" | " + extra if extra else ""))
CS = E.AlignedCompositor
# --- cheek zone
for S in (128, 256, 512):
    Z = CS._cheek_zone(S)
    check(f"zone {S}: in [0,1], eyes row empty, centre line (nose/mouth) empty, cheeks full",
          Z.min() >= 0 and Z.max() <= 1.0 and Z[int(.30*S)].max() == 0 and Z[:, int(.5*S)].max() == 0 and Z[int(.65*S), int(.2*S)] > 0.99 and Z[int(.65*S), int(.8*S)] > 0.99,
          f"max {Z.max():.2f}")
    mouth = Z[int(.78*S):int(.86*S), int(.38*S):int(.62*S)]
    check(f"zone {S}: mouth corners/lips untouched (<=0.02)", mouth.max() <= 0.02, f"max {mouth.max():.3f}")
# --- smoothing
S = 128; A = np.full((S, S, 3), 100, np.uint8); B = np.full((S, S, 3), 200, np.uint8)
tr = E.TrackState(0)
check("no track: returns the same object", CS.cheek_smooth(None, A) is A)
out = CS.cheek_smooth(tr, A); check("first frame seeds and returns the swap as it is", np.array_equal(out, A))
seq = [CS.cheek_smooth(tr, B) for _ in range(6)]
cheek = [int(o[int(.65*S), int(.2*S), 0]) for o in seq]; centre = [int(o[int(.65*S), int(.5*S), 0]) for o in seq]; eye = [int(o[int(.3*S), int(.2*S), 0]) for o in seq]
b_ = E._P["cheek_ema"]; exp = [100 + 100 * (1 - (1 - b_) ** (i + 1)) for i in range(6)]
cheek_ok = all(abs(c - e) <= 1.5 for c, e in zip(cheek, exp))
cheek_ok_name = "cheek: a step in the swap becomes the geometric ramp 100 + 100*(1-(1-b)^n)"
check(cheek_ok_name, cheek_ok, str(cheek) + " expected " + str([round(e) for e in exp]))
check("centre (nose/mouth) and eyes jump at once", set(centre) == {200} and set(eye) == {200}, f"centre {centre[:3]} eye {eye[:3]}")
check("monotone ramp, no overshoot", all(b >= a for a, b in zip(cheek, cheek[1:])) and max(cheek) <= 200)
tr2 = E.TrackState(0); CS.cheek_smooth(tr2, A)
bigA = np.full((256, 256, 3), 50, np.uint8); r = CS.cheek_smooth(tr2, bigA); check("crop size change re-seeds without error", np.array_equal(r, bigA))
E._P['cheek_ema'] = 1.0; tr3 = E.TrackState(0); CS.cheek_smooth(tr3, A); check("cheek_ema=1 -> exact no-op", np.array_equal(CS.cheek_smooth(tr3, B), B))
E._P['cheek_ema'] = 0.0; check("cheek_ema=0 -> no-op", np.array_equal(CS.cheek_smooth(tr3, B), B)); E._P['cheek_ema'] = 0.5
tr4 = E.TrackState(0); CS.cheek_smooth(tr4, A); tr4.reset_for_cut(); check("reset_for_cut clears the smoothing state", tr4.fake_s is None)
# --- graph cut safety
h = w = 192; img = np.full((h, w, 3), (120, 150, 200), np.uint8)
check("no object: unchanged", np.array_equal(E._gc_refine(np.zeros((h, w), np.uint8), img, h, w), np.zeros((h, w), np.uint8)))
tiny = np.zeros((h, w), np.uint8); tiny[100:104, 100:104] = 255
check("a speck is left alone", np.array_equal(E._gc_refine(tiny, img, h, w), tiny))
disc = np.zeros((h, w), np.uint8); cv2.circle(disc, (96, 140), 30, 255, -1)
pic = img.copy(); cv2.circle(pic, (96, 140), 34, (240, 240, 240), -1)           # the object is a little bigger than the blob found
res = E._gc_refine(disc, pic, h, w)
a0 = (disc > 0).sum(); a1 = (res > 0).sum()
check("blob is pulled out to the real object edge (area grows 0-60%)", a0 < a1 <= 1.6 * a0, f"{a0} -> {a1}")
pic2 = img.copy(); cv2.circle(pic2, (96, 140), 24, (240, 240, 240), -1)         # blob too big: object smaller
res2 = E._gc_refine(disc, pic2, h, w); a2 = (res2 > 0).sum()
check("blob that is too large is pulled in to the object, never below 60 %", 0.6 * a0 <= a2 <= a0, f"{a0} -> {a2}")
E._P['occ_gc'] = 0; check("occ_gc=0 -> unchanged", np.array_equal(E._gc_refine(disc, pic, h, w), disc)); E._P['occ_gc'] = 3
print("ALL PASS" if ok else "SOME FAILED")
