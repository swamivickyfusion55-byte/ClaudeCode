"""SDOS-206: one table from two sets of bench runs (identity_check.py + contact_lag.py logs). usage: cmp_table.py BENCHDIR prefixA prefixB clip ..."""
import sys, re, json, os
D, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
def parse(label):
    out = {}
    try:
        for l in open(os.path.join(D, label + ".bench.log")):
            if '"wall_s"' in l:
                out["time"] = json.loads(l)["wall_s"]
    except OSError:
        pass
    try:
        txt = open(os.path.join(D, label + ".id.log")).read()
    except OSError:
        return out
    for who in ("A", "B"):
        m = re.search(r"^%s \(.*?correct (\d+), MIX-UP (\d+), unclear (\d+).*?REVERT (\d+) .*?weak (\d+) .*?flicker x([\d.]+)" % who, txt, re.M)
        if m:
            out[who] = dict(correct=int(m.group(1)), mix=int(m.group(2)), unclear=int(m.group(3)), revert=int(m.group(4)), weak=int(m.group(5)), flick=float(m.group(6)))
        m = re.search(r"^%s: off mean ([\d.]+) px, 90th ([\d.]+) px, worst ([\d.]+) px \| jump ([\d.]+) px \| unpainted (\d+)" % who, txt, re.M)
        if m:
            out.setdefault(who, {}).update(off=float(m.group(1)), worst=float(m.group(3)), jump=float(m.group(4)), unpainted=int(m.group(5)))
    return out
def row(lab, d):
    if "time" not in d:
        return "| %s | (not finished) |" % lab
    a, b = d.get("A", {}), d.get("B", {})
    s = lambda k: a.get(k, 0) + b.get(k, 0)
    return ("| %s | %.0f s | %d / %d | %d | %d | %d | %.2f / %.2f | %.1f / %.1f | %.1f / %.1f | %d |" % (
        lab, d["time"], s("correct"), s("unclear"), s("mix"), s("revert"), s("weak"), a.get("flick", 0), b.get("flick", 0),
        a.get("off", 0), b.get("off", 0), a.get("worst", 0), b.get("worst", 0), s("unpainted")))
print("| Run | Time | Identity correct / unclear (A+B) | Mix-ups | Reverts | Weak frames | Flicker A / B (x footage) | Placement mean A / B (px) | Worst A / B (px) | Unpainted |")
print("|---|---|---|---|---|---|---|---|---|---|")
for c in sys.argv[4:]:
    print(row("%s %s" % (c, A), parse("%s_%s" % (A, c))))
    print(row("%s %s" % (c, B), parse("%s_%s" % (B, c))))
