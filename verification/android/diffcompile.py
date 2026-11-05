import sys, re, subprocess, collections, os
D = os.path.dirname(os.path.abspath(__file__))
def errors(path):
    out = subprocess.run([os.path.join(D, "kc.sh"), "out_dc", path], capture_output=True, text=True).stdout.splitlines()
    res = collections.Counter()
    i = 0
    while i < len(out):
        m = re.match(r".*?:\d+:\d+: error: (.*)", out[i])
        if m:
            code = out[i + 1].strip() if i + 1 < len(out) else ""
            res[(m.group(1)[:160], code[:140])] += 1
        i += 1
    return res
old, new = errors(sys.argv[1]), errors(sys.argv[2])
added = new - old; removed = old - new
print("old errors %d, new errors %d; only in new: %d, only in old: %d" % (sum(old.values()), sum(new.values()), sum(added.values()), sum(removed.values())))
for (msg, code), n in sorted(added.items()):
    print("NEW  x%d  %s\n        | %s" % (n, msg, code))
