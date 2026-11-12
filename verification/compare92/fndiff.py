import ast, sys, difflib
def funcs(path):
    src = open(path).read(); tree = ast.parse(src); lines = src.splitlines(); out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
            par = getattr(node, "_parent", None)
            out.setdefault(name, "\n".join(lines[node.lineno - 1:node.end_lineno]))
    return out, len(lines)
for f in ("core_pipeline.py", "swap_engine.py"):
    a, na = funcs(sys.argv[1] + "/" + f); b, nb = funcs(sys.argv[2] + "/" + f)
    same = [k for k in a if k in b and a[k] == b[k]]
    changed = [k for k in a if k in b and a[k] != b[k]]
    added = [k for k in b if k not in a]; removed = [k for k in a if k not in b]
    print("== %s: %d -> %d lines; functions/classes: %d unchanged, %d changed, %d new, %d removed" % (f, na, nb, len(same), len(changed), len(added), len(removed)))
    sizes = []
    for k in changed:
        d = sum(1 for l in difflib.unified_diff(a[k].splitlines(), b[k].splitlines(), lineterm="", n=0) if l[:1] in "+-" and not l.startswith(("+++", "---")))
        sizes.append((d, k))
    print("   most changed:", ", ".join("%s(%d)" % (k, d) for d, k in sorted(sizes, reverse=True)[:14]))
    print("   new:", ", ".join(added[:40]))
    if removed: print("   removed:", ", ".join(removed[:20]))
