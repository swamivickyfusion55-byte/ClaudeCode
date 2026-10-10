import json, base64, subprocess, sys, os, time, urllib.request
P = sys.argv[1]; PACK = sys.argv[2]
T = "../ktools"; AND = f"{T}/android-all-15-robolectric-12650502.jar"
CPB = f"{AND}:{T}/okhttp-4.12.0.jar:{T}/okio-jvm-3.6.0.jar:{T}/okhttp-dnsoverhttps-4.12.0.jar:{T}/kotlinx-coroutines-core-jvm-1.9.0.jar:{T}/kotlinc/lib/kotlin-stdlib.jar"
H = {"Authorization": "Bearer tok_write"}
def ctl(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{P}{path}", data=(json.dumps(body).encode() if body is not None else None), headers={**H, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))
def run(which, repo, special=False):
    out = subprocess.run(["java", "-cp", f"out_{which}:{CPB}", "FixedTestKt", P, PACK, repo] + (["special"] if special else []), capture_output=True, text=True, timeout=120).stdout
    line = [l for l in out.splitlines() if l.startswith("RESULT")][-1]
    return line
def wire(repo):
    return [r for r in ctl("/api/control/state")["_records"] if r["repo"] == repo and r["kind"] == "commit_wire"]
RES = []
def check(n, c, e=""):
    RES.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"  [{e}]" if e and not c else ""))

names = ["app.py", "config.py", "core_pipeline.py", "phoenix_api_adapter.py", "swap_engine.py", "packages.txt", "requirements.txt", "README.md"]
def reference_body(summary, files):
    d = lambda o: json.dumps(o, separators=(",", ":"), ensure_ascii=False)
    lines = [d({"key": "header", "value": {"summary": summary, "description": ""}})]
    for path, data in files:
        lines.append(d({"key": "file", "value": {"content": base64.b64encode(data).decode(), "path": path, "encoding": "base64"}}))
    return ("\n".join(lines) + "\n").encode("utf-8")
files = [(n, open(os.path.join(PACK, n), "rb").read()) for n in names]

ctl("/api/control/reset"); ctl("/api/control/set", {"deny_commit": ["Swamivicky/SwamitechGradio13"], "strict_ct": True})
old = run("old", "Swamivicky/SwamitechGradio2")
check("S1 their ORIGINAL code, against a Hub that selects the body format by exact Content-Type, fails with the very error in your screenshot",
      old.startswith("RESULT FAIL") and "HTTP 400 for Swamivicky/SwamitechGradio2" in old and "Invalid input: expected string, received undefined" in old and "value.summary" in old, old[:200])
w = wire("Swamivicky/SwamitechGradio2")
check("S1 ...and the wire shows why: their request carried 'application/x-ndjson; charset=utf-8'", w and w[0]["headers"]["content-type"] == "application/x-ndjson; charset=utf-8", w[0]["headers"]["content-type"] if w else "no wire")
new = run("new", "Swamivicky/SwamitechGradio3")
check("S2 the FIXED code succeeds against the same Hub model", new == "RESULT OK", new[:200])
w = wire("Swamivicky/SwamitechGradio3")
check("S2 Content-Type on the wire is exactly 'application/x-ndjson'", len(w) == 1 and w[0]["headers"]["content-type"] == "application/x-ndjson", w[0]["headers"]["content-type"] if w else "none")
body = base64.b64decode(w[0]["body"]) if w else b""
check("S2 request body is BYTE-IDENTICAL to what huggingface_hub's JSON serialisation writes for the same files (compact form)", body == reference_body("Phoenix pack update from the phone", files), f"{len(body)} vs {len(reference_body('Phoenix pack update from the phone', files))}")
st = ctl("/api/control/state")["Swamivicky/SwamitechGradio3"]["files"]
check("S2 server ends up with the exact pack bytes", all(base64.b64decode(st[n]) == d for n, d in files if n != "README.md"))
# special characters in names / bytes: byte equality with the reference serialiser
sp = run("new", "Swamivicky/SwamitechGradio4", special=True)
w = wire("Swamivicky/SwamitechGradio4"); body = base64.b64decode(w[0]["body"]) if w else b""
sfiles = files + [("we\"ird\\na/me\tü.txt", "bytes /// \u0001 é".encode())]
check("S3 quotes, backslash, tab, '/', control chars and non-ASCII in a path serialise exactly like the reference", sp == "RESULT OK" and body == reference_body("Phoenix pack update from the phone", sfiles))
ref_lines = [json.loads(l) for l in body.decode().splitlines()]
check("S3 and parse back to the original path", ref_lines[-1]["value"]["path"] == "we\"ird\\na/me\tü.txt")
# fallback
ctl("/api/control/reset"); ctl("/api/control/set", {"deny_commit": ["Swamivicky/SwamitechGradio13"], "reject_ndjson": True})
fb = run("new", "Swamivicky/SwamitechGradio5")
w = wire("Swamivicky/SwamitechGradio5")
check("S4 if the Hub still refuses the NDJSON form with the 'summary' error, the commit is retried once as plain JSON and succeeds",
      fb == "RESULT OK" and [r["headers"]["content-type"] for r in w] == ["application/x-ndjson", "application/json"], (fb[:120], [r["headers"]["content-type"] for r in w]))
st = ctl("/api/control/state")["Swamivicky/SwamitechGradio5"]["files"]
check("S4 ...and the Space got every file intact via the JSON form", all(base64.b64decode(st[n]) == d for n, d in files if n != "README.md"))
ctl("/api/control/set", {"reject_ndjson": True, "reject_json": True})
both = run("new", "Swamivicky/SwamitechGradio6")
check("S5 if both forms are refused, ONE error reports both answers", both.startswith("RESULT FAIL") and "(NDJSON)" in both and "JSON retry" in both, both[:200])
ctl("/api/control/reset")
r403 = run("new", "Swamivicky/SwamitechGradio13"); w = wire("Swamivicky/SwamitechGradio13")
check("S6 a 403 is reported as a token-permission problem and is not retried", r403.startswith("RESULT FAIL") and "write permission" in r403 and "HTTP 403" in r403 and len(w) == 0, r403[:160])
miss = run("new", "Swamivicky/DoesNotExist")
check("S6 an unknown Space gives a clear HTTP 404", miss.startswith("RESULT FAIL") and "HTTP 404" in miss, miss[:160])
print(f"\n{sum(RES)}/{len(RES)} passed"); sys.exit(0 if all(RES) else 1)
