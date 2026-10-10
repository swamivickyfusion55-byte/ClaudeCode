import json, base64, subprocess, sys, os, urllib.request
P = sys.argv[1]; PACK = sys.argv[2]
T = "../ktools"; AND = f"{T}/android-all-15-robolectric-12650502.jar"
CPB = f"{AND}:{T}/okhttp-4.12.0.jar:{T}/okio-jvm-3.6.0.jar:{T}/okhttp-dnsoverhttps-4.12.0.jar:{T}/kotlinx-coroutines-core-jvm-1.9.0.jar:{T}/kotlinc/lib/kotlin-stdlib.jar"
H = {"Authorization": "Bearer tok_write", "Content-Type": "application/json"}
def ctl(path, body=None):
    return json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{P}{path}", data=(json.dumps(body).encode() if body is not None else None), headers=H)))
def run(repo):
    return subprocess.run(["java", "-cp", f"out_one:{CPB}", "OneDeployKt", P, PACK, repo], capture_output=True, text=True, timeout=120).stdout.split("my deploy:", 1)[-1].join(["my deploy:", ""]).replace("\n", " ")
def wire(repo): return [r for r in ctl("/api/control/state")["_records"] if r["repo"] == repo and r["kind"] == "commit_wire"]
R = []
def check(n, c, e=""): R.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"  [{e}]" if e and not c else ""))
ctl("/api/control/reset"); ctl("/api/control/set", {"deny_commit": [], "strict_ct": True})
out = run("Swamivicky/SwamitechGradio5"); w = wire("Swamivicky/SwamitechGradio5")
check("M1 my deploy succeeds against a Hub that picks the body format by exact Content-Type", out.startswith("my deploy: true"), out)
check("M1 Content-Type on the wire is exactly 'application/x-ndjson'", len(w) == 1 and w[0]["headers"]["content-type"] == "application/x-ndjson")
raw = base64.b64decode(w[0]["body"]).decode()
lines = raw.split("\n"); assert lines[-1] == ""; lines = lines[:-1]
ref = lambda o: json.dumps(o, separators=(",", ":"), ensure_ascii=False)
check("M1 every line is byte-identical to the reference compact serialisation of itself (key order, escaping, no '\\/')",
      all(ref(json.loads(l)) == l for l in lines) and json.loads(lines[0])["key"] == "header" and "\\/" not in raw, [l[:60] for l in lines if ref(json.loads(l)) != l][:1])
check("M1 header carries a summary", json.loads(lines[0])["value"]["summary"] == "wire")
ctl("/api/control/reset"); ctl("/api/control/set", {"deny_commit": [], "reject_ndjson": True})
out = run("Swamivicky/SwamitechGradio6"); w = wire("Swamivicky/SwamitechGradio6")
check("M2 if NDJSON is refused with the summary error, my deploy retries once as JSON and succeeds, saying so",
      out.startswith("my deploy: true") and "JSON form" in out and [r["headers"]["content-type"] for r in w] == ["application/x-ndjson", "application/json"], (out, [r["headers"]["content-type"] for r in w]))
ctl("/api/control/set", {"reject_ndjson": True, "reject_json": True})
out = run("Swamivicky/SwamitechGradio7")
check("M3 both refused -> one result reporting both answers", out.startswith("my deploy: false") and "(NDJSON)" in out and "retry as JSON" in out, out[:200])
print(f"\n{sum(R)}/{len(R)} passed"); sys.exit(0 if all(R) else 1)
