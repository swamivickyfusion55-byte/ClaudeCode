import json, sys, os, base64, urllib.request
from huggingface_hub import HfApi, CommitOperationAdd
P = sys.argv[1]; pack = sys.argv[2]
files = ["app.py", "config.py", "core_pipeline.py", "phoenix_api_adapter.py", "swap_engine.py", "packages.txt", "requirements.txt"]
api = HfApi(endpoint=f"http://127.0.0.1:{P}", token="tok_write")
info = api.create_commit(repo_id="Swamivicky/HubConformance", repo_type="space",
                         operations=[CommitOperationAdd(path_in_repo=f, path_or_fileobj=open(os.path.join(pack, f), "rb").read()) for f in files],
                         commit_message="conformance")
req = urllib.request.Request(f"http://127.0.0.1:{P}/api/control/state", headers={"Authorization": "Bearer tok_write"})
st = json.load(urllib.request.urlopen(req))
recs = st["_records"]
def get(repo, kind): return [r for r in recs if r["repo"] == repo and r["kind"] == kind]
hp, ap = get("Swamivicky/HubConformance", "preupload"), get("Swamivicky/AppConformance", "preupload")
hc, ac = get("Swamivicky/HubConformance", "commit"), get("Swamivicky/AppConformance", "commit")
ok = True
def check(n, c, e=""):
    global ok; ok &= bool(c); print(("PASS " if c else "FAIL ") + n + (f"  [{e}]" if e and not c else ""))
check("huggingface_hub made exactly one preupload and one commit", len(hp) == 1 and len(hc) == 1, (len(hp), len(hc)))
check("the app made exactly one preupload and one commit", len(ap) == 1 and len(ac) == 1, (len(ap), len(ac)))
norm = lambda fs: sorted((f["path"], f["sample"], f["size"]) for f in fs)
check("preupload body: same keys", set(hp[0]["json"]) == set(ap[0]["json"]), (set(hp[0]["json"]), set(ap[0]["json"])))
check("preupload body: identical path/sample/size for every file", norm(hp[0]["json"]["files"]) == norm(ap[0]["json"]["files"]))
check("preupload file entries: identical key sets", all(set(f) == {"path", "sample", "size"} for f in hp[0]["json"]["files"] + ap[0]["json"]["files"]), [set(f) for f in hp[0]["json"]["files"]][:1])
H, A = hc[0]["lines"], ac[0]["lines"]
check("commit header line identical", H[0] == A[0], (H[0], A[0]))
fl = lambda L: sorted((l["key"], l["value"]["path"], l["value"]["encoding"], l["value"]["content"], tuple(sorted(l["value"]))) for l in L[1:])
check("commit file lines identical (key, path, encoding, base64 content, field set)", fl(H) == fl(A))
same_files = all(st["Swamivicky/HubConformance"]["files"][f] == st["Swamivicky/AppConformance"]["files"][f] for f in files)
check("both Spaces end up with byte-identical files", same_files)
print("ALL CONFORMANT" if ok else "MISMATCH"); sys.exit(0 if ok else 1)
