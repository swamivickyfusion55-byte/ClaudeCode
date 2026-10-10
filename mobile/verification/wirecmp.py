import json, sys, urllib.request, os
from huggingface_hub import HfApi, CommitOperationAdd
P = sys.argv[1]; pack = sys.argv[2]
names = ["app.py", "config.py", "core_pipeline.py", "phoenix_api_adapter.py", "swap_engine.py", "packages.txt", "requirements.txt"]
api = HfApi(endpoint=f"http://127.0.0.1:{P}", token="tok_write")
api.create_commit(repo_id="Swamivicky/HubConformance", repo_type="space",
                  operations=[CommitOperationAdd(path_in_repo=f, path_or_fileobj=open(os.path.join(pack, f), "rb").read()) for f in names],
                  commit_message="wire")
st = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{P}/api/control/state", headers={"Authorization": "Bearer tok_write"})))
W = {r["repo"]: r for r in st["_records"] if r["kind"] == "commit_wire"}
for repo, label in (("Swamivicky/HubConformance", "huggingface_hub (official, Python requests)"),
                    ("Swamivicky/SwamitechUAT2", "THEIR commitSpaceFiles (OkHttp, Android org.json)"),
                    ("Swamivicky/SwamitechGradio5", "MY HfDeploy (OkHttp, Android org.json)")):
    r = W.get(repo)
    if not r: print(f"{label}: no wire record"); continue
    h = r["headers"]
    print(f"{label}\n   content-type : {h.get('content-type')!r}\n   transfer-enc : {h.get('transfer-encoding')!r}   content-length: {h.get('content-length')}\n   accept-enc   : {h.get('accept-encoding')!r}   content-enc: {h.get('content-encoding')!r}\n   first 120 B  : {r['body_head'][:120]!r}")
