"""A mock of the Hugging Face Hub endpoints Phoenix Mobile's deploy uses, strict
about the wire format (it mirrors what huggingface_hub 0.27 sends)."""
import json, base64, hashlib, threading, re, sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

def git_sha(b): return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()

README_OLD = b"---\ntitle: SG UAT2\nemoji: x\nsdk: gradio\nsdk_version: 4.44.1\napp_file: app.py\npinned: false\n---\nold\n"
TOKENS = {"tok_write": ("Swamivicky", "write"), "tok_read": ("Swamivicky", "read"), "tok_fine": ("Swamivicky", "fineGrained")}
SPACES = {}
def reset():
    SPACES.clear()
    for n in list(range(2, 21)):
        SPACES[f"Swamivicky/SwamitechGradio{n}"] = {"files": {"README.md": README_OLD, "app.py": b"old app", ".gitattributes": b"*.bin filter=lfs\n"}, "commits": [], "stage": "RUNNING"}
    for n in ("SG_UAT2", "SwamitechUAT2"):
        SPACES[f"Swamivicky/{n}"] = {"files": {"README.md": README_OLD, "app.py": b"old app"}, "commits": [], "stage": "RUNNING"}
    SPACES["SwamiOrg/OrgSpace"] = {"files": {}, "commits": [], "stage": "RUNNING"}
    SPACES["Swamivicky/HubConformance"] = {"files": {}, "commits": [], "stage": "RUNNING"}
    SPACES["Swamivicky/AppConformance"] = {"files": {}, "commits": [], "stage": "RUNNING"}
reset()
STATE = {"deny_commit": {"Swamivicky/SwamitechGradio13"}, "lfs_paths": set(), "records": [], "list_calls": 0, "strict_ct": False, "reject_ndjson": False, "reject_json": False}
LOCK = threading.Lock()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj, headers=None):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        for k, v in (headers or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)
    def _auth(self):
        h = self.headers.get("Authorization", "")
        if not h.startswith("Bearer "): return None
        return TOKENS.get(h[7:])
    def _body(self):
        n = int(self.headers.get("Content-Length") or 0); return self.rfile.read(n)
    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query); who = self._auth()
        if who is None: return self._send(401, {"error": "Invalid credentials in Authorization header"})
        if u.path == "/api/control/reset":
            reset(); STATE["records"].clear(); STATE["deny_commit"] = {"Swamivicky/SwamitechGradio13"}; STATE["lfs_paths"] = set(); STATE["list_calls"] = 0; STATE["strict_ct"] = STATE["reject_ndjson"] = STATE["reject_json"] = False
            return self._send(200, {"ok": True})
        if u.path == "/api/whoami-v2":
            return self._send(200, {"type": "user", "name": who[0], "orgs": [{"name": "SwamiOrg"}], "auth": {"type": "access_token", "accessToken": {"displayName": "t", "role": who[1]}}})
        if u.path == "/api/spaces":
            STATE["list_calls"] += 1
            author = q.get("author", [""])[0]; limit = int(q.get("limit", ["1000"])[0]); expand = q.get("expand", [])
            ids = sorted(i for i in SPACES if i.split("/")[0] == author)
            page = int(q.get("cursor", ["0"])[0]); size = 7       # force several pages
            chunk = ids[page * size:(page + 1) * size]
            items = []
            for i in chunk:
                d = {"id": i}
                # the Hub returns subdomain when asked; omit it for a couple to exercise the fallback
                if "subdomain" in expand and not i.endswith("Gradio7") and not i.endswith("UAT2"):
                    d["subdomain"] = i.replace("/", "-").replace("_", "-").lower()
                items.append(d)
            hdr = {}
            if (page + 1) * size < len(ids):
                hdr["Link"] = f'<http://{self.headers["Host"]}/api/spaces?author={author}&limit={limit}&expand=subdomain&cursor={page+1}>; rel="next"'
            return self._send(200, items, hdr)
        m = re.fullmatch(r"/api/spaces/([^/]+/[^/]+)/runtime", u.path)
        if m:
            sp = SPACES.get(m.group(1))
            if not sp: return self._send(404, {"error": "Repository not found"})
            d = {"stage": sp["stage"], "hardware": {"current": "cpu-basic", "requested": "cpu-basic"}}
            if sp["stage"] == "RUNTIME_ERROR": d["errorMessage"] = "Exit code: 1. Reason: Traceback...\nmore"
            return self._send(200, d)
        if u.path == "/api/control/state":
            return self._send(200, {k: {"files": {p: base64.b64encode(b).decode() for p, b in v["files"].items()}, "commits": v["commits"], "stage": v["stage"]} for k, v in SPACES.items()} | {"_records": STATE["records"], "_list_calls": STATE["list_calls"]})
        return self._send(404, {"error": "not found " + u.path})
    def do_POST(self):
        u = urlparse(self.path); who = self._auth(); body = self._body()
        if who is None: return self._send(401, {"error": "Invalid credentials in Authorization header"})
        if u.path == "/api/control/set":
            d = json.loads(body); STATE["deny_commit"] = set(d.get("deny_commit", [])); STATE["lfs_paths"] = set(d.get("lfs_paths", []))
            for k in ("strict_ct", "reject_ndjson", "reject_json"):
                if k in d: STATE[k] = bool(d[k])
            for k, v in d.get("stage", {}).items(): SPACES[k]["stage"] = v
            return self._send(200, {"ok": True})
        m = re.fullmatch(r"/api/spaces/([^/]+/[^/]+)/preupload/main", u.path)
        if m:
            sp = SPACES.get(m.group(1))
            if not sp: return self._send(404, {"error": "Repository not found"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json": return self._send(400, {"error": "content-type"})
            d = json.loads(body)
            with LOCK: STATE["records"].append({"repo": m.group(1), "kind": "preupload", "json": d})
            out = []
            for f in d["files"]:
                assert set(f) >= {"path", "sample", "size"}, f
                base64.b64decode(f["sample"], validate=True)
                mode = "lfs" if (f["path"] in STATE["lfs_paths"] or f["size"] > 10 * 1024 * 1024) else "regular"
                e = sp["files"].get(f["path"])
                out.append({"path": f["path"], "uploadMode": mode, "shouldIgnore": False, "oid": git_sha(e) if e is not None else None})
            return self._send(200, {"files": out, "commitOid": "0" * 40})
        m = re.fullmatch(r"/api/spaces/([^/]+/[^/]+)/commit/main", u.path)
        if m:
            repo = m.group(1); sp = SPACES.get(repo)
            if not sp: return self._send(404, {"error": "Repository not found"})
            if who[1] == "read" or repo in STATE["deny_commit"]:
                return self._send(403, {"error": "You don't have the rights to create a commit on this repo"})
            with LOCK: STATE["records"].append({"repo": repo, "kind": "commit_wire", "headers": {k.lower(): v for k, v in self.headers.items()}, "body_len": len(body), "body_head": body[:200].decode("utf-8", "replace"), "body": base64.b64encode(body).decode()})
            ctype = self.headers.get("Content-Type", "").strip().lower()
            SUMMARY_ERR = {"error": "\u2716 Invalid input: expected string, received undefined\n  \u2192 at value.summary"}
            if ctype == "application/json":
                if STATE["reject_json"]: return self._send(400, {"error": "json mode rejected (test)"})
                try: dj = json.loads(body)
                except Exception: return self._send(400, SUMMARY_ERR)
                if not isinstance(dj.get("summary"), str): return self._send(400, SUMMARY_ERR)
                lines = [{"key": "header", "value": {"summary": dj["summary"], "description": dj.get("description", "")}}] + [{"key": "file", "value": f} for f in dj.get("files", [])]
            elif ctype == "application/x-ndjson" and not STATE["reject_ndjson"]:
                lines = [json.loads(l) for l in body.decode().split("\n") if l.strip()]
            elif STATE["strict_ct"] or STATE["reject_ndjson"]:
                # MODEL OF THE SUSPECTED HUB BEHAVIOUR (unverified): the body format is chosen by the exact
                # Content-Type; anything else is read as the plain-JSON form, whose top-level `summary` is absent.
                return self._send(400, SUMMARY_ERR)
            else:
                lines = [json.loads(l) for l in body.decode().split("\n") if l.strip()]
            if not lines or lines[0].get("key") != "header" or "summary" not in lines[0]["value"]:
                return self._send(400, {"error": "first line must be the header"})
            new = {}
            for l in lines[1:]:
                if l.get("key") != "file": return self._send(400, {"error": "unsupported op " + str(l.get("key"))})
                v = l["value"]
                if v.get("encoding") != "base64": return self._send(400, {"error": "encoding"})
                new[v["path"]] = base64.b64decode(v["content"], validate=True)
            if "README.md" in new:
                hdr = new["README.md"].decode().split("\n---", 1)[0]
                if not re.search(r"(?m)^sdk:", hdr) or not re.search(r"(?m)^app_file:", hdr):
                    return self._send(400, {"error": "Invalid metadata in README.md"})
            with LOCK:
                STATE["records"].append({"repo": repo, "kind": "commit", "lines": lines})
                sp["files"].update(new)
                oid = hashlib.sha1(body).hexdigest()
                sp["commits"].append({"oid": oid, "summary": lines[0]["value"]["summary"], "paths": sorted(new)})
                sp["stage"] = "BUILDING"
            return self._send(200, {"commitUrl": f"https://huggingface.co/spaces/{repo}/commit/{oid}", "commitOid": oid, "pullRequestUrl": None})
        return self._send(404, {"error": "not found " + u.path})

if __name__ == "__main__":
    port = int(sys.argv[1])
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print("mock hub on", port, flush=True); srv.serve_forever()
