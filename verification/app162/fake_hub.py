# a stand-in for the Hugging Face Hub's Space runtime routes (shape as huggingface_hub 0.27.1 reads them); logs every request
import json, sys, http.server
stage = {"owner/SwamitechGradio3": "RUNNING"}
LOG = sys.argv[2]
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def _route(self, method):
        open(LOG, "a").write("%s %s auth=%s\n" % (method, self.path, self.headers.get("Authorization")))
        if method == "POST":
            n = int(self.headers.get("Content-Length", 0) or 0); self.rfile.read(n)
        p = self.path.split("?")[0].split("/")       # ['', 'api', 'spaces', owner, name, action]
        repo, action = p[3] + "/" + p[4], p[5]
        if self.headers.get("Authorization") != "Bearer hf_TESTTOKEN": return self._send(401, {"error": "Invalid credentials"})
        if repo == "owner/ReadOnly" and method == "POST": return self._send(403, {"error": "You don't have the rights"})
        if repo not in stage and repo != "owner/ReadOnly": return self._send(404, {"error": "Repository not found"})
        if method == "POST" and action == "pause": stage[repo] = "PAUSED"
        elif method == "POST" and action == "restart": stage[repo] = "BUILDING"
        elif not (method == "GET" and action == "runtime"): return self._send(405, {"error": "no"})
        self._send(200, {"stage": stage.get(repo, "RUNNING"), "hardware": {"current": "cpu-basic", "requested": "cpu-basic"}, "gcTimeout": 172800})
    def do_GET(self): self._route("GET")
    def do_POST(self): self._route("POST")
http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
