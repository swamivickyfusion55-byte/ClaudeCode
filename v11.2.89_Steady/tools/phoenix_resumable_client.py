#!/usr/bin/env python3
"""
Reference client for the Phoenix resumable transport (Swamitech Phoenix v11.2.80).

NOT part of the Space - do not upload it there. It is the working, tested
description of the protocol for whoever maintains the phone app, and a way to
check from a laptop that a Space really has the routes (`probe`).

Only `requests` is needed. Every step can be killed at any moment (the process,
the network, the whole machine) and re-run: state lives in --state, and the
server is always asked how far it got before anything is sent again.

  python phoenix_resumable_client.py --space https://owner-space.hf.space \
      [--hf-token hf_xxx] probe
  ... run --video clip.mp4 --face a.jpg [--face b.jpg] \
      [--settings '{"quality":"Best"}'] [--split-seconds 30] --out ./results
  ... upload clip.mp4          # just the resumable upload; prints the FileData
  ... wait JOB_OR_TOKEN        # poll /phoenix/status until the job finishes
  ... get  JOB_OR_TOKEN out.mp4  # Range-resumable download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import time
import uuid

import requests

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class Client:
    def __init__(self, space: str, hf_token: str = "", state_path: str = "", log=print, timeout=60):
        self.base = space.rstrip("/")
        self.h = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        self.s = requests.Session()
        self.s.trust_env = False if self.base.startswith("http://127.") else self.s.trust_env
        self.log = log
        self.timeout = timeout
        self.state_path = state_path
        self.state: dict = {}
        if state_path and os.path.isfile(state_path):
            try:
                self.state = json.load(open(state_path))
            except Exception:
                self.state = {}

    # -- persistence: what a phone app keeps in its database --------------------
    def save(self):
        if self.state_path:
            tmp = self.state_path + ".tmp"
            json.dump(self.state, open(tmp, "w"))
            os.replace(tmp, self.state_path)

    def token(self, name: str) -> str:
        """A stable token per logical step, created once and stored BEFORE use."""
        t = self.state.setdefault("tokens", {})
        if name not in t:
            t[name] = secrets.token_urlsafe(12)
            self.save()
        return t[name]

    # -- one HTTP call with retry; the caller decides what a status means --------
    def call(self, method, path, tries=8, **kw):
        delay = 0.5
        for i in range(tries):
            try:
                r = self.s.request(method, self.base + path, headers={**self.h, **kw.pop("headers", {})},
                                   timeout=kw.pop("timeout", self.timeout), **kw)
            except requests.RequestException as e:           # minimised, no signal, socket killed...
                self.log(f"  network: {type(e).__name__}; retry in {delay:.1f}s")
                time.sleep(delay); delay = min(delay * 2, 15); continue
            if r.status_code in RETRY_STATUS and i < tries - 1:
                time.sleep(delay); delay = min(delay * 2, 15); continue
            return r
        raise RuntimeError(f"{method} {path}: gave up after {tries} tries")

    def probe(self) -> bool:
        r = self.call("GET", "/phoenix/upload/probe0000000000000", tries=2)
        ok = r.status_code == 404 and (r.headers.get("content-type", "").startswith("application/json"))
        self.log("resumable routes: " + ("PRESENT" if ok else f"MISSING (HTTP {r.status_code})"))
        return ok

    # -- upload ------------------------------------------------------------------
    def upload(self, path: str, token_name: str | None = None, chunk: int = 0) -> dict:
        size = os.path.getsize(path)
        name = os.path.basename(path)
        shas = self.state.setdefault("sha", {})
        key = f"{path}:{size}:{int(os.path.getmtime(path))}"
        if key not in shas:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for blk in iter(lambda: f.read(1 << 20), b""):
                    h.update(blk)
            shas[key] = h.hexdigest(); self.save()
        tok = self.token(token_name or f"upload:{key}")
        r = self.call("POST", "/phoenix/upload/init", json={"name": name, "size": size, "token": tok, "sha256": shas[key]})
        if r.status_code != 200:
            raise RuntimeError(f"upload init: {r.status_code} {r.text[:200]}")
        info = r.json()
        uid = info["upload_id"]
        chunk = chunk or int(info.get("chunk_bytes") or 4 * 1024 * 1024)
        if info.get("complete"):
            self.log(f"  {name}: already on the server"); return info
        off = int(info["offset"])
        if off:
            self.log(f"  {name}: resuming at {off}/{size}")
        t0, sent0 = time.time(), off
        with open(path, "rb") as f:
            while off < size:
                f.seek(off)
                data = f.read(min(chunk, size - off))
                try:
                    r = self.s.put(f"{self.base}/phoenix/upload/{uid}", data=data,
                                   headers={**self.h, "Upload-Offset": str(off)}, timeout=self.timeout)
                except requests.RequestException as e:
                    self.log(f"  network: {type(e).__name__} at {off}; asking the server where it got to")
                    off = self._offset(uid); continue
                if r.status_code == 200:
                    j = r.json(); off = int(j["offset"])
                    if j.get("complete"):
                        info = j; break
                elif r.status_code == 409:                  # the server knows better: continue from ITS offset
                    off = int(r.json()["offset"])
                elif r.status_code == 422:
                    raise RuntimeError("the file arrived damaged; the upload was discarded - run again")
                elif r.status_code in RETRY_STATUS:
                    time.sleep(1.0); off = self._offset(uid)
                else:
                    raise RuntimeError(f"upload PUT: {r.status_code} {r.text[:200]}")
        if not info.get("complete"):
            info = self.call("GET", f"/phoenix/upload/{uid}").json()
        self.log(f"  {name}: uploaded {size} bytes in {time.time() - t0:.1f}s (sent {size - sent0})")
        return info

    def _offset(self, uid: str) -> int:
        r = self.call("GET", f"/phoenix/upload/{uid}")
        if r.status_code == 404:
            raise RuntimeError("the server no longer has this upload; run again")
        return int(r.json()["offset"])

    # -- submit (Gradio queue protocol, idempotent via client_token) --------------
    def _queue_call(self, api_name: str, data: list, wait: float = 600.0):
        cfg = self.call("GET", "/config").json()
        fn = [d["id"] for d in cfg["dependencies"] if d.get("api_name") == api_name][0]
        sh = uuid.uuid4().hex[:11]
        r = self.call("POST", "/queue/join", json={"data": data, "fn_index": fn, "session_hash": sh,
                                                   "event_data": None, "trigger_id": 0})
        if r.status_code != 200:
            raise RuntimeError(f"queue/join {r.status_code} {r.text[:200]}")
        deadline = time.time() + wait
        with self.s.get(self.base + "/queue/data", params={"session_hash": sh}, headers=self.h,
                        stream=True, timeout=(15, 120)) as st:
            for line in st.iter_lines():
                if time.time() > deadline:
                    break
                if line.startswith(b"data:"):
                    m = json.loads(line[5:])
                    if m.get("msg") == "process_completed":
                        if not m.get("success"):
                            raise RuntimeError(f"{api_name} failed: {str(m.get('output'))[:200]}")
                        return m["output"]["data"][0]
        raise requests.ConnectionError("stream ended before the result")

    def submit(self, video_info: dict, face_infos: list, settings: dict, name: str = "submit") -> dict:
        tok = self.token(name)
        settings = {**settings, "client_token": tok}
        faces = [(i["file"] if i else None) for i in (face_infos + [None] * 4)[:4]]
        for attempt in range(6):
            try:
                # Did the first attempt land? (A queued call that was dropped never ran; a running one did.)
                st = self.call("GET", f"/phoenix/status/{tok}", tries=2)
                if st.status_code == 200 and not settings.get("split_seconds"):
                    self.log(f"  submit already landed → job {st.json()['job_id']}")
                    return {**st.json(), "ok": True}
                out = self._queue_call("phoenix_submit_video", [video_info["file"], *faces, json.dumps(settings)])
                return out
            except (requests.RequestException, RuntimeError) as e:
                self.log(f"  submit: {type(e).__name__}: {str(e)[:100]}; checking, then retrying")
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError("submit: gave up")

    # -- wait / download ------------------------------------------------------------
    def wait(self, ref: str, poll: float = 3.0, timeout: float = 6 * 3600) -> dict:
        t0 = time.time(); last = None
        while time.time() - t0 < timeout:
            r = self.call("GET", f"/phoenix/status/{ref}")
            if r.status_code == 404:
                raise RuntimeError(f"job {ref} not found")
            j = r.json()
            if (j.get("status"), j.get("progress")) != last:
                self.log(f"  {ref[:8]} {j.get('status')} {j.get('progress')}%"); last = (j.get("status"), j.get("progress"))
            if j.get("status") in ("done", "error", "cancelled"):
                return j
            time.sleep(poll)
        raise TimeoutError(ref)

    def get(self, ref: str, out: str) -> str:
        part, tagf = out + ".part", out + ".part.etag"
        etag = open(tagf).read().strip() if os.path.isfile(tagf) else ""
        delay = 0.5
        for _ in range(200):
            have = os.path.getsize(part) if os.path.isfile(part) else 0
            hdrs = dict(self.h)
            if have:
                hdrs["Range"] = f"bytes={have}-"
                if etag:
                    hdrs["If-Range"] = etag
            try:
                with self.s.get(f"{self.base}/phoenix/download/{ref}", headers=hdrs, stream=True, timeout=(15, 60)) as r:
                    if r.status_code == 416:                    # we already hold everything, or the file changed
                        total = int(r.headers.get("Content-Range", "bytes */0").split("/")[-1])
                        if have == total:
                            break
                        os.remove(part); continue
                    if r.status_code == 409:
                        time.sleep(3); continue
                    if r.status_code == 410:
                        raise RuntimeError("the result has expired on the server; submit again")
                    if r.status_code not in (200, 206):
                        raise RuntimeError(f"download: {r.status_code} {r.text[:150]}")
                    new_tag = r.headers.get("ETag", "")
                    if r.status_code == 200 and have:
                        have = 0                                # server ignored/refused the Range: start over
                    if new_tag:
                        open(tagf, "w").write(new_tag); etag = new_tag
                    total = int(r.headers["Content-Length"]) + have
                    with open(part, "ab" if have else "wb") as f:
                        for blk in r.iter_content(256 * 1024):
                            f.write(blk)
                    if os.path.getsize(part) >= total:
                        break
            except requests.RequestException as e:
                now = os.path.getsize(part) if os.path.isfile(part) else 0
                self.log(f"  network: {type(e).__name__}; resuming at {now}")
                if now > have:                  # it got somewhere: that is not a failing network, do not back off
                    delay = 0.5
                time.sleep(delay); delay = min(delay * 2, 15)
        else:
            raise RuntimeError("download: gave up")
        os.replace(part, out)
        try:
            os.remove(tagf)
        except OSError:
            pass
        return out

    # -- everything --------------------------------------------------------------------
    def run(self, video: str, faces: list, settings: dict, out_dir: str, split_seconds: float = 0):
        os.makedirs(out_dir, exist_ok=True)
        self.log("upload video"); v = self.upload(video)
        fi = []
        for i, p in enumerate(faces):
            self.log(f"upload face {i + 1}"); fi.append(self.upload(p))
        if split_seconds:
            settings = {**settings, "split_seconds": split_seconds}
        self.log("submit")
        rec = self.submit(v, fi, settings)
        if not rec.get("ok"):
            raise RuntimeError(f"submit refused: {rec}")
        jobs = rec.get("jobs") or [rec]
        self.state["jobs"] = [j["job_id"] for j in jobs]; self.save()
        self.log(f"{len(jobs)} job(s): {self.state['jobs']}")
        outs = []
        for j in jobs:                                     # parts finish in order; fetch each as it is ready
            st = self.wait(j["job_id"])
            if st["status"] != "done":
                raise RuntimeError(f"job {j['job_id']} ended {st['status']}: {st.get('message')}")
            name = f"Phoenix_{j['job_id']}" + (f"-{j['part_index']}" if j.get("part_index") else "") + ".mp4"
            outs.append(self.get(j["job_id"], os.path.join(out_dir, name)))
            self.log(f"saved {outs[-1]}")
        return outs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True); ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--state", default="phoenix_client_state.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    p = sub.add_parser("upload"); p.add_argument("path")
    p = sub.add_parser("wait"); p.add_argument("ref")
    p = sub.add_parser("get"); p.add_argument("ref"); p.add_argument("out")
    p = sub.add_parser("run"); p.add_argument("--video", required=True); p.add_argument("--face", action="append", default=[])
    p.add_argument("--settings", default="{}"); p.add_argument("--split-seconds", type=float, default=0); p.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    c = Client(a.space, a.hf_token, a.state)
    if a.cmd == "probe":
        return 0 if c.probe() else 1
    if a.cmd == "upload":
        print(json.dumps(c.upload(a.path))); return 0
    if a.cmd == "wait":
        print(json.dumps(c.wait(a.ref))); return 0
    if a.cmd == "get":
        print(c.get(a.ref, a.out)); return 0
    c.run(a.video, a.face, json.loads(a.settings), a.out, a.split_seconds); return 0


if __name__ == "__main__":
    sys.exit(main())
