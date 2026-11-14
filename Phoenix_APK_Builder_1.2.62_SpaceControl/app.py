import os
import json
import shlex
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import gradio as gr

ROOT = os.environ.get("PHOENIX_ROOT", "/workspace")
VER = "1.2.62"
APK = f"{ROOT}/PhoenixMobile-{VER}-debug.apk"
SHA = f"{ROOT}/PhoenixMobile-{VER}-debug.sha256"
INFO = f"{ROOT}/PhoenixMobile-{VER}-install-info.txt"

INSTALL_TIP = (
    "**Install this APK over the Phoenix that is already on the phone. Do not uninstall first** - "
    "that would delete History and the job numbers of jobs still running on the Spaces. The signing key is "
    "the same on every build from this pack, so Android updates in place and keeps your token, Spaces and "
    "History. Only if the phone says \"App not installed\" or \"Something went wrong\" does an older copy "
    "signed with a different key need to be uninstalled - open the Spaces in the browser and download any "
    "finished jobs first."
)
BUILD = f"{ROOT}/build.sh"

CSS = """
.gradio-container {max-width: 940px !important;}
#phx-head {background:linear-gradient(135deg,#071A35,#10305A);border-radius:16px;
  padding:20px 22px;color:#fff;margin-bottom:14px;}
#phx-head h1 {margin:0;font-size:22px;font-weight:800;letter-spacing:-.2px;}
#phx-head p {margin:6px 0 0;color:#A8C2DE;font-size:13px;}
#phx-log textarea {font-family:ui-monospace,SFMono-Regular,Menlo,monospace !important;
  font-size:12px !important;line-height:1.45 !important;background:#0d1a2b !important;
  color:#d5e4f2 !important;}
"""


def _env_report() -> str:
    """Verify the toolchain before Gradle is invoked, so failures are legible."""
    probes = [
        ("Java", "java -version"),
        ("Gradle", "gradle --version"),
        ("SDK manager", "sdkmanager --version"),
    ]
    out = []
    for name, cmd in probes:
        try:
            r = subprocess.run(
                shlex.split(cmd), text=True, capture_output=True, timeout=90
            )
            blob = (r.stdout + r.stderr).strip().splitlines()
            first = next((l for l in blob if l.strip()), "(no output)")
            out.append(f"{name:<12} {first.strip()}")
        except Exception as exc:  # noqa: BLE001 - report, never crash the UI
            out.append(f"{name:<12} UNAVAILABLE - {exc}")

    out.append(f"{'ANDROID_HOME':<12} {os.environ.get('ANDROID_HOME', '(unset)')}")
    out.append(f"{'GRADLE_HOME':<12} {os.environ.get('GRADLE_USER_HOME', '(default)')}")
    out.append(f"{'Workspace':<12} {ROOT} (writable: {os.access(ROOT, os.W_OK)})")
    src = os.path.join(ROOT, "source")
    out.append(f"{'Source':<12} {src} (present: {os.path.isdir(src)})")
    return "\n".join(out)


def check_env():
    return "```\n" + _env_report() + "\n```"


def _zip_complete(path: str) -> bool:
    """A phone says 'package appears to be invalid' when the download was cut off.
    A finished APK is a zip whose end-of-directory record is in the last 64 KB."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    if size < 1_000_000:
        return False
    with open(path, "rb") as fh:
        fh.seek(0)
        if fh.read(2) != b"PK":
            return False
        fh.seek(max(0, size - 65536))
        return b"PK\x05\x06" in fh.read()


def build():
    """
    Streams the build log line by line.

    The previous version blocked for the entire Gradle run and only returned
    output at the end, so a 10-20 minute build looked like a hung page with no
    way to tell whether anything was happening.
    """
    for stale in (APK, SHA, INFO):
        try:
            os.remove(stale)
        except FileNotFoundError:
            pass

    if not os.path.isfile(BUILD):
        yield f"build.sh not found at {BUILD}", "**Failed** - builder script missing", None, None
        return

    started = time.time()
    header = [
        "=== Phoenix Mobile 1.2.62 build ===",
        _env_report(),
        "",
        "=== Launching build.sh ===",
        "",
    ]
    lines = list(header)
    yield "\n".join(lines), "**Building...** starting Gradle", None, None

    try:
        proc = subprocess.Popen(
            ["bash", BUILD],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            cwd=ROOT,
        )
    except Exception as exc:  # noqa: BLE001
        yield f"Could not start build: {exc}", "**Failed** - could not start build", None, None
        return

    last_flush = 0.0
    assert proc.stdout is not None
    for raw in proc.stdout:
        lines.append(raw.rstrip("\n"))
        now = time.time()
        if now - last_flush >= 0.5:
            last_flush = now
            mins, secs = divmod(int(now - started), 60)
            yield (
                "\n".join(lines[-800:]),
                f"**Building...** {mins}m {secs:02d}s elapsed",
                None,
                None,
            )

    proc.wait()
    elapsed = int(time.time() - started)
    mins, secs = divmod(elapsed, 60)
    log = "\n".join(lines[-800:])

    if proc.returncode != 0:
        yield (
            log,
            f"**Build failed** (exit {proc.returncode}) after {mins}m {secs:02d}s. "
            "Scroll the log for the first ERROR or FAILURE line.",
            None,
            None,
        )
        return

    apk_out = APK if os.path.isfile(APK) else None
    sha_out = SHA if os.path.isfile(SHA) else None
    if apk_out is None:
        yield (
            log,
            "**Build reported success but PhoenixMobile-1.2.62-debug.apk was not produced.** "
            "An older APK is not offered. A phone with 1.2.49 would call that a downgrade and say the package is invalid.",
            None,
            None,
        )
        return

    size_mb = os.path.getsize(apk_out) / 1_048_576
    if not _zip_complete(apk_out):
        yield (
            log,
            f"**Not offering this file.** It is {size_mb:.1f} MB and is not a complete APK. "
            "A phone would say the package is invalid. Build again.",
            None,
            None,
        )
        return
    info = ""
    try:
        with open(INFO, encoding="utf-8") as fh:
            info = fh.read().strip()
    except OSError:
        pass
    if info and ("1.2.62" not in info or "versionCode 182" not in info):
        yield (
            log,
            "**Not offering this file.** It is not versionCode 182 (1.2.62). "
            "Installing an older build over the copy on the phone is what the phone calls an invalid package.\n\n```\n"
            + info + "\n```",
            None,
            None,
        )
        return
    nbytes = os.path.getsize(apk_out)
    yield (
        log,
        f"**Build passed** in {mins}m {secs:02d}s - APK is {nbytes:,} bytes "
        f"(the phone's Downloads list shows {nbytes / 1e6:.2f} MB). "
        "Download it with the button below only. Install it only if the phone's Downloads shows exactly that size "
        "and not \"Download failed\": a cut-off file is what the phone calls an invalid package."
        + (f"\n\n```\n{info}\n```" if info else "")
        + "\n\n" + INSTALL_TIP,
        _apk_button(nbytes),
        sha_out,
    )


def _apk_url():
    host = (os.environ.get("SPACE_HOST") or "").strip()
    return f"https://{host}/phoenix-apk" if host else "/phoenix-apk"


def _apk_button(nbytes):
    return (
        f"<div style='padding:14px;border-radius:14px;background:#FDF2F6;border:1px solid #F3C6D8'>"
        f"<a href='{_apk_url()}' target='_blank' rel='noopener' download "
        f"style='display:block;text-align:center;padding:14px;border-radius:12px;background:#9B1B4A;color:#fff;"
        f"font-weight:800;font-size:17px;text-decoration:none'>Download PhoenixMobile-{VER}-debug.apk</a>"
        f"<p style='margin:10px 2px 0;font-size:14px'>Size: <b>{nbytes:,} bytes</b> - the phone's Downloads list shows "
        f"<b>{nbytes / 1e6:.2f} MB</b>. Install only a file of exactly that size. If Downloads says \"Download failed\", "
        "delete it and tap the button again (it resumes).</p></div>"
    )


def _hf_json(token, method, url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token.strip())
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def refresh_spaces(token, author, name_filter):
    """Restart every matching Space. Restart stops its jobs and wakes it."""
    token = (token or "").strip()
    author = (author or "").strip()
    needle = (name_filter or "").strip().lower()
    if not token:
        yield "Paste a Hugging Face token with write access."
        return
    if not author:
        yield "The account name is empty."
        return
    url = "https://huggingface.co/api/spaces?" + urllib.parse.urlencode(
        {"author": author, "limit": "100"}
    )
    code, raw = _hf_json(token, "GET", url)
    if code == 401:
        yield "The token was refused. Use a token that can write to these Spaces."
        return
    if code != 200:
        yield f"Could not list Spaces (HTTP {code})."
        return
    try:
        spaces = json.loads(raw)
    except json.JSONDecodeError:
        yield "Hugging Face did not return a Space list."
        return
    if not isinstance(spaces, list):
        yield "Hugging Face did not return a Space list."
        return
    chosen = []
    for item in spaces:
        sid = str(item.get("id") or "")
        if not sid or "/" not in sid:
            continue
        if needle and needle not in sid.lower():
            continue
        chosen.append(sid)
    if not chosen:
        yield f"No Spaces under {author} matched \"{needle or 'anything'}\"."
        return
    lines = [f"{len(chosen)} Space(s). Restarting each one stops its jobs and wakes it.", ""]
    yield "\n".join(lines)
    for sid in chosen:
        restart = "https://huggingface.co/api/spaces/" + urllib.parse.quote(sid) + "/restart"
        status_code, body = _hf_json(token, "POST", restart)
        if status_code in (200, 202):
            lines.append(f"{sid}  restarted")
        else:
            detail = " ".join(body.split())[:140]
            lines.append(f"{sid}  HTTP {status_code}  {detail}")
        yield "\n".join(lines)
        time.sleep(0.4)
    lines.append("")
    lines.append("Done. Give each Space a minute, then open it. The log must say v11.2.173 before you start a job.")
    yield "\n".join(lines)


with gr.Blocks(title="Phoenix Mobile 1.2.62 Builder", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        "<div id='phx-head'><h1>Phoenix Mobile 1.2.62</h1>"
        "<p>versionCode 182. New look: a calm light theme, dark text on light, one blue accent. Redo part works again (cuts the part again if needed); a job a Space lost shows Lost. Skin tone step (brighten or darken a chosen person, Preview with Before / After; needs Space v11.2.205). "
        "Tap draws a free shape on the dark patch, at most 4 square inches. "
        "<a href='/phoenix-apk'>Download PhoenixMobile-1.2.62-debug.apk</a>.</p></div>"
    )

    with gr.Row():
        build_btn = gr.Button("Build APK", variant="primary", scale=3)
        env_btn = gr.Button("Check toolchain", scale=1)

    status = gr.Markdown("Idle - press **Build APK** to start.")
    log = gr.Textbox(
        label="Build / audit log",
        lines=26,
        max_lines=26,
        elem_id="phx-log",
        show_copy_button=True,
        autoscroll=True,
    )

    with gr.Row():
        # 1.2.58: the APK is offered ONLY through the direct, resumable /phoenix-apk link. Gradio's own file link
        # (/gradio_api/file=...) broke off again and again on the phone (206, 206, 206 ...) and left a cut-off file.
        apk = gr.HTML(_apk_button(os.path.getsize(APK)) if os.path.isfile(APK) and _zip_complete(APK) else "")
        sha = gr.File(label="SHA-256 checksum", interactive=False)

    gr.Markdown(
        "The output is a **debug** build, signed with the local debug key. It is for "
        "testing on your own device, not for distribution."
    )

    build_btn.click(build, outputs=[log, status, apk, sha], concurrency_limit=1)
    env_btn.click(check_env, outputs=[status])

    gr.Markdown(
        "### Clear jobs and refresh Spaces\n"
        "Restarts every matching Space. A restart stops the jobs on it and wakes it if it was asleep. "
        "The token is used once and is not saved. It needs **write** access."
    )
    with gr.Row():
        hf_token = gr.Textbox(label="Hugging Face token", type="password")
        hf_author = gr.Textbox(label="Account", value="swamivicky")
        hf_filter = gr.Textbox(label="Name contains", value="swamitech")
    refresh_btn = gr.Button("Clear jobs and refresh all Spaces")
    refresh_log = gr.Textbox(label="Space refresh", lines=12, elem_id="phx-log")
    refresh_btn.click(
        refresh_spaces,
        inputs=[hf_token, hf_author, hf_filter],
        outputs=[refresh_log],
    )

def _official_apk():
    if os.path.isfile(APK) and _zip_complete(APK):
        return APK
    return None


def _apk_bytes(path, start, end):
    with open(path, "rb") as fh:
        fh.seek(start)
        left = end - start + 1
        while left > 0:
            chunk = fh.read(min(1024 * 1024, left))
            if not chunk:
                break
            left -= len(chunk)
            yield chunk


def _make_server():
    """The download route is registered on the FastAPI app that is actually served.
    Adding it to demo.app before launch is wiped on Gradio 5, so the link never appeared."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, StreamingResponse

    api = FastAPI()

    @api.get("/phoenix-apk")
    def phoenix_apk(request: Request):
        path = _official_apk()
        if path is None:
            raise HTTPException(status_code=404, detail="No complete PhoenixMobile-1.2.62-debug.apk yet. Press Build APK first.")
        size = os.path.getsize(path)
        name = os.path.basename(path)
        common = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'attachment; filename="{name}"',
            "Content-Type": "application/vnd.android.package-archive",
        }
        header = request.headers.get("range")
        if header and header.startswith("bytes="):
            spec = header.split("=", 1)[1].split(",")[0].strip()
            start_s, _, end_s = spec.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            if start < 0 or start >= size or end < start:
                raise HTTPException(status_code=416, detail="Range not satisfiable")
            end = min(end, size - 1)
            headers = dict(common)
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            headers["Content-Length"] = str(end - start + 1)
            return StreamingResponse(_apk_bytes(path, start, end), status_code=206, headers=headers)
        return FileResponse(path, media_type=common["Content-Type"], filename=name)

    return gr.mount_gradio_app(api, demo, path="/")


demo.queue(max_size=4)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(_make_server(), host="0.0.0.0", port=7860)
