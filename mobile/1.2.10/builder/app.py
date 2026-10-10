import os
import shlex
import subprocess
import time

import gradio as gr

ROOT = os.environ.get("PHOENIX_ROOT", "/workspace")
APK = f"{ROOT}/PhoenixMobile-1.2.10-debug.apk"
SHA = f"{ROOT}/PhoenixMobile-1.2.10-debug.sha256"
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


def build():
    """
    Streams the build log line by line.

    The previous version blocked for the entire Gradle run and only returned
    output at the end, so a 10-20 minute build looked like a hung page with no
    way to tell whether anything was happening.
    """
    for stale in (APK, SHA):
        try:
            os.remove(stale)
        except FileNotFoundError:
            pass

    if not os.path.isfile(BUILD):
        yield f"build.sh not found at {BUILD}", "**Failed** - builder script missing", None, None
        return

    started = time.time()
    header = [
        "=== Phoenix Mobile 1.2.10 build ===",
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
            "**Build reported success but no APK was produced.** "
            "Check the `assembleDebug` output path in the log.",
            None,
            None,
        )
        return

    size_mb = os.path.getsize(apk_out) / 1_048_576
    yield (
        log,
        f"**Build passed** in {mins}m {secs:02d}s - APK is {size_mb:.1f} MB. "
        "Download both files below; verify the checksum before installing.",
        apk_out,
        sha_out,
    )


with gr.Blocks(title="Phoenix Mobile 1.2.10 Builder", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        "<div id='phx-head'><h1>Phoenix Mobile 1.2.10</h1>"
        "<p>Builds Phoenix Mobile 1.2.10, then hands back the debug APK and its SHA-256.</p></div>"
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
        apk = gr.File(label="Debug APK", interactive=False)
        sha = gr.File(label="SHA-256 checksum", interactive=False)

    gr.Markdown(
        "The output is a **debug** build, signed with the local debug key. It is for "
        "testing on your own device, not for distribution."
    )

    build_btn.click(build, outputs=[log, status, apk, sha], concurrency_limit=1)
    env_btn.click(check_env, outputs=[status])

demo.queue(max_size=4).launch(server_name="0.0.0.0", server_port=7860, show_error=True)
