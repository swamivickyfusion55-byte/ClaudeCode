"""
Swamitech Beauty Studio - HDR video editor UI (Gradio, HF Spaces entry point).

Run locally:   python -m beauty_studio.app
On Spaces:     set app_file to beauty_studio/app.py (see beauty_studio/README.md)

The layout is one page: source and results on the left, every control on the
right. Sliders are generated from a single spec table so the UI, the Settings
dataclass and the preset system cannot drift apart - adding a control is one
line here and one field in settings.py, not four places to keep in sync.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import traceback
import uuid

import cv2
import gradio as gr

if __package__ in (None, ""):
    # Run as a plain script (`python beauty_studio/app.py`, which is how a
    # Hugging Face Space with app_file pointing here starts it). Put the
    # parent directory on the path and adopt this folder as the package, so
    # the relative imports below resolve whatever the folder is called.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
    __package__ = os.path.basename(_here)

from . import retention
from .mp_backend import mediapipe_ready, mediapipe_status
from .pipeline import (Cancelled, FrameProcessor, capability_report, grab_frame,
                       probe, process_image, render_video)
from .settings import DEFAULT_PRESET, PRESETS, Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("beauty_studio")

VERSION = "v1.5.0 (Aurora)"


# --------------------------------------------------------------- control spec
# (field, label, bipolar, tooltip). Order inside a group is the display order;
# the flat order of this table is the order every callback receives values in.
GROUPS: list[tuple[str, list[tuple[str, str, bool, str]]]] = [
    ("HDR & colour", [
        ("hdr_strength", "HDR strength", False, "Local tone mapping: opens shadows and recovers highlights"),
        ("shadows", "Shadows", True, "Extra lift (right) or crush (left)"),
        ("highlights", "Highlights", True, "Recover (left) or push (right)"),
        ("clarity", "Clarity", False, "Mid-frequency local contrast"),
        ("contrast", "Contrast", False, "Gentle S-curve"),
        ("vibrance", "Vibrance", False, "Saturation, weighted toward the duller colours"),
        ("saturation", "Saturation", True, "Flat saturation on top of vibrance"),
        ("warmth", "Warmth", True, "White balance: blue to amber"),
        ("tint", "Tint", True, "White balance: green to magenta"),
        ("bloom", "Highlight bloom", False, "Soft glow around the brightest areas"),
        ("sharpen", "Output sharpening", False, "Final edge-aware sharpen"),
    ]),
    ("Skin & face detail", [
        ("skin_smooth", "Skin smoothing", False, "Frequency-separated - it flattens tone, not pores"),
        ("texture", "Keep texture", False, "How much real skin detail is put back (high = most natural)"),
        ("blemish", "Blemish removal", False, "Suppresses small dark spots only"),
        ("skin_even", "Even skin tone", False, "Evens colour blotches, leaves the lighting alone"),
        ("under_eye", "Under-eye circles", False, "Lifts and de-blues the shadow under the eyes"),
        ("glow", "Soft glow", False, "Softbox-style luminosity on skin"),
        ("eye_brighten", "Eye brightness", False, "Whitens the sclera, defines the iris and lashes"),
        ("teeth_whiten", "Teeth whitening", False, "Bright pixels inside the mouth only"),
        ("lip_enhance", "Lip definition", False, "Colour depth and edge definition, no shape change"),
    ]),
    ("Face shape", [
        ("face_slim", "Slim face", False, "Narrows jaw and cheeks toward the face's own centre line"),
        ("face_round", "Rounder face", False, "The opposite: fuller cheeks and a softer jaw"),
        ("chin_shape", "Chin taper", False, "Shortens and tapers the chin"),
        ("nose_slim", "Nose width", False, "Narrows the nostril wings"),
        ("eye_enlarge", "Eye size", False, "Enlarges the eyes slightly"),
    ]),
    ("Body shape", [
        ("body_slim", "Slim silhouette", False, "Narrows the whole visible body below the shoulders"),
        ("body_fuller", "Fuller body", False, "The opposite: widens the whole visible body"),
        ("waist_shape", "Waist", False, "Pinches at the waist line"),
        ("curve_shape", "Curvy (hourglass)", False,
         "One control for the whole shape: in at the waist, out at bust and hips"),
        ("bust_shape", "Bust", False, "Fills out the chest line on its own"),
        ("hip_shape", "Hips", False, "Widens the hips and upper thighs on its own"),
        ("posture", "Posture", True, "Lifts (right) or drops (left) the shoulders"),
    ]),
    ("Hair", [
        ("hair_detail", "Strand definition", False, "Brings out individual strands"),
        ("hair_shine", "Shine", False, "Gloss along the light that is already there"),
        ("hair_richness", "Colour depth", False, "Richer colour, deeper shadow"),
        ("hair_frizz", "Flyaway control", False, "Tidies the outline without touching the inside"),
        ("hair_volume", "Volume", False, "Grows the hair silhouette a few pixels"),
    ]),
    ("Finish", [
        ("naturalness", "Naturalness", False, "Scales every person-effect at once. 100 = as set above"),
    ]),
]

FIELDS: list[str] = [f for _, items in GROUPS for f, *_ in items]
BIPOLAR: dict[str, bool] = {f: bi for _, items in GROUPS for f, _, bi, _ in items}

SCALE_CHOICES = [("Source resolution (slowest)", 0), ("2160p / 4K", 2160),
                 ("1440p", 1440), ("1080p (recommended)", 1080),
                 ("720p (fastest)", 720)]


def settings_from(values, protect_skin, stabilise, scale, out_long, quality, hdr10) -> Settings:
    kw = {}
    for field, v in zip(FIELDS, values):
        kw[field] = float(v) / 100.0
    return Settings(
        protect_skin_colour=bool(protect_skin),
        stabilise=bool(stabilise),
        process_scale=int(scale) if int(scale) > 0 else 100000,
        out_long_edge=int(out_long),
        quality=int(quality),
        hdr10=bool(hdr10),
        **kw,
    ).normalised()


def preset_values(name: str) -> list[float]:
    """Preset amounts as whole-number slider positions.

    Rounded because 0.55 * 100 is 55.00000000000001 in binary floating point,
    and that is exactly what the number box next to the slider would display.
    """
    s = PRESETS.get(name, PRESETS[DEFAULT_PRESET])
    return [round(float(getattr(s, f)) * 100.0) for f in FIELDS]


# --------------------------------------------------------------- cancellation
# A render is minutes of work, so the Stop button has to reach inside the loop
# rather than rely on the request being dropped. Each render gets a token; the
# renderer polls this set once per frame, and stopping is a set membership
# test instead of anything shared and mutable crossing threads.
_CANCEL_LOCK = threading.Lock()
_CANCELLED: set[str] = set()


def _cancel_requested(job_id: str) -> bool:
    with _CANCEL_LOCK:
        return job_id in _CANCELLED


def on_cancel(job_id):
    if not job_id:
        return _status("Nothing is rendering.")
    with _CANCEL_LOCK:
        _CANCELLED.add(job_id)
    return _status("Stopping after the current frame…")


# ------------------------------------------------------------------ callbacks

def on_video(path):
    if not path:
        return "", gr.update(value=None), gr.update(value=None)
    try:
        info = probe(path)
        msg = f"**Loaded** · {info.label}"
        if info.frames == 0:
            msg += " · frame count unknown (the progress bar will be approximate)"
        frame = grab_frame(path, 0.35)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
        return msg, rgb, None
    except Exception as e:
        return f"**Could not read that file** · {e}", None, None


def on_preview(path, position, *args):
    """Process a single frame so settings can be judged without a render."""
    if not path:
        raise gr.Error("Load a video first.")
    frame = grab_frame(path, float(position) / 100.0)
    if frame is None:
        raise gr.Error("Could not read a frame from that video.")
    s = settings_from(args[:len(FIELDS)], *args[len(FIELDS):])
    fp = FrameProcessor(s, static=True)
    try:
        out = fp.process(frame)
    finally:
        fp.close()
    bits = []
    if fp.max_face_shift_px > 0.05:
        bits.append(f"face reshaped by {fp.max_face_shift_px:.0f} px")
    if fp.frames_with_person:
        bits.append(f"body reshaped by {fp.max_body_shift_px:.0f} px")
    elif s.touches_body():
        bits.append("no body outline in this frame — body shaping did nothing")
    extra = (" · " + " · ".join(bits)) if bits else ""
    if fp.frames_with_face:
        found = "face found"
    elif not mediapipe_ready():
        # Distinguish "your footage" from "your install" - they need different
        # fixes, and the preview is where a user first notices either.
        found = f"face features unavailable — {mediapipe_status()}"
    else:
        found = "no face found in this frame"
    return (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
            f"Preview at {float(position):.0f}% · {found}{extra}")


def on_photo(image, *args):
    if image is None:
        raise gr.Error("Load a photo first.")
    s = settings_from(args[:len(FIELDS)], *args[len(FIELDS):])
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(process_image(bgr, s), cv2.COLOR_BGR2RGB)


def on_render(path, start, end, *args, progress=gr.Progress()):
    """Generator so the UI keeps updating while the render runs."""
    if not path:
        raise gr.Error("Load a video first.")
    s = settings_from(args[:len(FIELDS)], *args[len(FIELDS):])
    a, b = sorted((float(start) / 100.0, float(end) / 100.0))
    if b - a < 0.01:
        a, b = 0.0, 1.0

    job_id = uuid.uuid4().hex
    yield None, _status("Starting render…"), gr.update(interactive=False), job_id
    progress(0.0, desc="Starting…")

    def report(done, total, message):
        progress(done / max(total, 1), desc=message)

    try:
        res = render_video(path, s, start=a, end=b, progress=report,
                           should_cancel=lambda: _cancel_requested(job_id))
    except Cancelled:
        yield None, _status("Render stopped."), gr.update(interactive=True), ""
        return
    except Exception as e:
        log.error("render failed: %s", traceback.format_exc())
        yield None, _status(f"Render failed: {e}", bad=True), gr.update(interactive=True), ""
        return
    finally:
        with _CANCEL_LOCK:
            _CANCELLED.discard(job_id)

    w, h = res["size"]
    seen = max(res["frames_seen"], 1)
    face_pct = 100.0 * res["faces_seen"] / seen
    body_pct = 100.0 * res["persons_seen"] / seen
    # gr.HTML, so the emphasis is markup rather than markdown asterisks.
    # The second line is what makes a disappointing result diagnosable: it
    # separates "the subject was never tracked" from "the effect ran and you
    # wanted more of it", which are the two things a user cannot tell apart
    # by looking at the output.
    lines = [
        f"<b>Done</b> · {res['frames']} frames at {w}×{h} in {res['seconds']:.1f}s "
        f"({res['fps']:.1f} fps)",
        f"Face on {face_pct:.0f}% of frames · body outline on {body_pct:.0f}%",
        f"Reshaped: face by {res['max_face_shift_px']:.0f} px · "
        f"body by {res['max_body_shift_px']:.0f} px",
    ]
    lines += [f"⚠️ {n}" for n in res["notes"]]
    yield res["path"], _status("<br>".join(lines)), gr.update(interactive=True), ""


def on_purge():
    items, freed = retention.purge_now()
    if not items:
        return _status("Nothing left to delete - the working directories are already empty.")
    return _status(f"Deleted {items} item{'s' if items != 1 else ''} "
                   f"({freed / 1e6:.1f} MB) — uploads, renders and previews.")


def _status(msg: str, bad: bool = False) -> str:
    colour = "#F2A3B3" if bad else "#94B8D8"
    return (f"<div class='status' style='color:{colour}'>{msg}</div>")


# ------------------------------------------------------------------------ CSS
CSS = """
:root{ --navy-deep:#020B16; --navy:#0A1E3A; --navy2:#07172E;
       --burg:#7B1935; --burg2:#97144D; --gold:#C9A227; --gold-light:#E4BA3E; }
.gradio-container,gradio-app,.dark,.app{
  --body-background-fill:#020B16!important; --body-text-color:#FFFFFF!important;
  --body-text-color-subdued:#94B8D8!important; --background-fill-primary:#0A1E3A!important;
  --background-fill-secondary:#07172E!important; --block-background-fill:#0A1E3A!important;
  --block-border-color:rgba(255,255,255,.09)!important; --block-label-text-color:#94B8D8!important;
  --block-title-text-color:#E4BA3E!important; --input-background-fill:#07172E!important;
  --input-border-color:rgba(255,255,255,.13)!important; --input-text-color:#FFFFFF!important;
  --border-color-primary:rgba(255,255,255,.09)!important; --color-accent:#C9A227!important;
  --slider-color:#C9A227!important; --checkbox-background-color-selected:#97144D!important;
  --checkbox-border-color-selected:#C9A227!important;
  color-scheme:dark!important;
}
/* Accordion headers ship as muted grey on this palette - unreadable on navy. */
.label-wrap > span,.label-wrap span,button.label-wrap span,details > summary span{
  color:#E4BA3E!important; font-weight:800!important; font-size:12.5px!important;
  letter-spacing:.4px!important;
}
.label-wrap{ padding:6px 2px!important; }
body,.gradio-container{
  background:radial-gradient(120% 70% at 50% 0%,#0A1E3A 0%,#020B16 60%)!important;
  color:#fff!important;
}
.gradio-container{ max-width:1180px!important; margin:0 auto!important; }
footer{ display:none!important; }
.app-hdr{
  background:linear-gradient(150deg,#7B1935 0%,#97144D 60%,#C4185E 120%);
  padding:16px 18px; border-radius:0 0 20px 20px; color:#fff; text-align:center;
}
.app-hdr h1{ font-size:21px; font-weight:900; margin:0; letter-spacing:.3px; }
.app-hdr p{ color:#FFD6E0; font-size:12px; margin:5px 0 0; }
.caps{ font-size:11px; color:#94B8D8; text-align:center; margin:6px 0 2px; }
.sec-lbl{ font-size:10.5px; font-weight:900; letter-spacing:.7px; text-transform:uppercase;
          color:#E4BA3E!important; margin:10px 0 4px; display:block; }
.status{ font-size:12.5px; line-height:1.55; padding:10px 12px; border-radius:10px;
         background:#07172E; border:1px solid rgba(255,255,255,.09); min-height:42px; }
.btn-p{ background:linear-gradient(135deg,#7B1935 0%,#97144D 55%,#C4185E 100%)!important;
        color:#fff!important; border:none!important; border-radius:10px!important;
        font-weight:900!important; min-height:46px!important; }
.btn-s{ background:linear-gradient(180deg,#122E55,#0D2647)!important; color:#CFE2F5!important;
        border:1px solid rgba(255,255,255,.12)!important; border-radius:10px!important;
        font-weight:800!important; min-height:46px!important; }
.note{ font-size:11.5px; color:#94B8D8; }
@media (max-width:820px){ .gradio-container{ max-width:100%!important; padding:0 6px!important; } }
"""


def build() -> gr.Blocks:
    init = preset_values(DEFAULT_PRESET)

    with gr.Blocks(title="Swamitech Beauty Studio", css=CSS, analytics_enabled=False) as demo:
        gr.HTML(
            "<div class='app-hdr'><h1>Swamitech Beauty Studio</h1>"
            "<p>HDR grading · natural skin retouch · face &amp; body shaping · hair enhancement</p></div>"
            f"<div class='caps'>{VERSION} · {capability_report()}</div>"
        )

        with gr.Row():
            # ---------------------------------------------------- left column
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Video"):
                        video_in = gr.Video(label="Source video", height=260)
                        info_md = gr.Markdown("", elem_classes=["note"])
                        with gr.Row():
                            trim_a = gr.Slider(0, 100, 0, step=1, label="Trim start (%)")
                            trim_b = gr.Slider(0, 100, 100, step=1, label="Trim end (%)")
                        preview_pos = gr.Slider(0, 100, 35, step=1, label="Preview position (%)")
                        with gr.Row():
                            preview_btn = gr.Button("Preview frame", elem_classes=["btn-s"])
                            render_btn = gr.Button("Render video", elem_classes=["btn-p"], variant="primary")
                            stop_btn = gr.Button("Stop", elem_classes=["btn-s"])
                        job_state = gr.State("")
                        with gr.Row():
                            before_img = gr.Image(label="Before", height=250, interactive=False)
                            after_img = gr.Image(label="After", height=250, interactive=False)
                        preview_note = gr.Markdown("", elem_classes=["note"])
                        status = gr.HTML(_status("Load a video, pick a preset, preview a frame, then render."))
                        video_out = gr.Video(label="Result", height=300, interactive=False)

                    with gr.Tab("Photo"):
                        photo_in = gr.Image(label="Source photo", type="numpy", height=300)
                        photo_btn = gr.Button("Enhance photo", elem_classes=["btn-p"], variant="primary")
                        photo_out = gr.Image(label="Result", height=340, interactive=False)

            # --------------------------------------------------- right column
            with gr.Column(scale=4):
                preset = gr.Dropdown(list(PRESETS.keys()), value=DEFAULT_PRESET, label="Preset")
                gr.HTML("<span class='sec-lbl'>Adjustments</span>")
                sliders: list[gr.Slider] = []
                for gi, (group, items) in enumerate(GROUPS):
                    with gr.Accordion(group, open=(gi == 0)):
                        for field, label, bipolar, tip in items:
                            lo = -100 if bipolar else 0
                            sliders.append(gr.Slider(
                                lo, 100, value=init[FIELDS.index(field)], step=1,
                                label=label, info=tip))

                gr.HTML("<span class='sec-lbl'>Output</span>")
                with gr.Accordion("Output & quality", open=False):
                    scale = gr.Dropdown(SCALE_CHOICES, value=1080,
                                        label="Working resolution",
                                        info="The effect stack runs here; the render is written at "
                                             "this size (never upscaled past the source)")
                    out_long = gr.Number(0, label="Output long edge (0 = same as working)",
                                         precision=0)
                    quality = gr.Slider(12, 30, 18, step=1, label="Encode quality (CRF)",
                                        info="Lower is better quality and a bigger file")
                    protect_skin = gr.Checkbox(True, label="Protect skin from the colour boost")
                    stabilise = gr.Checkbox(True, label="Temporal stabilisation (video)")
                    purge_btn = gr.Button("Delete my files now", elem_classes=["btn-s"])
                    hdr10 = gr.Checkbox(False, label="HDR10 export (experimental)",
                                        info="BT.2020 + PQ, if this ffmpeg build supports it. "
                                             "An inverse tone map of SDR - it does not recover "
                                             "detail the source never had.")

        extras = [protect_skin, stabilise, scale, out_long, quality, hdr10]
        controls = sliders + extras

        preset.change(lambda name: preset_values(name), inputs=preset, outputs=sliders)
        video_in.change(on_video, inputs=video_in, outputs=[info_md, before_img, after_img])
        preview_btn.click(on_preview, inputs=[video_in, preview_pos] + controls,
                          outputs=[before_img, after_img, preview_note])
        render_btn.click(on_render, inputs=[video_in, trim_a, trim_b] + controls,
                         outputs=[video_out, status, render_btn, job_state])
        stop_btn.click(on_cancel, inputs=job_state, outputs=status)
        photo_btn.click(on_photo, inputs=[photo_in] + controls, outputs=photo_out)
        purge_btn.click(on_purge, outputs=status)

        gr.Markdown(
            retention.policy_text() + " "
            "Edits are applied to the video you upload, on this machine. "
            "Shape adjustments are capped so the result stays believable - "
            "this is a retouching tool for your own footage, not an identity editor.",
            elem_classes=["note"])
    return demo


def main():
    retention.start()
    demo = build()
    demo.queue(max_size=8)
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
                server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
                show_api=False)


if __name__ == "__main__":
    main()
