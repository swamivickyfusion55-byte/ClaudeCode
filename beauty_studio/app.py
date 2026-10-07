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
import time

import cv2
import gradio as gr
import numpy as np

if __package__ in (None, ""):
    # Run as a plain script (`python beauty_studio/app.py`, which is how a
    # Hugging Face Space with app_file pointing here starts it). Put the
    # parent directory on the path and adopt this folder as the package, so
    # the relative imports below resolve whatever the folder is called.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
    __package__ = os.path.basename(_here)

from . import jobs, retention
from .landmarks import FaceTracker, PersonSegmenter
from .mp_backend import mediapipe_ready, mediapipe_status
from .pipeline import (FrameProcessor, capability_report, grab_frame, probe,
                       process_image)
from .settings import (COMPLEXIONS, DEFAULT_PRESET, HAIR_COLOURS, MAX_STACK, PRESETS,
                       Settings, combine_presets, stack_label)
from .spots import (MIN_SURROUND_SHARE, SpotMark, anchor_to_face, detect_spots,
                    dominant_surround, heal, marks_from, marks_from_stroke,
                    resolve_face, skin_region, stroke_mask)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("beauty_studio")

VERSION = "v2.4.0 (Aurora)"


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
        ("texture", "Keep texture", False, "How much real skin detail survives the smoothing (high = most natural)"),
        ("skin_texture", "Add skin texture", False,
         "The other direction: brings micro-detail back to skin that arrived flat "
         "— a phone's own beauty mode, heavy denoise, a low-bitrate upload"),
        ("auto_spots", "Find and remove marks", False,
         "Auto-detect: finds moles and dark patches anywhere on the person and "
         "takes them out, no tapping. Low finds only what is unmistakable, high "
         "finds every freckle. 0 = off"),
        ("skin_tone_amount", "Complexion strength", False,
         "How far toward the complexion chosen above. Nothing happens while it "
         "is \"none\" — use the depth control below for a nudge without picking one"),
        ("skin_depth", "Complexion depth", True,
         "Deeper (left) or lighter (right), keeping your own undertone. Works "
         "on its own or on top of a chosen complexion"),
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
        ("hair_colour_amount", "Colour strength", False,
         "How far toward the colour chosen above. Nothing happens while the "
         "colour is \"none\""),
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


def settings_from(values, skin_tone, hair_colour, hair_hue, protect_skin, stabilise,
                  scale, out_long, quality, hdr10, spots=None) -> Settings:
    kw = {}
    for field, v in zip(FIELDS, values):
        kw[field] = float(v) / 100.0
    return Settings(
        spots=list(spots or []),
        skin_tone=str(skin_tone or "none"),
        hair_colour=str(hair_colour or "none"),
        hair_hue=float(hair_hue),
        protect_skin_colour=bool(protect_skin),
        stabilise=bool(stabilise),
        process_scale=int(scale) if int(scale) > 0 else 100000,
        out_long_edge=int(out_long),
        quality=int(quality),
        hdr10=bool(hdr10),
        **kw,
    ).normalised()


def preset_extras(names):
    """The preset's named settings - the ones that are not numbers, so not
    part of the generated slider set."""
    s = combine_presets(names)
    return s.skin_tone, s.hair_colour, float(s.hair_hue)


def preset_values(names) -> list[float]:
    """Slider positions for a preset, or for a stack of up to three.

    Rounded because 0.55 * 100 is 55.00000000000001 in binary floating point,
    and that is exactly what the number box next to the slider would display.
    """
    s = combine_presets(names)
    return [round(float(getattr(s, f)) * 100.0) for f in FIELDS]


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
        return msg, _editor(frame, []), None
    except Exception as e:
        return f"**Could not read that file** · {e}", None, None


def on_scrub(path, position, marks):
    """Move the preview position: show THAT frame, not the one from before.

    This was a trap. The Before image was loaded once at 35% and left there,
    while a tap read the frame at wherever the slider had since been moved to.
    Slide to a different moment, tap the mole you can see, and the mark lands
    on whatever happens to be at those coordinates a few seconds away - so the
    tap appears to do nothing, which is exactly the report that found it.
    """
    if not path:
        return gr.update(), gr.update()
    frame = grab_frame(path, float(position) / 100.0)
    if frame is None:
        return gr.update(), gr.update()
    return _editor(frame, marks), _healed_preview(frame, marks)


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
    if s.auto_spots > 0:
        bits.append(f"auto-detect found {fp.auto_found} mark(s) on this frame")
    extra = (" · " + " · ".join(bits)) if bits else ""
    # Keep the rings: the Before frame is the only place that says which marks
    # are in play, and a preview must not quietly wipe them.
    return (_editor(frame, s.spots),
            cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
            f"Preview at {float(position):.0f}% · {found}{extra}")


def on_rub(path, position, marks, canvas, sensitivity):
    """
    A rub, or a touch, on the Before surface.

    Both arrive the same way - as paint on a layer - so there is one handler
    and one gesture. What the band is used for is aiming, not measuring: the
    marks inside it are found and sized from the picture, exactly as a tap's
    were, and what you rubbed over only decides which of them to take. So a
    sloppy stroke works, and so does catching the edge of a mole.
    """
    if not path:
        return (list(marks or []), _marks_text(marks), *_notice(
            "Load a video first.", bad=True), gr.update(), gr.update())
    frame = grab_frame(path, float(position) / 100.0)
    if frame is None:
        return (list(marks or []), _marks_text(marks), *_notice(
            "Could not read that frame.", bad=True), gr.update(), gr.update())
    painted = stroke_mask((canvas or {}).get("layers") if isinstance(canvas, dict) else None,
                          frame.shape)
    marks = list(marks or [])
    if painted is None:
        # The layer came back empty: either nothing was drawn, or this is the
        # echo of us clearing it. Either way there is nothing to do.
        return marks, _marks_text(marks), gr.update(), gr.update(), gr.update(), gr.update()

    level = (float(sensitivity) / 100.0) or 0.5
    h, w = frame.shape[:2]
    found = marks_from_stroke(frame, painted, level, limit=12)

    tracker = FaceTracker(static=True)
    try:
        faces = tracker(frame)
    finally:
        tracker.close()

    added, refused = 0, 0
    for (mx, my, radius) in found:
        if any((mx - m["x"] * w) ** 2 + (my - m["y"] * h) ** 2
               < (radius + m["radius"] * max(w, h)) ** 2 for m in marks):
            continue
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - mx) ** 2 + (yy - my) ** 2)
        _, share = dominant_surround(frame.astype(np.float32) / 255.0, dist, radius)
        if share and share < MIN_SURROUND_SHARE:
            refused += 1
            continue
        mark = SpotMark(x=mx / w, y=my / h, radius=radius / max(w, h),
                        frame=_frame_index(path, position),
                        label=f"({mx:.0f},{my:.0f})")
        marks.append(anchor_to_face(mark, faces, frame.shape).to_dict())
        added += 1

    if added:
        html, until = _notice(
            f"Rubbed out {added} mark{'s' if added != 1 else ''} — gone in "
            f"<b>② After</b>, and gone on every frame of the render. "
            f"Keep rubbing to take out more; <b>Clear spot marks</b> puts "
            f"everything back."
            + (f" {refused} spot sat on an edge and was left alone." if refused else ""))
    elif refused:
        html, until = _notice(
            f"That rub only covered an edge — where skin meets hair, a collar "
            f"or a sleeve there is no single colour to carry across. Rub over "
            f"the skin side of it.", bad=True)
    else:
        html, until = _notice("That rub landed on something already marked.")
    return (marks, _marks_text(marks), html, until,
            _editor(frame, marks), _healed_preview(frame, marks))


def _frame_index(path, position) -> int:
    try:
        info = probe(path)
        return int(np.clip(float(position) / 100.0, 0, 0.999) * max(info.frames - 1, 0))
    except Exception:
        return 0


def _marks_text(marks) -> str:
    if not marks:
        return ("**No spots marked.** Rub your finger over a mole or dark patch on "
                "**①** — roughly is fine, and a single touch works too. It "
                "disappears in **②** as you go, and from every frame of the "
                "render. Or press **Find marks on this frame** and let it look "
                "for you.")
    bits = [f"{i}. {m.get('label', '')} "
            f"{'(face)' if m.get('anchor') == 'face' else '(tracked)'}"
            for i, m in enumerate(marks or [], 1)]
    return f"**{len(marks)} spot(s):** " + " · ".join(bits)


def on_auto_find(path, position, marks, sensitivity):
    """
    Find the marks on this frame and add them as if they had been tapped.

    The slider beside this runs the same detector on every frame of the
    render, which is what catches a mole that only comes into view halfway
    through. This button is the other half of the same idea: run it once, here,
    so you can SEE what it found before committing an hour of rendering to it -
    and so you can clear it if it found something you wanted kept.
    """
    if not path:
        raise gr.Error("Load a video first.")
    frame = grab_frame(path, float(position) / 100.0)
    if frame is None:
        raise gr.Error("Could not read that frame.")
    h, w = frame.shape[:2]
    marks = list(marks or [])

    tracker, seg = FaceTracker(static=True), PersonSegmenter(static=True)
    try:
        faces = tracker(frame)
        person = seg(frame)
    finally:
        tracker.close()
        seg.close()

    allow, ref = skin_region(frame, faces, person, frame.shape)
    if allow is None:
        html, until = _notice(
            "No face or body found in this frame, so there is nowhere to look. "
            "Try a frame where the person is clearly in shot.", bad=True)
        return (marks, _marks_text(marks), html, until, _editor(frame, marks),
                _healed_preview(frame, marks))

    # The slider is the sensitivity; at zero, use a middling setting rather
    # than finding nothing, because pressing the button IS the request.
    level = (float(sensitivity) / 100.0) or 0.5
    found = detect_spots(frame, allow, level, limit=24, reference=ref)

    added = 0
    for (mx, my, radius, _strength) in found:
        if any((mx - m["x"] * w) ** 2 + (my - m["y"] * h) ** 2
               < (radius + m["radius"] * max(w, h)) ** 2 for m in marks):
            continue                      # already marked, by hand or by this
        mark = SpotMark(x=mx / w, y=my / h, radius=radius / max(w, h),
                        frame=_frame_index(path, position),
                        label=f"({mx:.0f},{my:.0f})")
        marks.append(anchor_to_face(mark, faces, frame.shape).to_dict())
        added += 1

    where = "on the face" if faces and person is None else "on the face and body"
    if added:
        html, until = _notice(
            f"Found {added} mark{'s' if added != 1 else ''} {where} and added "
            f"{'them' if added != 1 else 'it'} to the list — press <b>Preview "
            f"frame</b> to see the result, or <b>Clear spot marks</b> to undo. "
            f"Each one is followed through the whole clip.")
    else:
        html, until = _notice(
            "Nothing stood out on this frame at that sensitivity. Raise "
            "<b>Find and remove marks</b> to look harder, or tap a mark yourself.")
    return (marks, _marks_text(marks), html, until, _editor(frame, marks),
            _healed_preview(frame, marks))


def _healed_preview(frame, marks, strength: float = 0.9):
    """The frame with ONLY the spot pass applied, for the After image.

    The complaint this answers is the obvious one: you tap a mole, and the
    picture you are looking at still has the mole in it - because the Before
    frame is the before. Waiting for a full preview render to find out whether
    the tap did anything is too long a loop for something you do five times in
    a row, and the spot pass on its own takes milliseconds.
    """
    if frame is None:
        return None
    if not marks:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    spots = []
    for m in marks_from(marks):
        h, w = frame.shape[:2]
        pos = None
        if m.anchor == "face":
            tracker = FaceTracker(static=True)
            try:
                pos = resolve_face(m, tracker(frame), frame.shape)
            finally:
                tracker.close()
        if pos is None:
            pos = (m.x * w, m.y * h, m.radius * max(w, h))
        spots.append(pos)
    out = heal(frame.astype(np.float32) / 255.0, spots, strength)
    return cv2.cvtColor(np.clip(out * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def _editor(frame, marks):
    """The Before surface's value: the frame with its marks ringed, and no
    paint on it. Handing back empty layers is what wipes the stroke you just
    made, so the next rub starts on a clean picture rather than on top of the
    last one."""
    rgb = _draw_marks(frame, marks)
    if rgb is None:
        return None
    return {"background": rgb, "layers": [], "composite": rgb}


def _draw_marks(frame, marks):
    """Ring every mark on the Before frame.

    On a phone especially, a list of coordinates under the picture answers the
    wrong question. What someone needs to know after tapping - or after the
    detector has gone looking by itself - is WHICH thing it took, and the only
    honest way to say that is to draw it on the frame.
    """
    if frame is None:
        return None
    img = frame.copy()
    h, w = img.shape[:2]
    for i, m in enumerate(marks or [], 1):
        cx, cy = int(m["x"] * w), int(m["y"] * h)
        r = max(6, int(m["radius"] * max(w, h) * 1.9))
        cv2.circle(img, (cx, cy), r + 1, (12, 12, 12), 3, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, (39, 162, 201), 2, cv2.LINE_AA)   # gold, BGR
        cv2.putText(img, str(i), (cx + r + 4, cy - r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (12, 12, 12), 3, cv2.LINE_AA)
        cv2.putText(img, str(i), (cx + r + 4, cy - r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (39, 162, 201), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def on_clear_spots(path, position):
    html, until = _notice("Spot marks cleared.")
    frame = grab_frame(path, float(position) / 100.0) if path else None
    return ([], _marks_text([]), html, until, _editor(frame, []),
            _healed_preview(frame, []))


def on_photo(image, *args):
    if image is None:
        raise gr.Error("Load a photo first.")
    s = settings_from(args[:len(FIELDS)], *args[len(FIELDS):])
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(process_image(bgr, s), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ rendering
#
# The request submits; a worker thread renders. Everything below reads job
# state by id, which is what lets a render survive the tab that started it.

IDLE_HTML = "Load a video, pick a preset, preview a frame, then render."

# How long a message written by a button survives before the status line goes
# back to reporting the job. Without this the poller - which fires every two
# seconds - wipes "Stopping 78281b…" off the screen before it can be read, and
# the button looks like it did nothing.
NOTICE_SECONDS = 8.0


def _notice(msg: str, bad: bool = False):
    return _status(msg, bad), time.time() + NOTICE_SECONDS


def on_submit(path, preset_name, start, end, *args):
    """Queue a render and return at once - the work is not this request's."""
    if not path:
        raise gr.Error("Load a video first.")
    s = settings_from(args[:len(FIELDS)], *args[len(FIELDS):])
    a, b = sorted((float(start) / 100.0, float(end) / 100.0))
    if b - a < 0.01:
        a, b = 0.0, 1.0
    job = jobs.manager().submit(path, s, stack_label(preset_name), a, b)
    html, until = _notice(f"Queued as <b>{job.id[:6]}</b>. This runs on the server — "
                          f"you can close this tab and come back to it in History.")
    return job.id, html, gr.update(choices=_history_choices()), until


def on_cancel(job_id):
    """Stop a render.

    Falls back to whatever is actually running when this tab does not know a
    job id - which is the normal case for a tab opened after the render
    started, and exactly when someone most wants to stop it.
    """
    manager = jobs.manager()
    job = manager.get(job_id)
    if job is None or not job.active:
        job = manager.latest_active()
    if job is None:
        return _notice("Nothing is rendering.")
    if manager.cancel(job.id):
        return _notice(f"Stopping {job.id[:6]} after the current frame…")
    return _notice(f"Job {job.id[:6]} has already finished.")


def on_cancel_selected(job_id):
    """Stop the job picked in History, whichever tab or device started it."""
    manager = jobs.manager()
    job = manager.get(job_id)
    if job is None:
        html, until = _notice("Pick a render from the list first.")
        return html, _history_rows(), until
    if not job.active:
        html, until = _notice(f"Job {job.id[:6]} is already {job.status} — nothing to stop.")
        return html, _history_rows(), until
    manager.cancel(job.id)
    html, until = _notice(f"Stopping {job.id[:6]} after the current frame…")
    return html, _history_rows(), until


def _job_html(job) -> str:
    if job.status == "running":
        pct = job.progress * 100.0
        bar = (f"<div style='background:#07172E;border:1px solid rgba(255,255,255,.12);"
               f"border-radius:6px;height:10px;margin:6px 0;overflow:hidden'>"
               f"<div style='background:linear-gradient(90deg,#97144D,#C9A227);"
               f"height:100%;width:{pct:.1f}%'></div></div>")
        # Before the first frame lands there is no percentage to report - the
        # time is going into opening the file and starting the models - and a
        # confident "0%" reads like something is stuck.
        head = f"{pct:.0f}%" if job.done_frames else "starting"
        return (f"<b>Rendering {job.id[:6]} — {head}</b> · {job.source_name} · "
                f"{job.preset}{bar}{job.message}")
    if job.status == "queued":
        ahead = jobs.manager().queue_position(job.id)
        place = f" · {ahead} ahead of it" if ahead else " · next"
        return (f"<b>Queued</b> · {job.source_name} · waiting for the renderer{place}")
    if job.status == "done":
        lines = [f"<b>Done</b> · {job.id[:6]} · {job.done_frames} frames at "
                 f"{job.width}×{job.height} in {job.seconds:.1f}s",
                 f"Face on {job.face_pct:.0f}% of frames · body outline on "
                 f"{job.body_pct:.0f}%",
                 f"Reshaped: face by {job.face_shift:.0f} px · body by "
                 f"{job.body_shift:.0f} px"]
        lines += [f"⚠️ {n}" for n in job.notes]
        return "<br>".join(lines)
    if job.status == "cancelled":
        return f"<b>Stopped</b> · {job.id[:6]}"
    if job.status == "interrupted":
        return (f"<b>Interrupted</b> · {job.id[:6]} — the app restarted while this "
                f"was rendering. Submit it again.")
    if job.status == "expired":
        return f"<b>Expired</b> · {job.id[:6]} — deleted by the retention policy."
    return f"<b>Failed</b> · {job.id[:6]} · {job.error or job.message}"


def _history_choices():
    return [(j.label(), j.id) for j in jobs.manager().list_jobs()]


def _history_rows():
    rows = []
    for j in jobs.manager().list_jobs():
        if j.status == "running":
            if not j.done_frames:
                progress = "starting"
            else:
                progress = f"{j.progress * 100:.0f}%"
                if j.total_frames:
                    progress += f" ({j.done_frames}/{j.total_frames})"
        elif j.status == "queued":
            progress = "waiting"
        elif j.status == "done":
            progress = "100%"
        else:
            # A stopped or failed job froze somewhere; where it got to is the
            # useful thing to show, not a dash.
            progress = f"{j.progress * 100:.0f}%" if j.done_frames else "—"
        rows.append([j.id[:6], j.source_name, j.preset, j.status, progress,
                     f"{j.width}×{j.height}" if j.width else "—",
                     f"{j.size_mb():.1f} MB" if j.size_mb() else "—",
                     j.age_text()])
    return rows


def poll(job_id, shown_path, notice_until):
    """Called on a timer: the only thing keeping the page in step with the
    worker. Cheap by construction - it reads in-memory job state."""
    manager = jobs.manager()
    job = manager.get(job_id) or manager.latest_active()
    rows = _history_rows()
    choices = gr.update(choices=_history_choices())
    # A button said something recently: leave it on screen.
    quiet = bool(notice_until) and time.time() < float(notice_until)
    if job is None:
        status = gr.update() if quiet else _status(IDLE_HTML)
        return status, gr.update(), shown_path, rows, choices
    html = (gr.update() if quiet
            else _status(_job_html(job), bad=job.status in ("failed", "interrupted")))
    if (job.status == "done" and job.out_path and os.path.exists(job.out_path)
            and job.out_path != shown_path):
        # Only when it changes: handing the same path back every two seconds
        # would reload the player under whoever is watching it.
        return html, gr.update(value=job.out_path), job.out_path, rows, choices
    return html, gr.update(), shown_path, rows, choices


def on_load_history(job_id):
    job = jobs.manager().get(job_id)
    if job is None:
        raise gr.Error("Pick a render from the list first.")
    if job.status != "done" or not job.out_path or not os.path.exists(job.out_path):
        raise gr.Error(f"That render is {job.status} — there is no file to load.")
    html, until = _notice(_job_html(job))
    return job.out_path, job.out_path, html, until


def on_purge():
    items, freed = retention.purge_now()
    jobs.manager().forget_missing()
    if not items:
        return _notice("Nothing left to delete - the working directories are already empty.")
    return _notice(f"Deleted {items} item{'s' if items != 1 else ''} "
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
/* Dataframe: Gradio ships it light-on-light, which on this palette renders as
   an empty white box with invisible headers. */
.table-wrap,.table-wrap table{ background:#07172E!important; border-radius:10px!important;
  border-color:rgba(255,255,255,.10)!important; }
.table-wrap thead th,.table-wrap th{ background:#0F2D52!important; color:#E4BA3E!important;
  font-weight:800!important; font-size:11.5px!important; letter-spacing:.4px!important;
  border-color:rgba(255,255,255,.10)!important; }
.table-wrap tbody td,.table-wrap td{ background:#0A1E3A!important; color:#EAF2FB!important;
  font-size:12px!important; border-color:rgba(255,255,255,.08)!important; }
.table-wrap td span,.table-wrap th span,.cell-wrap span{ color:inherit!important; }
.table-wrap tbody tr:hover td{ background:#122E55!important; }
/* The Before image is a tap target, not a picture. On a phone a browser waits
   300 ms on every tap in case a second one follows (double-tap to zoom), and
   long-presses offer to save the image - both make tapping a mole feel broken.
   touch-action:manipulation drops the wait, and the rest stops the tap being
   taken for a drag, a text selection or a save. */
.tappable img,.tappable canvas{
  touch-action:none!important; -ms-touch-action:none!important;
  cursor:crosshair!important; -webkit-user-select:none!important; user-select:none!important;
  -webkit-touch-callout:none!important; -webkit-tap-highlight-color:rgba(201,162,39,.35)!important;
}
/* The surface is drawn on, so the browser must not claim the gesture: with
   anything but touch-action:none a drag across it scrolls the page instead of
   painting, which is the whole feature gone on a phone. */
.tappable{ touch-action:none!important; }
.tappable .image-frame,.tappable [data-testid="image"]{ cursor:crosshair!important; }

.tappable label span,.tappable .block-label{ color:#E4BA3E!important; }
@media (max-width:820px){
  .gradio-container{ max-width:100%!important; padding:0 6px!important; }
  /* The editor sizes its canvas to the component's height ONCE, on mount,
     and writes the result inline - so the picture is grown with the `height`
     argument, not from here. All this has to do is stop the two frames
     sharing a row, which left each of them half a phone wide. */

  .btn-s,.btn-p{ min-height:50px!important; }
}
"""


def build() -> gr.Blocks:
    init = preset_values(DEFAULT_PRESET)

    with gr.Blocks(title="Swamitech Beauty Studio", css=CSS, analytics_enabled=False) as demo:
        # Tapped spots. Clip data rather than settings, so switching or
        # stacking presets never clears what someone has marked. Declared up
        # here because the control list below refers to it.
        spots_state = gr.State([])
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
                        # Stacked, not side by side. The editor sizes its
                        # canvas to its container once, when it mounts, and
                        # writes the result inline - so sharing a row with the
                        # After frame left it permanently half-width, and no
                        # amount of CSS afterwards moved it. Full width is the
                        # right answer regardless: the bigger the picture, the
                        # less precision a fingertip needs.
                        #
                        # An editor rather than a picture, because the gesture
                        # people reach for is rubbing a mark out, not aiming at
                        # it. A single touch is a one-dot stroke, so the same
                        # surface takes both and there is nothing to choose.
                        before_img = gr.ImageEditor(
                            label="① Rub out a mole",
                            height=430, type="numpy", interactive=True,
                            sources=(), transforms=(), layers=False,
                            show_download_button=False,
                            brush=gr.Brush(colors=["#C9A227"], default_color="#C9A227",
                                           color_mode="fixed", default_size=18),
                            eraser=gr.Eraser(default_size=18),
                            elem_classes=["tappable"])
                        after_img = gr.Image(label="② After", height=380,
                                             interactive=False)
                        gr.Markdown(
                            "Drag your finger over a mole or dark patch on ① — "
                            "roughly is fine, a single touch works too — then press "
                            "**Erase what I rubbed**.",
                            elem_classes=["note", "howto"])
                        marks_md = gr.Markdown(_marks_text([]), elem_classes=["note", "marks"])
                        with gr.Row():
                            erase_btn = gr.Button("Erase what I rubbed",
                                                  elem_classes=["btn-s"])
                            auto_find_btn = gr.Button("Find marks on this frame",
                                                      elem_classes=["btn-s"])
                            clear_spots_btn = gr.Button("Put everything back",
                                                        elem_classes=["btn-s"])
                        preview_note = gr.Markdown("", elem_classes=["note"])
                        status = gr.HTML(_status("Load a video, pick a preset, preview a frame, then render."))
                        video_out = gr.Video(label="Result", height=300, interactive=False)

                    with gr.Tab("Photo"):
                        photo_in = gr.Image(label="Source photo", type="numpy", height=300)
                        photo_btn = gr.Button("Enhance photo", elem_classes=["btn-p"], variant="primary")
                        photo_out = gr.Image(label="Result", height=340, interactive=False)

                    with gr.Tab("History"):
                        gr.Markdown(
                            "Every render on this server, newest first — including "
                            "ones started from another tab or device. Files are kept "
                            "for the retention window; after that a row stays as a "
                            "record and reads *expired*.",
                            elem_classes=["note"])
                        history_df = gr.Dataframe(
                            headers=["job", "source", "preset", "status",
                                     "progress", "resolution", "size", "when"],
                            datatype=["str"] * 8, interactive=False, wrap=True,
                            label="Renders")
                        history_dd = gr.Dropdown(choices=[], label="Pick a render",
                                                 interactive=True)
                        with gr.Row():
                            history_load = gr.Button("Load it", elem_classes=["btn-p"],
                                                     variant="primary")
                            history_cancel = gr.Button("Cancel this job",
                                                       elem_classes=["btn-s"])
                            history_refresh = gr.Button("Refresh", elem_classes=["btn-s"])
                        history_video = gr.Video(label="Saved render", height=280,
                                                 interactive=False)
                        history_file = gr.File(label="Download", interactive=False)

            # --------------------------------------------------- right column
            with gr.Column(scale=4):
                preset = gr.Dropdown(
                    list(PRESETS.keys()), value=[DEFAULT_PRESET], multiselect=True,
                    max_choices=MAX_STACK, label=f"Presets — stack up to {MAX_STACK}",
                    info="Each preset writes only the part of the picture it is "
                         "about, so Chubby + HDR Cinematic gives you both. Order "
                         "does not matter: the specific one always wins over the "
                         "general one.")
                gr.HTML("<span class='sec-lbl'>Adjustments</span>")
                sliders: list[gr.Slider] = []
                skin_tone = None
                hair_colour = None
                hair_hue = None
                for gi, (group, items) in enumerate(GROUPS):
                    with gr.Accordion(group, open=(gi == 0)):
                        if group == "Skin & face detail":
                            # A depth and an undertone, by name. Deliberately
                            # not nationalities: there is no "Indian" or
                            # "Brazilian" skin colour - each of those names
                            # covers most of the human range, so a preset
                            # carrying one would be a guess frozen into three
                            # numbers, wrong for nearly everyone it claims to
                            # describe. The axes below are what such a label
                            # is reaching for, and they are honest about it.
                            skin_tone = gr.Dropdown(
                                list(COMPLEXIONS.keys()), value="none",
                                label="Complexion",
                                info="Shifts your skin's depth and undertone across "
                                     "the whole person — face, neck, shoulders, arms. "
                                     "Your own variation is kept, so it reads as skin "
                                     "rather than as paint.")
                        if group == "Hair":
                            # The colour is a name, not a number, so it is not
                            # part of the generated slider set - and it goes
                            # first, because the strength slider below it means
                            # nothing until a colour is chosen.
                            hair_colour = gr.Dropdown(
                                list(HAIR_COLOURS.keys()), value="none",
                                label="Hair colour",
                                info="Recolours the hair and nothing else. Going "
                                     "darker (blonde to black) is the strong "
                                     "direction; lightening dark hair is limited, "
                                     "because near-black pixels hold little detail "
                                     "to carry.")
                            hair_hue = gr.Slider(0, 359, 30, step=1,
                                                 label="Custom hue (degrees)",
                                                 info="Used when the colour above is "
                                                      "\"custom\"")
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

        extras = [skin_tone, hair_colour, hair_hue, protect_skin, stabilise, scale,
                  out_long, quality, hdr10, spots_state]
        controls = sliders + extras
        # Which finished file the player is already showing, so the poller can
        # leave it alone until it actually changes.
        shown_state = gr.State("")
        # When a button's message expires; until then the poller leaves the
        # status line alone.
        notice_state = gr.State(0.0)

        preset.change(lambda names: preset_values(names), inputs=preset, outputs=sliders)
        preset.change(lambda names: preset_extras(names), inputs=preset,
                      outputs=[skin_tone, hair_colour, hair_hue])
        video_in.change(on_video, inputs=video_in, outputs=[info_md, before_img, after_img])
        # `change`, not `release`: a slider is also a number box, and typing a
        # position into it has to refresh the frame for the same reason
        # dragging does - otherwise the picture and the frame a tap reads come
        # apart again, by another route.
        preview_pos.change(on_scrub, inputs=[video_in, preview_pos, spots_state],
                           outputs=[before_img, after_img], show_progress="minimal")
        preview_btn.click(on_preview, inputs=[video_in, preview_pos] + controls,
                          outputs=[before_img, after_img, preview_note])
        # `change` rather than `input`: on this pinned Gradio, `input` does not
        # fire for a brush stroke. The echo of handing the cleared canvas back
        # is harmless because a canvas with no paint on it returns early.
        rub_args = dict(fn=on_rub,
                        inputs=[video_in, preview_pos, spots_state, before_img,
                                sliders[FIELDS.index("auto_spots")]],
                        outputs=[spots_state, marks_md, status, notice_state,
                                 before_img, after_img])
        before_img.change(**rub_args)
        erase_btn.click(**rub_args)
        auto_find_btn.click(
            on_auto_find,
            inputs=[video_in, preview_pos, spots_state,
                    sliders[FIELDS.index("auto_spots")]],
            outputs=[spots_state, marks_md, status, notice_state, before_img,
                     after_img])
        clear_spots_btn.click(on_clear_spots, inputs=[video_in, preview_pos],
                              outputs=[spots_state, marks_md, status, notice_state,
                                       before_img, after_img])
        render_btn.click(on_submit,
                         inputs=[video_in, preset, trim_a, trim_b] + controls,
                         outputs=[job_state, status, history_dd, notice_state])
        stop_btn.click(on_cancel, inputs=job_state, outputs=[status, notice_state])
        history_load.click(on_load_history, inputs=history_dd,
                           outputs=[history_video, history_file, status, notice_state])
        history_cancel.click(on_cancel_selected, inputs=history_dd,
                             outputs=[status, history_df, notice_state])
        history_refresh.click(lambda: (_history_rows(), gr.update(choices=_history_choices())),
                              outputs=[history_df, history_dd])

        # The heartbeat. Everything the page knows about a running render comes
        # through here, which is why closing the tab costs nothing: the work is
        # not in the request, only the view of it is.
        demo.load(poll, inputs=[job_state, shown_state, notice_state],
                  outputs=[status, video_out, shown_state, history_df, history_dd],
                  every=2)
        photo_btn.click(on_photo, inputs=[photo_in] + controls, outputs=photo_out)
        purge_btn.click(on_purge, outputs=[status, notice_state])

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
