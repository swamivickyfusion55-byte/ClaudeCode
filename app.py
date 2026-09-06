"""
Swamitech Phoenix — Gradio UI entry point (HF Spaces app_file).

Core pipeline logic lives in core_pipeline.py
Tunable constants live in config.py
"""
from __future__ import annotations

import logging
import os

import gradio as gr

# Stable machine API for Phoenix Mobile / external clients.
from phoenix_api_adapter import api_submit_video, api_job_status, api_download, api_cancel, api_detect_video_frame

# Import pipeline (models, jobs, swap, temporal, device helpers, …)
from core_pipeline import (  # noqa: F401
    HAS_SPACES,
    _device_status_text,
    _hist_html,
    _status_banner_html,
    _video_progress_html,
    _get_done_choices,
    _running_job_choices,
    _parse_device_mode,
    _device_pref,
    _load,
    _cuda_available,
    _loaded_device,
    detect_image,
    detect_video,
    show_frame,
    capture_face,
    swap_image,
    submit_video,
    cancel_running,
    load_result,
    delete_now,
    clear_history,
    _load_ui_session,
    _safe_session_id,
    RES,
    jobs,
    _lock,
)

try:
    from config import VERSION_FULL, DUR as _DUR, FPS as _FPS, RES as _RES
except Exception:
    VERSION_FULL = "v11.2.3 (SolidFace)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")



def _calc_trim_duration(video, start, end):
    if video is None:
        return "Full video"
    try:
        import cv2
        cap = cv2.VideoCapture(video)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
        cap.release()
        return f"{total * (float(end) - float(start)) / 100:.1f}s trimmed"
    except Exception as e:
        logging.warning("trim duration: %s", e)
        return "Full video"


# ==================== DARK THEME ====================
CSS = """
:root{
  --navy-deep:#001428; --navy-mid:#0A2040; --burg:#7B1935; --burg2:#97144D;
  --gold:#C9A227; --gold-light:#E4BA3E;
  --btn-h:52px; --banner-h:76px; --prog-h:128px; --hist-h:260px;
}
.gradio-container,gradio-app,.dark,.app{
  --body-background-fill:#020B16!important; --body-text-color:#FFFFFF!important;
  --body-text-color-subdued:#94B8D8!important; --background-fill-primary:#0A1E3A!important;
  --background-fill-secondary:#07172E!important; --block-background-fill:#0A1E3A!important;
  --block-border-color:rgba(255,255,255,.09)!important; --block-label-text-color:#94B8D8!important;
  --block-title-text-color:#E4BA3E!important; --input-background-fill:#07172E!important;
  --input-border-color:rgba(255,255,255,.13)!important; --input-text-color:#FFFFFF!important;
  --border-color-primary:rgba(255,255,255,.09)!important; --color-accent:#C9A227!important;
  color-scheme:dark!important;
}
body,.gradio-container{ background:radial-gradient(120% 70% at 50% 0%,#0A1E3A 0%,#020B16 60%)!important; color:#FFFFFF!important; }
.gradio-container{ max-width:520px!important; margin:0 auto!important; }
footer{ display:none!important; }

/* ---------- Header ---------- */
.app-hdr{
  background:linear-gradient(150deg,#7B1935 0%,#97144D 60%,#C4185E 120%);
  padding:calc(env(safe-area-inset-top)+14px) 14px 14px; text-align:center;
  border-radius:0 0 20px 20px; color:white;
}
.app-hdr h1{ font-size:18px; font-weight:900; margin:0; }
.app-hdr p{ color:#FFD6E0; font-size:11px; margin-top:4px; }

/* ---------- Tabs: flex row, equal pills ---------- */
.tabs .tab-nav,div[role="tablist"]{
  display:flex!important; flex-direction:row!important; flex-wrap:nowrap!important;
  align-items:stretch!important; gap:6px!important;
  background:linear-gradient(180deg,#0F2D52,#0A2040)!important;
  border:1px solid rgba(255,255,255,.09)!important; border-radius:13px!important; padding:5px!important;
}
.tabs .tab-nav button,button[role="tab"]{
  flex:1 1 0!important; min-height:40px!important;
  background:linear-gradient(180deg,#122E55,#0D2647)!important; color:#94B8D8!important;
  border:1px solid rgba(255,255,255,.10)!important; border-radius:10px!important; font-weight:800!important;
}
.tabs .tab-nav button.selected,button[role="tab"][aria-selected="true"]{
  background:linear-gradient(150deg,#7B1935,#97144D)!important; color:#fff!important; border-color:transparent!important;
}
.sec-lbl{ font-size:10.5px; font-weight:900; letter-spacing:.7px; text-transform:uppercase; color:#E4BA3E!important; margin:10px 0 6px; display:block; }

/* ---------- Button rows: flex, nowrap, fixed height (no jump) ---------- */
/* Maps: display:flex + flex-wrap:nowrap + align-items:stretch + flex:1 0 0 on children */
.btn-bar{
  display:flex!important;
  flex-direction:row!important;
  flex-wrap:nowrap!important;
  align-items:stretch!important;
  justify-content:flex-start!important;
  gap:8px!important;
  min-height:var(--btn-h)!important;
  margin:6px 0 10px 0!important;
  width:100%!important;
  box-sizing:border-box!important;
}
/* Gradio wraps Row children — force inner form row to flex too */
.btn-bar.row, .btn-bar > .form, .btn-bar > div{
  display:flex!important; flex-direction:row!important; flex-wrap:nowrap!important;
  align-items:stretch!important; width:100%!important; gap:8px!important;
  min-height:var(--btn-h)!important;
}
.btn-p{
  background:linear-gradient(135deg,#7B1935 0%,#97144D 55%,#C4185E 100%)!important;
  color:#fff!important; border:none!important; border-radius:10px!important;
  font-weight:900!important; font-size:15px!important;
  min-height:var(--btn-h)!important; height:var(--btn-h)!important;
}
.btn-s{
  background:linear-gradient(180deg,#122E55,#0D2647)!important; color:#B8CFEA!important;
  border:1.5px solid rgba(255,255,255,.13)!important; border-radius:10px!important;
  font-weight:800!important; font-size:14px!important;
  min-height:var(--btn-h)!important; height:var(--btn-h)!important;
}
.btn-bar button, .btn-fixed{
  flex:1 0 0!important;           /* equal share, don't shrink below basis messily */
  flex-shrink:0!important;
  min-height:var(--btn-h)!important; height:var(--btn-h)!important;
  width:100%!important; max-height:var(--btn-h)!important;
  font-size:15px!important; font-weight:800!important;
  box-sizing:border-box!important;
}
.btn-danger{
  background:linear-gradient(180deg,#3A1520,#2A0F18)!important; color:#F0A0B0!important;
  border:1.5px solid rgba(200,80,100,.35)!important; border-radius:10px!important;
  font-weight:800!important; font-size:15px!important;
  flex:1 0 0!important; min-height:var(--btn-h)!important; height:var(--btn-h)!important; width:100%!important;
}
.btn-defaults{
  background:linear-gradient(135deg,#8B6A10 0%,#C9A227 55%,#E4BA3E 100%)!important;
  color:#001428!important; border:none!important; border-radius:10px!important;
  font-weight:900!important; min-height:40px!important;
}
.actions-lock{ position:relative!important; z-index:5!important; flex:0 0 auto!important; }

/* ---------- Status banner: column flex, fixed height slot ---------- */
/* Maps: flex-direction:column + justify-content:center + fixed height + overflow hidden */
.stat-banner{
  display:flex!important;
  flex-direction:column!important;
  justify-content:center!important;
  align-items:stretch!important;
  gap:4px!important;
  background:linear-gradient(90deg,#1A0A14 0%,#0F2D52 40%,#07172E 100%);
  border:1.5px solid rgba(201,162,39,.55); border-radius:12px;
  padding:8px 12px; margin:0 0 10px 0;
  height:var(--banner-h)!important; min-height:var(--banner-h)!important; max-height:var(--banner-h)!important;
  box-sizing:border-box!important; overflow:hidden!important;
}
.stat-banner .sb-left{
  display:flex!important; flex-direction:row!important; flex-wrap:nowrap!important;
  align-items:center!important; gap:8px!important; min-height:24px;
}
.stat-banner .sb-pct{
  flex:0 0 auto; min-width:52px;
  font-size:20px; font-weight:900; color:#FFE08A!important;
  text-shadow:0 1px 2px rgba(0,0,0,.9); letter-spacing:0.3px;
}
.stat-banner .sb-id{ flex:0 0 auto; font-size:11px; font-weight:800; color:#E4BA3E!important; white-space:nowrap; }
.stat-banner .sb-msg{
  flex:1 1 auto; min-width:0;          /* allow shrink for ellipsis */
  font-size:11px; font-weight:600; color:#D7E6F8!important;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; line-height:1.2;
}
.stat-banner .sb-bar{
  flex:0 0 6px; width:100%; height:6px; background:#07172E; border-radius:6px; overflow:hidden;
  border:1px solid rgba(201,162,39,.3);
}
.stat-banner .sb-fill{ height:6px; border-radius:6px; background:linear-gradient(90deg,#C9A227,#F5D76E); }
.stat-banner.idle{ opacity:0.9; }
.stat-banner.idle .sb-pct{ color:#8FA8C4!important; font-size:16px; }

.banner-slot, .banner-slot > .wrap, .banner-slot > div,
.banner-slot .html-container, .banner-slot .prose{
  height:var(--banner-h)!important; min-height:var(--banner-h)!important; max-height:var(--banner-h)!important;
  overflow:hidden!important; box-sizing:border-box!important;
}

/* ---------- Progress card: fixed height column ---------- */
.vprog{
  display:flex!important; flex-direction:column!important; justify-content:flex-start!important;
  background:linear-gradient(165deg,#0F2D52 0%,#07172E 100%);
  border:1.5px solid rgba(201,162,39,.45); border-radius:14px; padding:10px 12px;
  height:var(--prog-h)!important; min-height:var(--prog-h)!important; max-height:var(--prog-h)!important;
  box-sizing:border-box!important; overflow:hidden!important;
}
.vp-pct{
  flex:0 0 auto;
  font-size:28px; font-weight:900; line-height:1.1; color:#FFE08A!important;
  text-shadow:0 1px 2px rgba(0,0,0,.85); letter-spacing:0.5px; margin:6px 0 4px 0;
}
.vp-head{ flex:0 0 auto; color:#B8CFEA!important; font-size:12px; font-weight:700; }
.vp-id{ color:#FFE08A!important; font-weight:800; margin-right:8px; }
.vp-msg{
  flex:0 0 auto; color:#D7E6F8!important; font-size:12.5px; font-weight:600; margin-top:6px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.vprog-bar{ flex:0 0 12px; background:#07172E; border-radius:7px; height:12px; overflow:hidden; border:1px solid rgba(201,162,39,.25); }
.vprog-fill{ height:12px; border-radius:7px; background:linear-gradient(90deg,#C9A227,#E4BA3E,#F5D76E); }
.prog-slot, .prog-slot > .wrap, .prog-slot > div, .prog-slot .html-container{
  height:var(--prog-h)!important; min-height:var(--prog-h)!important; max-height:var(--prog-h)!important;
  overflow:hidden!important; box-sizing:border-box!important;
}

/* ---------- History list: flex child that scrolls inside fixed height ---------- */
/* Maps: flex:1 1 auto + max-height + overflow-y:auto */
.hist-panel{
  display:block!important;
  min-height:180px!important; max-height:var(--hist-h)!important; height:var(--hist-h)!important;
  overflow-y:auto!important; overflow-x:hidden!important;
  box-sizing:border-box!important;
}
.hcard{
  display:flex; flex-direction:column; gap:4px;
  background:linear-gradient(180deg,#0F2D52,#0A2040);
  border:1px solid rgba(255,255,255,.09); border-left:4px solid #7B1935;
  border-radius:0 12px 12px 0; padding:10px 12px; margin-bottom:8px;
}
.hid{ background:linear-gradient(135deg,#001428,#0A2040)!important; color:#fff!important; font-size:10.5px; font-weight:800; padding:3px 9px; border-radius:6px; }
.bdg-ok{ background:linear-gradient(135deg,#10B981,#34D399); color:#001428; }
.bdg-err{ background:linear-gradient(135deg,#7B1935,#97144D); color:#fff; }
.bdg-run{ background:linear-gradient(135deg,#003974,#1A3566); color:#fff; }

/* ---------- Cancel dropdown: capped height ---------- */
.cancel-box, div:has(> .cancel-box){
  min-height:56px!important; max-height:120px!important;
}

/* ---------- Radios / hints / status / footer ---------- */
label.svelte-1b6s6s, .wrap > label, .gr-radio label, div[data-testid="radio"] label {
  background:#0D2647!important; border:1.5px solid rgba(255,255,255,0.15)!important; border-radius:10px!important;
  color:#94B8D8!important; padding:8px 14px!important; margin:4px 3px!important; font-weight:700!important;
}
label.svelte-1b6s6s:has(input:checked), .wrap > label:has(input:checked), .gr-radio label:has(input:checked),
div[data-testid="radio"] label:has(input:checked), label.selected {
  background:linear-gradient(135deg,#7B1935,#97144D)!important; border-color:#C9A227!important; color:#FFFFFF!important;
  box-shadow:0 0 0 2px rgba(201,162,39,0.45)!important;
}
.hint{ font-size:11px; color:#94B8D8!important; background:linear-gradient(180deg,#0D2647,#07172E); border:1px solid rgba(201,162,39,.22); border-radius:10px; padding:9px 11px; margin-bottom:9px; }
.hint b{ color:#E4BA3E!important; }
.status-bx textarea,.status-bx input{
  background:#07172E!important; border:1.5px solid rgba(201,162,39,.30)!important;
  color:#E4BA3E!important; font-weight:800!important; border-radius:10px!important;
  min-height:42px!important; max-height:64px!important;
}
.app-ftr{ text-align:center; padding:13px 14px; color:#B8D0E8!important; font-size:11px; }
.hmsg,.vp-msg,.sb-msg,.empty p{ color:#EAF2FF!important; }
.sb-pct,.vp-pct{ color:#FFE566!important; font-weight:900!important; text-shadow:0 1px 2px rgba(0,0,0,.5); }
.hint{ color:#C5D9F0!important; }
.hint b{ color:#FFD666!important; }
.settings-card{
  padding:12px 14px; margin:8px 0 12px; border-radius:14px; font-size:12.5px; font-weight:700;
  color:#EAF2FF!important;
  background:linear-gradient(145deg,#0F2A4D 0%,#0A1C36 55%,#071428 100%);
  border:1px solid rgba(201,162,39,.28);
  box-shadow:0 10px 28px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.06);
}
.settings-card b{ color:#FFD666!important; }
.vprog,.stat-banner,.hcard{
  box-shadow:0 10px 28px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.06)!important;
  border:1px solid rgba(201,162,39,.25)!important; border-radius:14px!important;
}
.status-bx textarea,.status-bx input{
  color:#FFE566!important; background:#0A1F3A!important;
  border:1.5px solid rgba(201,162,39,.45)!important; font-weight:800!important;
}
label{ color:#C8DCF0!important; font-weight:700!important; }

textarea,input,select{ background:#07172E!important; border:1.5px solid rgba(255,255,255,.13)!important; color:#FFFFFF!important; border-radius:10px!important; }
label{ color:#D8E8F8!important; font-weight:800!important; }

/* ---------- High-contrast 3D polish ---------- */
.gradio-container .block, .gradio-container .form, .gradio-container .panel, .gradio-container .tabs{
  border-color:rgba(255,255,255,.10)!important;
}
.stat-banner,.vprog,.hcard,.settings-card{
  box-shadow:0 12px 26px rgba(0,0,0,.52), inset 0 1px 0 rgba(255,255,255,.10), inset 0 -1px 0 rgba(0,0,0,.35)!important;
}
.btn-p,.btn-s,.btn-danger,.btn-defaults{
  box-shadow:0 7px 14px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.16)!important;
  transition:transform .12s ease, box-shadow .12s ease, filter .12s ease!important;
}
.btn-p:hover,.btn-s:hover,.btn-danger:hover,.btn-defaults:hover{
  transform:translateY(-1px)!important; filter:brightness(1.08)!important;
  box-shadow:0 10px 18px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.18)!important;
}
.btn-p:active,.btn-s:active,.btn-danger:active,.btn-defaults:active{
  transform:translateY(1px)!important; box-shadow:0 4px 8px rgba(0,0,0,.38)!important;
}
.sb-meta,.vp-meta{
  color:#BFE7FF!important; font-size:10.5px!important; font-weight:800!important; margin-left:auto!important; white-space:nowrap!important;
}
.vp-head{ display:flex!important; align-items:center!important; gap:7px!important; color:#D8E8F8!important; }
.vp-idle-text{ color:#D8E8F8!important; font-weight:800!important; }
.bdg-run{ background:linear-gradient(135deg,#0B6B9A,#168AC0)!important; color:#FFFFFF!important; border:1px solid rgba(255,255,255,.18)!important; }
.bdg-ok{ color:#05251A!important; font-weight:900!important; }
.bdg-err{ color:#FFFFFF!important; font-weight:900!important; }
.gradio-container .wrap{ box-shadow:inset 0 1px 0 rgba(255,255,255,.035)!important; }
"""

DUR = [10,20,30,60,90,120,150,180,240,300,360]
FPS = [15,24,30,40,50,60]

def apply_speed():
    # Fast path for CPU
    return "640p (Fast)", gr.update(), 30, "Balanced", "None", "3", "3", "6"

def apply_balanced():
    # Practical balance — Swap=1 for less flicker
    return "720p (HD)", gr.update(), 30, "Best", "None", "1", "2", "4"

def apply_mobile_hq():
    # Mobile HQ: 680p, Swap=1, no polish (avoids shiny patch)
    return "680p", gr.update(), 30, "Ultra", "None", "1", "2", "4"

def apply_hq():
    # HQ: higher res, Swap=1, no polish by default
    return "900p (HD+)", gr.update(), 30, "Ultra", "None", "1", "1", "4"

def apply_quality():
    # Quality: 720p, Swap=1, no polish by default
    return "720p (HD)", gr.update(), 30, "Ultra", "None", "1", "1", "4"

def apply_optimized():
    # HF Pro CPU default: 720p, self-managed skip/det, no enhancer, fast encode.
    return "720p (HD)", gr.update(), 30, "Optimized", "None", "Auto", "Auto", "2"

def apply_stable():
    # Maximum temporal stability — Ultra quality, swap every frame, det every keyframe.
    # (Best previously still allowed adaptive skip; Ultra+swap_n=1 locks gap=1.)
    return "720p (HD)", gr.update(), 30, "Ultra", "None", "1", "1", "1"

with gr.Blocks(title="Swamitech Phoenix", analytics_enabled=False, css=CSS) as demo:
    img_refs = gr.State([None, None, None, None])
    vid_refs = gr.State([None, None, None, None])

    gr.HTML('<div class="app-hdr"><h1>🎬 Swamitech Phoenix</h1><p>Aligned-space compositing · Every-frame reuse · Phoenix v11.2.3 (SolidFace)</p></div>')

    with gr.Tabs():
        with gr.Tab("Image"):
            gr.HTML('<span class="sec-lbl">Target Photo</span>')
            im_tgt = gr.Image(type="numpy", sources=["upload"], height=160)
            with gr.Row(elem_classes="btn-bar"):
                im_detect = gr.Button("🔍 Detect Faces", elem_classes="btn-fixed btn-s", scale=1, min_width=160)
            im_prev = gr.Image(interactive=False, height=140)
            gr.HTML('<span class="sec-lbl">Replacement Faces</span>')
            with gr.Row():
                im_df1 = gr.Image(label="#1", interactive=False, height=90)
                im_s1 = gr.Image(label="Replace #1", type="numpy", sources=["upload"], height=90)
            with gr.Row():
                im_df2 = gr.Image(label="#2", interactive=False, height=90)
                im_s2 = gr.Image(label="Replace #2", type="numpy", sources=["upload"], height=90)
            with gr.Row():
                im_df3 = gr.Image(label="#3", interactive=False, height=90)
                im_s3 = gr.Image(label="Replace #3", type="numpy", sources=["upload"], height=90)
            with gr.Row():
                im_df4 = gr.Image(label="#4", interactive=False, height=90)
                im_s4 = gr.Image(label="Replace #4", type="numpy", sources=["upload"], height=90)
            im_q = gr.Radio(["Fast","Balanced","Best","Ultra"], value="Best", label="Quality")
            gr.HTML('<span class="sec-lbl">Actions</span>')
            with gr.Row(elem_classes="btn-bar"):
                im_go = gr.Button("✨ Swap Faces", elem_classes="btn-fixed btn-p", scale=2, min_width=160)
            with gr.Row(elem_classes="btn-bar"):
                im_clr = gr.Button("Clear image inputs", elem_classes="btn-fixed btn-s", scale=1, min_width=160)
            im_out = gr.Image(interactive=False, height=200)
            im_st = gr.Textbox(interactive=False, elem_classes="status-bx")

        with gr.Tab("Video"):
            v_banner = gr.HTML(value=_status_banner_html(), elem_classes="banner-slot")
            gr.HTML('<span class="sec-lbl">Target Video</span>')
            v_vid = gr.Video(sources=["upload"], height=150)
            with gr.Row(elem_classes="btn-bar"):
                v_detect = gr.Button("🔍 Detect Faces", elem_classes="btn-fixed btn-s", scale=1, min_width=160)
            v_prev = gr.Image(interactive=False, height=140)

            gr.HTML('<span class="sec-lbl">Trim (Optional)</span>')
            with gr.Row():
                v_trim_start = gr.Slider(0, 100, value=0, step=0.5, label="Start %")
                v_trim_end = gr.Slider(0, 100, value=100, step=0.5, label="End %")
            v_trim_duration = gr.Textbox(value="Full video", interactive=False, elem_classes="status-bx")

            # ===== ALWAYS VISIBLE: Pick faces from different frames =====
            gr.HTML('<span class="sec-lbl">🎞️ Pick faces from different frames</span>')
            gr.HTML('<div class="hint">Use this when the two people never appear together. Scrub the slider → Show frame → then Capture the face into the correct slot.</div>')
            v_scrub = gr.Slider(0, 100, value=0, step=1, label="Frame position (%)")
            with gr.Row(elem_classes="btn-bar"):
                v_show = gr.Button("Show frame", elem_classes="btn-fixed btn-s", scale=1, min_width=160)
            v_frame = gr.Image(label="Frame at this position", interactive=False, height=170)
            with gr.Row(elem_classes="btn-bar"):
                v_cap1 = gr.Button("Capture → #1", elem_classes="btn-fixed btn-s", scale=1, min_width=120)
                v_cap2 = gr.Button("Capture → #2", elem_classes="btn-fixed btn-s", scale=1, min_width=120)
            with gr.Row(elem_classes="btn-bar"):
                v_cap3 = gr.Button("Capture → #3", elem_classes="btn-fixed btn-s", scale=1, min_width=120)
                v_cap4 = gr.Button("Capture → #4", elem_classes="btn-fixed btn-s", scale=1, min_width=120)

            gr.HTML('<span class="sec-lbl">Replacement Faces</span>')
            with gr.Row():
                v_df1 = gr.Image(label="#1", interactive=False, height=90)
                v_s1 = gr.Image(label="Replace #1", type="numpy", sources=["upload"], height=90)
            with gr.Row():
                v_df2 = gr.Image(label="#2", interactive=False, height=90)
                v_s2 = gr.Image(label="Replace #2", type="numpy", sources=["upload"], height=90)
            with gr.Row():
                v_df3 = gr.Image(label="#3", interactive=False, height=90)
                v_s3 = gr.Image(label="Replace #3", type="numpy", sources=["upload"], height=90)
            with gr.Row():
                v_df4 = gr.Image(label="#4", interactive=False, height=90)
                v_s4 = gr.Image(label="Replace #4", type="numpy", sources=["upload"], height=90)

            gr.HTML('<span class="sec-lbl">Quick Start Presets</span>')
            preset = gr.Radio(
                ["⚡ Speed", "⭐ Balanced", "🚀 Optimized (CPU · Recommended)", "🛡️ Stable (most detail)", "📱 Mobile HQ (680p)", "💎 Quality", "🏆 HQ"],
                value="🚀 Optimized (CPU · Recommended)",
                label="Choose a preset — the selected one stays highlighted"
            )
            gr.HTML('''<div class="hint">
• <b>Speed</b>: 640p · Swap=3 (fastest; expression refreshes less often)<br>
• <b>Balanced</b>: 720p · Swap=1<br>
• <b>Optimized (Recommended)</b>: 720p · self-managed skip/det · HF Pro CPU default<br>
• <b>Stable</b>: 720p · Swap=1 · re-detect every 1 (face refreshed every frame)<br>
• <b>Mobile HQ</b>: 680p · Ultra · Swap=1 · no enhancer<br>
• <b>Quality / HQ</b>: higher quality · Swap=1<br>
💡 Lower <b>Swap every N</b> = smoother face, slower. Use Speed Controls below to fine-tune.
</div>''')

            with gr.Row():
                v_sec = gr.Dropdown(DUR, value=30, label="Duration (sec) — 180/240/300/360 available")
                v_fps = gr.Dropdown(FPS, value=30, label="FPS")
            v_res = gr.Dropdown(list(RES.keys()), value="720p (HD)", label="Resolution")
            v_q = gr.Radio(["Fast","Balanced","Optimized","Best","Ultra"], value="Optimized", label="Quality")
            gr.HTML('''<div class="hint">
💡 <b>Optimized</b> — HF Pro CPU default (v11.2.3 SolidFace): the swap network runs every ~5th frame, but <b>every</b> output frame is still composited from the cached aligned result using its own geometry, mask and lighting. Raising the skip interval costs expression freshness, not face presence or placement.
</div>''')
            gr.HTML('<span class="sec-lbl">Speed settings</span>')
            v_settings_summary = gr.HTML(
                value='<div class="settings-card">Swap every N=<b>Auto</b> · Re-detect every N=<b>Auto</b> · Det interval=<b>2</b></div>'
            )
            with gr.Row():
                v_swap = gr.Dropdown(["Auto","1","2","3","4","6","8","10"], value="Auto", label="Swap every N (manual)")
                v_det = gr.Dropdown(["Auto","1","2","3","4","6","8"], value="Auto", label="Re-detect every N (manual)")
                v_det_int = gr.Dropdown(["2","4","6","8","12","16"], value="2", label="Det Interval (manual)")
            gr.HTML('<div class="hint"><b>Auto</b> lets the Quality preset above drive the frame-skip (this is what makes Fast/Balanced/Optimized actually faster). Setting an explicit number overrides the preset — <b>Swap=1</b> + <b>Re-detect=1</b> refreshes the face on every frame: the most expression detail, and the slowest. Since v11.1.0 every output frame is composited whatever these are set to, so a higher number costs expression freshness rather than face presence or placement. Multi-face jobs always process every frame regardless, for correctness.</div>')
            v_enh = gr.Dropdown(
                [
                    "None",
                    "Cinematic (clarity + smooth)",
                    "Soft polish (fast)",
                    "Bilateral smooth",
                    "Mild sharpen",
                    "Detail boost",
                    "OpenCV DNN (full)",
                    "OpenCV DNN (fast)",
                    "GFPGAN",
                    "CodeFormer",
                    "Real-ESRGAN (face)",
                    "GPEN",
                ],
                value="None",
                label="Face Enhancer / polish",
            )
            gr.HTML('''<div class="hint">
• <b>None (recommended)</b> — cleanest, no shiny patches<br>
• <b>Soft polish / Bilateral / Sharpen / Detail</b> — bilateral only (no unsharp/CLAHE — SDOS)<br>
• <b>OpenCV DNN (full)</b> — EDSR ×2 on face ROI + denoise<br>
• <b>OpenCV DNN (fast)</b> — FSRCNN ×2 (lighter)<br>
• <b>GFPGAN / CodeFormer / Real-ESRGAN / GPEN</b> — heavier AI (may be unavailable on HF)
</div>''')
            v_enh_scope = gr.Radio(
                ["Primary face only (faster)", "All faces"],
                value="Primary face only (faster)",
                label="Enhancer scope (2+ faces)"
            )
            gr.HTML('''<div class="hint">
• <b>Primary only</b> — GFPGAN on face #1; other faces get swap + color match only (default, faster)<br>
• <b>All faces</b> — restore every swapped face (slower, most uniform look)
</div>''')

            gr.HTML('<span class="sec-lbl">Compute device</span>')
            v_device = gr.Radio(
                ["GPU if available", "CPU only"],
                value="CPU only",
                label="Where should the models run?"
            )
            v_device_status = gr.Textbox(
                value=_device_status_text(),
                label="Device status (auto-refresh 10s)",
                interactive=False,
                elem_classes="status-bx",
            )
            gr.HTML('''<div class="hint">
• <b>GPU if available</b> — uses ZeroGPU/CUDA when the Space has it; auto-falls back to CPU if not<br>
• <b>CPU only</b> — never requests GPU (best for long jobs on free CPU Spaces)<br>
• ZeroGPU free quota is limited; long clips may fall back to CPU mid-job
</div>''')

            gr.HTML('<span class="sec-lbl">Faces to swap (efficiency)</span>')
            v_face_mode = gr.Radio(
                ["1 face (fastest)", "2 faces", "Multiple faces"],
                value="1 face (fastest)",
                label="How many faces should be swapped?"
            )
            gr.HTML('''<div class="hint">
• <b>1 face</b> — only the main subject (largest / matched). ~50% of clips · least CPU<br>
• <b>2 faces</b> — two people max · recommended for couple scenes<br>
• <b>Multiple</b> — up to 4 faces (slowest)
</div>''')


            v_pw = gr.Textbox(label="Password (optional · enables encrypted autosave)", type="password", placeholder="Leave blank for normal mp4")

            gr.HTML('<div class="actions-lock"><span class="sec-lbl">Actions</span></div>')
            with gr.Row(elem_classes="btn-bar"):
                v_go = gr.Button("🚀 Start Swap", elem_classes="btn-fixed btn-p", scale=2, min_width=180)
            with gr.Row(elem_classes="btn-bar"):
                v_cancel = gr.Button("✋ Cancel selected / all running", elem_classes="btn-fixed btn-s", scale=1, min_width=180)
            with gr.Row(elem_classes="btn-bar"):
                v_clr = gr.Button("Clear video inputs", elem_classes="btn-fixed btn-s", scale=1, min_width=180)
            with gr.Row(elem_classes="btn-bar"):
                v_refresh = gr.Button("⟳ Refresh status", elem_classes="btn-fixed btn-s", scale=1, min_width=180)

            # Dropdown keeps fixed height (CheckboxGroup grew and shoved buttons around)
            v_cancel_pick = gr.Dropdown(
                choices=[],
                value=[],
                multiselect=True,
                label="Running jobs to cancel (optional — empty = all)",
                elem_classes=["cancel-box"],
            )

            gr.HTML('<span class="sec-lbl">Progress</span>')
            v_prog = gr.HTML(value=_video_progress_html(), elem_classes="prog-slot")
            v_st = gr.Textbox(interactive=False, elem_classes="status-bx")

        with gr.Tab("History"):
            h_banner = gr.HTML(value=_status_banner_html(), elem_classes="banner-slot")
            # Fixed controls FIRST so live HTML refresh never moves the click targets
            gr.HTML('<span class="sec-lbl">History actions (fixed)</span>')
            with gr.Row(elem_classes="btn-bar"):
                h_ref = gr.Button("⟳ Refresh list", elem_classes="btn-fixed btn-s", scale=1, min_width=140)
            h_dd = gr.Dropdown(choices=[], label="Completed jobs")
            with gr.Row(elem_classes="btn-bar"):
                h_btn = gr.Button("⬇ Load selected", elem_classes="btn-fixed btn-p", scale=2, min_width=160)
                h_del = gr.Button("🗑 Delete selected", elem_classes="btn-fixed btn-s", scale=1, min_width=140)
            with gr.Row(elem_classes="btn-bar"):
                h_clr = gr.Button("🗑 Clear finished only", elem_classes="btn-danger", scale=1, min_width=200)
            gr.HTML('<div class="hint">Clear removes <b>finished / failed / cancelled</b> jobs only — running jobs are kept. Refresh is on its own row so it cannot be mis-tapped.</div>')
            h_msg = gr.Textbox(interactive=False, elem_classes="status-bx")
            h_vid = gr.Video(label="Preview", visible=False)
            h_file = gr.File(label="⬇ Download Video", visible=False)
            gr.HTML('<span class="sec-lbl">Job list</span>')
            h_html = gr.HTML(value=_hist_html(), elem_classes="hist-panel")

    # -----------------------------------------------------------------------
    # Phoenix Mobile / external API bridge
    # Keep these hidden controls INSIDE the Blocks context.
    # Gradio 4.44.1 rejects event registration outside Blocks.
    # -----------------------------------------------------------------------
    with gr.Row(visible=False):
        _api_video = gr.File(label="API target video", type="filepath")
        _api_face1 = gr.Image(type="numpy", label="API face 1")
        _api_face2 = gr.Image(type="numpy", label="API face 2")
        _api_face3 = gr.Image(type="numpy", label="API face 3")
        _api_face4 = gr.Image(type="numpy", label="API face 4")
        _api_settings = gr.Textbox(label="API settings JSON")
        _api_job_id = gr.Textbox(label="API job ID")
        _api_submit_btn = gr.Button("API Submit")
        _api_detect_video = gr.File(label="API detect video", type="filepath")
        _api_detect_pct = gr.Number(value=0, label="API frame position %")
        _api_detect_btn = gr.Button("API Detect Frame")
        _api_detect_image = gr.Image(label="API detected frame")
        _api_detect_msg = gr.Textbox(label="API detect status")
        _api_detect_face1 = gr.Image(label="API detected face 1")
        _api_detect_face2 = gr.Image(label="API detected face 2")
        _api_detect_face3 = gr.Image(label="API detected face 3")
        _api_detect_face4 = gr.Image(label="API detected face 4")
        _api_status_btn = gr.Button("API Status")
        _api_download_btn = gr.Button("API Download")
        _api_cancel_btn = gr.Button("API Cancel")
        _api_json_out = gr.JSON(label="API response")
        _api_file_out = gr.File(label="API output file")

        # Image API compatibility controls. These remain hidden and exist only
        # to publish the stable named Gradio routes required by Phoenix Mobile.
        _api_image_target = gr.Image(type="numpy", label="API image target")
        _api_image_face1 = gr.Image(type="numpy", label="API image face 1")
        _api_image_face2 = gr.Image(type="numpy", label="API image face 2")
        _api_image_face3 = gr.Image(type="numpy", label="API image face 3")
        _api_image_face4 = gr.Image(type="numpy", label="API image face 4")
        _api_image_quality = gr.Textbox(value="Best", label="API image quality")
        _api_image_refs = gr.JSON(value=None, label="API image refs")
        _api_image_detect_btn = gr.Button("API Detect Image")
        _api_image_swap_btn = gr.Button("API Swap Image")
        _api_image_detect_out = gr.Image(label="API detected image")
        _api_image_detect_msg = gr.Textbox(label="API image detect status")
        _api_image_detect_refs = gr.JSON(label="API image refs output")
        _api_image_detect_face1 = gr.Image(label="API detected face 1")
        _api_image_detect_face2 = gr.Image(label="API detected face 2")
        _api_image_detect_face3 = gr.Image(label="API detected face 3")
        _api_image_detect_face4 = gr.Image(label="API detected face 4")
        _api_image_swap_out = gr.Image(label="API image swap output")
        _api_image_swap_msg = gr.Textbox(label="API image swap status")

    def _api_detect_image_compat(target):
        # core_pipeline.detect_image returns seven UI values: annotated image,
        # status, internal face-reference embeddings, and four face crops. The
        # mobile client only needs the image/status/crops; do not expose raw
        # numpy embeddings through the public API response.
        result = detect_image(target)
        return result[0], result[1], None, result[3], result[4], result[5], result[6]

    def _api_swap_image_compat(target, s1, s2, s3, s4, quality, refs):
        # Preserve the Phoenix Mobile contract: 7 inputs and 2 outputs.
        return swap_image(target, s1, s2, s3, s4, quality or "Best", refs)

    _api_submit_btn.click(api_submit_video, inputs=[_api_video, _api_face1, _api_face2, _api_face3, _api_face4, _api_settings], outputs=[_api_json_out], api_name="phoenix_submit_video")
    _api_detect_btn.click(api_detect_video_frame, inputs=[_api_detect_video, _api_detect_pct], outputs=[_api_detect_image, _api_detect_msg, _api_detect_face1, _api_detect_face2, _api_detect_face3, _api_detect_face4], api_name="phoenix_detect_video_frame")
    # queue=False: these three are near-instant (a dict lookup, serving an
    # already-rendered file, setting a cancel flag) and were previously
    # sharing the same default_concurrency_limit=2 pool as the actual video
    # job. phoenix_job_status is polled every ~2.5s by the client for the
    # entire duration of a render that can run for well over an hour -
    # competing for a queue slot that long stretch means a status check can
    # end up waiting long enough for Gradio to discard its own session
    # before ever serving it, which surfaces to the client as
    # "404: Session not found" - not a real server crash, just contention
    # for a queue these calls never needed to be in.
    _api_status_btn.click(api_job_status, inputs=[_api_job_id], outputs=[_api_json_out], api_name="phoenix_job_status", queue=False)
    _api_download_btn.click(api_download, inputs=[_api_job_id], outputs=[_api_file_out], api_name="phoenix_download", queue=False)
    _api_cancel_btn.click(api_cancel, inputs=[_api_job_id], outputs=[_api_json_out], api_name="phoenix_cancel", queue=False)

    # Required image routes for Phoenix Mobile / external clients.
    _api_image_detect_btn.click(
        _api_detect_image_compat,
        inputs=[_api_image_target],
        outputs=[
            _api_image_detect_out,
            _api_image_detect_msg,
            _api_image_detect_refs,
            _api_image_detect_face1,
            _api_image_detect_face2,
            _api_image_detect_face3,
            _api_image_detect_face4,
        ],
        api_name="detect_image",
    )
    _api_image_swap_btn.click(
        _api_swap_image_compat,
        inputs=[
            _api_image_target,
            _api_image_face1,
            _api_image_face2,
            _api_image_face3,
            _api_image_face4,
            _api_image_quality,
            _api_image_refs,
        ],
        outputs=[_api_image_swap_out, _api_image_swap_msg],
        api_name="swap_image",
    )

    gr.HTML('<div class="app-ftr">Swamitech Phoenix v11.2.3 “SolidFace” · Every frame composited · Occlusion-aware tracking · CPU throughput</div>')

    # Events
    im_detect.click(detect_image, [im_tgt], [im_prev, im_st, img_refs, im_df1, im_df2, im_df3, im_df4], api_name=False)
    im_go.click(swap_image, [im_tgt, im_s1, im_s2, im_s3, im_s4, im_q, img_refs], [im_out, im_st], api_name=False)
    im_clr.click(lambda: [None]*12, outputs=[im_tgt, im_s1, im_s2, im_s3, im_s4, im_prev, im_out, im_df1, im_df2, im_df3, im_df4, img_refs], api_name=False)

    v_detect.click(detect_video, [v_vid], [v_prev, v_st, vid_refs, v_df1, v_df2, v_df3, v_df4], api_name=False)
    v_show.click(show_frame, [v_vid, v_scrub], [v_frame, v_st], api_name=False)
    v_cap1.click(lambda v,p,r: capture_face(v,p,0,r), [v_vid, v_scrub, vid_refs], [vid_refs, v_df1, v_st], api_name=False)
    v_cap2.click(lambda v,p,r: capture_face(v,p,1,r), [v_vid, v_scrub, vid_refs], [vid_refs, v_df2, v_st], api_name=False)
    v_cap3.click(lambda v,p,r: capture_face(v,p,2,r), [v_vid, v_scrub, vid_refs], [vid_refs, v_df3, v_st], api_name=False)
    v_cap4.click(lambda v,p,r: capture_face(v,p,3,r), [v_vid, v_scrub, vid_refs], [vid_refs, v_df4, v_st], api_name=False)

    v_trim_start.change(_calc_trim_duration, [v_vid, v_trim_start, v_trim_end], [v_trim_duration], api_name=False)
    v_trim_end.change(_calc_trim_duration, [v_vid, v_trim_start, v_trim_end], [v_trim_duration], api_name=False)
    v_vid.change(_calc_trim_duration, [v_vid, v_trim_start, v_trim_end], [v_trim_duration], api_name=False)

    def apply_preset(choice):
        if "Speed" in choice:
            return apply_speed()
        if "Optimized" in choice:
            return apply_optimized()
        if "Stable" in choice:
            return apply_stable()
        if "Mobile HQ" in choice:
            return apply_mobile_hq()
        if "HQ" in choice and "Mobile" not in choice:
            return apply_hq()
        if "Quality" in choice:
            return apply_quality()
        return apply_balanced()

    preset.change(apply_preset, inputs=[preset], outputs=[v_res, v_sec, v_fps, v_q, v_enh, v_swap, v_det, v_det_int], api_name=False)
    def _upd_settings(sw, de, di):
        return (f'<div class="settings-card">Swap every N=<b>{sw}</b> · '
                f'Re-detect every N=<b>{de}</b> · Det interval=<b>{di}</b></div>')
    for _c in (v_swap, v_det, v_det_int):
        _c.change(_upd_settings, inputs=[v_swap, v_det, v_det_int], outputs=[v_settings_summary], api_name=False)


    def _on_device_change(mode):
        _device_pref[0] = _parse_device_mode(mode)
        return _device_status_text()

    v_device.change(_on_device_change, inputs=[v_device], outputs=[v_device_status], api_name=False)

    def _refresh_all_status(request: gr.Request = None):
        # Jobs are application-level resources; browser refresh may create a new
        # Gradio session id. Always query the global Phoenix job registry.
        return (
            _video_progress_html(),
            _status_banner_html(),
            _status_banner_html(),
            _hist_html(),
            gr.update(choices=_get_done_choices()),
            gr.update(choices=_running_job_choices()),
        )

    v_go.click(submit_video,
               [v_s1, v_s2, v_s3, v_s4, v_vid, v_sec, v_fps, v_res, v_q, v_enh, v_swap, v_det, v_det_int, v_pw, vid_refs, v_trim_start, v_trim_end, v_face_mode, v_enh_scope, v_device],
               [v_st, h_html, h_dd, v_prog, v_device_status], api_name=False)
    v_cancel.click(
        cancel_running,
        inputs=[v_cancel_pick],
        outputs=[v_st, h_html, v_prog, v_banner, v_cancel_pick],
        api_name=False,
    )
    v_refresh.click(
        _refresh_all_status,
        outputs=[v_prog, v_banner, h_banner, h_html, h_dd, v_cancel_pick],
        api_name=False,
    )
    v_clr.click(lambda: [None]*11, outputs=[v_s1,v_s2,v_s3,v_s4,v_vid,v_prev,v_df1,v_df2,v_df3,v_df4,v_st], api_name=False)

    def _refresh_history(request: gr.Request = None):
        return (
            _hist_html(),
            gr.update(choices=_get_done_choices()),
            _status_banner_html(),
            _video_progress_html(),
            gr.update(choices=_running_job_choices()),
        )

    h_ref.click(
        _refresh_history,
        outputs=[h_html, h_dd, h_banner, v_prog, v_cancel_pick],
        api_name=False,
    )
    h_btn.click(load_result, [h_dd], [h_vid, h_file, h_msg], api_name=False)
    h_del.click(delete_now, [h_dd], [h_msg, h_html, h_dd, h_vid, v_prog], api_name=False)
    h_clr.click(clear_history, outputs=[h_msg, h_html, h_dd, v_prog], api_name=False)

    # Restore last form (video/faces/settings) if session still valid (<5 min after job)
    def _restore_session(request: gr.Request = None):
        vid, f1, f2, f3, f4, stg = _load_ui_session(request)
        if not stg and vid is None and all(x is None for x in (f1, f2, f3, f4)):
            return [gr.update()] * 18

        def g(key, default=None):
            if key in stg and stg[key] is not None:
                return stg[key]
            return gr.update() if default is None else default

        return [
            vid if vid is not None else gr.update(),
            f1 if f1 is not None else gr.update(),
            f2 if f2 is not None else gr.update(),
            f3 if f3 is not None else gr.update(),
            f4 if f4 is not None else gr.update(),
            g("secs"),
            g("fps"),
            g("res"),
            g("quality"),
            g("enhancer"),
            str(g("swap_n")) if "swap_n" in stg else gr.update(),
            str(g("det_n")) if "det_n" in stg else gr.update(),
            str(g("det_int")) if "det_int" in stg else gr.update(),
            g("face_mode"),
            g("enhance_scope"),
            g("device_mode"),
            g("trim_start"),
            g("trim_end"),
        ]

    try:
        demo.load(
            _restore_session,
            outputs=[
                v_vid, v_s1, v_s2, v_s3, v_s4,
                v_sec, v_fps, v_res, v_q, v_enh, v_swap, v_det, v_det_int,
                v_face_mode, v_enh_scope, v_device, v_trim_start, v_trim_end,
            ],
            api_name=False,
        )
    except TypeError:
        # Older Gradio may not accept api_name on load
        try:
            demo.load(
                _restore_session,
                outputs=[
                    v_vid, v_s1, v_s2, v_s3, v_s4,
                    v_sec, v_fps, v_res, v_q, v_enh, v_swap, v_det, v_det_int,
                    v_face_mode, v_enh_scope, v_device, v_trim_start, v_trim_end,
                ],
            )
        except Exception as e:
            logging.warning(f"demo.load restore skipped: {e}")
    except Exception as e:
        logging.warning(f"demo.load restore skipped: {e}")

    # Optional live refresh (disabled if Timer unsupported)
    try:
        def _tick(request: gr.Request = None):
            # Do not key polling to the current Gradio session. A page refresh
            # creates a new session while the worker/job remains alive.
            with _lock:
                active = any(j.get('status') not in ('done','error','cancelled') for j in jobs.values())
            banner = _status_banner_html()
            prog = _video_progress_html()
            picks = gr.update(choices=_running_job_choices())
            if not active:
                return gr.update(), gr.update(), banner, banner, prog, picks
            return (
                _hist_html(),
                gr.update(choices=_get_done_choices()),
                banner,
                banner,
                prog,
                picks,
            )

        def _tick_device():
            # If user prefers GPU but CUDA vanished → status shows CPU fallback
            if _device_pref[0] == "gpu" and not _cuda_available() and _loaded_device[0] == "gpu":
                try:
                    _load(prefer_gpu=False)
                except Exception:
                    pass
            return _device_status_text()

        if hasattr(gr, "Timer"):
            gr.Timer(3).tick(
                _tick,
                outputs=[h_html, h_dd, v_banner, h_banner, v_prog, v_cancel_pick],
                api_name=False,
            )
            gr.Timer(10).tick(_tick_device, outputs=[v_device_status], api_name=False)
    except Exception:
        pass


# HF Spaces imports `demo` from this module
demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    try:
        demo.launch(show_api=False)
    except TypeError:
        demo.launch()
