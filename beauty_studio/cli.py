"""
Headless renderer - the same pipeline the UI drives, from a terminal.

    python -m beauty_studio.cli clip.mp4 -o out.mp4 --preset "HDR Cinematic"
    python -m beauty_studio.cli clip.mp4 --set face_slim=35 --set skin_smooth=60
    python -m beauty_studio.cli photo.jpg -o photo_out.jpg
    python -m beauty_studio.cli --selftest

Useful for batches, for anything long enough that a browser tab is the wrong
place to keep it, and for reproducing a render exactly: every argument that
changed the result is printed with the report.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import cv2

if __package__ in (None, ""):
    # See the note in app.py: this also has to work when run as a script.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
    __package__ = os.path.basename(_here)

from .pipeline import capability_report, probe, process_image, render_video
from .settings import DEFAULT_PRESET, PRESETS, Settings

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _apply_overrides(s: Settings, pairs: list[str]) -> Settings:
    """--set name=value, where the value is in the same 0..100 the UI shows."""
    known = set(s.to_dict().keys())
    kw = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects name=value, got {pair!r}")
        name, _, value = pair.partition("=")
        name = name.strip()
        if name not in known:
            raise SystemExit(f"unknown setting {name!r}. Known: {', '.join(sorted(known))}")
        current = getattr(s, name)
        if isinstance(current, bool):
            kw[name] = value.strip().lower() in ("1", "true", "yes", "on")
        elif name in ("process_scale", "out_long_edge", "quality"):
            kw[name] = int(float(value))
        else:
            kw[name] = float(value) / 100.0
    return s.with_(**kw)


def build_settings(args) -> Settings:
    s = PRESETS.get(args.preset, PRESETS[DEFAULT_PRESET])
    s = s.with_(process_scale=args.scale, quality=args.quality, hdr10=args.hdr10,
                out_long_edge=args.out_long_edge, stabilise=not args.no_stabilise)
    if args.set:
        s = _apply_overrides(s, args.set)
    return s.normalised()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="beauty_studio.cli",
                                 description="HDR grading and natural retouching for video and photos")
    ap.add_argument("source", nargs="?", help="input video or image")
    ap.add_argument("-o", "--output", help="output path (default: alongside the input)")
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS.keys()))
    ap.add_argument("--set", action="append", metavar="NAME=VALUE",
                    help="override one setting, 0-100 like the UI sliders (repeatable)")
    ap.add_argument("--scale", type=int, default=1080, metavar="PX",
                    help="working/output long edge (default 1080; 0 = source size)")
    ap.add_argument("--out-long-edge", type=int, default=0, metavar="PX",
                    help="resize the output further (0 = same as working)")
    ap.add_argument("--trim", metavar="A:B", help="percent range of the clip to render, e.g. 10:90")
    ap.add_argument("--quality", type=int, default=18, metavar="CRF", help="x264 CRF (default 18)")
    ap.add_argument("--hdr10", action="store_true", help="experimental BT.2020/PQ export")
    ap.add_argument("--no-stabilise", action="store_true", help="disable temporal smoothing")
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="render a synthetic clip to check the install end to end")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(capability_report())

    if args.list_presets:
        for name, preset in PRESETS.items():
            print(f"  {name:24s} hdr={preset.hdr_strength:.2f} skin={preset.skin_smooth:.2f} "
                  f"face={preset.face_slim:.2f} body={preset.body_slim:.2f}")
        return 0

    if args.selftest:
        from .selftest import run as run_selftest
        return run_selftest()

    if not args.source:
        ap.error("a source file is required (or use --selftest)")
    if not os.path.exists(args.source):
        ap.error(f"no such file: {args.source}")

    if args.scale == 0:
        args.scale = 100000
    s = build_settings(args)
    stem, ext = os.path.splitext(args.source)

    if ext.lower() in IMAGE_EXT:
        img = cv2.imread(args.source, cv2.IMREAD_COLOR)
        if img is None:
            ap.error("could not read that image")
        out = args.output or f"{stem}_enhanced{ext}"
        t = time.time()
        cv2.imwrite(out, process_image(img, s))
        print(f"wrote {out} in {time.time() - t:.1f}s")
        return 0

    a, b = 0.0, 1.0
    if args.trim:
        try:
            lo, _, hi = args.trim.partition(":")
            a, b = float(lo) / 100.0, float(hi) / 100.0
        except ValueError:
            ap.error("--trim expects A:B in percent, e.g. 10:90")

    print(f"input: {probe(args.source).label}")
    out = args.output or f"{stem}_enhanced.mp4"
    res = render_video(args.source, s, out_path=out, start=a, end=b,
                       progress=lambda d, t_, m: print(f"\r  {m}   ", end="", flush=True))
    print()
    w, h = res["size"]
    face_pct = 100.0 * res["faces_seen"] / max(res["frames_seen"], 1)
    print(f"wrote {res['path']} · {res['frames']} frames at {w}x{h} · "
          f"{res['seconds']:.1f}s ({res['fps']:.1f} fps) · face on {face_pct:.0f}% of frames")
    for note in res["notes"]:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
