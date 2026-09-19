---
title: Swamitech Beauty Studio
emoji: ✨
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
---

# Swamitech Beauty Studio

An HDR video editor with natural retouching: it grades your footage, cleans up
skin without erasing it, shapes the face and silhouette within believable
limits, enhances hair, and writes a finished MP4 with the original audio.

Everything runs locally (or in your own Space) on CPU. Nothing is uploaded
anywhere, and no generative model is involved - every pixel in the output comes
from your footage.

```
python -m beauty_studio.app                   # web UI on http://localhost:7860
python -m beauty_studio.cli clip.mp4 -o out.mp4 --preset "HDR Cinematic"
python -m beauty_studio.cli --selftest        # check the install end to end
```

---

## What it does

**HDR grade.** A real local tone-mapping operator, not a contrast curve: the
luminance is split in the log domain into a base layer (the lighting, via an
edge-preserving guided filter) and a detail layer (the texture). Compressing
the base while keeping the detail is what opens shadows and recovers a blown
window in the same shot. Clarity, vibrance, bloom, white balance and an
edge-aware final sharpen sit on top as grading.

Detail-boosting stages are gated by a texture-energy map, so skies, walls and
studio backdrops do not hand back amplified noise and JPEG blocking - the
single clearest tell of a phone "HDR" filter.

**Skin.** Frequency separation. The face is split into a smooth tone layer and
a texture layer; the tone layer is evened out and the texture is put back at
whatever level you choose (default: two thirds). Blemish removal only touches
pixels that are both darker than their surroundings and smaller than the local
median kernel, so a spot goes and the shadow under the nose stays. Eyes, brows,
lips and nostrils are masked out of all of it.

Also: under-eye circle reduction (weighted by how dark the shadow actually is,
so it does not leave pale rectangles), sclera whitening, iris and lash
definition, teeth whitening inside the mouth only, and lip definition.

**Face and body shape.** Every adjustment writes into one smooth displacement
field that is applied with a single `remap`. Jaw and cheek slimming pull toward
the face's own centre line, so a tilted head slims correctly. Body work follows
the silhouette from the segmentation mask - not the pose skeleton, which says
nothing about how wide a coat is - with the waist, bust and hip bands located
from the pose.

Three rules keep it from bending the person, all learned the hard way:

- **Nothing above the shoulders.** The person mask includes the head, so a
  whole-silhouette squeeze used to narrow the skull and jaw along with the
  body. The field is gated at the shoulder line now, ramped over the neck.
- **Torso features use torso width.** Waist, bust and hips are measured
  against the shoulder span, not the whole outline. Measured against the
  outline, their peak displacement lands out on the arms, which then bend
  with the torso.
- **A band whose centre is off-screen sits out.** On a chest-up shot the
  estimated waist lands below the frame, and its tail was squeezing the
  shoulders at the bottom edge. No waist in shot, no waist shaping - the
  render report says when this happens.

Row widths are measured about the body's axis, which comes from the pose
(shoulder, hip, knee and ankle midpoints) rather than from the silhouette.
Below the knees that axis runs down the gap *between* the legs, where there is
no silhouette at all - and a profiler that picks the widest run instead latches
onto one leg, warps it about its own centre, and leaves the other where it was.
Runs too far from the axis are still excluded, so an arm held away from the
body is not counted as torso width.

Legs also taper out of the warp below the knee: a calf is narrow, moves fast
and sits against background, so shaping it buys nothing and only risks calves
that do not line up with the knees. Thighs - where slimming actually reads -
keep the full amount.

A fuller **face** is a different mechanism from a slimmer one. Slimming only
has to move the jaw line inward, but dragging an outline outward leaves the
nose and mouth where they were and stretches the picture between them. So
**Rounder face** scales the lower face about its own axis instead: everything
inside widens together, which is what a fuller face actually is.

Bust, waist and hips are separate bands, positioned from the shoulder line and
the torso length rather than read straight off the pose - a subject framed from
the chest up still gets a waist in a sensible place instead of one extrapolated
below the bottom of the picture. **Curvy (hourglass)** moves all three at once;
**Bust** and **Hips** move one without the other.

Measured on a standing subject:

| | bust | waist | hips |
| --- | --- | --- | --- |
| Natural | – | −10% | – |
| Glam | – | −14% | – |
| Curvy | +8% | −18% | +10% |
| Curvy (strong) | +13% | −23% | +13% |

Two ceilings keep that honest: the per-row total is capped at 30% of the body's
own half-width, so three sliders pushed up together cannot compound into a
caricature, and the field as a whole is capped relative to the frame, which is
why straight lines in the background stay straight.

**Hair.** The mask is seeded on the crown - which is hair beyond argument -
and grown outward through pixels that are both joined to it and the same
colour. Elimination alone was not enough: the earlier version leaned on a
skin-tone test, and blonde hair sits squarely inside the skin-tone window, so
on the reference frame it scored real hair at 0.13 while a space helmet's
visor stayed at 1.0. Seed-and-grow gets the hair and nothing else.

On that mask: strand definition, gloss that follows the light already in the
shot, colour depth, flyaway control on the outer edge only, a volume pass that
grows the silhouette by a few pixels of real re-sampled hair - and
**recolouring**.

**Hair colour.** Pick a target (black, dark brown, brown, light brown, auburn,
red, blonde, platinum, grey, or a custom hue) and a strength. The hair's
average colour is moved to the target while every pixel keeps its own
departure from that average, so the variation between strands survives instead
of becoming one flat colour; lightness is scaled about the mean rather than
offset, because dark hair genuinely has a narrower range than blonde; and
specular highlights are held back so gloss survives the change.

Going darker is the strong direction. Lightening dark hair is limited and
honestly so: near-black pixels hold little detail to carry, so the result is
flatter than life. Nothing here is generated - every pixel still comes from
your footage.

**Video, not stills.** Landmarks, silhouette profiles and exposure statistics
are all smoothed over time; a lost face is coasted for a few frames and a newly
found one ramps up over a few, so effects never pop on and off between frames.
`--selftest` reports the resulting frame-to-frame stability as a number.

## Presets

| Preset | For |
| --- | --- |
| **Natural** | The default. Should read as good lighting, not as an edit. |
| **Natural+ (subtle)** | Half strength again - very lightly graded footage. |
| **Professional Portrait** | Interviews, corporate, talking heads. Clean skin, almost no shape work. |
| **HDR Cinematic** | Strong tone mapping and local contrast, warm, filmic. |
| **Glam** | Everything up, still inside the caps. |
| **Curvy** | The hourglass by name: waist in, bust and hips out, lightly graded. |
| **Curvy (strong)** | The same shape, pushed. |
| **Chubby (light / medium / heavy)** | The other direction: fuller face and silhouette, waist left alone. Face widens ~3 / 5 / 7%. |
| **Hair colour only** | Recolours the hair and changes nothing else at all. |
| **Skin texture only** | Adds skin micro-detail back and changes nothing else at all. |
| **Shape Only** | Reshaping with no grade or retouch. |
| **HDR Only (no retouch)** | Grade only - landscapes, product, b-roll. |

### Stacking presets

Up to three presets can be combined - *Chubby (medium) + HDR Cinematic*, say.
Each preset writes only the part of the picture it is about, so they add up
instead of overwriting each other:

| Preset | Writes |
| --- | --- |
| Natural, Natural+, Professional Portrait, Glam | everything (full looks) |
| HDR Cinematic, HDR Only | the grade |
| Curvy, Curvy (strong) | the body |
| Chubby (light / medium / heavy) | the body and the face |
| Hair colour only | the hair |
| Skin texture only | the skin |
| Shape Only | the face and the body |

**Order does not matter.** The broadest preset is applied first and the most
specific last, whichever order you picked them in - so adding a full look on
top of a shape preset brings its skin and hair without quietly undoing the
shaping. In the CLI: `--preset "Chubby (medium),HDR Cinematic"`.

Every slider is 0-100 and every preset is just a set of slider positions, so
you can start from one and adjust. **Naturalness** scales every person-effect
at once without touching the grade.

## Using it

**Web UI.** Load a video, pick a preset, hit *Preview frame* to judge the
settings on one frame (a second or two), then *Render video*. Trim start/end
render a section instead of the whole clip. The Photo tab runs the same stack
on a still.

**Renders run on the server, not in your tab.** *Render video* queues a job and
returns immediately; a worker thread owns it from there. Close the tab, lock
the phone, lose the wifi - the render carries on, and any tab that comes back
picks it up by polling. The **History** tab lists every render on the server,
newest first, with the file to download and the stats for each; renders started
on another device show up there too. One render at a time, because a second
concurrent one only makes both slower.

A job whose files have been swept by the retention policy stays in the list as
a record and reads *expired*. A job that was mid-render when the app restarted
reads *interrupted* rather than pretending to still be running.

**CLI.**

```bash
python -m beauty_studio.cli clip.mp4 -o out.mp4 \
    --preset "Professional Portrait" \
    --set skin_smooth=55 --set face_slim=30 --set waist_shape=25 \
    --scale 1080 --trim 10:90 --quality 18
```

`--set NAME=VALUE` takes any field in `settings.py` on the same 0-100 scale as
the UI. `--list-presets` prints what each preset does; `--selftest` renders a
synthetic clip and reports tracking and stability.

## MediaPipe: both generations work

Everything except the HDR grade needs MediaPipe to find the face and body, and
there are now two incompatible MediaPipe APIs:

| Installed | API used | Models | Needs |
| --- | --- | --- | --- |
| **≤ 0.10.21** | legacy `solutions` | inside the wheel | nothing extra |
| **≥ 0.10.30** (1.x included) | `tasks` | downloaded once (~10 MB) and cached | outbound network on first run, and `libEGL` |

The app detects which is present and uses it; the header line and `--doctor`
say which is active. All three of 0.10.14, 0.10.35 and 1.0.1 are tested, and
give the same result to within half a percent of a pixel value.

**`solutions` was removed at 0.10.30, not at 1.0.** That is the trap: a
`mediapipe<0.11` pin looks conservative and still resolves to 0.10.35, which
does not have it. An app written against the old API (including this one
before v1.1.0) then fails on the first frame with:

```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

If you are seeing that, update to this version. `requirements.txt` here pins
`mediapipe>=0.10.14,<0.10.22` - the last release that needs nothing at runtime.
To run on a current MediaPipe instead, relax it to `mediapipe>=0.10.30` and add
`libegl1` and `libgles2` to `packages.txt`.

Model downloads land in `$BEAUTY_STUDIO_MODELS`, else `$HF_HOME/beauty_studio`,
else `~/.cache/beauty_studio/models`, else the system temp directory - the
first one that is writable. Set `BEAUTY_STUDIO_MODELS` to bake them into an
image and skip the runtime download.

When none of this works - no MediaPipe, no network for the models, no libEGL -
the app does not fail. It grades the video, says why the rest is off in the
header and in the render report, and renders.

## When something is off: `--doctor`

```
$ python -m beauty_studio.cli --doctor
python        3.10.14 (x86_64)
mediapipe     1.0.1
backend       tasks
status        MediaPipe 1.0.1 (tasks API)
face features ON
model cache   /home/user/.cache/beauty_studio/models
  face        cached  face_landmarker.task
  pose        will download  pose_landmarker_lite.task
  segment     cached  selfie_segmenter.tflite
ffmpeg        /usr/bin/ffmpeg
ffprobe       /usr/bin/ffprobe
opencv        4.10.0
```

That block answers, in one place, every "why is it only grading?" and "why is
there no audio?" question this app can raise.

## Deploying to Hugging Face Spaces

Copy the **folder** into the Space and point the Space at it - the modules
import each other as a package, so keep them together in a directory rather
than spilling them into the repository root:

```bash
git clone https://huggingface.co/spaces/<you>/<space> && cd <space>
cp -r /path/to/beauty_studio .
cp beauty_studio/requirements.txt beauty_studio/packages.txt .
git add -A && git commit -m "Beauty Studio" && git push
```

Then in the Space's root `README.md` YAML header:

```yaml
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: beauty_studio/app.py
```

The header at the top of this file is the same thing for a Space whose root
*is* this folder.

`packages.txt` installs **ffmpeg**, which carries the original audio into the
output and re-encodes to browser-safe H.264 (OpenCV writes video only, and
many OpenCV builds have no H.264 encoder at all), plus **libgl1**. Without
ffmpeg the render still completes, silent, and the UI says so.

That file is fed straight to `xargs apt-get install`, so it takes **bare
package names only** - one per line, no comments and no apostrophes. A `#`
comment in it does not get ignored, it gets installed, and the build fails
with `E: Unable to locate package #`.

**On a Docker Space** `packages.txt` is ignored - your Dockerfile owns the
system packages, so install them there:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libegl1 libgles2 && rm -rf /var/lib/apt/lists/*
```

## Performance

Measured on a 4-core cloud CPU, per frame, with a face in shot:

| Working resolution | Grade only | Full stack |
| --- | --- | --- |
| 720p | ~0.4 s | ~0.6 s |
| 1080p | ~0.7 s | ~1.0 s |

So roughly 4-6 frames/second at 720p: a 30-second clip takes a few minutes.
Drop **Working resolution** to 720p for speed, and use *Preview frame* rather
than repeated renders while you dial settings in. The stack is all OpenCV and
NumPy - no GPU is required and none is used.

## Limits, stated plainly

- **HDR10 export is experimental.** It converts to BT.2020 primaries and the PQ
  transfer function and tags the file so a display switches into HDR mode
  (verified: HEVC, `yuv420p10le`, `color_primaries=bt2020`,
  `color_transfer=smpte2084`). It is an inverse tone map of SDR material: it
  does not recover highlight detail that was never captured. If this ffmpeg
  build cannot do it, the render automatically falls back to standard H.264
  and the report says so - it never hands back a file you cannot play.
- **Faces need to be findable.** Everything except the grade depends on
  MediaPipe finding the face. Very small, heavily backlit or extremely
  motion-blurred faces will be skipped - the report after each render says on
  what percentage of frames the face was tracked.
- **Body shaping needs the body in frame.** Waist and hourglass adjustments
  need the hips visible for the pose model to place the bands; with only a
  head-and-shoulders framing, the overall slimming still works. The render
  report says what was found - face percentage, body-outline percentage, and
  the largest reshape actually applied in pixels - so "too subtle" and "never
  ran" are not the same message.
- **Shape amounts are capped** (see `CAPS` in `settings.py`) at the point where
  each effect starts to read as an edit rather than a flattering adjustment.

## Layout

| File | What is in it |
| --- | --- |
| `app.py` | Gradio UI (Spaces entry point) |
| `cli.py` | Headless renderer |
| `pipeline.py` | Per-frame orchestration, video render, ffmpeg encode/mux |
| `grade.py` | HDR tone mapping and colour |
| `retouch.py` | Skin, eyes, teeth, lips |
| `reshape.py` | Warp field, face and body shaping |
| `hair.py` | Hair mask and enhancement |
| `landmarks.py` | MediaPipe trackers, smoothing, coasting |
| `imaging.py` | Guided filter, blend modes, masks, EMA |
| `settings.py` | Every knob, the caps, the presets |
| `selftest.py` | Synthetic end-to-end check |

## Jobs and history

| State | What it means |
| --- | --- |
| queued | waiting for the renderer (one runs at a time) |
| running | in progress; the page shows frames, fps and an ETA |
| done | finished; the file is in History until the retention window passes |
| stopped | you pressed Stop or Cancel this job; the row keeps the percentage it reached |
| interrupted | the app restarted mid-render - submit it again |
| expired | the retention policy deleted the file; the row is the record |
| failed | the reason is on the row |

**Progress** is on both tabs: the Video tab shows a bar with the percentage,
frames done and an ETA, and History has a progress column that updates while
you watch it - so a render started on your phone can be followed from a laptop.

**Stopping** works from either tab and from any device. *Stop* on the Video
tab halts whatever this tab started, or, in a tab that did not start anything,
whatever is currently rendering. *Cancel this job* in History stops the render
you picked from the list. A stopped render finishes the frame it is on and
then gives up, and its row keeps the percentage it reached.

Messages from buttons stay on screen for a few seconds before the status line
goes back to reporting the job - otherwise the two-second refresh wipes
"Stopping…" off the screen before it can be read.

`BEAUTY_JOBS_DIR` sets where job outputs and the index live (default: a
`beauty_jobs` directory in the system temp directory).

## Data retention

Uploads, renders, previews and working files are deleted automatically **three
hours** after they are last touched. A janitor thread sweeps every ten minutes;
the UI has a **Delete my files now** button, and the CLI has `--purge`.

- `BEAUTY_RETENTION_HOURS` changes the window (e.g. `0.5` for thirty minutes).
- Swept: this app's render directories, finished jobs (one at a time, so the
  history index survives and the row can say the file expired), and Gradio's
  cache, which is where uploads and the files served back to the browser live. Both are addressed by
  name - sweeping a whole temp directory would risk another process's files.
- Not swept by default: the MediaPipe model cache. Those are weights, not
  anyone's data, and dropping them only forces a re-download. Add
  `BEAUTY_PURGE_MODELS=1` to include them, or use `--purge --purge-models`.

## A note on what this is for

This edits video you provide, on your own machine, into a more flattering
version of itself - the same thing a colourist and a retoucher do by hand. The
caps exist so that the result still looks like the person who was filmed. It is
not a face swap, it does not synthesise anyone, and it should not be used to
make footage of someone who has not agreed to it.
