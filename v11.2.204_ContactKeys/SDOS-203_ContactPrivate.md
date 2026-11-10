# SDOS-203 — ContactPrivate: 202 with the full contact fixes, a quality-neutral speed step, and encrypted saved uploads

**Pack:** v11.2.203, built on the uploaded v11.2.202 "Private24h". Engine `aequus-1.2.203-contactprivate`.

**Asked:**
1. "Is it still possible to improve the face occlusion logic to make it cut to cut segregated."
2. "Is there any opportunity to enhance processing speed without downgrading quality."
3. "Could you check whether the input data is protected, I have a private HF space.. Is there any additional encryption in
   this."

Then: "would that mean any quality compromise, specially in occlusion", and "Yes".

All tests use synthetic clips and files made from public sample photos. No user footage was looked at.

---

## 1. What 202 was

- `core_pipeline.py` is v11.2.179's (the 176-179 paint-side contact fixes: a steady front / behind decision, two faces
  told apart by their boxes, two people in contact never merged) plus the 24-hour saved-upload code.
- `swap_engine.py` (the tracker) was put back to 11.2.92, with only `_two_boxes` kept ("The 11.2.92 swap never calls
  this"). The tracker half of 176-179 was therefore off: the elimination rule (2 people), the face behind in a kiss, the
  carry in contact, the fast-step snap, picking a crossing face up again, and the slow-sliver guard.
- `patch_manual.py` is 11.2.92's. The adapter has 179's mask reader and Preview, with two extra guards.

## 2. The changes

### 2.1 The tracker half of 176-179 is back (`swap_engine.py` = v11.2.179's)

Each part has its own switch in `swap_engine._P`; 0 gives the 11.2.92 behaviour for that part:

| Switch | Part |
|---|---|
| `two_elim` | elimination of 2 people |
| `contact_follow` | the face behind and the contact rules |
| `contact_carry_max` | how long a face missed in contact is carried |
| `fast_snap` | a fast step is followed without smoothing lag |

What they do is in SDOS-176 and SDOS-177. Measured 202 -> 203 in section 3.

### 2.2 Speed, quality-neutral (`core_pipeline._arc_reuse`, switch `arc_skip`)

Measured first, at 2 CPU threads like the Space:

| Model, per call | ms |
|---|---|
| Swap model (`inswapper_128`) | 1056 |
| XSeg occlusion model | 133 |
| ArcFace recognition | 119 |
| Face detection (`SCRFD`) | 123 |

On a 2-face frame that is about 2.1 s of swap, 0.27 s of XSeg, 0.24 s of ArcFace and 0.12 s of detection.

- **Done: the ArcFace pass is skipped where it cannot change who is who.** The conditions:
  - exactly two people and two faces;
  - each face on its own person's box from the frame before (overlap 0.5 or more, under 0.1 with the other);
  - the two far apart (centres 1.5 face sizes or more);
  - nobody missed or crossing.

  Every 3rd frame is still checked. Contact, a crossing, a third face, a missed person or a new shot always run ArcFace.
  The masks and the occlusion model never use these embeddings.
- **Not done, and why:**
  - **Re-running XSeg only on change** could miss an object entering for a frame or two. XSeg still runs on every frame
    for every face.
  - **Batching XSeg and ArcFace** for both faces gains a few percent of those two models only, and would need the
    compositing order rewritten.
  - **The swap model cannot be batched:** its input is fixed to one face.
  - **A newer onnxruntime (1.31)** was not faster: swap model 1098 ms against 1056 ms, the same outputs to 1e-5.
  - **Two frames in parallel** (`PHOENIX_VIDEO_WORKERS=2`) is not quality-neutral: the per-face mask smoothing depends on
    frame order.
  - **8-bit models** would change the result.

### 2.3 Privacy

- **Saved uploads encrypted at rest.** The video, the replacement faces and the painted patch frames saved for a restart
  (`/data/phoenix_jobs/inputs/<job>/`) are written AES-256-GCM encrypted when the Space secret **`PHOENIX_STORE_KEY`** is
  set.
  - Each file gets its own key (HKDF-SHA256 from the secret and a random salt) and is sealed in 4 MiB chunks. A changed,
    cut or reordered file does not open.
  - The key is never written to disk.
  - On a restart the files are decrypted into `/tmp` (the container's temporary disk) for the job, and that copy is
    deleted when the job ends.
  - Without the secret, files are kept as in 202 (plain), and the job log says so.
- **The result-ZIP password is no longer saved.** 202 kept it in plain text in `resume.json`, next to the video. A job
  resumed after a restart therefore makes no encrypted ZIP (the log says so). The result itself is served as usual.
- **Painted patch frames and byte uploads go to `/tmp`.** Before, they were on the persistent `/data`, and byte uploads
  there were never deleted. The job keeps its own sealed copy for a restart, and deletes the folder when it ends.
- **Gradio serves files only from the result and temporary folders.** 202 allowed all of `/data`, including the saved
  inputs, to anyone who can open the Space and knows a path.

What stays as in 202:
- The Space being private is the access control: only accounts with access can open it or call it.
- Results are kept 12 hours and are not encrypted at rest. The optional result ZIP is AES-256 when you set a password.
- How Hugging Face stores the disk is outside the code and not verified here.

## 3. Verification (from the pack's own files)

**Unit checks** (stand-in faces and files, and insightface's own `Face` objects), run from the final files: **44 of 44 pass**.
- `unit176.py` 12 and `unit176_pipe.py` 14: the 176-179 tracker and contact rules.
- `unit203.py` 18: the ArcFace skip and the encryption.

- two people far apart: ArcFace skipped on about 2 calls in 3, every face still given to its own person  [40 faces skipped, 20 embedded, 0 wrong assignments]
- two people close (a kiss, a hug): ArcFace runs on every face  [0 skipped]
- a third face in the picture: ArcFace runs on every face  [0 skipped]
- arc_skip 0: every face embedded (v11.2.92)  [0 skipped]
- a face that is not on its person's last box: no reuse
- a new shot with the two people swapped: the skip relabels no later than without it  [wrong frames 1, last wrong 1 (no skip: 0, 0)]
- with insightface's own Face objects the skip works too (normed_embedding comes from embedding)  [16 faces reused]
- round trip: the decrypted file is identical (9.5 MB, 3 chunks)
- the sealed file does not contain the plain bytes
- a changed byte is refused
- a cut file (last chunk missing) is refused
- chunks in another order are refused
- another secret cannot open it
- saved inputs are sealed files (.enc)  [['face0.jpg.enc', 'face1.jpg.enc', 'patch0_frame.jpg.enc', 'patch0_mask.png.enc', 'resume.json', 'video.mp4.enc']]
- the result-ZIP password is not saved
- a restart decrypts the video, faces and patch frames into /tmp, identical to the originals  [/tmp/phoenix_resume/UNIT0203/video.mp4]
- without the secret a sealed job cannot be resumed (and says so), nothing crashes
- without the secret: kept as before (plain), still without the ZIP password  [['face0.jpg', 'face1.jpg', 'patch0_frame.jpg', 'patch0_mask.png', 'resume.json', 'video.mp4']]

**The real job, v11.2.202 -> v11.2.203:**

The real job (`bench_pipeline.py`: the Space's own pipeline on CPU, 2 threads, the paste masks recorded), then
`identity_check.py` and `contact_lag.py` on the output. "Revert" means under 40 % of the usual paint on the person's visible
face; "weak" 40-70 %; "mix-up" means the face shows the other person's replacement. All 203 numbers are from the final files.

| Clip | v11.2.202 | v11.2.203 |
|---|---|---|
| **Side-view kiss**, A behind | revert 44, mix-up 44, weak 10; worst 62 px off | **revert 0, mix-up 0**, weak 12; worst 16 px off |
| **Side-view kiss**, B in front | revert 35, mix-up 0, weak 15; worst 33 px off | **revert 7**, mix-up 0, weak 15; worst 8 px off |
| Side-view kiss: the other person's paint on a face | up to 82 % (5 % or more on 92 face-frames) | **at most 4 %** (0 face-frames) |
| **Harsh step** (180 px in 3 frames) | worst 50 px off, a 41 px jump | **worst 4.8 px, jump 2.9 px** |
| **Hide** (A a sliver behind B) | A weak 30 | **A weak 3**; no revert or mix-up in either |
| **Cross** (A walks behind B) | A weak 7, B weak 4; A jump 25 px | A weak 8, B weak 4; A jump 5.7 px |

**Speed** (the harsh step clip, the same 2 cores, run after each other):

| Build | Time | ArcFace passes |
|---|---|---|
| v11.2.202 | 531 s | 168 |
| v11.2.203 with the skip off (it did not act in the first run, see section 4) | 509 s | 168 |
| v11.2.203 | **498 s** | **74** (94 skipped) |

- The skip saves about 2 % on a clip where the two are apart throughout. ArcFace is only about 4 % of a job here (0.12 s a
  pass against about 3.3 s of swap work per face per frame).
- On contact clips it skips little: 8 passes on the kiss, 24 on the cross, 0 on the hide.
- The rest of the 202 -> 203 difference (about 4 %) is within run-to-run noise.
- The output quality is the same: the step clip gives identical scores with and without the skip.

**Privacy (unit checks above):**
- The saved files are sealed (`.enc`), and the plain bytes are not in them.
- A changed byte, a cut file, swapped chunks or another secret are refused.
- The ZIP password is not saved.
- A restart decrypts into `/tmp` and gets identical files.
- Without the secret, the 202 behaviour is kept, still without the password.

## 4. What did not work first

1. **The first 203 build's ArcFace skip never acted in the real pipeline** (log: "skipped 0"). insightface's `Face` objects
   derive `normed_embedding` from `embedding` (a read-only property), so setting it raised, and the code caught the error and
   embedded as before. The stand-in faces of the unit test allowed it, which is why the test passed. Fixed: the skip sets
   `embedding`, and a unit check now uses insightface's own `Face`. The step clip and the kiss were run again on the fixed
   files (the table).
2. **A shot cut with the two people on swapped sides:** the first skip would have kept reusing the wrong identity, because
   every skipped frame cancelled the tracker's relabelling vote. Fixed with the disagreeing-check stop, before any bench run.

## 5. Limits

1. **Synthetic tests only.** Cut-out faces from public photos, not real people moving. The numbers show the direction and size
   of a change, not a promise for every clip.
2. **Speed:** the quality-neutral gain is small (about 2 %, up to about 6 % with run noise). The swap model dominates and
   cannot be skipped or batched without changing the result. Real speed needs more CPU (an upgraded Space) or fewer swapped
   frames (a quality choice).
3. **The ArcFace skip, at a shot cut where the two people land exactly on each other's previous spots:** 1 frame shows the
   other person's replacement (stand-in test). Without the skip: 0. From the next frame it is right, because a disagreeing
   check stops the skip for 60 frames. Switch `arc_skip: 0` restores the 11.2.92 behaviour.
4. **From 176-179 (unchanged here):**
   - The face in front of a side-view kiss thins on 7 frames (likely the 11.2.92 lip masks, not confirmed).
   - Smooch: 1 frame wrong at the first touch.
   - Hug: 2 more weak frames than 11.2.92.
   - Step clip: one interpolated frame 19 px ahead.
5. **Encryption protects the saved uploads on the disk, not the running app.** The Space process holds the key while it
   runs. Anyone who can change the Space's code or settings (a leaked write token) could read new uploads. Results (12 hours)
   are not encrypted at rest, unless you set a ZIP password. A job resumed after a restart makes no password ZIP.

## 6. Install

- Upload the pack's files. Changed from 202: `swap_engine.py`, `core_pipeline.py`, `phoenix_api_adapter.py`, `config.py`,
  `app.py`. No new package: the AES library (pycryptodomex) comes with `pyzipper`, already in `requirements.txt`.
- **Add the secret:** Space settings -> Variables and secrets -> New secret, name `PHOENIX_STORE_KEY`, value a long random
  text (32 characters or more). Keep a copy only if you want jobs saved before a change of the secret to resume. Changing
  or removing it makes saved uploads unreadable: those jobs are then not resumed, and are deleted after 24 hours as before.
- Log at start: `Phoenix v11.2.203`, `swap_engine aequus-1.2.203-contactprivate`. Each job logs
  `Kept job ... ENCRYPTED (AES-256-GCM)`, or `NOT encrypted - add the Space secret PHOENIX_STORE_KEY`.
- Do not restart a Space while a job runs on it.
