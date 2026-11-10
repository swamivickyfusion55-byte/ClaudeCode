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

UNITS

**The real job, v11.2.202 -> v11.2.203:**

TABLE

## 4. Limits

LIMITS

## 5. Install

- Upload the pack's files. Changed from 202: `swap_engine.py`, `core_pipeline.py`, `phoenix_api_adapter.py`, `config.py`,
  `app.py`. No new package: the AES library (pycryptodomex) comes with `pyzipper`, already in `requirements.txt`.
- **Add the secret:** Space settings -> Variables and secrets -> New secret, name `PHOENIX_STORE_KEY`, value a long random
  text (32 characters or more). Keep a copy only if you want jobs saved before a change of the secret to resume. Changing
  or removing it makes saved uploads unreadable: those jobs are then not resumed, and are deleted after 24 hours as before.
- Log at start: `Phoenix v11.2.203`, `swap_engine aequus-1.2.203-contactprivate`. Each job logs
  `Kept job ... ENCRYPTED (AES-256-GCM)`, or `NOT encrypted - add the Space secret PHOENIX_STORE_KEY`.
- Do not restart a Space while a job runs on it.
