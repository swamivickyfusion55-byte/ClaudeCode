# SDOS-079 — Resume: uploads, split jobs and downloads stop when the app is minimised

**Pack:** v11.2.80 "Resume" (includes everything in v11.2.79 ReID — if you never installed 79, you do not need it).
**Reported:** "when I minimise the swamitech mobile app, any upload or split job or download stops, and at times if it happens during upload the job gets aborted."

---

## 1. Short answer

Minimising the app does **not** stop processing on the Space. A job runs on a background worker (`VIDEO_EXECUTOR`) that nothing in the code ties to the phone's connection; I found no code that cancels a job when a client disconnects.

What stops is every step that needs the **phone** to keep a connection open or keep working:

| Step | Who does the work | What happens on minimise | Fault on the server? |
|---|---|---|---|
| Upload | phone → Space, one Gradio `/upload` POST | The OS freezes/kills the app's sockets; the POST dies at a random byte; every byte already sent is wasted; the app sees an error | Partly: one big un-resumable POST |
| Split | the phone cuts the clip (`config.py` notes the Android client uploads `phoenix_trim_*.mp4`) | The cutting is app CPU work; it stops with the app | The server could do the cut, and now does |
| Submit | phone → `phoenix_submit_video` over Gradio's queue | See §2: the job may never start, or start with its id lost | Yes: not safe to retry |
| Status | phone polls | Polling stops; nothing is lost | No |
| Download | Space → phone, one GET | Connection dies; restarts from byte 0 | Partly: no `Range` |
| Result file | Space | Deleted 3 h after it finishes | Yes: shorter than a night |

I do **not** have the Android app's source (this repository is the Space only). So this pack fixes the server half completely and ships a tested reference client that defines the other half. **A minimised phone will keep transferring only after the app is changed to use these routes** (§5). Until then the server behaves exactly as v11.2.79.

---

## 2. Measured, not assumed — what Gradio does when a client drops

Probe on the pinned stack (Gradio 4.44.1, Starlette 0.37.2), one worker slot, real sockets:

| Client drops while the event is… | Result |
|---|---|
| **waiting in the queue** | Gradio **discards it. The function never runs.** |
| **running** | The function **runs to completion; its result is thrown away.** |

For `phoenix_submit_video` that means a phone minimised mid-submit either never starts the job, or starts it and never learns the job id — and then submits again. On a one-worker CPU Space that is the same clip processed twice. (The "queued" case needs two submits in flight at once, because Gradio applies `default_concurrency_limit` per event; the "running" case is the common one.)

---

## 3. What this pack adds (server side)

### 3.1 Resumable upload — `POST/GET/PUT/DELETE /phoenix/upload…`
Append-only chunks at an explicit offset (tus-style). **The server's file size is the offset**, so after any failure the client asks, then continues.

| Call | Purpose |
|---|---|
| `POST /phoenix/upload/init` `{name,size,token?,sha256?}` | Start an upload, or **resume it by `token`** after the app was killed. Returns `upload_id`, `offset`, `chunk_bytes` (advised 4 MB) |
| `GET /phoenix/upload/{id}` | `{offset, complete}`; also the `Upload-Offset` header |
| `PUT /phoenix/upload/{id}` + header `Upload-Offset: N`, raw bytes | Append. `200` → new `offset`. `409` → body has the server's real `offset`; continue from it |
| `DELETE /phoenix/upload/{id}` | Abandon |

Behaviours that matter on a phone:
- **A cut chunk is not wasted.** Bytes received before the connection died are kept (measured: 24 mid-chunk disconnects, file bit-exact, partial bytes kept).
- **A dead connection cannot block the new one.** If the OS killed the socket but the server has not noticed, a new `PUT` takes the upload over immediately and the old writer is stopped. (Measured: the new `PUT` returned in under 2 s while the old one was stalled; the old one's late bytes never touched the file.)
- A finished upload lands **in Gradio's own upload folder** and the reply carries a **FileData-compatible dict**, so `phoenix_submit_video` takes it unchanged. Tested through Gradio's real queue (its upload-folder check accepts it).
- Optional `sha256`: a damaged file is refused (`422`) and discarded rather than processed.
- Limits (`config.py`): 2 GB per file, 32 MB per request, 16 unfinished uploads (idle ones are evicted to make room), unfinished uploads dropped after 6 h, video/image extensions only, names sanitised.

### 3.2 Idempotent submit — `settings.client_token`
A token (12–64 chars `A-Z a-z 0-9 _ -`), generated **once per user action and stored before the call**. A retry with the same token returns the job the first attempt started (`"deduplicated": true`). `phoenix_job_status`, `phoenix_download`, `phoenix_cancel` and `GET /phoenix/status/{ref}` accept the token in place of the job id, so a phone that lost the response can ask "did my submit land?" and resubmit only on `not_found`. Six simultaneous submits with one token start one job (tested). A malformed token is an **error**, never silently ignored (ignoring it would quietly turn retries back into duplicates). No token → exactly the old behaviour.

### 3.3 Split on the server — `settings.split_seconds`
Upload the clip **once**; the Space cuts it (the pipeline already cuts by `trim_start`/`trim_end` with ffmpeg). `split_seconds: 30` on a 100 s clip makes 4 equal, **contiguous** 25 s parts (`end(i) == start(i+1)`), one job each, queued back-to-back on the one worker, named `Phoenix_<job>-<n>.mp4` as today. The reply has `jobs: [...]` (top-level `job_id` is part 1). Honours your own `trim_start`/`trim_end`. Each part's `secs` cap is raised to cover the part so the default 30 s cut-off does not truncate it. Parts carry tokens `<token>-p<n>`, so repeating the whole call, or calling again after one part failed to start, never duplicates a part (tested). Max 10 parts (`SPLIT_MAX_PARTS`); more → `too_many_parts` with the `split_seconds` to use. The old way (phone-side split with `part_count`/`part_index`) is untouched (differential test: 48 settings × face combinations give identical arguments to the engine).

### 3.3b Status without Gradio — `GET /phoenix/status/{job-or-token}`
Plain HTTP, no Gradio session/queue. Never exposes server paths.

### 3.4 Resumable download — `GET|HEAD /phoenix/download/{job-or-token}`
`Range` (open-ended, suffix, clamped), `If-Range` + `ETag` (a changed file is re-sent whole, never spliced), `416` with `Content-Range: bytes */size`, correct filename incl. `-<n>` for parts, `409 not_ready` / `410 expired` / `404`. Streams in 256 KB blocks (no copy, no hashing — the Gradio route copies the file into its cache on every call). A download cut at 1.2 MB and resumed gave an identical file; a file deleted mid-stream still completes.

### 3.5 Results wait 12 h, not 3 h
`RETAIN_SEC` and `SERVER_OUTPUT_TTL_SEC` 10800 → 43200 (keep them equal). Start a long split, minimise, sleep: the first part used to be deleted before you opened the app. Costs disk on `/data` (a 30 s 720p result is typically tens of MB).

### 3.6 Wiring
`app.py` now launches with `prevent_thread_lock=True`, adds the routes to the live server, then blocks (`_serve()`). The routes are imported inside `_serve`, so a Space that got `app.py` but not the matching adapter still starts and serves what it served before (tested), and `install_http_routes` never raises. `RESUMABLE_ENABLED = False` switches the transport off.

---

## 4. Verification (all on the real pinned Gradio 4.44.1 + uvicorn + real sockets; the video engine is stubbed)

| Suite | Result |
|---|---|
| Upload: happy path, 24 random mid-chunk cuts, stalled-connection takeover, offset discipline, token resume, limits/validation/eviction, janitor, Gradio queue accepts the file, 200 MB streamed | 38/38 |
| Download/Range, idempotent submit, real-queue drop (running and queued) → token recovery, split (contiguity, retries, partial failure, concurrency, trim, caps) | 66/66 |
| Differential vs v11.2.79; new app + old adapter; new adapter + old config | 16/16 |
| Reference client through a proxy that kills connections (100 cuts on a 30 MB upload, 26 on a 24 MB download), client process `kill -9`'d mid-upload then resumed, full split flow | 10/10 |

`swap_engine.py` and `core_pipeline.py` are byte-identical to v11.2.79.

**Not tested — please read:**
- **Through Hugging Face's edge proxy.** Everything above is local. After deploying, run `python tools/phoenix_resumable_client.py --space https://<owner>-<space>.hf.space --hf-token hf_… probe`; it must say `PRESENT`. If it says `MISSING`, tell me the HTTP code.
- **The Android app itself.** Not in this repo; nothing here changes its behaviour until it uses the routes.
- **The real engine on a split** (stubbed). Each part is a normal job with trim percentages, which the pipeline already supports.
- Python 3.10 (the Space) vs 3.11 (here): nothing 3.11-specific is used, but I could not run 3.10.

---

## 5. What the phone app must do (the other half)

The reference client `tools/phoenix_resumable_client.py` is the executable spec (requests only; port it). The app needs:

1. **Persist state before acting** (database, not memory): the run token, per-file upload tokens, `upload_id`s, job ids, partial download path + `ETag`. Survives the app being killed.
2. **Upload** with the chunk loop in §3.1; on *any* failure call `GET /phoenix/upload/{id}` and continue from the server's offset. Never restart from 0.
3. **Submit** with `client_token` (and `split_seconds` instead of cutting on the phone). On a lost reply: `GET /phoenix/status/{token}`; `not_found` → resubmit with the same token.
4. **Poll** `GET /phoenix/status/{id}` (no Gradio session).
5. **Download** each part to a `.part` file with `Range`/`If-Range`; rename when complete.
6. **Keep running while minimised** — this is Android's rule, not the server's: run the transfer as a long-running worker/foreground service with a visible notification, and retry when the network returns. Check the current Android documentation for the version you target (newer versions have user-initiated data-transfer jobs, and foreground-service types have time limits). I have not written or tested any Android code.

With 1–5 in place, even if the OS kills the app outright, reopening it resumes from the last byte; 6 only decides whether it keeps going while hidden.

---

## 6. Trade-offs and limits

- **Disk:** each split part copies the source (`/tmp/vid_<job>.mp4`), so N parts of a 500 MB clip hold N×500 MB until their jobs end. Fine on a 50 GB Space; it is why a single upload is capped at 2 GB and parts at 10.
- **Each part is its own job**, so the tracker starts fresh at each part boundary — identical to splitting on the phone.
- A part that fails after the others started is reported; calling again with the same token finishes the set.
- The transport routes are open to anyone who can reach the Space (it is private, so a token is needed at HF's edge). Concurrent `init` calls can over-commit disk; the cap of 16 and the 6 h/15 min eviction bound it.
- A token identifies **one attempt**. After a Space restart, a job interrupted by the restart is returned (as `error`) for its token; use a **new** token to resubmit.
- The queued-and-dropped Gradio case still discards a submit that never ran; the token turns that into `not_found` + resubmit instead of a phantom or duplicate job.

## 7. Install / rollback

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to the **root** of the Space, commit, normal restart. `tools/` is **not** uploaded. Startup log must show `Phoenix v11.2.80` and `resumable transport on: /phoenix/upload, /phoenix/status, /phoenix/download`; `swap_engine` still reads `aequus-1.2.79-reid` (the engine did not change).

Rollback: `RESUMABLE_ENABLED = False` (routes off), `RETAIN_SEC`/`SERVER_OUTPUT_TTL_SEC` back to 10800, don't send `split_seconds`/`client_token`.

## 8. Changed files
`phoenix_api_adapter.py` (idempotent submit, split, token lookups, transport routes), `app.py` (`_serve`), `config.py` (version, 12 h retention, `RESUMABLE_*`, `SPLIT_MAX_PARTS`). New: `tools/phoenix_resumable_client.py` (reference, not for the Space), this document. Unchanged: `core_pipeline.py`, `swap_engine.py`, `packages.txt`, `requirements.txt`, `README.md`.
