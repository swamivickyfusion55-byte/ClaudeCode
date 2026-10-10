# Phoenix Mobile 1.2.10 — "Push to all Spaces" fix

**Your screenshot:** `Updated 0, failed 21. HTTP 400 for swamivicky/SwamitechUAT2. {"error":"✖ Invalid input: expected string, received undefined → at value.summ…`

## What I found

`1.2.9b` is a different build from my 1.2.9: its source holds the "Push to all Spaces" button and the
landscape layout. Its commit code builds the right body: the first line has
`{"key":"header","value":{"summary":"Phoenix pack update from the phone",…}}`. The Hub still says
`value.summary` is missing, so the body is being read in the wrong form.

I ran its real code and Hugging Face's official Python client against a local server and compared the bytes on the wire:

| | Content-Type sent |
|---|---|
| `huggingface_hub` (official) | `application/x-ndjson` |
| `huggingface.js` (official, from its source) | `application/x-ndjson` |
| **1.2.9b** | `application/x-ndjson; charset=utf-8` |

OkHttp appends `; charset=utf-8` whenever a body is built from a `String`. The Hub has two body formats
(NDJSON and plain JSON) and chooses by Content-Type. A third-party reference for the endpoint describes
both formats, so with the extra parameter the NDJSON body is most likely read as the JSON form, whose
top-level `summary` does not exist. That would produce exactly "expected string, received undefined at value.summary" on every Space.

**My 1.2.9 source had the same flaw** (it also built the body from a String). Fixed too; see below.

## What changed (1.2.10 = 1.2.9b + this fix; versionCode 130)

- `HfApi.commitSpaceFiles`: body is sent as bytes, so the header is exactly `application/x-ndjson`; lines are written
  the way `JSON.stringify` writes them (org.json writes `/` as `\/`).
- If the Hub still answers 400 mentioning `summary`, the same commit is retried once in the Hub's plain-JSON form,
  and both answers are shown if both fail.
- The failure line shows the **first** error in full (it showed the last, cut at 120 characters).
- Nothing else changed: same 37 files, same Dockerfile / `gradle.properties`; only version strings changed in `build.sh`,
  `app.py` and `README.md`.

## Verified, and what is not

- **Wire format:** the fixed request is byte-identical to the reference serialisation of the same files, including
  quotes, backslashes, tabs, `/`, control characters and non-ASCII in paths.
- **Reproduction:** against a local server that models "body format chosen by exact Content-Type", the old code fails
  with your exact error and the new code succeeds. The model is my hypothesis, not the Hub's code.
- **Not verified:** the real huggingface.co. It is blocked from my sandbox, so I cannot confirm the cause or the fix.
  The Content-Type difference is measured. That it is *the* cause is an inference; if it is wrong, the plain-JSON
  retry may still work, and otherwise the new error text will show the real reason.
- **Compile:** all 15 Kotlin files compile with 0 errors (same off-device toolchain as 1.2.9); pack audit 40/40. No APK was built here.

## If it still fails

First tick **one** Space only, or push to one. Then send me the whole line that starts `First error:`.
