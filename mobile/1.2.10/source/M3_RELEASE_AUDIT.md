# Phoenix Mobile M3 — Release Audit Gate

## Release status
**M3-rc1 is a release-candidate engineering pack, not a claim that the local face-swap model graph is complete.**

### Gate A — static engineering audit
Run:

```bash
python3 tools/audit/m3_audit.py
```

The gate checks versioning, dual-engine declaration, secret leakage, explicit local-engine failure behavior, and removal of the synthetic remote percentage algorithm.

### Gate B — device/visual audit (mandatory before APK release)
A real APK must be tested on the target Android phone with:

1. Person A → source A, Person B → source B.
2. A/B crossing and partial/full occlusion.
3. Face leaving/re-entering the frame.
4. Front → 3/4 → side → front transitions.
5. No-face frames: output must preserve original pixels.
6. Similar-looking faces: identity must not switch.
7. 720p/1080p clips and long clips.
8. Audio preservation and final mux validation.
9. CPU-only run and, if supported, XNNPACK/NNAPI benchmark.
10. Memory peak and thermal/throttling observation.

**Fail-closed rule:** uncertain target visibility/identity must preserve the original frame rather than paint a guessed face or rectangle.

## Why the local engine remains gated
The project intentionally does not embed or redistribute a third-party face-swap weight whose license has not been verified for this application. The M3 boundary is ready for the selected, licensed detector/recognizer/landmark/swapper graph.
