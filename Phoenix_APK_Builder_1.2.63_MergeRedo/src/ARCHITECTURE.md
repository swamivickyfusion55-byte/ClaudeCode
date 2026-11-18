# Phoenix Mobile Architecture

## Core invariants

### 1. Identity is explicit

Every target face has:
- trackId
- identityId
- sourceFaceId
- embedding
- confidence
- visibility state

A face cannot inherit another person's identity merely because another face is nearby.

### 2. Occlusion is conservative

When a target face is occluded:
- stop swapping that target
- preserve original target pixels
- retain track state
- reacquire only after confidence is restored

No speculative white/brown patch should ever be painted over an occluded area.

### 3. Side-profile hysteresis

Do not immediately change identity because one frame has a weak embedding.

Use:
- embedding similarity
- landmark geometry
- track continuity
- motion
- face-size stability
- confidence hysteresis

### 4. Detection is adaptive

Detector runs at key intervals.
Tracker handles intervening frames.
Re-detection is triggered by:
- confidence drop
- large motion
- track loss
- scene cut
- occlusion recovery

### 5. Progress is time-based and work-based

The final engine will expose:
- frames decoded
- frames processed
- frames skipped
- faces swapped
- detector calls
- tracker calls
- encode progress
- rolling processing FPS
- stable ETA

ETA must use a rolling window, not the instantaneous first few frames.

## Processing stages

1. Demux/decode
2. Lightweight motion analysis
3. Face detection when required
4. Multi-object face tracking
5. Identity association
6. Occlusion gate
7. ROI swap inference
8. Temporal stabilization
9. Composite
10. Hardware encode
11. Mux audio
12. Atomic output commit

## Quality rule

If confidence is insufficient, preserve the original frame.

A visible imperfect original is preferable to a wrong face or artificial rectangle.
