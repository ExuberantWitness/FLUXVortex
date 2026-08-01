# Cancelled fresh V4.1 run

- timestamp: 2026-07-29 18:32 +08:00
- command: `platform/lb_sweep151_fresh.py --timestamp 20260729_135128 --resume`
- process group: `1169771`
- action: sent `SIGTERM` after the user explicitly cancelled further V4.1
  validation.
- outcome: process exited normally after the signal; no result, manifest, lock,
  or evidence artifact was deleted or rewritten by this note.
- caution: the existing manifest still reports `status=running`; it is an
  interrupted partial run and must not be interpreted as complete unless a
  future explicitly authorized resume completes its own integrity protocol.
