# Figure 12 independent visual review

Date: 2026-07-30  
Reviewer role: primary Codex execution thread  
Scope: digitization identity only; not candidate-model validation.

The persisted `roundtrip_overlay.svg` was rasterized independently and
compared side by side with PDF pages 17 and 18 of the audited article
(`SHA256 cc4970b38b3586affc4805a84e526fcb0049ba2dfa42219c01379e2a8f48fa84`).

Result: **PASS**.

- Panels `(a)` through `(h)` retain the published ordering:
  upstroke `12/15/18/19 deg`, then downstroke `12/15/18/19 deg`.
- The black source polylines and red CSV round-trip samples visually
  coincide on both surfaces in all eight panels.
- Blue experimental diamonds follow the published taps; the legend diamond
  is absent from every data series.
- The strong upper-surface separation plateaus in panels `(d)`, `(g)`, and
  `(h)` are neither swapped nor smoothed.
- The continuation from article page 201 to 202 is correctly mapped.
- The leading tap is correctly identified as Report 9221
  `x/c=0.00025`; its unsnapped vector coordinate remains separately stored.
- Both categorical 19-degree panels retain a null exact phase.

This pass authorizes use of the CSV as the published Figure 12 response
oracle. It does not say that N2.6-SVI-DW reproduces those curves; that remains
the later eight-panel source-response gate.
