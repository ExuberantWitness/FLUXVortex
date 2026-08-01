# N3.1j5 G1 data-exposure addendum

Date: 2026-07-29  
Status: written after diagnostic q16/q24 values were visible

## Why this addendum exists

`PLAN.md` froze the quadrature threshold at `0.5%`, but described the
observable only as “sentinel total force”.  It did not freeze whether the
percentage meant a body-force vector norm or the maximum of the reported
lift/thrust component changes.  That ambiguity was discovered only after the
diagnostic runs below had completed, so it must not be repaired silently or
represented as a preregistered definition.

## Exposed diagnostic data

- q16 plumbing run:
  `runs/20260729_204918/`
- q24 plumbing run:
  `../n3_spatial_edge_pressure_v1_shadow_q24/runs/20260729_205128/`
- high-twist sentinel:
  `U=8 m/s, f=2.6 Hz, nominal twist=45 deg, AoA=5 deg`
- q16: `L=5.7622896034 N`, `T=-3.1857096388 N`
- q24: `L=5.7994179293 N`, `T=-3.2029935510 N`

These runs predate the strengthened campaign source/environment lock and are
therefore plumbing evidence only.

## Non-adaptive interpretation

The formal scorer will report both:

1. each reported force component,
   `|F_q16-F_q24| / max(|F_q24|, 1e-12 N)`; and
2. the wind-axis force-vector difference,
   `||(L,T)_q16-(L,T)_q24|| / ||(L,T)_q24||`.

The candidate passes only if the maximum component percentage is at most
`0.5%`; this preserves the component-wise convention used to report the v0
quadrature family and is stricter than the vector norm.  No threshold is
changed.

The already exposed diagnostic values fail under both interpretations:

- maximum component change: approximately `0.64%`;
- force-vector change: approximately `0.62%`.

Therefore the ambiguity cannot be resolved in a way that changes the
scientific decision.  A strengthened formal replay is permitted only to
authenticate the result.  If it reproduces the failure, G1 terminates the
candidate: do not run `dt/2`, `representative32`, or `confirmed151`.

