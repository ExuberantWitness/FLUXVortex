# FluxV v5h3 native Ptera feedback vertical slice

Status: `go_one_way_native_feedback_mechanics_only`.

This non-target auxiliary run verifies that a live v5h2 dyadic cumulative
rVPM cloud can enter the existing FluxV `UVPMHybridSolver` exactly once at
Ptera collocation points and exactly once at each of Ptera's four native load
LineVortex-centre batches. Ptera remains the only bound-circulation, TE
ring-wake, panel-force, moment, and airplane-load owner.

The smoke uses a generic NACA0001 rectangular API wing, three Ptera steps,
`dt=0.02 s`, two DVM circulation signs, and a two-report live parent chain.
It reads no Yang, Izraelevitz, or Baik observation and makes no load-accuracy
claim.

Headline evidence:

- hard-off factory is the exact native `UVPMHybridSolver`;
- enabled-empty run is bitwise equal to native FluxV state and loads;
- active step ledger is `1` collocation evaluation, `4` load-leg evaluations,
  and `1` native Ptera load-processor call;
- maximum native no-penetration residual is
  `2.7755575615628914e-17`;
- both DVM signs and the step-1 to step-2 live cloud chain pass;
- 11 focused and 117 related regression tests pass.

This does not close bound/wake-to-rVPM transport feedback. Target-paper
scoring remains blocked.
