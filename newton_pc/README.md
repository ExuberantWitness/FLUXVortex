# newton_pc — window predictor-corrector coupling (problem-agnostic)

A minimal, solver-agnostic implementation of the **two-pass window
predictor-corrector** strong-coupling scheme, packaged so it can drive either
a hand-rolled physics code or a [newton-physics](https://github.com/newton-physics/newton)
solver through public force buffers (`State.particle_f` / `State.body_f`) —
the integration path recommended by the Newton team for external physics.

```
predictor:  march the window with forces extrapolated from previous solves
solve:      evaluate the force provider at the predicted window-end state
rewind:     restore the window-start snapshot
corrector:  re-march with forces interpolated old -> new per substep
            (linear or quadratic; optional Picard re-iterations)
```

## Why: the two capabilities existing coupled frameworks lack

Verified against newton-physics PR #2848 source (experimental coupled-solver
framework): coupling forces are **zero-order-held across substeps**
(`solver_coupled.py:2152,2169`) and there is **no predictor pass**. Both
matter enormously for tightly-coupled multi-rate problems (aeroelasticity,
added-mass-dominated FSI):

| evidence (same arena, same coupler, only `mode` differs) | lagged (ZOH) | two-pass |
|---|---|---|
| Aeroelastic plate, t*=3, vs MATLAB 1e-6 reference | **46.8%** worst amplitude error | **5.9e-5** |
| Aeroelastic plate, t*=6 (multiple flapping reversals) | — | **5.5e-4** |
| Newton VBD cloth in wind, strongly coupled, equal solve budget | 2.6e-2 | **2.4e-3 (10.5×)** |
| Newton VBD cloth, mild coupling | 1.9e-4 | 1.3e-4 (1.5×) |

The FLUXV arena numbers are against an independent MATLAB reference matched
to 1e-6 (full validation chain in `../docs/`); the Newton arena uses a
per-step-coupled run as reference. The cost difference: two-pass ≈ 2×
structural marches per window, same number of force solves.

## Architecture

```
newton_pc/
├── protocols.py     StructuralEntry / ForceProvider / ForceSet (3 methods each)
├── coupler.py       WindowPredictorCorrector
│                      mode      = two-pass | lagged
│                      interp    = linear | quad
│                      iterations, adaptive_tol   (Picard re-iteration)
├── adapters/
│   ├── fluxv.py        ANCF shell + UVLM (the MATLAB-validated arena)
│   └── newton_vbd.py   newton SolverVBD cloth + ring-vortex UVLM provider
├── examples/
│   ├── fluxv_longrun.py    long-horizon stability vs MATLAB truth
│   └── plate_in_wind.py    Newton demo, lagged vs two-pass
└── tests/
    ├── test_coupler.py      unit tests (toy analytic arena, <1s)
    └── test_regression.py   red line: reproduces the validated chain
                             bit-identically (ratio=1.000000)
```

## Mapping to newton-physics #2848 concepts

| this package | #2848 counterpart | status |
|---|---|---|
| `StructuralEntry.snapshot/restore` | `coupling_notify_input_state_update(restart=True)` + proxy `iteration_restart` | generalizes per-step restart to multi-substep windows |
| `StructuralEntry.substep(t, dt, forces)` | per-entry `substeps` | **adds force interpolation across substeps (missing upstream)** |
| `ForceProvider.solve/commit` | proxy feedback harvest | adds repeatable-solve + explicit-commit semantics for auxiliary state (wake) |
| coupler `mode="lagged"` | `SolverProxyCoupled(mode="lagged")` | built-in baseline for comparisons |
| force operators in `ForceSet` | `coupling_eval_effective_mass_block` / virtual inertia | added-mass operators ride the same interpolation |

## Quick start

```bash
pytest newton_pc/tests/test_coupler.py -q          # unit tests
python -m newton_pc.tests.test_regression          # MATLAB red line (needs fixtures)
python newton_pc/examples/plate_in_wind.py         # Newton demo (pip install -e newton)
python newton_pc/examples/fluxv_longrun.py --tstar 3 [--mode lagged]
```

## Scheme provenance

The scheme is the window coupling of Yamano et al.'s aeroelastic solver,
reverse-engineered line-by-line and validated to 1e-6 against the original
MATLAB implementation (free-running GPU port matches to 0.0005% over 500
substeps). Three systematic studies (exact complex-step derivatives, clean
single-window convergence metrics) established that the scheme needs no
gradient enhancement — a good predictor plus window-interpolated Picard is
already near-optimal across this system's physical envelope. See
`../docs/grad_pc_study.md` and `../docs/newton_comparison.md`.
