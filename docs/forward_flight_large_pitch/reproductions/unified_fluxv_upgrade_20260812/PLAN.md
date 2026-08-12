# Unified FluxV UVLM-preserving upgrade experiment plan

## 1. Objective

- run id: `20260812_unified_uvlm_polar`
- selected idea in `1-2` sentences: retain the existing prescribed-wake UVLM history and replace only its finite-wing linear quasi-steady polar contribution by a full-angle finite-wing flat-plate polar.  The same geometry-derived law and constants are used for Izraelevitz 2017 and Yang 2025; no paper name, case id, or observation-derived residual is accepted by the implementation.
- user's core requirements: cross-compare existing algorithms on both papers; improve FluxV while retaining UVLM; compare old and improved FluxV; plot lift and drag versus the paper parameter and over representative cycles.
- non-negotiable user constraints: use the reconstructed paper geometry and conditions; show curves rather than only aggregate metrics; distinguish author/reference curves from executable local models.
- research question: can a source-frozen nonlinear finite-wing polar residual improve Yang wind-tunnel lift/drag while retaining useful agreement with the Izraelevitz UVLM reference?
- null hypothesis: the shared residual does not improve Yang and/or materially degrades the Izraelevitz reference comparison.
- alternative hypothesis: the shared residual substantially reduces Yang mean-load error while remaining competitive with the old UVLM channel on Izraelevitz lift/drag phase histories.

## 2. Baseline And Comparability

- baseline id: `fluxv_uvpm` from the frozen prior runs.
- baseline variant: `UVPMHybridSolver` with prescribed ring wake; VPM particles are one-way and do not feed back to loads.
- dataset / split: Yang 2025 rigid wing at 0, 5, 10, 15, 20, 25 deg; Izraelevitz 2017 Figure 11 AR=3 heave-pitch case.  All published observations/reference curves are evaluation-only.
- primary metric: Yang lift/drag MAE in gf; Izraelevitz phase-history RMSE and peak error against digitized author UVLM, reported separately for lift and drag.
- required metric keys: `mae`, `rmse`, `bias`, `max_abs_error`, `range_nrmse`, `prediction_half_amplitude`, `reference_half_amplitude`, `half_amplitude_error`, and separate positive/negative `peak_phase_error_cycle`; complete provenance and force-sign metadata.
- comparability risks: Yang uses nominal four-bar motion because the LDS trace is unavailable; Izraelevitz Figure 11 is numerical UVLM validation rather than experiment; digitization uncertainty; old FluxV and Ptera prescribed UVLM are the same load channel; Ptera Figure 11 time refinement is unstable beyond the documented range.
- independent experimental gate added after the exploratory model was frozen: Izraelevitz Figure 14 / Scherer 1968 cycle-mean thrust observations.  This is not Figure 11 and the 2017 paper has no Figure 17.

## 3. Code Translation Plan

| Path | Current role | Planned change | Why this is needed | Risk |
|---|---|---|---|---|
| `platform/forward_flight_benchmarks/uvlm_polar_correction.py` | absent | implement geometry-only full-angle polar residual and load-history augmentation | one shared UVLM-preserving improvement for both papers | force axes, local velocity, and double-counting errors |
| `platform/forward_flight_benchmarks/run_unified_fluxv_upgrade.py` | absent | run both paper matrices, metrics, manifests, and CSV artifacts | reproducible cross-paper execution | runtime and incomplete reference digitization |
| `platform/forward_flight_benchmarks/plot_unified_fluxv_upgrade.py` | absent | render data-driven PDF/PNG comparisons | satisfy curve-first request | overcrowded legends or misleading model identity |
| `platform/tests/test_uvlm_polar_correction.py` | absent | regression, zero-angle, symmetry, geometry-only, and sign tests | prevent hidden case switching and force-ledger errors | insufficient movement coverage |
| `docs/.../unified_fluxv_upgrade_20260812/source_data/` | absent | store digitized Izraelevitz reference with audit notes | enable quantitative author-curve comparison | raster digitization uncertainty |

## 4. Execution Design

- minimal experiment: formula unit tests plus Yang smoke at 0/15/25 deg and Izraelevitz smoke.
- smoke / pilot plan: verify finite outputs, old FluxV exact reproduction, no case identifier enters the correction, and correct drag convention.
- full run plan: reuse frozen old histories where hashes/settings match; compute the shared correction on the exact movement histories; generate six-angle Yang means and Izraelevitz Figure 11 phase curves.  One-factor strip/time sensitivity remains follow-up work and is not represented as a completed artifact.
- expected outputs: run manifest, model identity table, mean/phase CSVs, accuracy CSV, publication PDF/PNG figures, Chinese verification report, and an integrity audit.  No sensitivity CSV was produced in this exploratory run.
- stop condition: complete artifacts with all model-condition cells finite and claims classified.
- abandonment condition: shared correction worsens either benchmark beyond the preregistered baseline tolerance or requires case-specific coefficients.
- strongest alternative hypothesis: Yang's advantage comes from PLEV/AWS and unpublished measured kinematics, so a static polar residual may improve drag but cannot reproduce the author model's complete dynamics.

## 5. Runtime Strategy

- command for smoke: targeted unit/smoke tests in `platform/tests/test_augmented_uvpm.py`, `test_uvlm_polar_correction.py`, and `test_ullt_attached.py`; the final matrix runner intentionally accepts `--quality full` only.
- command for main run: first run `python -m forward_flight_benchmarks.run_unified_fluxv_upgrade --quality full`, then run `python -m forward_flight_benchmarks.run_unified_ullt_extension --base-run <v1-run> --output-dir <v2-run>` and render with `python -m forward_flight_benchmarks.plot_unified_fluxv_upgrade --run-dir <v2-run>`.
- independent experiment command: `python -m forward_flight_benchmarks.run_izraelevitz_scherer_experiment --quality full`, then `python -m forward_flight_benchmarks.plot_izraelevitz_scherer_experiment`.
- expected runtime / budget: minutes on CPU; no GPU required.
- log / artifact locations: `docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/runs/20260812_periodic_v1_full` and `.../20260812_periodic_v2_ullt_full`.
- safe efficiency levers to use first: reuse frozen old UVLM CSV histories; calculate the correction from movement geometry without rerunning an equivalent prescribed solver; vectorize strip evaluation.
- how existing tooling will be used efficiently: existing movement builders, frozen Yang full run, and Izraelevitz production histories remain the source of geometry, phase, and old-force baselines.

Monitoring and sleep plan:

- wait cadence: `60s`, `120s`, `300s`, `600s`, `1800s`
- health signals that justify continuing to monitor: new model-condition rows, finite loads, stable memory, and advancing log timestamps.
- conditions that trigger kill / relaunch: non-finite loads, repeated solver divergence, no artifact progress, or a baseline mismatch.

## 6. Fallbacks And Recovery

- if the intended model / endpoint / download path fails: use the already local PDFs and frozen run artifacts; do not substitute unrelated web values.
- if hardware or memory is tighter than expected: run conditions sequentially and reuse old histories.
- if the code path is wrong after smoke: stop before full execution, repair sign/geometry invariants, then rerun smoke.
- if the first full run becomes non-comparable: retain it as diagnostic only and issue a new run id after updating this plan.

## 7. Checklist Link

- checklist path: `CHECKLIST.md`
- next unchecked item: perform frozen-parameter blind validation and one-factor numerical sensitivity checks on a new, previously unseen forward-flight case.

## 8. Revision Log

| Time | What changed | Why it changed | Impact on comparability or runtime |
|---|---|---|---|
| 2026-08-12 | initial frozen plan | user selected Izraelevitz 2017 and Yang 2025 | establishes evaluation-only references and bans case-specific tuning |
| 2026-08-12 | v0 full-angle residual rejected by the joint gate | after the first Yang/Izraelevitz diagnostic, the all-phase 3/4-chord residual improved Yang but degraded Izraelevitz mean thrust | v0 is retained as a failed diagnostic; it cannot support a unified-improvement claim |
| 2026-08-12 | exploratory v1 uses a quarter-chord incidence proxy and 15--20 deg smooth activation | keep pitch-rate/non-circulatory loads in UVLM and limit the polar residual to separated incidence; 15 deg is the paper's attached Figure-11 limit | this was introduced after viewing v0 results and is explicitly exploratory, not preregistered or confirmatory |
| 2026-08-12 | exploratory v2 adds the source-constrained 1-state ULLT as attached alternating-load owner | the local ULLT independently reproduces Figure 11, whereas the full-angle residual is needed for Yang; UVLM/polar retains the cycle mean and separated alternating load | same observation-free gate in both papers; periodic two-pass only, so no online transient claim |
| 2026-08-12 | VPM diagnostic fast path enabled for the full matrix | current particles are one-way and made the first full attempt needlessly slow | a paired smoke regression proves identical lift/drag with particles on/off; model-load comparability is unchanged |
| 2026-08-12 | phase-correct movement-boundary derivative replaces a global history roll | the integrity audit showed that a duplicated phase-zero endpoint contaminated the Yang ULLT/v2 last phase sample | two new regression tests freeze endpoint correctness; the 15-degree lift closure jump fell from 0.3002 N to 0.00103 N; all v2 phase artifacts were regenerated |
| 2026-08-12 | added Izraelevitz Figure 14 / Scherer 1968 as a true experimental gate | Figure 11 is a numerical UVLM reference, not an experiment; the paper has no Figure 17 | the previously frozen v1/v2 fails the new experiment (CT RMSE 0.22260 versus old FluxV 0.05115), so the unified-generalization claim is refuted and production promotion is blocked |
| 2026-08-12 | opened `20260812_fig14_load_ownership_fix` as a new main/test repair pass | the additive instantaneous full-angle drag residual double-counts the mean streamwise load in symmetric heave--pitch propulsion | preserve v1/v2 as failed controls; implement a paper-agnostic periodic mean-load ownership rule, then rerun Yang 2025, Izraelevitz Figure 11, and the independent Figure-14 experiment before accepting the repair |

## 9. Figure-14 Repair Run Contract

- run id: `20260812_fig14_load_ownership_fix`
- tier: `main/test`
- research question: can the nonlinear-polar contribution be given a physically explicit periodic mean-load owner, without a paper/case switch, so that it retains the useful Yang correction without reversing Scherer mean thrust?
- baseline: frozen old FluxV, exploratory v1, and exploratory v2 artifacts above; their evaluation code and force conventions remain unchanged.
- hypothesis: the Figure-14 failure is a load-ledger error rather than a tunable coefficient error.  The instantaneous separated-polar residual may shape alternating loads, but its cycle mean may only be admitted where separation is persistent in the cycle-mean local incidence; zero-mean heave--pitch excitation remains owned by UVLM/ULLT.
- forbidden changes: no Figure-14 force fitting, no condition/phase lookup, no paper/case identifier, no change to experimental points, no phase shift, and no change to `Cd0=0.057` in the primary Figure-14 comparison.
- changed code surface: one reusable mean-ownership transform in `uvlm_polar_correction.py`, an explicit new model version in the runners, focused tests, new run directories, regenerated plots, and a revision note.  Historical v1/v2 artifacts remain immutable.
- primary acceptance keys:
  - Figure 14: repaired model RMSE must be no worse than old FluxV `0.05115`, and must improve substantially over v1/v2 `0.22260` without selecting `Cd0` by score;
  - Yang 2025: repaired lift and drag MAE must each remain below old FluxV (`6.855 gf`, `12.922 gf`);
  - Figure 11: repaired raw-phase lift and drag RMSE must not materially exceed old FluxV under the existing `0.060` coefficient trace-width tolerance.
- minimal falsification: one Figure-14 15-degree condition, Yang 0/15/25 degrees, and Figure-11 full case.  Abandon the rule if it needs a benchmark name or observation-derived threshold.
- full execution: all 12 unique Scherer conditions / 14 observations, all six Yang angles, and the complete Figure-11 phase history; output CSV, JSON manifests, PNG/PDF, and an integrity note.

The v1 joint acceptance rule is now frozen before rendering the final figures:

- Yang: neither lift nor drag MAE may be worse than old FluxV.
- Izraelevitz: neither raw-phase lift nor drag RMSE may be worse than old FluxV by more than the vector-digitization trace width (`0.060` coefficient units), and the paper-scaled peak-to-peak amplitude error must not increase in both channels.
- A model is called a successful unified upgrade only if both paper gates pass.  A Yang-only improvement is reported as such and is not promoted to the production FluxV model.
- The later Figure-14 experimental gate takes precedence over the two development-task gates.  Its failure means the current v1/v2 is retained as a diagnostic branch only.
