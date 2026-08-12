# GitHub publication scope

Target remote: `https://github.com/ExuberantWitness/FLUXVortex.git`  
Intended base branch: `aero-rvpm-lev`  
Intended topic branch: `agent/unified-fluxv-experimental-benchmarks`

The current worktree contains unrelated tracked and untracked research work.
Publishing must use an isolated worktree and an explicit allow-list; never use
`git add .` or force-add the repository root.

## Code allow-list

- `platform/forward_flight_benchmarks/`
- `platform/robofalcon2_aero.py` (required by the published Yang cross-case adapter/tests)
- `platform/tests/test_augmented_uvpm.py`
- `platform/tests/test_forward_flight_benchmarks.py`
- `platform/tests/test_periodic_load_ownership.py`
- `platform/tests/test_ullt_attached.py`
- `platform/tests/test_uvlm_polar_correction.py`
- `platform/tests/test_yang_plev.py`

## Documentation/result allow-list

- `docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/`
  - include the full v1, v2, Scherer Figure-14, and v3 result directories;
  - exclude both `runs/20260812_scherer_fig14_experiment_smoke/` and
    `runs/20260812_periodic_v3_persistent_smoke/` from the public commit;
- `docs/forward_flight_large_pitch/reproductions/plev2025/source_data/`
- inputs required by the current v1 runner:
  - `docs/forward_flight_large_pitch/reproductions/plev2025/runs/20260808_multimodel_full/mean_characteristics.csv`
  - `docs/forward_flight_large_pitch/reproductions/plev2025/runs/20260808_multimodel_full/phase_histories.csv`
  - `docs/forward_flight_large_pitch/reproductions/runs/20260807_rigid_firstpass/izraelevitz_fig11_exact/phase_histories.csv`
- `docs/forward_flight_large_pitch/literature/candidates_20260812/LITERATURE_ACCESS_AND_STEVENS_AUDIT.md`

The literature audit is text-only. Publisher PDFs, large theses, and the
third-party Fortran archive remain local research inputs and are not
redistributed by this branch.

JSON result files are ignored by the repository-wide `*.json` rule and require
file-by-file force-add.  Never use `git add -f .`.

## Explicit exclusions

- `platform/_v2_robo.py` and `platform/claim_nodes/*.yaml` (pre-existing user changes);
- RoboFalcon, paper5, DeLaurier and flapping_models_v3 work;
- literature PDFs and all files larger than 10 MiB;
- credentials, local caches, `__pycache__`, and the smoke-run directory.

## Suggested commits

1. `feat(benchmarks): add unified UVLM-ULLT forward-flight models`
2. `docs(results): add Yang and Izraelevitz experimental benchmarks`

The Figure-14 result must be described in two stages: it is a failed
independent experimental gate for frozen v1/v2; the subsequently introduced
v3 post-hoc load-owner repair passes this already-seen gate but is neither an
independent generalization result nor an LEV/dynamic-stall closure.
