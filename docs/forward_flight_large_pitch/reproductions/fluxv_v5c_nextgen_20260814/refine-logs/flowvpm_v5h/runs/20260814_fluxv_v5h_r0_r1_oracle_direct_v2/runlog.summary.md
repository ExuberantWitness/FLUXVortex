# v5h R0–R1 schema-v2 run log summary

The run used the `experiment` skill's frozen-gate workflow. Managed `bash_exec`, memory, and artifact services were unavailable; ordinary terminal execution and repository-resident evidence were used instead. No raw byte-addressed terminal log is claimed.

Execution:

1. Pinned Julia Project/Manifest retained FLOWVPM `4f433fb...` and FastMultipole `adc4f26...`; no resolver change occurred.
2. The schema-v2 exporter wrote a fresh HDF5 and JSON mirror without overwrite.
3. The evaluator checked schema/config, full and probe U/J, nearfield spacing, fixed/affine-Uinf RK, relaxation, finite values, and clip count; all gates passed.
4. Pytest recomputed `evaluate(payload)` and required exact field-for-field equality with stored `metrics.json`; 29 tests passed.
5. Black and Ruff passed.
6. A second same-basename export produced byte-identical JSON and semantic-zero-difference HDF5; a second evaluation produced byte-identical metrics.

The environment incident remains documented: registry FastMultipole 2.0.4 was API-incompatible with the pinned FLOWVPM port, so the single source-justified recovery pinned upstream FastMultipole commit `adc4f26`. No target observations, Ptera coupling, force path, clipping, SFS, or viscosity were used.
