# v5h R0–R1 run log summary

## Execution route

The `experiment` skill was followed for planning, hard gates, fresh outputs, and closeout. Its managed `bash_exec`, memory, and artifact services were unavailable in this runtime, so commands were executed with the ordinary terminal tool and durable evidence was written into this repository. No raw byte-addressed terminal log is claimed.

## Environment incident and recovery

The first isolated environment resolved registry FastMultipole v2.0.4, which failed FLOWVPM precompilation because the required `metadata_per_body` API was missing. Exactly one recovery was made: FastMultipole was pinned to upstream commit `adc4f264732de3dbbd492758e729af0b35db54b2`, the revision targeted by this FLOWVPM port. No numerical model default was changed.

## Commands and outcomes

1. Official upstream suite: the exact command is stored in `run_manifest.json`. It asserted both FLOWVPM and FastMultipole commit/tree identities before calling `Pkg.test("FLOWVPM"; allow_reresolve=false)`. Result: exit 0; 10 single-ring and 4 leapfrog testsets passed.
2. Julia oracle export: single-threaded Float64 direct interactions. The canonical HDF5 and JSON mirror were written into `oracle/` without overwrite.
3. Python parity evaluation: current code reproduced every frozen gate in `metrics.json`; exit 0.
4. Python tests: `24 passed in 0.19s` with third-party pytest plugin autoload disabled.
5. Formatting/static checks: system Black 23.11.0 reported five files unchanged; Ruff 0.15.11 reported all checks passed.
6. Fresh replay: current Python evaluation generated a byte-identical metrics file. A fresh same-basename Julia export generated a byte-identical JSON mirror. `h5diff` returned exit 0 with no semantic dataset difference between fresh and canonical HDF5.

## HDF5 reader decision

The base Python `h5py` installation had a NumPy ABI mismatch, while the fluxvortex environment did not include h5py. The canonical source artifact remains HDF5 written by Julia; a deterministic JSON mirror is exported from that same HDF5 in the same Julia process. Python tests read JSON without skipping parity. HDF5 semantic replay is checked with `h5diff`.

## Route discipline

- No target-paper observations were loaded.
- No Ptera or legacy particle file was changed.
- No clip, NaN replacement, FMM, SFS, viscosity, or force channel was enabled.
- No target metric was used to choose a parameter.
