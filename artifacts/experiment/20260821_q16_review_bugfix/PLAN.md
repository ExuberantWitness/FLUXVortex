# Q16 review bug-fix plan

- run id: `20260821_q16_review_bugfix`
- tier: auxiliary/dev integration repair
- baseline: branch `run/q16-lev-tev-pc-fsi-20260821`, commit `dc43d4c`
- scope: only bugs exposed by the four-angle review; no Q9, no generic material expansion

## Research question

Can the existing Q16 components be made compositionally correct for a shared-node
wing mesh and be placed behind an exact, mandatory separated-LEV/joint-TEV/free-wake
transaction boundary without weakening the CUDA float64 or rollback contracts?

## Frozen changes

1. Replace element-local `element_count*96` transfer ownership with the exact
   shared `Q16Mesh` connectivity and global DOF count.
2. Add an explicit stable-node boundary constructor so swept/rotated Q16 wings
   need not infer their root from a global coordinate minimum.
3. Add a Q16 aerodynamic adapter contract which rejects disabled separated LEV,
   non-joint TEV and prescribed wake, and whose snapshot covers the real CUDA
   particle fields plus every solver state mutated by a trial.
4. Add a joint structural/aerodynamic step owner only after the above two
   interfaces pass exact rollback and one-commit tests.
5. Remove only provably redundant inner-loop host checks; retain fail-closed
   numerical checks at transaction boundaries.  Do not trade correctness for a
   broad "GPU-only" label.

## Acceptance gates

- a two-element shared mesh has transfer/global structural DOF count `168`, not
  `192`, while virtual work, force and moment remain within the frozen tolerance;
- shared-edge point/load ownership is counted exactly once;
- explicit root-node ownership is invariant under rigid coordinate rotation;
- the Q16 aerodynamic production contract mechanically requires separated LEV,
  joint TEV and free wake;
- two predictor trials leave the real adapter parent hashes unchanged; injected
  failure restores every registered field; one accepted proposal commits once;
- the existing Q16 focused surface remains green;
- no claim of complete FSI or co-design is made until a real one-step solver test
  passes.

## Stop conditions

- a fix requires changing the frozen Q16 kinematics, MITC/ANS/EAS equations,
  separated-LEV model or wake integrator;
- rollback cannot cover a mutable real solver field exactly;
- Torch/Warp interop would require a hidden CPU numerical copy;
- any existing focused Q16 test regresses.

## Expected files

- `src/fluxvortex/q16_work_conjugate_transfer.py`
- `src/fluxvortex/warp_fsi/kernels_q16_transfer.py`
- `src/fluxvortex/q16_boundary_constraints.py`
- Q16 transaction/adapter module only if the real owner contract can be closed
- focused tests for the same modules

## Logging limitation

The selected `experiment` skill requests managed `bash_exec`, which is not
available in this session.  Commands are therefore run through the local
terminal fallback and must be copied into the final run summary.

## Revision log

- During tests-first repair, the shared-edge audit showed that global assembly
  alone did not prevent the same algebraic surface point from being registered
  through both adjacent elements.  The scope was therefore narrowed further by
  adding duplicate algebraic-row rejection.
- The Q16 scientific-abstraction review also showed that a uniform material
  scalar could not represent the requested early-design rigid/flexible and mass
  distribution.  A per-element homogenized macro-property field was added;
  detailed rib/spar/laminate topology remains explicitly out of scope.
- A generic `deepcopy` real-solver branch passed shallow state tests but failed
  the first actual Ptera run because panel-vortex vertices were absent.  The
  accepted transaction uses Ptera's executable pickle state and retains the
  real-run regression as a mandatory gate.
