# Q16 review bug-fix results

## Outcome

The review exposed real integration bugs, not merely missing documentation.
This repair closes the shared-node transfer, stable boundary ownership,
homogenized macro-property distribution and real aerodynamic branch ownership.
It does **not** claim a completed Q16 fluid--structure trajectory.

## Closed defects

1. **Disconnected structural transfer DOFs.**  `Q16SurfaceTransferMap` now owns
   the exact shared `Q16Mesh`.  A 2x1 mesh therefore has 28 unique nodes and 168
   global DOFs, not two disconnected 96-DOF element blocks.  CPU and CUDA
   transpose maps assemble shared-node contributions exactly once.
2. **Ambiguous interface ownership.**  Algebraically duplicate surface points,
   including the two parametric representations of a shared edge, are rejected
   before a load can be counted twice.
3. **Coordinate-inferred root.**  `make_clamped_q16_nodes` clamps explicit,
   topology-owned node IDs.  A 120-degree rigid rotation demonstrates that the
   node owner is invariant while the old global-axis-minimum heuristic is not.
4. **No early-design stiffness/mass field.**  `Q16MacroPropertyField` assigns
   homogenized Young's modulus, Poisson ratio and density to each Q16 macro
   element.  This supports rigid/flexible and mass-distribution exploration
   without pretending to resolve explicit ribs, spars or laminate details.
5. **Optional LEV/TEV/wake modes.**  the Q16-specific live guard accepts only
   exact `separated LEV=True`, `joint TEV=True`, `prescribed wake=False`, CUDA
   float64 and the same configuration/particle owners before and after a trial.
6. **Mock-only aerodynamic rollback.**  `Q16CudaAeroSolverOwner` now branches the
   real `CudaJointLEVTEVSolver` through its executable pickle state, evaluates
   trials away from the live parent, and advances the unique owner only when the
   latest exact proposal is committed.  Failure, parent mutation, proposal
   mutation and force mutation fail closed.  A real two-step CUDA free-wake run
   verifies separated LEV shedding, joint TEV state, particle advance and wake
   convection on the selected branch only.
7. **Redundant inner validation.**  the shared MITC16/EAS mesh uses prevalidated
   element force/Jv entry points inside Newton--CG, while geometry, EAS and
   output-finiteness gates remain active.

An attempted generic `deepcopy` was rejected during development: it copied the
Python object graph but lost panel-vortex vertex state, so the first real Ptera
run failed.  The production branch path uses a pickle round trip, and that exact
failure now has a regression test.

## Verification

Registered joint command:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:platform/warp_vpm pytest -q \
  tests/test_q16_ancf_element.py \
  tests/test_q16_ancf_continuum_gpu.py \
  tests/test_q16_mitc16_projection.py \
  tests/test_q16_ans_eas_continuum.py \
  tests/test_q16_ans_eas_continuum_gpu.py \
  tests/test_q16_ancf_shared_mesh_gpu.py \
  tests/test_q16_boundary_constraints_gpu.py \
  tests/test_q16_work_conjugate_transfer.py \
  tests/test_q16_structural_step_gpu.py \
  tests/test_q16_mandatory_aero_mode.py \
  tests/test_aero_step_transaction.py \
  platform/warp_vpm/test_q16_real_aero_branch_transaction.py \
  platform/warp_vpm/test_ptera_gpu_active_lev.py
```

Final result: **102/102 passed**.  Black, Ruff, pycompile and whitespace checks
passed on the 13 changed source/test files.

## Remaining blocker

The scientific data path from the real aerodynamic solver into Q16 is still
missing.  In particular, a completed implementation must:

- update the real Ptera wing geometry from Q16 trial position and velocity;
- retain the CUDA aerodynamic force application points/leg forces (or an
  equivalent exact wrench), then map them through the shared Q16 transpose
  without a CPU numerical fallback;
- run the Newmark predictor/corrector against those forces;
- commit the selected aerodynamic branch and structural result atomically.

The current real-branch test returns a CUDA placeholder generalized-force array
after the actual aerodynamic trajectory.  It proves ownership/rollback only;
it is deliberately not evidence of a coupled Q16 time step.  The serialized
branch is also a correctness-first transaction and has not been performance
profiled.  No FSI accuracy, flexible-wing benchmark or co-design result is
unlocked by this checkpoint.
