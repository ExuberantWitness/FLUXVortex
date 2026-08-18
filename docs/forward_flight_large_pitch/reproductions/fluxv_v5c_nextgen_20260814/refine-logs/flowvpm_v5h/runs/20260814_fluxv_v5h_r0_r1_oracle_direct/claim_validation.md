# v5h R0–R1 claim validation

```yaml
claim: >-
  The isolated Python direct backend numerically reproduces the pinned
  FLOWVPM Gaussian-erf U/J, reformulated f=0 g=0.2 RHS, low-storage RK3,
  and corrected Pedrizzetti relaxation on the frozen deterministic fixtures.
claim_supported: yes
evidence_type: pinned_julia_oracle_plus_deterministic_cross_language_replay
scope:
  precision: Float64
  interaction: direct_O_N2
  kernel: Gaussian-erf
  formulation: reformulated_VPM_f0_g0p2
  integration: FLOWVPM_three_stage_low_storage_RK3
  relaxation: corrected_Pedrizzetti
  exclusions:
    - FMM
    - SFS_or_viscosity
    - TE_or_LEV_birth
    - Ptera_coupling
    - aerodynamic_force
    - experimental_accuracy
integrity_status: pass_with_documented_runtime_provenance_warnings
promotion:
  direct_transport_parity: go
  conservative_TE_bridge_R2: eligible_to_start
  Ptera_or_target_paper_scoring: blocked
reason: >-
  All frozen numerical thresholds pass by four to five orders of magnitude,
  clip and nonfinite counts are zero, current-code replay is deterministic at
  the JSON/metrics level, and pinned upstream tests pass. FLOWVPM contains no
  LEV/LESP birth closure, so this result cannot support any stronger aerodynamic
  or cross-paper claim.
```

The null hypothesis for R0–R1 is rejected. The v5f birth instability remains an independent unresolved problem; transport parity must not be used to conceal or regularize it.
