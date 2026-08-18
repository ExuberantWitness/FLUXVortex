# v5h R0–R1 schema-v2 claim validation

```yaml
claim_supported: yes
claim: >-
  On the frozen Float64 schema-v2 fixtures, the isolated Python direct backend
  reproduces pinned FLOWVPM Gaussian-erf full/probe/nearfield U/J,
  reformulated f=0 g=0.2 low-storage RK3 under fixed and affine step-time
  freestream, and corrected Pedrizzetti relaxation.
evidence_type: pinned_julia_oracle_plus_deterministic_cross_language_replay
integrity_status: pass_with_documented_runtime_provenance_warnings
review_independence: same_family
promotion:
  direct_transport_parity: go
  conservative_TE_shadow_bridge: eligible_to_start
  exclusive_ownership_or_Ptera_coupling: blocked
  aerodynamic_or_cross_paper_performance: blocked
exclusions:
  - long_time_stability
  - FMM_SFS_viscosity
  - TE_or_LEV_birth
  - Ptera_or_force_coupling
  - experimental_accuracy
reason: >-
  Every preregistered schema, configuration, full/probe/nearfield numerical,
  RK clock, relaxation, finite, and no-clip gate passes. Stored metrics are
  automatically recomputed and deterministic. FLOWVPM has no LEV/LESP source
  closure, so stronger aerodynamic claims remain unsupported.
```
