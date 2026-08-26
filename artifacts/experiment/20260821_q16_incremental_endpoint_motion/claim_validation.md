# Claim Validation

| Claim | Evidence | Observed | Verdict |
|---|---|---|---|
| `dq_trial` is consumed by the real aerodynamic equations | same `q`, different `dq`; direct LE relative-velocity oracle | load/wake/receipt differ and LE velocity matches | supported |
| Position is not changed by the velocity intervention | current-geometry SHA | exact equal | supported |
| Predictor branch leaves committed parent unchanged | parent pickle SHA and step | exact | supported |
| Mandatory separated modes remain active | live solver particles/TEV/wake | 12 LEV particles in both branches, TEV and free wake active | supported |
| Endpoint motion equals a higher-order interval integrator | no internal time quadrature was added | not tested/implemented | unsupported |
| Complete Q16 generalized force can be committed | unresolved active-LEV impulse remains | nonzero global-only impulse | blocked |
