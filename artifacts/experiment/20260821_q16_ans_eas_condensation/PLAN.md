# Q16 transverse-normal ANS+EAS condensation plan

## Scope

Extend the verified MITC16 covariant projection with a distinct
transverse-normal treatment:

1. sample compatible `E_zetazeta` at the sixteen Q16 surface nodes and
   interpolate it with the fixed cubic Q16 basis (ANS);
2. add one element-local thickness-linear enhanced covariant mode
   `E_zetazeta^EAS = zeta*alpha`;
3. solve the Hu--Washizu stationarity equation for `alpha` and statically
   condense it from the nodal system;
4. derive energy, internal force and analytic condensed tangent action from
   exactly the same projected strain field.

This is an independent NumPy formulation oracle. CUDA production parity is a
later gate. No Q9, selective reduced integration or case-dependent switch is
allowed.

## Acceptance gates

- exact rigid-motion null energy/force and zero enhanced parameter;
- element-local EAS stationarity residual at roundoff;
- condensed energy no larger than the uncondensed `alpha=0` energy;
- internal force equals the condensed-energy directional derivative;
- analytic condensed Jv equals centered force difference;
- thin cylindrical bending selects a nonzero EAS mode and remains finite over
  `h/L = 1e-3,1e-2,5e-2,1e-1`;
- exact float64/finite/domain guards fail closed.

