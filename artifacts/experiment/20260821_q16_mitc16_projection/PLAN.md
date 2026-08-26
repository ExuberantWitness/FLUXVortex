# Q16 MITC16/ANS projection plan

## Claim boundary

This slice implements and verifies only the assumed-covariant-strain
projection for the fixed 16-node Q16 ANCF macro element.  It does **not** yet
claim a locking-free structural operator: transverse-normal ANS/EAS,
element-local EAS condensation, internal force, tangent action, CUDA parity,
and nonlinear solve integration remain later gates.

## Primary formulation sources

- M. L. Bucalem and K. J. Bathe, *Higher-order MITC general shell
  elements*, IJNME 36 (1993), 3729--3754.  Use the published MITC16 tying
  layout: 3x4 for `E_xixi`/`2E_xizeta`, symmetric 4x3 for
  `E_etaeta`/`2E_etazeta`, and 3x3 for `2E_xieta`; use full integration.
- H. Yamashita et al., *Continuum Mechanics Based Bilinear Shear
  Deformable Shell Element Using Absolute Nodal Coordinate Formulation*,
  J. Comput. Nonlinear Dyn. 10 (2015), 051012.  Use this only to freeze the
  later separation of responsibilities: ANS for transverse shear, and a
  combined ANS+EAS treatment for transverse-normal/thickness locking.
- J. C. Simo and M. S. Rifai, *A class of mixed assumed strain methods and
  the method of incompatible modes*, IJNME 29 (1990), 1595--1638.  The later
  EAS field must obey its orthogonality/patch-test conditions and be statically
  condensed element-wise.

## Fixed Q16 tying spaces

Let `a=sqrt(3/5)` and let `(g1,g2,g3,g4)` be the four-point Gauss--Legendre
abscissae on `[-1,1]`.

- `E_xixi` and `2E_xizeta`: `xi=(-a,0,a)`, `eta=(g1..g4)`; interpolate in
  `P2(xi) tensor P3(eta)`.
- `E_etaeta` and `2E_etazeta`: `xi=(g1..g4)`, `eta=(-a,0,a)`; interpolate in
  `P3(xi) tensor P2(eta)`.
- `2E_xieta`: `xi,eta=(-a,0,a)`; interpolate in `P2 tensor P2`.
- Transverse shear is sampled on the middle surface and is independent of
  the thickness coordinate, following the MITC16 shell assumption.
- `E_zetazeta` remains the compatible strain in this slice.  It must not be
  relabelled as ANS/EAS until the separate thickness gate is implemented.

## Acceptance gates

1. Exact tying-point reproduction for all five projected components.
2. Exact component ordering `[E11,E22,E33,2E12,2E23,2E13]`.
3. Rigid-body translation/rotation gives zero projected strain to roundoff.
4. A manufactured cylindrical Kirchhoff bending field reduces the parasitic
   transverse-shear norm by at least 50x relative to the unprojected Q16
   field; this is a diagnostic, not a convergence claim.
5. Exact float64/C-contiguous/finite state and coordinate guards fail closed.
6. No Q9 path, reduced-integration switch, runtime polynomial order, or CPU
   fallback is introduced.

