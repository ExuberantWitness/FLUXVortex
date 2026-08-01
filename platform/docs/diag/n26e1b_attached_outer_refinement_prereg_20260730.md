# N2.6e1b attached-outer numerical refinement gate — preregistration

Date frozen: 2026-07-30  
Claim boundary: `N2.6e1b` attached, unsteady, one-wake outer-flow equation gate  
Decision role: numerical admissibility only; this experiment cannot validate
the IBL, separated double wake, pressure, force, Fig. 12, or RoboEagle response.

## 1. Research question

Does the current attached SVI outer-flow march have a numerically converged
solution for a smooth, target-independent pitching-airfoil problem when panel
count, time resolution, and material-vortex core radius are refined one axis
at a time?

The null hypothesis is that at least one refinement axis fails the frozen
two-percent final-state gate, or an equation residual exceeds its frozen
algebraic gate. The alternative is that all three axes pass both gates.

Passing this experiment permits only the statement that the attached outer
march is numerically admissible for the frozen problem. It does not promote
`N2.6e1`, select a closure, or authorize target-response testing.

## 2. Frozen source-free problem

- Section: closed NACA 0015 actual two-sided surface.
- Chord: \(c=1\,\mathrm{m}\).
- Freestream: \(U_\infty=9\,\mathrm{m\,s^{-1}}\), fixed in the inertial
  \(+x\) direction.
- Motion: pitch about the quarter chord, \(x_p/c=0.25\), with no translation.
- Aerodynamic angle-of-attack schedule:

  \[
  \alpha(t)=\frac{\alpha_f}{2}
  \left[1-\cos\left(\pi t/T\right)\right],
  \quad
  \alpha_f=6^\circ,\quad T=0.4\,\mathrm{s}.
  \]

- The matching angular rate is

  \[
  \dot{\alpha}(t)=\frac{\alpha_f\pi}{2T}
  \sin\left(\pi t/T\right).
  \]

- In the runtime coordinate convention, positive aerodynamic angle is a
  clockwise body rotation, so `RigidKinematics2D.angle_rad=-alpha` and
  `angular_velocity_rad_s=-alpha_dot`. This is the rigid-motion equivalent of
  the already tested stationary-wall freestream vector
  \(U_\infty(\cos\alpha,\sin\alpha)\).
- Initial bound and wake circulation are exactly zero. Wall transpiration is
  exactly zero. No measured force, pressure, separation, or response datum is
  read by the runner.

For `steps=N`, \(\Delta t=T/N\). The owned march is evaluated at
\(t=0,\Delta t,\ldots,T-\Delta t\), advancing the material wake to \(T\).
One final attached solve is then evaluated at \(t=T\) without convecting past
the endpoint. The final wake inventory contains the old material blobs plus
the solved finite near-wake segment at \(T\).

The circulation-change predictor supplied to the runtime carries only the
analytic rising-ramp sign and cannot select the accepted TE branch. Both
orientation branches remain internally solved and the runtime's
sign-consistency rule remains authoritative.

## 3. Frozen refinement matrix

Only one numerical axis changes in each family; the other two remain at the
middle level.

| family | panels per side | ramp steps | blob core radius / c |
|---|---:|---:|---:|
| panel | 16, 32, 64 | 32 | 0.02 |
| time | 32 | 16, 32, 64 | 0.02 |
| core | 32 | 32 | 0.04, 0.02, 0.01 |

The duplicated middle configuration `(32, 32, 0.02)` is executed once and
referenced by all three families. Thus the frozen matrix has seven unique
runs. No level may be inserted, deleted, or replaced after observing the
result.

## 4. Frozen observables

The following final-state quantities are recorded:

1. bound circulation \(\Gamma_B\);
2. total wake circulation \(\Gamma_W\), including the final finite near-wake
   segment;
3. signed circulation centroid
   \(\mathbf{x}_\Gamma=\sum_i\Gamma_i\mathbf{x}_i/\Gamma_W\);
4. signed first circulation moment
   \(\mathbf{M}_\Gamma=\sum_i\Gamma_i\mathbf{x}_i\);
5. lower and upper downstream TE traces used by Eq. 7–8.

The finite near-wake segment contributes at its circulation centroid
(its midpoint). If \(\Gamma_W\) is numerically zero, the centroid is reported
as undefined and the run fails closed; the first moment remains defined.

Across every stage, the runner also records maxima of:

- \(|r_\mathrm{Kelvin}|/(U_\infty c)\);
- \(\max |r_\mathrm{BC}|/U_\infty\);
- \(|r_\mathrm{Eq7}|/c\);
- \(\max |r_\mathrm{linear}|/U_\infty\);
- \(|r_\mathrm{Eq8}|/U_\infty\).

The system condition number and geometry-iteration maximum are diagnostics,
not tunable acceptance variables.

## 5. Frozen go/no-go rule

For each of the panel, time, and core families, only the final two levels are
compared: `32 -> 64`, `32 -> 64`, and `0.02 -> 0.01`, respectively.

For each observable \(q\), define its physical scale \(S_q\):

- \(U_\infty c\) for circulation;
- \(c\) for centroid coordinates;
- \(U_\infty c^2\) for first-moment coordinates;
- \(U_\infty\) for TE traces.

The frozen change score is:

\[
\epsilon_q =
\begin{cases}
|q_f-q_m|/|q_f|, & |q_f|>0.02S_q,\\
|q_f-q_m|/S_q, & |q_f|\le 0.02S_q.
\end{cases}
\]

This fixes “near zero” before execution using the same two-percent gate; no
post-result small-denominator threshold is allowed. Every listed scalar
observable must satisfy \(\epsilon_q\le0.02\) on every refinement family.

Every normalized algebraic residual listed in Section 4 must be
\(\le10^{-9}\) in every unique run. A non-finite value, runtime exception,
undefined wake centroid, missing case, or failed branch selection is an
automatic failure.

Overall verdict:

- **GO** only if all seven unique runs finish, all algebraic gates pass, and
  all three final-two-level refinement comparisons pass.
- **NO-GO** otherwise.

The thresholds, matrix, motion, metric definitions, and verdict logic are
immutable after the run. A NO-GO is evidence against numerical admissibility
of the current `N2.6e1b` implementation; it must not be repaired by changing
this experiment's levels or gates.

## 6. Durable outputs and command

Runner:

```bash
/home/exuber/anaconda3/envs/fluxvortex/bin/python \
  platform/run_n26e1b_attached_outer_refinement.py
```

Durable outputs:

- `platform/docs/diag/n26e1b_attached_outer_refinement_result_20260730.json`
- `platform/docs/diag/n26e1b_attached_outer_refinement_result_20260730.md`

The JSON records the exact case matrix, per-case metrics and failures,
dimensionless residuals, comparison scores, runtime-source SHA-256, and final
GO/NO-GO. The Markdown is a human-readable rendering of the same decision.
