# Ramesh LDVM v2.5 source audit

## Scope and provenance

This is a clean-room, read-only audit of the author-distributed **LDVM v2.5**
Fortran package.  No Fortran source, executable, or large reference output is
copied into this repository.

Audited artifacts:

| Artifact | Local acquisition path | SHA-256 |
|---|---|---|
| LDVM v2.5 source archive | `docs/forward_flight_large_pitch/literature/ramesh_ldvm_v2_5_source.zip` in the acquisition workspace | `fd574792bfcceed61d1d7d890edf95081ff13d6745e4f89c1cbc0261c5643d03` |
| `ldvm.f95` inside that archive | archive member `LDVM_v2.5/ldvm.f95` | `eada8df2df605cc8bd929bbf9edf3672a85004d81c621a5637663d2a1b286c09` |
| Author force output | archive member `LDVM_v2.5/force_pr_amp45_k0.2_le.dat` | `8ce564d9358dcaf1c43fc433c969d2d3375eec9022eb4b7bf98ba759f0c620bb` |
| Ramesh dissertation | `candidates_20260812/ramesh_2013_phd_ldvm_foundation.pdf` | `735f0cb9af7636bf3fa3e21845f4a722b40a55004c409891af0687fa87f740c4` |

The source header identifies Ramesh and Gopalarathnam, version 2.5, and the
Ramesh et al. JFM 751 (2014) model.  `ldvm.f95:45-57` licenses the program under
GPLv3-or-later.  Therefore the proposed FluxV implementation must be an
equation-level clean-room implementation with behavioral tests; copying source
fragments into a differently licensed package requires a separate licensing
decision.

## Executable scope

The source says explicitly at `ldvm.f95:14-15` that it solves **two-dimensional
unsteady airfoil flows** using unsteady thin-airfoil theory plus an intermittent
LEV model.  It has no span coordinate, finite-wing influence matrix, tip wake,
or spanwise load state.  Airfoil coordinates are reduced to one mean camber
line (`ldvm.f95:676-717`); thickness affects the model only indirectly through
the empirical critical LESP supplied by the user.

The ten input fields are read at `ldvm.f95:129-139`:

1. chord;
2. reference speed;
3. pitch pivot as a chord fraction;
4. moment reference as a chord fraction;
5. airfoil coordinate file or `flat_plate`;
6. reference Reynolds number;
7. critical LESP;
8. motion file containing nondimensional time, pitch in degrees, `h/c`, and
   nondimensional airfoil speed;
9. force-output filename;
10. optional flow-output filename and sampling interval.

The force output (`ldvm.f95:588-592`) contains
`t, alpha, h/c, U, Gamma_bound, A0, Cn, Cs, Cl, Cd, Cm`.  The optional flow
output (`ldvm.f95:594-607`) contains LEV, TEV, and bound-segment circulation and
positions, separated by `NaN` rows.

## State and time layer

The persistent aerodynamic state is:

- Fourier coefficients `A0...An` and the previous `A0...A3`;
- bound-sheet segment circulations;
- all material TEV and LEV strengths and positions;
- previous induced velocities, although the active convection is first-order
  Euler;
- `levflag`, which distinguishes the first LEV after an attached interval from
  continuous shedding;
- `kelv_enf`, intended to retain circulation deleted from the finite wake.

For each motion sample after the initial state, the active order is:

1. move the airfoil in an inertial frame (`ldvm.f95:210-217`);
2. place one new TEV (`219-229`);
3. solve its strength from Kelvin circulation (`263-286`);
4. compute pre-LEV Fourier rates (`287-304`);
5. when the critical LESP is exceeded, add a new LEV and jointly solve LEV and
   TEV strengths (`310-413`);
6. rebuild the bound sheet (`419-445`);
7. calculate induced velocities and convect all free vortices
   (`448-522`);
8. delete distant vortices (`537-550`);
9. calculate and write loads (`555-607`).

The Adams-Bashforth formulas at `ldvm.f95:500-513` are commented out.  The
active update is explicit Euler.  The source moves the airfoil upstream through
an inertial fluid; free vortices therefore receive induced velocity only.  A
stationary-wing/freestream-frame implementation must also add the freestream to
material-wake convection.

## Fourier and LESP equations

On the cosine grid

\[
x=\frac{c}{2}(1-\cos\theta),\qquad 0\leq\theta\leq\pi,
\]

the source forms the local normal-flow residual $W(\theta)$, including body
motion, camber slope, and old/new wake induction.  The executable formulas at
`ldvm.f95:719-753` are

\[
A_0=-\frac{1}{\pi U_{ref}}\int_0^\pi W\,d\theta,
\qquad
A_n=\frac{2}{\pi U_{ref}}\int_0^\pi W\cos(n\theta)\,d\theta,
\]

and

\[
\Gamma_b=\pi U_{ref}c\left(A_0+\frac{A_1}{2}\right).
\]

The model identifies the leading-edge suction parameter with $A_0$.  If

\[
|A_{0,pre}|>L_{crit},
\]

it chooses the signed target

\[
A_{0,target}=\operatorname{sign}(A_{0,pre})L_{crit}.
\]

The two current-step unknowns are the newest TEV and LEV circulations.  The two
constraints solved at `ldvm.f95:353-409` are

\[
\Gamma_{deleted}+\Gamma_b+
\sum\Gamma_{TEV}+\sum\Gamma_{LEV}=0,
\]

and

\[
A_0-A_{0,target}=0.
\]

The source evaluates a finite-difference two-variable Newton update even though
these two constraints are linear in the new strengths.  A clean-room
implementation may solve the corresponding $2\times2$ system directly.

## Vortex birth, kernel, and wake

The source birth rules are:

- first TEV: trailing edge plus half a freestream step
  (`ldvm.f95:223-225`);
- later TEV: one third of the vector from the trailing edge to the previously
  born TEV (`227-228`);
- first LEV after an attached interval: leading edge plus half the local
  leading-edge velocity step (`322-324`);
- a continuously shed LEV: one third of the vector from the leading edge to the
  previously born LEV (`325-327`).

The regularized point-vortex kernel is the $n=2$ Vatistas form.  Its scalar
denominator is

\[
2\pi\sqrt{r^4+r_c^4},
\]

as implemented at `ldvm.f95:454-496` and `723-735`.  The supplied source fixes
`r_c/c=0.02`, uses 70 chord points and 45 Fourier terms, and deletes vortices at
10 chords (`ldvm.f95:100-106`).  The dissertation's convergence discussion
relates $r_c/c\simeq0.02$ to approximately $1.3\Delta t^*$ for
$\Delta t^*\simeq0.015$.  The value `0.02c` is not a universal physical
constant.

## Load equations and ownership

At `ldvm.f95:555-590` the executable load path is

\[
C_{N,c}=2\pi\frac{U\cos\alpha+\dot h\sin\alpha}{U_{ref}}
\left(A_0+\frac{A_1}{2}\right),
\]

\[
C_{N,nc}=\frac{2\pi c}{U_{ref}}
\left(\frac{3\dot A_0}{4}+\frac{\dot A_1}{4}
+\frac{\dot A_2}{8}\right),
\]

\[
C_S=2\pi A_0^2,
\]

\[
C_{N,ind}=\frac{2}{U_{ref}^2c}
\sum_i (u_{ind,i}\cos\alpha-w_{ind,i}\sin\alpha)\Delta\Gamma_{b,i},
\]

\[
C_N=C_{N,c}+C_{N,nc}+C_{N,ind},
\quad C_L=C_N\cos\alpha+C_S\sin\alpha,
\quad C_D=C_N\sin\alpha-C_S\cos\alpha.
\]

The active source moment expression uses coefficient $3/16$ for
$\dot A_1$, while dissertation Eq. 4.30 gives $11/64$.  Source-parity and
paper-equation modes must therefore be distinguished explicitly.

For a retained-UVLM FluxV implementation, UVLM pressure/Kutta-Joukowski plus
its unsteady term must remain the primary load owner.  Adding Ramesh's complete
`Cnc+Cnnc` path would count bound circulation and added mass twice.  Likewise:

- material LDVM LEV and Yang attached PLEV cannot both own leading-edge
  vorticity;
- a full-angle separated polar residual cannot remain active on top of the
  material LEV response;
- $C_S$ can be evaluated only as an explicitly exclusive alternative load
  owner during an ablation;
- source-defined profile drag, such as Figure 14's `Cd0=0.057`, must have a
  separate ledger and provenance.

## Reference-case reproduction

The distributed binary is linked to the unavailable `libgfortran.so.3`.
Recompiling `ldvm.f95` with system `gfortran-11 -O2` succeeded.  The distributed
case has:

- `c=Uref=1`;
- leading-edge pitch pivot and moment reference;
- SD7003, `Re=30000`, `Lcrit=0.18`;
- 45-degree, `K=0.2`, Wang-Eldredge ramp-hold-return motion;
- 500 motion samples and 499 force-output rows;
- `dt*=0.014123741...`.

The locally compiled output and a bounds-checked build with only the invalid
initialization bound repaired produced identical force files.  Compared with
the author-supplied file, time, motion, speed, circulation, and $A_0$ agree to
about $10^{-9}$; normal force and lift agree to about $10^{-7}$, consistent
with the supplied file's print precision.

Golden numeric landmarks from the author run are:

| Landmark | Author value |
|---|---:|
| First capped output row | 116 |
| First capped `t*` | 1.6383540 |
| First capped pitch | 14.630000 deg |
| First capped `A0` | 0.18000000 |
| First capped `CL` | 3.0860116 |
| First capped `CD` | 0.59517464 |
| First LEV circulation | 0.0052488387605094 |
| First LEV position `(X,Z)` | `(-2.6329878889248, 0.00904780375386)` |
| Same-step newest TEV circulation | -0.01357988564529845 |
| Same-block total-circulation residual | -2.40e-8 |
| Number of capped rows / final LEVs | 174 / 174 |
| Final TEVs | 499 |
| `CL` range | `[-0.49225872, 4.2483764]` |
| `CD` range | `[-0.042505791, 3.207459]` |
| Final `(t*, A0, CL, CD)` | `(7.047747, -0.0053614446, 0.46033769, -0.00018061072)` |

This is a source regression case, not an independent experiment and not a
finite-wing validation.

## Source defects that must not be copied

1. `aterm_prev` is declared with bounds `0:3` at `ldvm.f95:77` but initialized
   through `0:n_aterm` at line 193, where `n_aterm=45`.  A bounds-checked build
   fails immediately.  Limiting that initialization to `0:3` repairs the
   execution without changing optimized-case output.
2. At `ldvm.f95:538-550`, each vortex array is shifted before `tev(1)` or
   `lev(1)` is added to `kelv_enf`; the ledger therefore receives the old
   second vortex rather than the deleted first vortex.  The supplied short case
   does not reach the ten-chord deletion boundary.
3. `re_le` is calculated at `ldvm.f95:300-303` but is not used to update
   `Lcrit`; v2.5 therefore executes a fixed input threshold despite comments
   about velocity-dependent calibration.
4. The source zeros its first TEV after solving at `ldvm.f95:415-418`.  This is
   an undocumented start-up special case and must be retained for exact source
   parity or removed only in a separately named scientific ablation.

There is also conflicting flat-plate calibration evidence: the v2.5 README and
dissertation Table 4.1 give `Lcrit=0.19` at `Re=1000`, while dissertation
section 4.3.5 states `0.11`.  No implementation should hide that conflict.

## Existing Python status

`platform/ldvm_fourier.py` is a clean-room two-dimensional primitive and not a
licensed copy of the Fortran source.  The new reference test configures 70
chord points, 45 Fourier terms, the exact source motion and an SD7003 mean
camber reconstructed from the repository's coordinate file.

Measured current behavior is:

- all 499 steps and all returned states are finite;
- the first LEV occurs on the correct output row, time, and pitch;
- post-constraint $A_0$ is $0.18$ to machine precision;
- first-event `CL` and `CD` differ from the author values by roughly
  $1.2\times10^{-3}$ and $3.1\times10^{-4}$;
- the Kelvin ledger and force-definition identities close at machine precision;
- the Python history presently ends with 172 LEVs rather than 174.

Consequently the test is intentionally a **partial-parity reference guard**.
It must not be cited as proof that the full author history or validation curves
have been reproduced.  Likely contributors are the missing first-TEV zeroing,
the analytic rather than author-rounded input history, and remaining time-layer
or sign-convention differences.  Exact parity requires a dedicated clean-room
2D primitive before any 3D FluxV coupling claim.

Run the guard from the repository root with:

```bash
PYTHONPATH=platform python -m pytest \
  platform/tests/test_ramesh_ldvm_reference.py -q
```

## Minimal retained-UVLM integration contract

1. Add a diagnostic strip LESP extractor.  For strip $j$, compute $W_j$ on
   cosine chord nodes from freestream, body motion, old TE/LE wakes and
   nonlocal induction, excluding the candidate current newborn response.
   Return `A0_pre`, `A0_post`, local normalization speed, sign, and condition
   number.
2. Represent the LE shear layer as a spanwise material wake row/ring, not as an
   instantaneous attached PLEV.
3. For $N_s$ interacting strips, solve one global $2N_s$ system for current
   LE circulation and the existing newest-TE temporal shed state.  Enforce one
   Kelvin contour and one signed LESP constraint per strip.  Do not create a
   second TE wake.
4. Respect the current Ptera time layer: the current solve uses old material
   wake; after current loads, solve the temporal shed edge
   $s_n=g_{new}-g_{old}$, recover $g_{new}=s_n+g_{old}$, and commit it to the
   next state.
5. Keep one load owner and expose a ledger containing LESP values, LE/TE
   newborn circulation, Kelvin residual, LESP residual, and each exclusive
   force component.
6. Require explicit `LESPParameters(section_family, Re, Lcrit, provenance)`.
   Unknown calibration must fail closed rather than silently reuse `0.18`.

Mechanism forms that can be frozen across Yang/Figure-14/Stevens are the
Fourier $A_0$ definition, Kelvin-plus-LESP constraints, signed side selection,
one TE wake owner, intermittent shedding topology, and the kernel family.
Numerical `Lcrit`, leading-edge radius/thickness, profile drag, absolute core
size, deletion distance, and reattachment/viscous parameters cannot be shared
without independent evidence.

Recommended ablation order:

1. exact author 2D primitive parity;
2. diagnostic-only FluxV strip $A_0$, with exact module-off reduction;
3. onset gate only;
4. material LEV plus coupled existing TE state, UVLM as sole load owner;
5. axial-suction alternative owner;
6. Martínez variable-rate SVDVM as an alternative, not an additive mechanism;
7. span coupling, core and time-integration convergence.
