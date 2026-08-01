# N2.6e1 Riziotis--Voutsinas source-response contract

Date: 2026-07-30  
Claim: `N2.6e1`  
Status: `FROZEN BEFORE SOURCE-CASE EXECUTION`  
Role: source-method response oracle only; this is not a RoboEagle force fit.

## 1. Two distinct claims

The following claims must not be conflated:

1. **Original-program identity: NO-GO.** Riziotis and Voutsinas (2008) do
   not publish enough state-transfer and run-identity information to
   reproduce their executable bit for bit. Riziotis's 2003 upstream thesis
   now closes the formula-level gaps, but does not remove this distinction.
2. **Published source-response reproduction: GO.** The equations, the
   qualitative separation history, and the vector curves in their Figure 12
   provide an independent response-level oracle for a new implementation.

Passing the second claim never licenses wording such as "the original code
was reproduced". It licenses only the next, separately preregistered target
representation gate.

## 2. Source identity and provenance

Primary method source:

- V. A. Riziotis and S. G. Voutsinas, *Dynamic stall modelling on airfoils
  based on strong viscous--inviscid interaction coupling*, IJNMF 56,
  185--208 (2008), DOI `10.1002/fld.1525`.
- Audited PDF SHA256:
  `cc4970b38b3586affc4805a84e526fcb0049ba2dfa42219c01379e2a8f48fa84`.
- Figure 12 is on PDF pages 17--18, journal pages 201--202.

Primary upstream formula source:

- V. A. Riziotis, *Aerodynamic and aeroelastic analysis of dynamic stall on
  wind turbine rotors*, PhD thesis, NTUA (2003), DOI
  `10.12681/eadd/16690`.
- Official reader:
  `https://freader.ekt.gr/eadd/index.php?doc=16690&lang=el`.
- Audited reader configuration SHA256:
  `27bc7de99609a3540821ce111a25693f10e6ae5847a03e55a0d75b1385a3b4cc`.
- Exact page assets, hashes, equation transcription, and direct/inferred
  evidence boundaries are frozen in
  `research_n26e1_missing_closures_20260730.md`.

Primary experiment source:

- Galbraith, Gracey and Leitch, G.U. Aero Report 9221 (1992).
- Audited PDF SHA256:
  `0eb8842385c9f7e85c10826b87dac726cea4c707a54aac62609ed1c7797de8e9`.

Canonical case:

- NACA0015, Glasgow Model 5, chord `0.55 m`, span `1.61 m`;
- quarter-chord pitching axis;
- paper labels:
  `Re=1.5e6`, `Ma=0.12`, `alpha=11 deg + 8 deg sin`, `k=0.05`;
- Report 9221 complete run `05012721` (Orthodox/untripped):
  `Re=1.47e6`, `k=0.051`;
- 128 sweeps per cycle, ten continuously acquired cycles period averaged;
- 30 pressure taps, 15 on each side, at
  `x/c = 0.00025, 0.0025, 0.01, 0.025, 0.05, 0.10, 0.17, 0.26,
  0.37, 0.50, 0.59, 0.70, 0.83, 0.95, 0.98`.

The report and paper both leave possible three-dimensional experimental
effects. Experimental taps are therefore an observation channel with
uncertainty, not exact two-dimensional truth.

## 3. Equation ownership

The implementation must preserve these source roles.

### 3.1 Equivalent inviscid flow and moving body

- Eq. (1): `u_e = u_0 + u*`.
- Eqs. (2)--(3): actual body sources, attached/separated uniform bound
  vorticity, two newborn wake segments, and material vortex-blob history.
- Eq. (4): `(u_0 - U_B) dot n = 0`.
- Eqs. (5)--(6): wake pressure continuity and Kelvin/bound-circulation
  closure.
- Eq. (7): newborn segment length is emission mean velocity times `dt`.
- Eq. (8): newborn sheet strength is the tangential-velocity jump.

Riziotis Eq. (3) is clockwise-positive because its vortex kernel is
`(r cross k)/(2*pi*r**2)`.  The shadow implementation deliberately uses
physical counter-clockwise-positive circulation.  Therefore Eq. (8) must be
converted, not copied symbol for symbol:

```text
gamma_wake_ccw = w_lower_downstream - w_upper_downstream .
```

For a positive-angle impulsive start, decreasing bound CCW circulation emits
positive-CCW TE circulation along the lower-side outgoing tangent.  The
opposite-angle case must reverse both circulation and side.  This convention
is guarded independently by analytic segment traces, Kelvin conservation, and
time-step refinement.

The double-wake inviscid state therefore has `N+4` degrees of freedom:
`N` body source strengths, attached and separated body vorticities
`gamma_1/gamma_2`, and TE/separation newborn strengths
`gamma_W/gamma_S`. Old blobs retain their integrated circulation.

### 3.2 Unsteady integral boundary layer

For every attached-side station, Eq. (9) is registered as

```text
R_theta =
  Dt(rho_e*w_e_tau*delta_star)/(rho_e*w_e_tau**2)
  + Ds(theta)
  + (2+H)*theta/w_e_tau*Ds(w_e_tau)
  + theta/rho_e*Ds(rho_e)
  - Cf/2
```

and Eq. (10) is the kinetic-energy deficit residual containing the named
unsteady storage terms, `H_star`, `H_starstar`, rotation, normal momentum
transport, dissipation, wall shear, and the edge-velocity gradient. No term
may be silently discarded; an incompressible, non-rotating reduction must
name the terms that become identically zero.

Two easily confused Eq. (10) identities are frozen from the audited
600-dpi source page:

- the second left-side storage term is
  `Dt(rho_e*delta_star)/(rho_e*w_e_tau)`, not a time derivative of density
  thickness `delta_starstar`;
- the right-side local-acceleration term is
  `+2*a*delta_star/w_e**2`, again `delta_star`, not
  `delta_starstar`.

The velocity symbols are also distinct. Every other velocity in Eqs. (9)
and (10) is the edge tangential component `w_e_tau`, whereas the denominator
of that local-acceleration term is printed as the relative-speed magnitude
`w_e`. They coincide only when the converged transpiration/normal component
is zero. The equation oracle must therefore carry
`edge_tangential_velocity` and `edge_speed` separately; replacing both by
one scalar is not source-exact under strong interaction.

The 2008 paper's nomenclature only calls `a` “local flow acceleration.”
The upstream thesis resolves it in the moving `(s,n)` frame as

```text
a = Omega**2*R_OP,s + Dt(Omega)*R_OP,n - Dtt(R_O,s) .
```

It is distinct from `Dt(w_e_tau)`, which already appears separately on the
left side. `Omega` and `a` retain their coordinate signs; neither may be
replaced by an absolute pitch rate or fitted gain.

Density thickness remains present only through
`H_starstar=delta_starstar/theta` in the two explicitly printed
`H_starstar` terms. Implementations and tests must preserve all three
distinctions.

Each attached station owns exactly three solved states:

- displacement thickness `delta_star`;
- momentum thickness `theta`;
- laminar amplification `n`, or turbulent maximum-shear state `C_tau`.

`Cf` is a derived closure quantity and is not a substitute for `C_tau`.
The paper specifies two-point central spatial differences and a simultaneous
Newton solve. The upstream thesis further fixes BDF2 for IBL time derivatives
(`3A^n-4A^(n-1)+A^(n-2)` over `2dt`) and the interval-centred log form of the
two-point spatial residuals. Its documented backward-space stability branch
is source-owned; it is not a response-selectable discretization.

### 3.3 Strong interaction and separation

- Eq. (11):
  `w_T = rho_e**-1 Ds(rho_e*w_e_tau*delta_star)`.
- Eq. (12):
  `volume_deficit_flux = M_B = w_e_tau*delta_star`.
- Code may additionally store
  `mass_deficit_flux = rho_e*w_e_tau*delta_star`, but the two quantities must
  have different names and dimensions.
- Eq. (14): the same discrete derivative maps mass deficit to body
  transpiration.
- Eq. (16): the converged outer boundary condition is
  `(u_e-U_B) dot n = Ds(M)`.

For the double-wake method, boundary-layer integration stops at the
separation point. Downstream body and both wakes have zero transpiration;
the separated-wake induction represents the reverse-flow region. The
separation event is the converged `Cf=0` location. It is updated only after
the boundary-layer state converges, then the whole body is remeshed so
separation remains a grid node.

## 4. Direct closure source and frozen transcription decisions

Riziotis's 2003 thesis is the direct formula authority for the closure used
by this source-response candidate. The implementation must transcribe and
unit-test the equations registered in
`research_n26e1_missing_closures_20260730.md`, including:

- the signed East first approximation
  `Theta_n=(theta+delta_star)*Ds(delta_star)`;
- the complete laminar and turbulent `H*`, `H**`, `Cf` and `CD`
  relations;
- spatial transport of the turbulent maximum-shear state `C_tau`;
- the `e^N` transition equation with source-owned `n_crit=9`;
- transition initialization
  `sqrt(C_tau,tr)=0.7*sqrt(C_tau,eq)`;
- the moving-frame acceleration
  `a=Omega**2*R_OP,s+Dt(Omega)*R_OP,n-Dtt(R_O,s)`;
- BDF2 IBL storage, the interval-centred logarithmic spatial residuals,
  and the source-documented backward-space stability branch;
- nonstationary Bernoulli pressure using the
  `Phi-Phi_inf` gauge, with `-Delta h` inside the separated bubble;
- one and only one surface integration of pressure and wall shear into
  force and moment.

The thesis prints the turbulent branch breakpoint as
`H0=3+4/Re_theta` above `Re_theta=400`. That literal form is discontinuous.
The frozen transcription is instead:

```text
H0 = 4                         for Re_theta <= 400
H0 = 3 + 400/Re_theta          for Re_theta > 400 .
```

This is a source-error correction, not a response candidate: MIT's official
XFOIL 6.99 source uses `400/Re_theta`, the corrected branches meet at
`Re_theta=400`, and Agrawal et al. (2024) independently publish the same
relation. The literal and corrected variants must not both be run and
selected by Figure 12 or RoboEagle error.

Ramos Garcia (2011) is retained only as a related Drela-family cross-check.
It does not override the direct thesis formulas, BDF2 storage, or central/log
spatial discretization, and it does not establish identity with the 2008
executable.

## 5. Published information that is absent

The following are deliberately represented as numerical-convergence choices,
never as original-program parameters:

- body panel count and clustering;
- `dt`, steps per period, number of initialization cycles;
- vortex-blob kernel and core radius;
- wake truncation/history management;
- startup/ramp and initial IBL/wake state;
- Newton tolerance, damping, and iteration limits;
- boundary-layer and old-potential state transfer during whole-body
  remeshing;
- the Figure 12 paper curve's transition setting;
- experimental freestream turbulence intensity;
- exact sample phases for the two panels labelled 19 degrees.

No one of these may be chosen by minimizing Fig17/18/19 error.

The thesis directly defines the two-dimensional source-method separation
event at `Cf=0`. That evidence licenses this event only inside `N2.6e1`.
It does not revive the already falsified claim that instantaneous `Cf=0`
defines the production three-dimensional separation manifold.

The experiment is identified as Orthodox/untripped from complete run
`05012721`. This does not identify the paper calculation's transition
setting or a freestream-turbulence intensity. Source-owned `n_crit=9`
therefore remains fixed and may not be mapped from, or fitted to, either
unknown.

## 6. Figure 12 extraction contract

Figure 12 stores `-Cp`. Persistent data must retain both `minus_cp` and
`cp=-minus_cp`. The vector-model polyline and experimental diamonds are
independent paths and must remain separate series. Experimental markers are
snapped only to the 15 published tap locations.

Required provenance fields:

```text
source_id, source_sha256, doi, pdf_page, article_page, figure, panel,
series, surface, phase_branch, alpha_label_deg, phase_rad, phase_status,
tap_id, x_over_c, minus_cp, cp, vector_path_id, axis_bbox_pt,
digitization_sigma_xc, digitization_sigma_cp, extraction_version,
exclusion_flag, notes
```

The two 19-degree panels retain
`phase_status="reported categorical branch; exact phase undisclosed"` and
null `phase_rad`. A strict sinusoid has only one 19-degree turning point;
the paper does not disclose enough timing information to invent distinct
upstroke/downstroke phases.

Registered digitization uncertainty:

- generally `sigma_x/c=0.002`, `sigma_Cp=0.03`;
- for the steep leading-edge region `x/c<0.01`, `sigma_Cp=0.10`.

The deterministic extraction is now frozen at
`platform/docs/diag/n26e1_fig12/fig12_digitized.csv`:
`859` rows (`619` published double-wake curve points and `240`
experimental taps), with maximum nominal-tap coordinate residual
`1.08361e-4 x/c`.  Re-extraction from the PDF and from reversed-order SVG
pages is byte-identical.  The independent overlay review is recorded in
`n26e1_fig12/VISUAL_REVIEW_20260730.md`.

## 7. Frozen go/no-go gates

### 7.1 Algebraic and conservation gates

- converged normal/transpiration residual: `<1e-8` after physical scaling;
- converged Newton block residual: `<1e-8` after registered scaling;
- algebraic Kelvin and newborn-emission ledgers: `<1e-8`;
- old material-blob circulation changes only by named birth/removal flux;
- body pressure and wall shear are integrated once into panel force and
  reference-point moment.

Collocation quadrature errors such as integrated source-flux or surface-trace
circulation are reported separately as refinement errors. They cannot be
renamed algebraic Kelvin residuals, and they are not required to reach
roundoff on a constant-strength collocation mesh.

### 7.2 Numerical convergence gates

Panel count, `dt`, and vortex-core resolution are each refined independently.
Between the final two levels:

- cycle-mean `L/T/M` changes by at most `2%`;
- `Cp` relative L2 changes by at most `5%`;
- separation location changes by at most `0.02c`;
- event phase changes by at most `2 deg`;
- Figure 12 per-panel nRMSE changes by at most one percentage point.

These refinements are performed without looking at RoboEagle force residuals.

### 7.3 Published-model response gate

For every one of the eight Figure 12 panels and each surface separately,

```text
nRMSE = RMSE(Cp_candidate - Cp_published_DW)
        / (P95(Cp_published_DW)-P5(Cp_published_DW)) <= 0.05
```

All panels must pass. An aggregate average cannot hide a failed phase.

### 7.4 Separation-history topology gate

- upstroke 12, 15 and 18 degrees: practically attached;
- experiment at upstroke 19 degrees: rapid separation to the tap-supported
  interval `x_s/c in [0.10,0.17]`;
- the experiment retains that forward separation through downstroke
  18 degrees;
- the published double-wake model is delayed at upstroke 19 degrees and
  reaches its most forward separation near downstroke 18 degrees;
- from downstroke 18 to 15 to 12 degrees, separation moves monotonically
  toward the trailing edge;
- equal-angle upstroke/downstroke pressure fields must not collapse to the
  same curve.

For source reproduction, the published model's documented delay is the
reference response, not a failure. Experimental improvement is a later claim
and must be evaluated separately.

## 8. Promotion boundary

Passing this contract promotes only `N2.6e1` from `open` to a
source-response-validated state. It does not:

- alter frozen V4.1;
- authorize target-force fitting;
- identify the unknown RoboEagle dynamic tare;
- validate the NACA2406 target-section surrogate;
- validate stripwise mapping under spanwise flow;
- validate the final three-dimensional curved-surface IBL.

`N2.6e2` requires a separate actual-surface target representation and
co-design load-transfer contract before any Fig17/18/19 run.
