# FluxV v5b shared-wake smoke gate contract

## Evidence boundary

The first v5b implementation is a bounded, no-force shared-wake diagnostic.
It may validate the LESP event, LEV/TEV birth strengths, Kelvin identities,
material convection, and the disabled-module reduction.  It does not compute
pressure or aerodynamic force and therefore cannot produce Yang, Figure-14,
or Baik predictions.

Until a single conservative force coupling is implemented, the cross-paper
status must be exactly `blocked_not_scored`.  A no-force shadow history must
never be compared with experimental lift, drag, or thrust, and must never be
labelled a v5b accuracy result.

## Gate ladder

### G0: frozen Ramesh reference

Run the existing clean-room Ramesh LDVM reference suite without changing its
landmarks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
NUMBA_CACHE_DIR=/tmp/numba-v5b \
MPLCONFIGDIR=/tmp/mpl-v5b \
/home/exuber/anaconda3/envs/fluxvortex/bin/python \
  -m pytest tests/test_ramesh_ldvm_reference.py -q
```

Pass requires all three tests to pass.  This is a partial-parity 2-D primitive
guard; it is not evidence for finite-wing force coupling.

### G1: synthetic disabled/no-LEV reduction

1. Call `dispatch_v5b_or_parent(enable_shared_wake=False, ...)` with a sentinel
   parent callback.  The returned object must be the identical sentinel object
   and the parent callback must execute exactly once.  No shared-wake state may
   be constructed.
2. Run a stationary rectangular `nc=2, ns=2` half-wing for three steps with a
   deliberately super-high `LESPcrit=10`.  No LEV may be active or born, all
   LEV circulation must remain zero, and Eq.9/Kelvin/convection ledgers must
   close to their declared tolerances.

The second check establishes only a no-LEV wake-state limit.  Because the
module has no force output, it does not establish force equality to FluxV.

### G2: Hirato live shadow and birth limit

Run a two-step stationary 15-degree pitched rectangular half-wing with
`LESPcrit=0.05`.  Required checks:

- at least one strip activates and creates a new LEV sheet;
- the second step retains material history rather than rebirthing every ring;
- Eq.9 residual is at most `1e-14`;
- active LESP residual is at most `1e-12`;
- Kelvin residual is at most `1e-12`;
- material-vortex circulation changes are at most `1e-14`, with no missing
  material IDs;
- every reported birth strength is finite;
- the dedicated `dt -> 0` birth-limit diagnostic passes its frozen scaling
  and finiteness criteria.

The sequence is a live `HiratoLiveShadow` topology/conservation check.  Its
`force_coupling` field must remain `not_implemented` in this stage.

## Promotion and stop rules

| State | Cross-paper status | Allowed action |
|---|---|---|
| Any G0--G2 failure | `blocked_not_scored` | Fix only the failed identity; do not run paper cases. |
| G0--G2 pass, force coupling absent | `blocked_not_scored` | Freeze smoke diagnostics and report no accuracy result. |
| Force coupling later implemented but unverified | `blocked_not_scored` | Add force ownership/pressure tests before paper cases. |
| Force ownership, exact no-LEV force reduction, and small canonical case pass | eligible for a separate preregistered paper smoke | Do not reuse the no-force summary as an accuracy result. |

## Required outputs

The runner writes:

- `gate_results.csv`: one row per atomic gate, with measured value, threshold,
  pass/fail, and evidence role;
- `shadow_steps.csv`: G1/G2 step diagnostics only, explicitly marked
  `no_force_diagnostic_only`;
- `summary.json`: source/result hashes, exact command/environment, gate ladder,
  `force_coupling`, and `crosspaper_performance_status`;
- `g0_ramesh_pytest.log`: captured frozen reference-suite output.

If force coupling is absent, `summary.json` must contain no Yang, Figure-14, or
Baik accuracy metric fields.
