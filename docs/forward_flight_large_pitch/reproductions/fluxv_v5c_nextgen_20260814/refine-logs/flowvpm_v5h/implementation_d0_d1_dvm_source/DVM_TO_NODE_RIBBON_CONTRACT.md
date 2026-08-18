# DVM source to node-owned three-dimensional ribbon contract

Status: design-frozen, diagnostic shadow only.  Physical feedback and target
paper scoring remain prohibited until D0 source parity and all D1 ledgers pass.

## Ownership

- FluxV/Ptera owns the bound AIC and the final surface load.
- Ramesh LDVM owns only the sectional shedding decision, newborn LEV
  circulation, and the coupled newest-TEV source ledger.
- The topology manager owns stable node/edge identity but generates no
  vorticity or load.
- The FLOWVPM-derived particle backend owns free-vorticity position, vector
  strength, core, and transport only after an accepted source handoff.

No DVM `CL`, `CD`, `CN`, `CS`, impulse, or polar load may be added to the
Ptera result.

## Required fact sources

For every Ptera span cell, the DVM record supplies `A0_pre`, `A0_post`, signed
LESP target/residual, newborn LEV/TEV circulation, and a complete Kelvin
ledger.  Ptera supplies the current GP1 leading- and trailing-edge span nodes
and local chord/span/normal frame.  A node-birth record supplies the actual
node-local birth point or the unique previous-frontier identity.

A strip-centre DVM birth point is insufficient.  It may not be linearly
interpolated or averaged to construct shared span-node endpoints, because that
recreates the ragged per-strip seam found in v5f.

## Units and sign

The clean-room DVM backend uses `U=c=1`, so its circulation is

`Gamma_star = Gamma / (U_reference * chord)`.

The handoff converts it exactly once:

`Gamma_cell[m^2/s] = Gamma_star * U_reference[m/s] * chord[m]`.

Strip width is not part of this conversion.  The global edge bridge then
deposits vector particle strength

`Gamma_particle[m^3/s] = gamma_edge[m^2/s] * delta_l[m]`.

The source record must state the DVM positive out-of-plane axis and the GP1
axis mapping.  No implementation may infer circulation sign from coordinates.

## Node-owned birth topology

For cell `j=[k,k+1]`, an active source ring traverses

`birth_left -> birth_right -> anchor_right -> anchor_left -> birth_left`.

Stable IDs use `(wing, family, lineage_epoch, birth_step, span_node, role)`;
coordinate rounding is never an identity key.  Adjacent cells reference the
same endpoint ID.  Signed edge circulation is the full incidence sum

`gamma_edge = B.T @ Gamma_cell`,

with no half factor.  Equal adjacent cell strengths cancel their common side
edge exactly; unequal strengths retain the signed difference.  An
active/inactive boundary retains the active cell's full physical side edge.

Node activity is the logical union of its adjacent cell activity.  Each node
is classified independently:

- `first`: no previous LE frontier exists;
- `continuous`: the node was active last step and has one previous frontier;
- `restart`: it is active after an inactive step, with a new lineage epoch;
- `inactive`: no new source node or ring.

First/restart uses `x_birth=x_anchor+0.5*q_edge*dt`.  Continuous shedding uses
`x_birth=x_anchor+(x_previous_frontier-x_anchor)/3`.  A cell may therefore have
different left/right birth modes when an active region grows or shrinks.
Conflicting previous-frontier identities fail closed.

## Time layer and exclusive modes

The source uses Ptera's current `t_n` geometry, solved bound state, and old
free-vorticity field; the birth record is pre-convection.  It is consumed once
and the particle transport advances it to `t_(n+1)`.  Preallocated next-layer
flat arrays are not a fact source.

Two mutually exclusive modes are allowed:

1. `diagnostic_shadow`: Ptera bound/TE wake remains physical; DVM and
   particles have zero feedback and zero load writes.
2. future physical mode: either Ptera uniquely carries the TE source while
   particles carry LEV, or particles carry both LEV and TEV and the duplicate
   Ptera wake contribution is removed.  Both cannot be active.

Likewise, the DVM persistent 2-D wake and the three-dimensional particle wake
cannot both provide the same induced-velocity history.

## Mechanical promotion gates

1. schema, units, signs, IDs, finite values, and disabled input blindness;
2. shared-node determinism under shuffled input, with no coordinate key;
3. exact first/continuous/restart rules including mixed endpoint modes;
4. grow/shrink/split/merge active masks with zero numerical seam;
5. edge incidence residual `<=1e-12`, vector conservation `<=1e-14`
   absolute and `<=1e-12` relative;
6. independently recomputable per-cell Kelvin residual `<=1e-10 m^2/s`;
7. current/next time-layer and one-time source-consumption guards;
8. bitwise unchanged Ptera state/load in shadow mode;
9. non-target straight/taper/twist ribbon refinement with no clipping;
10. exactly one physical owner for every free, bound, and TE edge before any
    feedback is enabled.

