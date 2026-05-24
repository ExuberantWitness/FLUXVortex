# FLUXVortex Aeroelastic Extension — Architecture Analysis

## 1. Current Data Flow (Per Timestep)

```
for step in range(num_steps):
    1. _current_step = step
    2. Load pre-computed airplanes/operating_point for this step
       (all SteadyProblems are pre-built by Movement.generate_airplanes before run())
    3. _collapse_geometry()
       -- Flattens Wing.panels (2D ndarray) into 1D solver arrays
    4. _calculate_wing_wing_influences()
       -- Builds AIC matrix (N_panels x N_panels)
    5. _calculate_freestream_wing_influences()
    6. _calculate_wake_wing_influences()
    7. _calculate_vortex_strengths()
       -- Linear solve: AIC * Gamma = -(wake + freestream influence)
    8. _calculate_loads()
       -- Per-panel forces via Kutta-Joukowski + unsteady Bernoulli
       -- Stores panel.forces_GP1, panel.moments_GP1_CgP1
    9. _populate_next_airplanes_wake()
       -- Shed wake + VPM particles
```

**Key data per timestep:**
- `_current_bound_vortex_strengths`: (N_panels,) circulation per panel
- Per-panel forces/moments: `panel.forces_GP1`, `panel.moments_GP1_CgP1`
- Panel geometry: vertices, collocation points, normals, areas

## 2. Wing Geometry Hierarchy

```
Airplane -> Wing[] -> WingCrossSection[] (root to tip, >= 2)
  - chord, airfoil, Lp_Wcsp_Lpp (position), angles_Wcsp_to_Wcs_ixyz (orientation)
  - num_spanwise_panels (between sections)
  -> panels: 2D ndarray (num_chordwise x num_spanwise) of Panel objects
     Each Panel: 4 vertices, collocation point, ring vortex
```

**Critical**: Panel vertices are set at meshing time and made read-only. PteraSoftware recreates the entire Wing from scratch each timestep via `WingMovement.generate_wings()`.

## 3. Movement System (Pre-computed, Rigid-body Only)

- `Movement.__init__()` calls `generate_airplanes(num_steps, delta_time)` which pre-generates ALL wing configs before the solver runs
- Each `WingMovement.generate_wings()` oscillates `Lp_Wcsp_Lpp` and `angles_Gs_to_Wn_ixyz` per timestep
- No mechanism for solver-computed forces to feed back into geometry

## 4. Extension Point for Structural Coupling

**Injection point**: Between `_calculate_loads()` (step 8) and `_populate_next_airplanes_wake()` (step 9):

```
_calculate_loads()  -->  [STRUCTURAL SOLVER]  -->  _populate_next_airplanes_wake()
                              |
                              v
                  Map panel forces -> beam node loads
                  Solve beam FEM -> displacements/twists
                  Re-mesh Wing with deformed cross-sections
                  Replace steady_problems[step+1] with deformed geometry
```

## 5. What PteraSoftware Already Has / Doesn't Have

**Has:**
- Per-panel force and moment calculation (KJ + unsteady Bernoulli)
- Flexible movement system (sinusoidal, uniform, custom callable)
- Control surface deflections
- Symmetry handling (5 types)

**Does NOT have:**
- Any structural solver, beam model, or FEM capability
- Any aeroelastic coupling
- No "deformed geometry per timestep" within solver loop

## 6. What Needs to Be Added

| Component | Description |
|-----------|-------------|
| **A. Structural Solver** | Euler-Bernoulli or Timoshenko beam FEM; nodes at WingCrossSection positions; DOF: heave + twist per node |
| **B. Force Mapping** | Integrate per-panel forces spanwise → sectional lift/moment at beam nodes |
| **C. Geometry Update** | Beam displacements → WingCrossSection Lp_Wcsp_Lpp + angles → re-mesh |
| **D. Modified Solver Loop** | Override run(); generate geometry incrementally instead of pre-computed |
| **E. Coupling Scheme** | Loose (staggered) first; tight coupling + under-relaxation if needed |

**Main challenge**: The solver pre-generates all geometry. Aeroelastic coupling requires incremental generation (each step depends on previous loads). Must bypass or mutate `steady_problems` during the run.
