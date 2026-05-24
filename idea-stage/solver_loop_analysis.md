# Solver Loop Analysis: PteraSoftware URVLM for Aeroelastic Extension

## Recommended Approach: Override run() with on-the-fly geometry

Convert `steady_problems` from tuple to mutable list at start of run().
Between `_calculate_loads()` and `_populate_next_airplanes_wake()`:
1. Extract per-panel forces → map to beam node loads
2. Run BeamFE step → get deformations
3. Create new WingCrossSections with deformed positions
4. Create new Wing + Airplane + SteadyProblem
5. Replace steady_problems[step+1]
6. Re-initialize ring vortices for new problem

## Key Constraints
- Panel count must stay constant
- Must copy wake from old wing to new wing
- WingCrossSection positions are immutable (create new objects)
- Ring vortices must be re-initialized for deformed panels
- Wake population uses next step's panel vertices as anchoring points
