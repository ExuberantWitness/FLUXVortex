# Proposal: Aeroelastic Simulation via UVLM + Euler-Bernoulli Beam — Scope Fit for Newton?

## TL;DR

I've built [FLUXVortex](https://github.com/ExuberantWitness/FLUXVortex), a GPU-accelerated unsteady vortex lattice method (UVLM) coupled with an Euler-Bernoulli beam FE solver for aeroelastic simulation. It's built on PteraSoftware's UVLM and validated on the Goland Wing flutter benchmark (predicted 140.2 m/s vs. reference 137 m/s, 2.4% error). Before investing effort into a Newton integration, I'd like to understand whether aeroelastic simulation is within Newton's roadmap — and if so, what the right integration path looks like.

---

## What FLUXVortex Does

FLUXVortex is a partitioned aeroelastic solver that couples:

1. **Unsteady Vortex Lattice Method (UVLM)** — panel-method aerodynamics for lifting surfaces
2. **Euler-Bernoulli Beam FE** — coupled bending-torsion structural model (3 DOF/node: heave, slope, twist)

At each timestep:
- Aerodynamic panel forces → mapped to beam nodal loads
- Beam FE solves for structural deformation (Newmark-β time integration)
- Panel vertices are mutated to reflect deformation for the next timestep

### Validated Results

| Benchmark | Predicted | Reference | Error |
|-----------|-----------|-----------|-------|
| Goland Wing Flutter Speed | 140.2 m/s | ~137 m/s | 2.4% |

### Planned Warp GPU Acceleration

The computational bottleneck is Biot-Savart velocity induction (4 Numba `@njit` functions called ~32× per timestep, >90% of runtime). I have a concrete plan to convert these to NVIDIA Warp `@wp.kernel` GPU kernels, projected ~30× speedup for UVLM and ~200× for vortex particle methods.

---

## The Scope Question (I Know This Is Awkward)

Let me be upfront: **Newton's primary target is robotics rigid-body simulation**. FLUXVortex is an **aerodynamics + flexible-structure solver** for lifting surfaces. These are different domains.

However, I see some overlap points:

| Newton Feature | FLUXVortex Relevance |
|---|---|
| Built on NVIDIA Warp | FLUXVortex plans Warp GPU kernels (same compute backend) |
| `SolverBase` plugin architecture | FLUXVortex's time-stepping loop could map to `SolverBase.step()` |
| Deformable bodies (cloth, soft body, cables) | FLUXVortex handles flexible wing structures |
| Differentiable simulation | UVLM + beam coupling is differentiable in principle |
| MPM as a multi-physics solver | UVLM+beam is another multi-physics coupling pattern |
| OpenUSD scene representation | Wing geometries could be expressed as USD assets |

But the mismatch is also real:
- Newton operates on `Model` (rigid bodies + joints + particles) → FLUXVortex operates on panel meshes + beam elements
- Newton's `State` stores body transforms and joint coordinates → FLUXVortex tracks panel vertex positions and beam DOF displacements
- Newton's contact pipeline → no analogue in UVLM aerodynamics
- Newton targets robot learning (RL policy training) → FLUXVortex targets aerospace engineering analysis

---

## Specific Questions

**1. Is aeroelastic or aerodynamic simulation within Newton's roadmap?**

Newton already has cloth, soft body, cable, and MPM solvers — these go beyond rigid-body robotics. Is there interest in expanding to aerodynamic loads on deformable structures (e.g., drone propeller aeroelasticity, MAV wing flutter, wind loading on robotic systems)?

**2. Could FLUXVortex fit as a "domain solver" alongside MPM, VBD, etc.?**

Newton has 9 solver backends (`SolverMuJoCo`, `SolverXPBD`, `SolverVBD`, `SolverImplicitMPM`, `SolverKamino`, `SolverStyle3D`, `SolverFeatherstone`, `SolverSemiImplicit`), all inheriting from `SolverBase`. A hypothetical `SolverAeroelastic` would:
- Override `step(state_in, state_out, control, contacts, dt)`
- Accept wing geometry as shape data in the `Model`
- Write aerodynamic forces back to body force buffers

Is this kind of domain-specific solver extension what the `SolverBase` plugin pattern is designed for? Or is it meant only for rigid-body variants?

**3. If not Newton, is the Warp community a better fit?**

FLUXVortex's Warp GPU kernels (Biot-Savart, vortex particle advection, beam FE) are general-purpose scientific computing kernels that don't depend on Newton's `Model`/`State`/`Control` abstraction. Would contributing these as standalone Warp examples or utilities be more appropriate?

---

## What I've Done So Far

- Read through Newton's `SolverBase` API (`step()`, `notify_model_changed()`, `register_custom_attributes()`)
- Reviewed the solver plugin architecture (9 existing backends)
- Confirmed Newton is built on Warp — same GPU compute target as FLUXVortex's planned acceleration
- Validated FLUXVortex on the Goland Wing benchmark
- Drafted a Warp GPU kernel conversion plan for the Biot-Savart bottleneck

## What I Haven't Done

- I haven't started any Newton integration code — this Discussion is a scope check first
- I haven't mapped FLUXVortex's panel mesh data structures to Newton's `Model`/`State` pattern
- I don't know if the Newton team views aeroelasticity as in-scope or out-of-scope

---

I'm happy to adapt based on your feedback. If the answer is "interesting but not Newton's focus," that's completely fair — I'll look at contributing to the Warp ecosystem directly. If there's a viable integration path, I'd love guidance on the right architecture.

Thanks for your time.
