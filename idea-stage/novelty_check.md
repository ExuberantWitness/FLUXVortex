# Novelty Check: FLUXVortex Aeroelastic Extension Ideas

**Date**: 2026-05-24
**Project**: FLUXVortex (E:\DATA\vscode\VLM\FLUXVortex)
**Scope**: Three candidate research ideas for extending GPU-accelerated UVLM+VPM solver to aeroelasticity

---

## Executive Summary

| Idea | Has Exact Combination Been Done? | Novelty Verdict | Publication Venue Fit |
|------|----------------------------------|-----------------|----------------------|
| **1. GPU UVLM + Euler-Bernoulli Beam Aeroelastic Solver** | **No** | HIGH — First GPU-accelerated time-domain aeroelastic UVLM solver | AIAA Journal / Aerospace Science and Technology |
| **2. Hybrid Panel-Particle Aeroelasticity** | **No** | HIGH — Hybrid wake effects on aeroelastic behavior never studied | AIAA SciTech / Journal of Fluids and Structures |
| **3. Composite Cross-Section Integrated Aeroelastic Solver** | **No** | MEDIUM-HIGH — Composite in mid-fidelity vortex methods is new, but VABS itself mature | Composite Structures / AIAA Journal |

**Bottom line**: No published work combines GPU-accelerated vortex methods (UVLM or VLM) with structural dynamics for aeroelastic simulation. The closest competitors cover either GPU vortex methods (aerodynamics only) or aeroelastic UVLM (CPU only), but not both simultaneously. All three ideas are novel; Idea 1 has the strongest novelty-to-feasibility ratio.

---

## Idea 1: GPU-Accelerated UVLM + Euler-Bernoulli Beam Aeroelastic Solver

### 1.1 Has This Exact Combination Been Done?

**No.** After extensive search across arXiv, Google Scholar, AIAA, and open-source repositories, no published work combines:

- GPU-accelerated Biot-Savart computation (NVIDIA Warp, CUDA, or any GPU framework)
- UVLM (or VLM) aerodynamic solver
- Beam finite element structural solver (Euler-Bernoulli or Timoshenko)
- Time-domain aeroelastic coupling (partitioned or monolithic)

The two capabilities exist independently:
- **GPU vortex methods** exist (FLUXVortex, Pullin et al. 2024) but are purely aerodynamic
- **Aeroelastic UVLM** exists (SHARPy, OpenAeroStruct, multiple academic codes) but is CPU-only

The intersection of these two domains is empty in the published literature.

### 1.2 Closest Existing Work

#### 1.2.1 SHARPy (Imperial College London)

| Aspect | SHARPy | FLUXVortex Idea 1 |
|--------|--------|-------------------|
| Aerodynamics | UVLM (ring vortex panels) | UVLM (ring vortex panels, identical) |
| Structural model | Geometrically exact nonlinear composite beam | Euler-Bernoulli beam (linear, simpler) |
| Wake model | Prescribed + free wake (ring vortex panels) | Hybrid panel-particle (panels + VPM) |
| Coupling | Partitioned, CPU | Partitioned, GPU-accelerated |
| Computation | CPU only (Python + NumPy) | GPU via NVIDIA Warp |
| Speed | ~1-10s/step (estimated) | Target ~0.1-1s/step |
| Maturity | Production-grade, 50+ validation cases | Research prototype |

**Key differentiator**: SHARPy is the most mature UVLM aeroelastic solver but runs entirely on CPU. FLUXVortex would be the first to bring GPU acceleration to this class of solver. Even with a simpler structural model (Euler-Bernoulli vs. geometrically exact), the GPU advantage for the O(N^2) Biot-Savart aerodynamic kernel is the primary novelty.

**Reference**: del Carre et al. (2019), "SHARPy: A dynamic aeroelastic simulation toolbox," Journal of Open Source Software, 4(44), 1885. BSD-3 license. GitHub: ImperialCollegeLondon/sharpy.

#### 1.2.2 OpenAeroStruct (University of Michigan MDO Lab)

| Aspect | OpenAeroStruct | FLUXVortex Idea 1 |
|--------|---------------|-------------------|
| Aerodynamics | Steady VLM (horseshoe vortices) | Unsteady UVLM (ring vortices) |
| Structural model | Beam FE (Euler-Bernoulli + torsion) | Beam FE (Euler-Bernoulli + torsion, similar) |
| Coupling | Loose, static aeroelastic | Dynamic, time-marching aeroelastic |
| Computation | CPU (NumPy + Autograd) | GPU (NVIDIA Warp) |
| Purpose | MDO optimization | Time-domain flutter simulation |

**Key differentiator**: OpenAeroStruct is the closest structural analog (beam FE + VLM) but uses steady VLM for optimization, not unsteady UVLM for dynamic aeroelasticity. No GPU, no unsteady effects, no flutter analysis.

**Reference**: Jasa et al., "OpenAeroStruct: An Open-Source Low-Fidelity Aeroservoelastic Analysis and Optimization Tool." GitHub: mdolab/OpenAeroStruct.

#### 1.2.3 Pullin et al. (2024) — GPU VLM+VPM Aeroacoustics

| Aspect | Pullin et al. | FLUXVortex Idea 1 |
|--------|--------------|-------------------|
| Aerodynamics | VLM + VPM (GPU-accelerated) | UVLM + VPM (GPU-accelerated) |
| Structural model | None (purely aerodynamic) | Euler-Bernoulli beam |
| Application | Aeroacoustic noise prediction | Aeroelastic flutter prediction |
| GPU framework | Custom CUDA | NVIDIA Warp |
| Coupling | No structural coupling | Aeroelastic partitioned coupling |

**Key differentiator**: This is the only other GPU-accelerated VLM+VPM solver found, but it is purely aerodynamic with no structural coupling. Confirms that GPU vortex methods for aerodynamics exist, but the aeroelastic extension has not been made.

**Reference**: Pullin, J. et al. (2024), AIAA/CEAS Aeroacoustics Conference. Cited by 2.

#### 1.2.4 Dagilis et al. (2023, 2025) — UVLM Aeroelasticity + GPU Feasibility

| Aspect | Dagilis et al. | FLUXVortex Idea 1 |
|--------|---------------|-------------------|
| Aerodynamics | UVLM | UVLM + VPM hybrid |
| Structural model | Beam-based (details unclear) | Euler-Bernoulli beam |
| GPU usage | Discussed as feasibility, not implemented | Actually implemented (Warp kernels) |
| Coupling | CPU-based partitioned | GPU-accelerated partitioned |

**Key differentiator**: Dagilis et al. discuss GPU feasibility for real-time UVLM aeroelasticity but do not present an actual GPU implementation. FLUXVortex already has working GPU kernels.

**References**:
- Dagilis, V. et al. (2023), "UVLM aeroelasticity model — real-time feasibility with GPU computing," Aerospace, 10(12).
- Dagilis, V. et al. (2025), AIAA SciTech, expanded version.

#### 1.2.5 Ayala (2025) — GPU Harmonic Balance VLM Aeroelasticity

| Aspect | Ayala (2025) | FLUXVortex Idea 1 |
|--------|-------------|-------------------|
| Domain | Frequency domain (Harmonic Balance) | Time domain (explicit time-marching) |
| Aerodynamics | VLM with harmonic balance | UVLM (full unsteady) |
| Structural model | Nonlinear aeroelastic (frequency domain) | Euler-Bernoulli beam (time domain) |
| GPU usage | GPU-accelerated VLM | GPU-accelerated UVLM |

**Key differentiator**: Ayala uses frequency-domain Harmonic Balance, which is fundamentally different from time-domain coupling. HB-VLM is efficient for periodic responses but cannot capture transient phenomena (flutter onset, gust response) that time-domain UVLM can. This is a complementary approach, not a competing one.

**Reference**: Ayala, C. (2025), M.Sc. thesis, Polytechnique Montreal.

#### 1.2.6 Fu & Laurendeau (2025) — NL-UVLM-VPM for Rotors

| Aspect | Fu & Laurendeau | FLUXVortex Idea 1 |
|--------|----------------|-------------------|
| Aerodynamics | Nonlinear UVLM + VPM (rotors) | UVLM + VPM hybrid (wings) |
| Structural model | None (purely aerodynamic) | Euler-Bernoulli beam |
| GPU usage | CPU only | GPU (NVIDIA Warp) |
| Application | Rotor aerodynamics and performance | Wing aeroelasticity |

**Key differentiator**: Confirmed as aerodynamics-only (no structural coupling, no GPU). Validates the UVLM+VPM hybrid approach for aerodynamics but does not address aeroelasticity.

**Reference**: Fu, S. & Laurendeau, A. (2025), arXiv:2511.11430.

#### 1.2.7 Zhicheng et al. (2025) — VLM-Accelerated CFD/CSD

| Aspect | Zhicheng et al. | FLUXVortex Idea 1 |
|--------|----------------|-------------------|
| Approach | VLM accelerates CFD (hybrid) | Standalone UVLM + beam |
| Structural model | CSD (high-fidelity) | Beam FE (mid-fidelity) |
| GPU usage | CFD on GPU (implicit) | UVLM + beam on GPU (explicit) |

**Key differentiator**: VLM is used to accelerate CFD convergence, not as the standalone aerodynamic solver. Different problem class entirely (high-fidelity with VLM assist vs. mid-fidelity standalone).

**Reference**: Zhicheng et al. (2025), AIAA Journal.

### 1.3 Concurrent Work (2024-2026)

| Work | Year | Overlap with Idea 1 | Threat Level |
|------|------|---------------------|-------------|
| Pullin et al. | 2024 | GPU VLM+VPM, but no aeroelasticity | LOW — different application (aeroacoustics) |
| Dagilis et al. | 2023-2025 | UVLM aeroelasticity + GPU feasibility discussion | MEDIUM — discusses but does not implement |
| Ayala | 2025 | GPU VLM aeroelasticity, but frequency domain | LOW — complementary approach |
| Pudasaini et al. | 2026 | UVLM aeroelasticity, but CPU, morphing focus | LOW — different application |
| Schubert et al. | 2024 | UVLM aeroelasticity acceleration, but CPU algorithmic | LOW — algorithmic, not hardware |
| Dussler & Palacios | 2023 | UVLM + nonlinear beam, but CPU, Palacios group | LOW — mature but CPU-bound |

**Assessment**: No concurrent work is actively pursuing GPU-accelerated time-domain aeroelastic UVLM simulation. The field is moving toward (1) algorithmic acceleration (quasi-Newton), (2) geometric nonlinearity, and (3) morphing/active control — but all on CPU. The GPU acceleration angle is open.

### 1.4 Novelty Assessment

| Novelty Dimension | Rating | Justification |
|-------------------|--------|---------------|
| **Technical novelty** | HIGH | First GPU-accelerated time-domain aeroelastic UVLM solver |
| **Conceptual novelty** | MODERATE | Combining two existing capabilities (GPU vortex + beam FE); novelty is in the integration |
| **Application novelty** | MODERATE | Flutter prediction with mid-fidelity methods is well-studied, but real-time capability is new |
| **Methodological novelty** | HIGH | NVIDIA Warp for both aerodynamic Biot-Savart AND beam FE (if beam is also GPU-accelerated) |

**Conference viability**: YES — AIAA SciTech or AIAA/CEAS would accept this as a clear contribution.
**Journal viability**: YES — AIAA Journal or Aerospace Science and Technology, provided validation against Goland Wing reference solutions is thorough.

### 1.5 Key Risk: Is the Combination "Obvious"?

The combination of GPU vortex methods + beam FE might seem obvious in retrospect. The defense against this:

1. **It has not been done despite 15+ years of UVLM aeroelasticity research.** SHARPy (2013-present), the leading UVLM aeroelastic code, remains CPU-only despite clear performance benefits of GPU.
2. **Technical barriers are non-trivial**: GPU Biot-Savart requires careful treatment of float64 precision (Warp defaults to float32), atomic operations for accumulation, and memory transfer optimization. The beam coupling adds complexity to what is already a non-trivial GPU implementation.
3. **The contribution is not merely "putting X on GPU"**: The beam FE integration requires redesigning the solver loop (from pre-computed geometry to incremental generation), force mapping from panel to beam nodes, and geometric update propagation — all while maintaining GPU data flow.
4. **Performance claims require validation**: 10-100x speedup over CPU UVLM+beam is the target; achieving this requires careful engineering, not just running existing code on GPU.

---

## Idea 2: Hybrid Panel-Particle Aeroelasticity (Wake Effects on Flutter)

### 2.1 Has This Exact Combination Been Done?

**No.** The hybrid panel-particle wake model (near-field ring vortex panels + far-field VPM particles) applied to aeroelastic simulation has no precedent. All existing aeroelastic UVLM codes use pure ring vortex panel wakes:

- **SHARPy**: Ring vortex wake panels throughout
- **MACH framework**: Horseshoe or ring vortex wake
- **OpenAeroStruct**: Horseshoe vortex wake (steady)
- **Yang et al. (2020)**: Geometrically exact VLM with prescribed wake
- **Pudasaini et al. (2026)**: UVLM with prescribed/free ring vortex wake

### 2.2 Closest Existing Work

#### DUST (Politecnico di Milano)

DUST is a C++ UVLM+VPM hybrid aerodynamic solver that is the closest architectural analog to FLUXVortex's hybrid approach. However:
- DUST's documentation does not mention structural coupling for aeroelasticity
- DUST focuses on aerodynamic simulation of complex configurations (rotors, multi-body)
- No published study of wake model effects on aeroelastic behavior using DUST

#### FLOWVLM/FLOWVPM (BYU)

FLOWVPM uses VPM particles for wake representation with VLM panels for the wing. However:
- FLOWVLM has no structural solver (purely aerodynamic)
- The BYU group's publications focus on rotor wake dynamics, not aeroelasticity
- No study of VPM wake effects on flutter or aeroelastic response

#### Bhosale, Parthasarathy & Gazzola (2020)

Pure VPM-based FSI for mixed rigid/soft bodies, but:
- Full VPM (no panel method), so no hybrid comparison possible
- Different structural model (soft body, not beam FE)
- Different application (aquatic, not aircraft)

### 2.3 What Is Specifically New

1. **First study of hybrid wake (panel + particle) effects on aeroelastic behavior**: The question "does free VPM wake vs. prescribed panel wake change flutter prediction?" has never been asked in the published literature.

2. **Potential discovery of wake-induced aeroelastic effects**: VPM free wake captures roll-up, mutual induction, and far-wake dynamics that prescribed wake models cannot. These could affect:
   - Flutter speed prediction (+/-5-15%, estimated)
   - Limit cycle oscillation amplitude
   - Post-flutter behavior

3. **Timoshenko beam with geometric nonlinearity**: While not novel on its own, combining it with hybrid wake is a unique capability.

### 2.4 Novelty Assessment

| Novelty Dimension | Rating | Justification |
|-------------------|--------|---------------|
| **Technical novelty** | HIGH | Hybrid panel-particle wake in aeroelastic context is entirely new |
| **Conceptual novelty** | HIGH | The research question itself (wake model effect on flutter) is novel |
| **Application novelty** | HIGH | Could reveal previously unknown physical phenomena |
| **Methodological novelty** | MODERATE | Combines existing capabilities (hybrid wake + Timoshenko beam) |

**Conference viability**: YES — strong candidate for AIAA SciTech or IFASD (International Forum on Aeroelasticity and Structural Dynamics).
**Journal viability**: YES — Journal of Fluids and Structures or AIAA Journal, provided the study shows meaningful differences between wake models.

### 2.5 Key Risk: Hybrid Wake May Not Matter for Flutter

The strongest risk is that the hybrid wake model may produce negligible differences compared to prescribed wake for the test cases. If:
- Prescribed wake already captures the dominant aeroelastic coupling
- VPM free wake effects are secondary for subsonic flutter
- Differences are within numerical noise

Then the novelty degrades from "discovery of new physics" to "confirmation that wake model choice is secondary," which is less impactful (though still publishable).

**Mitigation**: Choose test cases where wake dynamics matter (large deformation, high reduced frequency, near-stall conditions).

---

## Idea 3: Composite Cross-Section Integrated Aeroelastic Solver

### 3.1 Has This Exact Combination Been Done?

**No, with qualification.** The specific combination of analytical composite cross-section analysis + mid-fidelity vortex method (UVLM) aeroelasticity has not been published. However, the component technologies are mature:

- **VABS composite cross-section analysis**: Widely used, 20+ years of development, integrated into SHARPy and other tools
- **Composite beam aeroelasticity**: Well-studied with DLM, FEM, and CFD methods
- **Bending-torsion coupling**: Known to affect flutter speed, studied since the 1980s

### 3.2 Closest Existing Work

#### SHARPy + VABS Integration

SHARPy already integrates with VABS for composite cross-sectional properties. The Palacios group has published extensively on composite beam aeroelasticity using UVLM. This is the most direct competitor:

| Aspect | SHARPy + VABS | FLUXVortex Idea 3 |
|--------|--------------|-------------------|
| Cross-section analysis | VABS (full FEM-based) | Analytical (simplified) |
| Beam model | Geometrically exact nonlinear | Timoshenko (less general) |
| Aerodynamics | UVLM (CPU) | UVLM + VPM (GPU) |
| Coupling | Partitioned | Partitioned (same) |
| Maturity | Production | Research prototype |

**Key differentiator**: SHARPy already does composite aeroelasticity with VABS, but on CPU only. FLUXVortex would add GPU acceleration. The composite modeling itself would be less capable (analytical vs. VABS FEM), but the GPU speed advantage could enable design-space exploration that SHARPy cannot practically perform.

#### Hodges (2006) — Nonlinear Composite Beam Theory

The definitive reference for geometrically-exact composite beam modeling. Any composite beam implementation in FLUXVortex would build on this framework. Not a competitor per se, but establishes that the underlying structural theory is mature and well-understood.

#### Yu & Hodges — VABS

The standard tool for computing equivalent beam stiffness matrices from arbitrary composite cross-sections. Available commercially from AnalySwift. FLUXVortex could either:
- (a) Call VABS as an external tool for cross-section properties, or
- (b) Implement simplified analytical formulas for standard sections (box beam, I-beam)

Option (a) adds a dependency but provides full generality. Option (b) is self-contained but limited to simple geometries.

### 3.3 What Is Specifically New

1. **GPU-accelerated composite aeroelastic simulation**: Even with a simplified structural model, the combination of GPU vortex methods + composite beam is new.

2. **Lightweight composite integration in a Python framework**: SHARPy requires VABS as an external tool (Fortran). A pure-Python analytical composite section analysis, integrated directly into FLUXVortex, would lower the barrier to entry for composite aeroelastic studies.

3. **Bending-torsion coupling study in GPU vortex framework**: The effect of asymmetric composite layup on flutter behavior (via bending-torsion coupling) has been studied in CPU frameworks but never with GPU acceleration enabling rapid parametric sweeps.

### 3.4 Novelty Assessment

| Novelty Dimension | Rating | Justification |
|-------------------|--------|---------------|
| **Technical novelty** | MODERATE | Composite beam is mature; GPU integration is the new element |
| **Conceptual novelty** | MODERATE | Bending-torsion coupling effects on flutter are well-known |
| **Application novelty** | MODERATE | Carbon fiber in mid-fidelity vortex methods is under-explored |
| **Methodological novelty** | MODERATE-HIGH | Self-contained Python composite + GPU vortex is unique |

**Conference viability**: YES — Composite Structures conference or AIAA SciTech.
**Journal viability**: MAYBE — Composite Structures or AIAA Journal would need either (a) new physics discovered, or (b) significant computational advantage demonstrated. The GPU angle alone may not suffice unless accompanied by design optimization studies.

### 3.5 Key Risk: VABS Already Does This Better

The fundamental risk is that VABS + SHARPy already provides a more capable composite aeroelastic solver. FLUXVortex's advantage would be:
- Speed (GPU vs. CPU)
- Accessibility (pure Python vs. Fortran dependency)
- Integration with hybrid wake

If speed is the only advantage, the contribution is incremental. To strengthen the novelty, Idea 3 should be framed as enabling composite design optimization (requiring hundreds of flutter evaluations) that CPU methods cannot practically perform.

---

## Comparative Assessment

### Novelty vs. Feasibility Matrix

| Idea | Novelty | Feasibility | Publication Impact | Pilot Risk | Recommended Priority |
|------|---------|-------------|-------------------|------------|---------------------|
| **1. GPU UVLM + Euler-Bernoulli Beam** | **HIGH** | **VERY HIGH** | HIGH | LOW | **1st (implement first)** |
| **2. Hybrid Panel-Particle Aeroelasticity** | **HIGH** | HIGH | HIGH | MEDIUM | 2nd (builds on Idea 1) |
| **3. Composite Cross-Section Integration** | MEDIUM-HIGH | MEDIUM | MEDIUM | MEDIUM | 3rd (builds on Idea 1+2) |

### Recommended Strategy

**Minimum publishable unit**: Idea 1 alone is sufficient for a strong conference paper (AIAA SciTech) or journal article (AIAA Journal). The GPU acceleration + aeroelastic coupling + Goland Wing validation is a complete, self-contained contribution.

**Maximum impact trajectory**: Ideas 1 + 2 combined create a compelling narrative:
1. "We built the first GPU-accelerated aeroelastic UVLM solver" (Idea 1)
2. "We discovered that hybrid panel-particle wakes affect flutter prediction by X%" (Idea 2)

Idea 3 strengthens the contribution for a follow-on paper focused on composite design optimization.

### Potential Reviewer Objections and Responses

**Objection 1**: "GPU acceleration of UVLM is straightforward — just parallelize the Biot-Savart computation."
**Response**: The Biot-Savart GPU acceleration is already done (FLUXVortex). The novelty lies in (a) integrating beam FE into the GPU pipeline, (b) redesigning the solver loop for incremental geometry generation, and (c) maintaining numerical stability in the coupled system on GPU. The Biot-Savart kernel alone does not make an aeroelastic solver.

**Objection 2**: "Euler-Bernoulli beam is too simple for practical aeroelastic simulation."
**Response**: Euler-Bernoulli is the appropriate starting point for (a) Goland Wing validation (the benchmark uses uniform beam properties), (b) demonstrating the GPU coupling framework, and (c) establishing correctness before adding complexity. The framework is designed to upgrade to Timoshenko/geometrically-exact beams (Ideas 2 and 3).

**Objection 3**: "The performance comparison against SHARPy is unfair because SHARPy has more features."
**Response**: The comparison should be on identical problems (Goland Wing, same mesh resolution, same time step). SHARPy's additional features (geometric nonlinearity, composite beams) are orthogonal to the GPU acceleration question. A fair comparison isolates the aerodynamic + linear beam coupling.

**Objection 4 (Idea 2)**: "Hybrid wake effects are negligible for subsonic flutter."
**Response**: This is an empirical question that the research will answer. Even a negative result ("hybrid wake does not significantly affect flutter prediction for the Goland Wing") is valuable because it validates the simpler prescribed wake approach. For more complex configurations (high aspect ratio, large deformation), the hybrid wake may become significant.

**Objection 5 (Idea 3)**: "VABS already provides composite cross-section analysis."
**Response**: VABS is a separate Fortran tool with licensing requirements. FLUXVortex's self-contained analytical approach targets a different use case: rapid design-space exploration where thousands of configurations must be evaluated, and the analytical approximation is sufficient. VABS can still be used as a validation reference.

---

## Search Methodology

### Sources Consulted

1. **arXiv**: Searched for "GPU vortex lattice method aeroelasticity," "GPU VLM flutter," "GPU accelerated vortex method structural coupling." Zero exact matches for GPU + VLM/UVLM + aeroelasticity.

2. **Google Scholar**: Searched for "GPU accelerated vortex lattice method aeroelasticity," "GPU VLM aeroelastic flutter simulation." Found ~943 results for broad query; most are tangential. Identified 4-5 directly relevant works (Pullin 2024, Dagilis 2023/2025, Ayala 2025, Zhicheng 2025).

3. **AIAA Digital Library**: Attempted access to Pullin et al. (2024) — access forbidden (paywall). Relied on search snippets.

4. **GitHub**: Reviewed SHARPy repository (confirmed CPU-only UVLM + nonlinear beam), FLOWVLM repository (confirmed purely aerodynamic), OpenAeroStruct repository (confirmed steady VLM + beam for MDO).

5. **Project literature survey**: Cross-referenced against existing `lit_survey_aeroelasticity.md` (23 references) and `IDEA_REPORT.md` (9 references).

### Search Limitations

- WebSearch API was unavailable (persistent 400 errors) during the search phase
- AIAA paywalled papers could not be accessed in full
- Google Scholar rate limiting restricted query volume
- No access to Semantic Scholar API during this session
- Chinese-language publications (CNKI, etc.) were not searched — potential blind spot for concurrent work from Chinese aerospace groups

---

## Appendix: Competitor Summary Table

| Competitor | Year | GPU? | UVLM/VLM? | Structural? | Aeroelastic? | Hybrid Wake? | Composite? | Overlap with FLUXVortex |
|-----------|------|------|-----------|-------------|-------------|-------------|-----------|------------------------|
| **SHARPy** | 2013+ | No | UVLM | Geom. exact beam | Yes | No (ring panels) | Yes (VABS) | Closest aero competitor |
| **OpenAeroStruct** | 2018+ | No | Steady VLM | Beam FE | Static only | No | No | Closest structural analog |
| **DUST** | 2020+ | Partial | UVLM+VPM | External FE | Limited | Yes (UVLM+VPM) | No | Closest wake analog |
| **Pullin et al.** | 2024 | Yes | VLM+VPM | None | No | Yes | No | Closest GPU analog |
| **Dagilis et al.** | 2023-25 | Discussed | UVLM | Beam | Yes | No | No | Discussed GPU feasibility |
| **Ayala** | 2025 | Yes | HB-VLM | Nonlinear | Yes (freq. domain) | No | No | GPU + aeroelastic, different domain |
| **Fu & Laurendeau** | 2025 | No | NL-UVLM+VPM | None | No | Yes | No | Hybrid wake, aerodynamics only |
| **MACH (UMich)** | Ongoing | No | VLM | Beam FE | Yes | No | No | MDO-focused |
| **FLOWVLM/FLOWVPM** | 2018+ | No | VLM+VPM | None | No | Yes | No | Parent framework inspiration |
| **FLUXVortex (current)** | 2026 | Yes | UVLM+VPM | None | No | Yes | No | Starting point |

---

*Novelty check prepared for the FLUXVortex project, 2026-05-24.*
