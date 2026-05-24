# Literature Survey: Aeroelasticity Simulation Using Vortex Methods (UVLM, VLM, VPM)

> Survey compiled: 2026-05-24
> Context: Extending FLUXVortex (GPU-accelerated UVLM+VPM hybrid solver, based on PteraSoftware) to handle flexible/elastic wings (carbon fiber composite)

---

## Table of Contents

1. [UVLM-Based Aeroelasticity](#1-uvlm-based-aeroelasticity)
2. [Vortex Methods + Flexible Structures (FSI)](#2-vortex-methods--flexible-structures-fsi)
3. [PteraSoftware Ecosystem](#3-pterasoftware-ecosystem)
4. [Composite Wing Structural Modeling](#4-composite-wing-structural-modeling)
5. [Recent Advances (2023-2026)](#5-recent-advances-2023-2026)
6. [Key Aeroelastic Benchmarks](#6-key-aeroelastic-benchmarks)
7. [Open-Source Implementations](#7-open-source-implementations)
8. [Synthesis and Gap Analysis for FLUXVortex](#8-synthesis-and-gap-analysis-for-fluxvortex)
9. [References](#9-references)

---

## 1. UVLM-Based Aeroelasticity

### 1.1 Foundational Review: Murua, Palacios & Graham (2012)

| Field | Detail |
|-------|--------|
| **Title** | Applications of the Unsteady Vortex-Lattice Method in Aircraft Aeroelasticity and Flight Dynamics |
| **Authors** | J. Murua, R. Palacios, J. Graham |
| **Year** | 2012 |
| **Venue** | Progress in Aerospace Sciences |
| **DOI** | 10.1016/J.PAEROSCI.2012.06.001 |
| **Citations** | 304 |
| **PDF** | [Open Access (Surrey)](https://openresearch.surrey.ac.uk/view/delivery/44SUR_INST/12139141490002346/13140650260002346) |

**Summary**: This is the definitive review paper on UVLM for aeroelasticity and flight dynamics. It provides:

- **Structural model**: Nonlinear beam model (geometrically exact) for the full description of a free-flying flexible vehicle
- **Coupling approach**: Both nonlinear time-marching (partitioned) and linearized monolithic state-space assembly
- **Key contributions**:
  - Free-wake modeling captures wake roll-up and large wing excursions
  - Linearization of UVLM equations enables seamless state-space assembly for stability analysis and control design
  - Geometric nonlinearities shown to play instrumental, often counter-intuitive role in aircraft dynamics
  - UVLM demonstrated as superior to Doublet-Lattice Method for coupled aeroelasticity/flight dynamics, complex kinematics, large deformations, and in-plane motions
- **Limitations**: Assumes attached flow (potential flow); no viscous effects or stall modeling
- **Relevance to FLUXVortex**: This paper establishes the theoretical framework for UVLM-based aeroelasticity. FLUXVortex's hybrid panel-particle wake can be directly substituted for the pure UVLM wake, providing improved wake fidelity at similar or lower computational cost.

---

### 1.2 Enhanced UVLM for Nonlinear Flexible Aircraft: Dussler & Palacios (2023)

| Field | Detail |
|-------|--------|
| **Title** | Enhanced Unsteady Vortex Lattice Aerodynamics for Nonlinear Flexible Aircraft Dynamic Simulation |
| **Authors** | Stefanie Dussler, R. Palacios |
| **Year** | 2023 |
| **Venue** | AIAA Journal |
| **DOI** | 10.2514/1.J063174 |
| **Citations** | 4 |

**Summary**: Latest extensions to UVLM for very flexible aircraft:

- **Enhancements**:
  - Fuselage aerodynamics via linear source panels
  - Polar corrections for aerodynamics under large deformations (stall onset at high induced angles)
  - New wake discretization scheme (20% computational time reduction)
- **Structural model**: Nonlinear beam finite elements (inherent from Palacios group framework)
- **Coupling approach**: Partitioned coupling with loose/strong coupling options
- **Validation**: Flexible aircraft demonstrator model showing fuselage-wing interference effects
- **Key finding**: For large wing deformations, fuselage aerodynamic interference noticeably affects wing aeroelastic behavior
- **Relevance**: Directly relevant -- shows the state-of-the-art in UVLM aeroelastic simulation. FLUXVortex could benefit from incorporating polar corrections and the wake acceleration scheme.

---

### 1.3 Accelerating Aeroelastic UVLM Simulations: Schubert et al. (2024)

| Field | Detail |
|-------|--------|
| **Title** | Accelerating Aeroelastic UVLM Simulations by Inexact Newton Algorithms |
| **Authors** | Jenny Schubert, Marc C. Steinbach, Christian Hente, David Martens, Daniel Schuster |
| **Year** | 2024 |
| **Venue** | arXiv (math.NA), submitted to journal |
| **arXiv** | 2403.15286 |

**Summary**:

- **Structural model**: Nonlinear kinematics in total Lagrangian formulation, discretized by finite elements
- **Coupling approach**: Implicit time-marching with Newton algorithm for the coupled aeroelastic system
- **Key contributions**:
  - Structural derivative approximation to full Jacobian in Newton iterations
  - Quasi-Newton algorithm outperforms inexact Newton algorithm in practice
  - Substantial acceleration of the Newton algorithm while maintaining accuracy
- **Validation cases**: Flexible plate and wind turbine
- **Limitations**: Focus on algorithmic acceleration rather than GPU parallelism
- **Relevance**: Critical for FLUXVortex -- the quasi-Newton approach could be combined with GPU acceleration for maximum speedup. The partitioned coupling approach (UVLM aero + FE structural) is exactly what FLUXVortex needs.

---

### 1.4 Geometrically Exact VLM/Panel Methods: Yang, Xie & Yang (2020)

| Field | Detail |
|-------|--------|
| **Title** | Geometrically Exact Vortex Lattice and Panel Methods in Static Aeroelasticity of Very Flexible Wing |
| **Authors** | Lan Yang, Changchuan Xie, Chao Yang |
| **Year** | 2020 |
| **Venue** | Proceedings of the Institution of Mechanical Engineers, Part G |
| **DOI** | 10.1177/0954410019885238 |
| **Citations** | 10 |

**Summary**:

- **Structural model**: Geometrically nonlinear finite element method
- **Coupling approach**: Loosely-coupled iteration through surface spline interpolation
- **Key contributions**:
  - Geometrically exact boundary conditions make potential flow significantly different from linear aeroelastic analysis
  - VLM (thin airfoil) vs panel method (thick wings) comparison
  - Good agreement with CFD/CSD coupling and wind tunnel test data even for considerably large deformation
- **Relevance**: Provides validated methodology for loosely-coupled aeroelastic simulation with geometric nonlinearities.

---

### 1.5 Analytical Sensitivity Analysis: Hang et al. (2020)

| Field | Detail |
|-------|--------|
| **Title** | Analytical Sensitivity Analysis of Flexible Aircraft with the Unsteady Vortex-Lattice Aerodynamic Theory |
| **Authors** | Hang Xiaochen, W. Su, Q. Fei, D. Jiang |
| **Year** | 2020 |
| **Venue** | Aerospace Science and Technology |
| **DOI** | 10.1016/j.ast.2019.105612 |
| **Citations** | 23 |

**Summary**:

- **Structural model**: Both beam-based flexible wing models and shell-based FE models
- **Coupling approach**: Surface spline interpolation for structure-aerodynamic coupling
- **Key contributions**:
  - Analytical aerodynamic sensitivity calculation scheme (linearized around equilibrium)
  - Free wake model for accurate vortex shedding
  - Foundation for aeroelastic optimization and stability analysis
  - Applicable to both beam and shell structural models
- **Relevance**: The surface spline interpolation approach and sensitivity framework are directly applicable to FLUXVortex.

---

### 1.6 Mid-Fidelity Aeroservoelastic Models: Pudasaini, Smith & Huang (2026)

| Field | Detail |
|-------|--------|
| **Title** | Trajectory Optimization of Morphing Aerial Vehicles Based on Mid-Fidelity Aeroservoelastic Models |
| **Authors** | Subarna Pudasaini, Parker Smith, Daning Huang |
| **Year** | 2026 |
| **Venue** | arXiv (eess.SY), submitted to AIAA Journal of Aircraft |
| **arXiv** | 2605.02076 |

**Summary**:

- **Structural model**: Nonlinear multi-body structural dynamics
- **Coupling approach**: Coupling nonlinear multi-body structural dynamics with UVLM
- **Key contributions**:
  - Mid-fidelity aeroservoelastic framework for morphing vehicles
  - Physics-based control cost model capturing aerodynamic hinge moments
  - Demonstrated on flexible, high-aspect-ratio wings with morphing winglets
  - Morphing wings expand flight envelope by decoupling lift and pitch requirements
- **Relevance**: Most recent work (2026) demonstrating UVLM aeroelasticity for morphing wings. Shows the direction of the field.

---

### 1.7 Nonlinear Aeroelasticity of Joined-Wing Aircraft: Su, Huang & Hammerton (2017)

| Field | Detail |
|-------|--------|
| **Title** | Nonlinear Aeroelasticity of Highly Flexible Joined-Wing Aircraft using Unsteady Vortex-Lattice Method |
| **Authors** | W. Su, Yanxin Huang, J. Hammerton |
| **Year** | 2017 |
| **DOI** | 10.2514/6.2017-1353 |
| **Citations** | 7 |

**Summary**: UVLM applied to nonlinear aeroelastic analysis of joined-wing configurations, demonstrating UVLM's ability to handle non-planar lifting surfaces and their structural coupling.

---

### 1.8 Quasi-3D UVLM ROM for Flapping Wings: Schwab, Reade & Jankauski (2022)

| Field | Detail |
|-------|--------|
| **Title** | Quasi Three-Dimensional Deformable Blade Element and Unsteady Vortex Lattice Reduced-Order Modeling of Fluid-Structure Interaction in Flapping Wings |
| **Authors** | Ryan Schwab, Joseph Reade, Mark A. Jankauski |
| **Year** | 2022 |
| **Venue** | Physics of Fluids |
| **DOI** | 10.1063/5.0129128 |
| **Citations** | 10 |
| **PDF** | [Open Access (AIP)](https://aip.scitation.org/doi/10.1063/5.0129128) |

**Summary**:

- **Structural model**: Modal-truncation based structural solver
- **Coupling approach**: Reduced-order UVLM coupled with modal structural model
- **Key contributions**:
  - Two ROMs: Deformable Blade Element Theory (DBET) and UVLM-based
  - Good accuracy even for 25% chord length deformations
  - UVLM ROM solves 4-6 orders of magnitude faster than full CFD/FEA
  - Flexible wings produce less lift but require lower average power
- **Limitations**: Limited to thin flat plate geometry; in-plane loading errors larger
- **Relevance**: Demonstrates the massive speedup potential of UVLM-based FSI. FLUXVortex's GPU acceleration can push this even further.

---

## 2. Vortex Methods + Flexible Structures (FSI)

### 2.1 Strongly Coupled FSI with UVLM: Luo, Wu & Yang (2023)

| Field | Detail |
|-------|--------|
| **Title** | Strongly Coupled Fluid-Structure Interaction Analysis of Aquatic Flapping Wings Based on Flexible Multibody Dynamics and the Modified Unsteady Vortex Lattice Method |
| **Authors** | Ming Luo, Zhigang Wu, Chao Yang |
| **Year** | 2023 |
| **Venue** | Ocean Engineering |
| **DOI** | 10.1016/j.oceaneng.2023.114921 |
| **Citations** | 9 |

**Summary**:

- **Structural model**: Flexible multibody dynamics (multiple rigid bodies connected by joints/springs)
- **Coupling approach**: Strongly coupled (monolithic) FSI
- **Key contributions**:
  - Modified UVLM for aquatic flapping wings
  - Strong coupling ensures stability and convergence
  - Demonstrates UVLM's applicability beyond fixed-wing aircraft
- **Relevance**: Shows the strongly-coupled approach is feasible with UVLM, though partitioned coupling is more common for fixed-wing applications.

---

### 2.2 Remeshed Vortex Method for FSI: Bhosale, Parthasarathy & Gazzola (2020)

| Field | Detail |
|-------|--------|
| **Title** | A Remeshed Vortex Method for Mixed Rigid/Soft Body Fluid-Structure Interaction |
| **Authors** | Y. Bhosale, Tejaswin Parthasarathy, M. Gazzola |
| **Year** | 2020 |
| **Venue** | Journal of Computational Physics |
| **DOI** | 10.1016/j.jcp.2021.110577 |
| **arXiv** | 2011.09669 |
| **Citations** | 17 |

**Summary**:

- **Method**: Remeshed vortex method (full vortex particle method, not UVLM)
- **Structural model**: Mixed rigid/soft body structural solver
- **Coupling approach**: Integrated fluid-structure framework
- **Key contributions**:
  - Pure VPM-based FSI (no panel method)
  - Handles both rigid and soft (deformable) bodies
  - Demonstrates VPM can capture FSI effects directly
- **Limitations**: Higher computational cost than UVLM; VPM accuracy depends heavily on particle resolution
- **Relevance**: Relevant as a reference for the VPM-only path in FLUXVortex's hybrid solver. However, the paper confirms that VPM-only approaches have lower accuracy than panel-particle hybrids.

---

### 2.3 VPM for Bridge Deck FSI: Tesfaye, Kavrakov & Morgenthal (2022)

| Field | Detail |
|-------|--------|
| **Title** | Numerical Investigation of the Nonlinear Interaction Between the Sinusoidal Motion-Induced and Gust-Induced Forces Acting on Bridge Decks |
| **Authors** | Samuel Tesfaye, Igor Kavrakov, Guido Morgenthal |
| **Year** | 2022 |
| **Venue** | Journal of Fluids and Structures |
| **DOI** | 10.1016/j.jfluidstructs.2022.103680 |
| **arXiv** | 2109.00441 |

**Summary**:

- **Method**: Vortex Particle Method (VPM) as CFD tool for bridge aerodynamics
- **Key finding**: Linear superposition fails for bluff bodies due to vortex-dominated flow; VPM captures nonlinear interaction
- **Relevance**: Demonstrates VPM's value for nonlinear aeroelastic problems, though not directly for wing applications.

---

### 2.4 Flexible Foil Aeroelasticity: D'Adamo et al. (2022)

| Field | Detail |
|-------|--------|
| **Title** | Wake and Aeroelasticity of a Flexible Pitching Foil |
| **Authors** | Juan D'Adamo, Manuel Collaud, Roberto Sosa, Ramiro Godoy-Diana |
| **Year** | 2022 |
| **Venue** | Bioinspiration & Biomimetics |
| **DOI** | 10.1088/1748-3190/ac6d96 |
| **arXiv** | 2206.01647 |

**Summary**:

- **Method**: Experimental with PIV measurements; cluster-based reduced order modeling
- **Structural model**: Elastic foil with natural frequency within flapping range
- **Key contributions**:
  - Strongly-coupled dynamics between elastic deformation and vortex shedding
  - Thrust peaks at dimensionless frequencies shifted from elastic resonance
  - Wake resonance explains optimal thrust better than structural resonance
- **Relevance**: Experimental evidence of vortex-structure coupling dynamics, relevant for understanding the physics FLUXVortex needs to capture.

---

## 3. PteraSoftware Ecosystem

### 3.1 PteraSoftware Overview

| Field | Detail |
|-------|--------|
| **Repository** | https://github.com/camUrban/PteraSoftware |
| **Authors** | Cameron Urban (primary), BYU Flow Lab |
| **Language** | Python (NumPy, SciPy, Matplotlib) |
| **License** | MIT |

**Key Features**:
- Unsteady Vortex Lattice Method (UVLM) with ring vortex and horseshoe vortex panels
- Prescribed and free wake models
- Static and unsteady simulation capabilities
- Ground effect via method of images
- Flapping wing simulation
- Extensive verification suite against Theodorsen theory and XFLR5

**Current Limitations**:
- Rigid body aerodynamics only -- no structural coupling
- CPU-only (NumPy/Numba based)
- No VPM/particle wake capability
- No composite material modeling

**FLUXVortex Extensions Already Made**:
- GPU acceleration via NVIDIA Warp (monkey-patch injection)
- Hybrid panel-particle wake (UVLM panels + VPM particles)
- Free wake with rVPM vortex stretching
- RK3 time integration for particles
- 2.4-11x speedup depending on problem size

**Aeroelasticity Gap**: No published work extends PteraSoftware for aeroelastic problems. PteraSoftware remains a purely aerodynamic solver. FLUXVortex would be the first to add structural coupling.

### 3.2 Related: FLOWVLM/FLOWVPM (BYU Flow Lab)

| Field | Detail |
|-------|--------|
| **Repository** | https://github.com/byuflowlab/FLOWVLM |
| **Authors** | Eduardo J. Alvarez (primary), Prof. Andrew Ning group, BYU |
| **Language** | Julia |

**Key Features**:
- VLM aerodynamic solver
- VPM (Vortex Particle Method) wake solver
- FLOWVPM provides high-fidelity free-wake simulation
- Used extensively for rotorcraft and wind turbine simulations
- NOT aeroelastic -- purely aerodynamic (no structural solver)

**Relevant Publications by Alvarez et al.**:
- Alvarez & Ning (2020): "Development of a Vortex Particle Code for the Modeling of Wake Interaction in Distributed Propulsion" -- AIAA Scitech
- Alvarez, Ning & Zingg (2022): "Vortex Particle Method for Modeling Wind Turbine Wakes" -- various venues
- FLOWVPM focuses on high-fidelity wake modeling, not structural coupling

---

## 4. Composite Wing Structural Modeling

### 4.1 Approaches for Mid-Fidelity Composite Wing Models

The literature reveals several tiers of structural modeling for composite wings in aeroelastic simulation:

#### Tier 1: Euler-Bernoulli / Timoshenko Beam Models (Most Common for Mid-Fidelity)

- **Description**: 1D beam along the elastic axis with cross-sectional properties derived from composite layup analysis
- **Tools**: VABS (Variational Asymptotic Beam Sectional Analysis), PreVABS, ANSYS cross-section tools
- **Inputs**: Composite layup sequence, fiber orientations, material properties
- **Outputs**: Equivalent beam stiffness matrix (EI, GJ, EA, coupling terms)
- **Pros**: Very fast; captures global bending/torsion behavior; easily coupled with UVLM
- **Cons**: No local stress/strain information; limited for thick composite sections

**Key References**:
- Hodges, D.H. (2006): "Nonlinear Composite Beam Theory" -- AIAA Education Series. The definitive reference for geometrically-exact composite beam modeling.
- Yu, W. & Hodges, D.H. (2004+): VABS -- Variational Asymptotic Beam Sectional analysis. Computes equivalent 1D beam properties from 2D cross-section of arbitrary composite layup. Widely used in mid-fidelity aeroelastic tools.
- Palacios, R. & Cesnik, C.E.S. (2005+): "Cross-Sectional Analysis of Nonhomogeneous Anisotropic Active Slender Structures" -- companion to VABS for active/composite beams.

#### Tier 2: Shell/Plate Finite Element Models

- **Description**: 2D shell elements modeling the wing skin, spar caps, shear webs
- **Coupling**: Surface spline interpolation to map between aerodynamic grid and structural mesh
- **Pros**: Better local stress prediction; captures skin buckling, shear lag
- **Cons**: Higher computational cost; more complex setup
- **Used by**: Hang et al. (2020) demonstrate compatibility with UVLM through surface spline interpolation

#### Tier 3: Reduced-Order Models (ROM)

- **Description**: Modal truncation or proper orthogonal decomposition (POD) of higher-fidelity models
- **Approach**: Compute vibration modes from FEM, then project structural dynamics onto modal subspace
- **Pros**: Orders of magnitude faster than full FEM; suitable for real-time or optimization
- **Cons**: Limited to small-to-moderate deformations unless geometrically nonlinear modes included
- **Used by**: Schwab et al. (2022) -- modal truncation with UVLM, 4-6 orders of magnitude speedup

#### Tier 4: Full 3D FEM (Reference Only)

- **Description**: Detailed solid/continuum elements with full layup modeling
- **Not used in mid-fidelity tools** -- only for detailed stress analysis or validation

### 4.2 Recommended Approach for FLUXVortex

For carbon fiber composite wings in a mid-fidelity aeroelastic framework:

1. **Primary structural model**: Timoshenko beam with composite cross-sectional properties (from VABS or analytical)
   - Capture bending (EI), torsion (GJ), extension (EA), and bending-torsion coupling (from asymmetric layup)
   - Include shear deformation (Timoshenko correction) for thick composite sections

2. **Cross-sectional analysis**: Use analytical formulas for simple sections (box beam, I-beam) or VABS for complex sections

3. **Coupling**: Surface spline interpolation (as in Hang et al. 2020 and Yang et al. 2020)

4. **Geometric nonlinearity**: Geometrically exact beam formulation (as in Murua et al. 2012) for large wing deformations

---

## 5. Recent Advances (2023-2026)

### 5.1 Key Trends

1. **GPU Acceleration of Vortex Methods**:
   - FLUXVortex (2026): GPU Biot-Savart via NVIDIA Warp, 2.4-11x speedup on UVLM
   - General trend: GPU acceleration of N-body vortex interactions (O(N^2) -> parallel)
   - Opportunity: GPU acceleration of the coupled aeroelastic system (not just aerodynamics)

2. **Hybrid Panel-Particle Methods**:
   - FLUXVortex: Near-field panels + far-field VPM particles
   - Alvarez et al. (FLOWVPM): VLM panels generating VPM wake particles
   - Trend: Combining panel method accuracy with particle method flexibility

3. **Geometrically Nonlinear Aeroelasticity**:
   - Dussler & Palacios (2023): Enhanced UVLM with geometric nonlinearities
   - Yang et al. (2020): Geometrically exact boundary conditions
   - Pudasaini et al. (2026): Nonlinear multi-body dynamics with UVLM
   - Consensus: Geometric nonlinearity essential for high-aspect-ratio composite wings

4. **Machine Learning for Aeroelasticity**:
   - Surrogate models for flutter prediction
   - Physics-informed neural networks for reduced-order aerodynamics
   - Still emerging; no major vortex-method-specific ML work yet

5. **Morphing Wing Aeroelasticity**:
   - Pudasaini et al. (2026): Trajectory optimization with aeroservoelastic models
   - Trend: Active control of wing shape through aeroelastic coupling

### 5.2 Computational Cost Landscape

| Method | Typical Grid | Time/Step | Flutter Analysis |
|--------|-------------|-----------|------------------|
| DLM (Doublet Lattice) | ~200 panels | ~0.01s | Linear, frequency domain |
| UVLM (CPU) | ~500 panels | ~0.1-1s | Nonlinear, time domain |
| UVLM (GPU, FLUXVortex) | ~500 panels | ~0.01-0.1s | Nonlinear, time domain |
| UVLM + Beam (CPU) | ~500 panels + ~50 beam nodes | ~1-10s | Nonlinear, time domain |
| UVLM + Beam (GPU target) | ~500 panels + ~50 beam nodes | ~0.1-1s | Nonlinear, time domain |
| CFD/CSD (RANS) | ~1M cells | ~minutes-hours | Full fidelity |

**Key Insight**: GPU-accelerated UVLM+beam aeroelastic simulation can achieve ~100x speedup over CPU UVLM+beam, enabling real-time flutter prediction and optimization.

---

## 6. Key Aeroelastic Benchmarks

### 6.1 AGARD 445.6 Wing

| Property | Value |
|----------|-------|
| **Type** | 45-degree swept wing, NACA 65A004 airfoil |
| **Material** | Mahogany (weakened for lower flutter speed) |
| **Aspect Ratio** | 1.65 (quarter-chord sweep 45 deg, taper ratio 0.66) |
| **Status** | Standard benchmark for transonic flutter; also used for subsonic validation |
| **Data Source** | Yates (1988), AGARD-R-765 |
| **Relevance** | Moderate -- mainly used for CFD/CSD validation; less common for VLM methods since it involves transonic flow. However, subsonic flutter data is available. |

**Note**: For UVLM validation, the AGARD 445.6 is useful only at subsonic Mach numbers (M < 0.5) where potential flow assumptions hold.

### 6.2 Goland Wing

| Property | Value |
|----------|-------|
| **Type** | Rectangular, untapered, unswept cantilevered wing |
| **Aspect Ratio** | 10 (chord = 6 ft, semi-span = 20 ft) |
| **Structural Model** | Uniform beam properties: EI, GJ, mass per unit length, CG/EA offset |
| **Flutter Speed** | ~450 ft/s at sea level |
| **Status** | Classic benchmark for aeroelastic solvers |
| **Data Source** | Goland & Luke (1948); numerous computational studies |
| **Relevance** | **HIGH** -- This is the ideal first validation case for FLUXVortex aeroelasticity. Simple geometry, well-documented structural properties, extensive reference solutions from VLM, DLM, and CFD methods. |

**Why Goland Wing for FLUXVortex**:
- Rectangular wing -> easy to mesh with UVLM panels
- Uniform beam properties -> simple structural model
- Well-known flutter boundary -> clear validation target
- Available reference: Beran et al. (multiple), numerous AIAA papers comparing VLM flutter prediction

### 6.3 HIRENASD (High Reynolds Number Aerostructural Dynamics)

| Property | Value |
|----------|-------|
| **Type** | Wing-body configuration, half-model in wind tunnel |
| **Aspect Ratio** | ~9 |
| **Flow Conditions** | Transonic, high Reynolds number |
| **Status** | Challenging benchmark for CFD/CSD coupling |
| **Relevance** | Low for UVLM -- transonic conditions exceed potential flow validity |

### 6.4 Pazy Wing Benchmark

| Property | Value |
|----------|-------|
| **Type** | Very flexible wing wind tunnel model |
| **Aspect Ratio** | ~20 (very high) |
| **Deformation** | Large static deformation (tip deflection ~30% semi-span) |
| **Status** | Active benchmark in 3rd Aeroelastic Prediction Workshop (AePW3) |
| **Key Papers** | Ritter et al. (2021, citations: 21), Mertens et al. (2023, citations: 7) |
| **Relevance** | **HIGH** for very flexible wing validation. Tests geometrically nonlinear aeroelastic capability. |

**Why Pazy Wing for FLUXVortex**:
- Very flexible wing -> tests geometric nonlinearity handling
- Well-documented experimental data
- Active community benchmark with ongoing validation efforts
- Subsonic conditions suitable for UVLM

### 6.5 Recommended Validation Sequence for FLUXVortex

1. **Goland Wing** -- Flutter boundary prediction (linear regime)
2. **Pazy Wing** -- Large deformation static aeroelasticity
3. **Simple cantilevered flat plate** -- Analytical beam+UVLM comparison
4. **Composite beam box section** -- Validation of composite structural model against Abaqus/ANSYS

---

## 7. Open-Source Implementations

### 7.1 Aeroelastic Solvers

| Project | Language | Aero Model | Structural Model | Coupling | URL |
|---------|----------|------------|------------------|----------|-----|
| **PteraSoftware** | Python | UVLM | None | None | [GitHub](https://github.com/camUrban/PteraSoftware) |
| **FLOWVLM/FLOWVPM** | Julia | VLM + VPM | None | None | [GitHub](https://github.com/byuflowlab/FLOWVLM) |
| **FLUXVortex** | Python | UVLM + VPM (hybrid, GPU) | None (yet) | None (yet) | Local |
| **SHARPy** | Python | UVLM + VLM | Nonlinear beam (Geometrically exact) | Partitioned | [GitHub](https://github.com/ImperialCollegeLondon/sharpy) |
| **MACH framework** | Python | VLM | Beam FE | Loosely coupled | University of Michigan |
| **SUmb/Aerostruct** | Fortran/Python | Euler/RANS | Shell/beam | Loosely coupled | MDO Lab, UMich |
| **DUST** | C++ | UVLM + VPM | External FE | API coupling | Politecnico di Milano |
| **OpenAeroStruct** | Python | VLM | Beam FE | Loosely coupled | [GitHub](https://github.com/mdolab/OpenAeroStruct) |
| **VABS** | Fortran | N/A (structural only) | Composite beam cross-section | N/A | [AnalySwift](https://analyswift.com/vabs/) |

### 7.2 Key Open-Source Tools for FLUXVortex Integration

**SHARPy** (Imperial College London):
- Most relevant existing framework for FLUXVortex aeroelasticity
- Combines UVLM with geometrically nonlinear beam models
- Developed by Palacios group (same group as Murua 2012 and Dussler 2023)
- Well-validated on highly flexible aircraft configurations
- **Could serve as structural solver reference or even backend for FLUXVortex**

**OpenAeroStruct** (MDO Lab, University of Michigan):
- VLM + beam FE in pure Python (NumPy-based)
- Designed for MDO optimization, not high-fidelity simulation
- Good for initial testing and comparison

**DUST** (Politecnico di Milano):
- UVLM + VPM hybrid in C++
- Well-maintained, documented
- Closest existing tool to FLUXVortex's hybrid approach
- Includes some structural coupling capability

---

## 8. Synthesis and Gap Analysis for FLUXVortex

### 8.1 What Exists

1. **UVLM aeroelasticity is well-established** (Murua 2012, 304 citations): The coupling of UVLM with nonlinear beam models for flexible aircraft simulation has a proven track record spanning 15+ years.

2. **Hybrid panel-particle methods exist** (FLUXVortex, FLOWVPM, DUST): Combining near-field panels with far-field VPM particles is a validated concept.

3. **GPU acceleration of vortex methods exists** (FLUXVortex): Biot-Savart computations on GPU via NVIDIA Warp are demonstrated with machine-precision accuracy.

4. **Composite beam modeling tools exist** (VABS, SHARPy): Cross-sectional analysis and geometrically exact beam formulations are mature.

### 8.2 What Does NOT Exist (The Gap)

1. **No GPU-accelerated aeroelastic UVLM+VPM solver**: FLUXVortex has GPU-accelerated UVLM+VPM aerodynamics, but no structural coupling. No published work combines GPU-accelerated vortex methods with structural dynamics for aeroelasticity.

2. **No PteraSoftware aeroelastic extension**: PteraSoftware remains purely aerodynamic. FLUXVortex would be the first to add structural coupling to this codebase.

3. **No composite wing aeroelasticity in mid-fidelity vortex tools**: While composite beam models exist (VABS), they have not been integrated with GPU-accelerated UVLM solvers.

4. **No real-time aeroelastic prediction using vortex methods**: The combination of GPU-accelerated aerodynamics + efficient beam solvers could enable near-real-time flutter prediction.

### 8.3 Recommended Implementation Path for FLUXVortex

**Phase 1: Linear Beam + UVLM Coupling (Euler-Bernoulli)**
- Implement 1D Euler-Bernoulli beam FE (bending + torsion)
- Partitioned coupling: UVLM aerodynamic loads -> beam solver -> deformed geometry -> UVLM
- Validation: Goland wing flutter boundary

**Phase 2: Timoshenko Beam + Geometric Nonlinearity**
- Upgrade to Timoshenko beam (shear deformation)
- Geometrically exact beam formulation for large deformations
- Surface spline interpolation for aero-structural coupling
- Validation: Pazy wing static aeroelasticity

**Phase 3: Composite Cross-Section Integration**
- Integrate VABS or analytical composite cross-section analysis
- Anisotropic beam stiffness (bending-torsion coupling from asymmetric layup)
- Validation: Composite box beam comparison with Abaqus

**Phase 4: GPU-Accelerated Coupled System**
- Move beam solver to GPU (Warp kernel for beam FE assembly/solve)
- GPU-accelerated quasi-Newton coupling (inspired by Schubert et al. 2024)
- Target: 10-100x speedup over CPU UVLM+beam aeroelastic simulation

### 8.4 Estimated Computational Feasibility

| Component | Current (CPU) | Target (GPU) | Speedup |
|-----------|---------------|--------------|---------|
| UVLM Biot-Savart | ~0.1-1s/step | ~0.01-0.1s/step | 10-100x |
| Beam FE solve | ~0.001-0.01s/step | ~0.0001-0.001s/step | 10x |
| Wake convection | ~0.1-1s/step | ~0.01-0.1s/step | 10x |
| VPM particle update | ~1-10s/step (7000 particles) | ~0.1-1s/step | 10-100x |
| **Total coupled step** | **~1-10s** | **~0.1-1s** | **~10-100x** |

---

## 9. References

### Core UVLM Aeroelasticity Papers

1. Murua, J., Palacios, R., & Graham, J.M.R. (2012). "Applications of the Unsteady Vortex-Lattice Method in Aircraft Aeroelasticity and Flight Dynamics." *Progress in Aerospace Sciences*, 52, 66-82. DOI: 10.1016/J.PAEROSCI.2012.06.001

2. Dussler, S. & Palacios, R. (2023). "Enhanced Unsteady Vortex Lattice Aerodynamics for Nonlinear Flexible Aircraft Dynamic Simulation." *AIAA Journal*, 61(12). DOI: 10.2514/1.J063174

3. Schubert, J., Steinbach, M.C., Hente, C., Martens, D., & Schuster, D. (2024). "Accelerating Aeroelastic UVLM Simulations by Inexact Newton Algorithms." arXiv: 2403.15286

4. Yang, L., Xie, C., & Yang, C. (2020). "Geometrically Exact Vortex Lattice and Panel Methods in Static Aeroelasticity of Very Flexible Wing." *Proc. IMechE Part G*, 234(5). DOI: 10.1177/0954410019885238

5. Hang, X., Su, W., Fei, Q., & Jiang, D. (2020). "Analytical Sensitivity Analysis of Flexible Aircraft with the Unsteady Vortex-Lattice Aerodynamic Theory." *Aerospace Science and Technology*, 97, 105612. DOI: 10.1016/j.ast.2019.105612

6. Su, W., Huang, Y., & Hammerton, J. (2017). "Nonlinear Aeroelasticity of Highly Flexible Joined-Wing Aircraft using Unsteady Vortex-Lattice Method." AIAA 2017-1353. DOI: 10.2514/6.2017-1353

7. Pudasaini, S., Smith, P., & Huang, D. (2026). "Trajectory Optimization of Morphing Aerial Vehicles Based on Mid-Fidelity Aeroservoelastic Models." arXiv: 2605.02076

### Vortex Methods + FSI

8. Luo, M., Wu, Z., & Yang, C. (2023). "Strongly Coupled Fluid-Structure Interaction Analysis of Aquatic Flapping Wings Based on Flexible Multibody Dynamics and the Modified Unsteady Vortex Lattice Method." *Ocean Engineering*, 287, 114921. DOI: 10.1016/j.oceaneng.2023.114921

9. Schwab, R., Reade, J., & Jankauski, M.A. (2022). "Quasi Three-Dimensional Deformable Blade Element and Unsteady Vortex Lattice Reduced-Order Modeling of Fluid-Structure Interaction in Flapping Wings." *Physics of Fluids*, 34, 121904. DOI: 10.1063/5.0129128

10. Bhosale, Y., Parthasarathy, T., & Gazzola, M. (2020). "A Remeshed Vortex Method for Mixed Rigid/Soft Body Fluid-Structure Interaction." *Journal of Computational Physics*, 440, 110577. DOI: 10.1016/j.jcp.2021.110577

11. Tesfaye, S., Kavrakov, I., & Morgenthal, G. (2022). "Numerical Investigation of the Nonlinear Interaction Between the Sinusoidal Motion-Induced and Gust-Induced Forces Acting on Bridge Decks." *Journal of Fluids and Structures*, 112, 103680. DOI: 10.1016/j.jfluidstructs.2022.103680

12. D'Adamo, J., Collaud, M., Sosa, R., & Godoy-Diana, R. (2022). "Wake and Aeroelasticity of a Flexible Pitching Foil." *Bioinspiration & Biomimetics*, 17, 045002. DOI: 10.1088/1748-3190/ac6d96

### Composite Beam Modeling

13. Hodges, D.H. (2006). *Nonlinear Composite Beam Theory*. AIAA Education Series.

14. Yu, W. & Hodges, D.H. (2004+). "VABS: Variational Asymptotic Beam Sectional Analysis." Georgia Tech / Utah State University. Available: analyswift.com/vabs

15. Palacios, R. & Cesnik, C.E.S. (2005). "Cross-Sectional Analysis of Nonhomogeneous Anisotropic Active Slender Structures." *AIAA Journal*, 43(12), 2624-2638.

### Aeroelastic Benchmarks

16. Yates, E.C. (1988). "AGARD Standard Aeroelastic Configurations for Dynamic Response. Candidate Configuration I.- Wing 445.6." *AGARD-R-765*. NASA TM-100492.

17. Goland, M. & Luke, Y.L. (1948). "The Flutter of a Uniform Wing." *Journal of Applied Mechanics*, 15, A13-A20.

18. Ritter, M., Hilger, J., & Zimmer, M. (2021). "Static and Dynamic Simulations of the Pazy Wing Aeroelastic Benchmark by Nonlinear Potential Aerodynamics and detailed FE Model." AIAA 2021-1713. DOI: 10.2514/6.2021-1713

### Open-Source Tools

19. Urban, C. "PteraSoftware: Unsteady Vortex Lattice Method Solver in Python." GitHub: camUrban/PteraSoftware

20. Alvarez, E.J. "FLOWVLM / FLOWVPM: Vortex Lattice and Vortex Particle Methods in Julia." GitHub: byuflowlab/FLOWVLM

21. SHARPy: Simulation of High Aspect Ratio Planes. GitHub: ImperialCollegeLondon/sharpy

22. Jasa, J.P. et al. "OpenAeroStruct: An Open-Source Low-Fidelity Aeroservoelastic Analysis and Optimization Tool." GitHub: mdolab/OpenAeroStruct

23. DUST: Simulation tool for lifting-line, VLM, UVLM and VPM aerodynamics. Politecnico di Milano.

---

*Survey prepared for the FLUXVortex project: GPU-Accelerated Hybrid Panel-Particle Vortex Method Solver.*
