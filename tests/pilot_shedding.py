"""
Pilot test: 多粒子脱落策略对比

策略 A: 4-particle ring (4 legs, adaptive sigma)
策略 B: 修正单粒子 (kf sweep, 找最优)
策略 D: 线涡段粒子 (line-vortex kernel instead of Gaussian)
策略 E: 前腿 + trailing vortex 组合

对 k=0.5 做快速测试 (1 cycle, 50 steps)
"""
import sys, os, time
import numpy as np
from scipy.special import hankel2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
np.trapezoid = np.trapz

import pterasoftware as ps
ps.set_up_logging(level="Warning")
from fluxvortex.particles import VortexParticleField

# ─── Theodorsen ───
def theo_Ck(k):
    if k < 1e-10: return 1.0+0j
    H0, H1 = hankel2(0,k), hankel2(1,k)
    return H1/(H1+1j*H0)

def theo_cl(k, h0c, omega, t, c=1.0):
    U = omega*c/(2*k) if k>1e-10 else 1e10
    h0 = h0c*c
    return np.real(np.pi*(-h0*omega**2*np.sin(omega*t))*c/(2*U**2)
                   - 2*np.pi*theo_Ck(k)*h0*omega*np.cos(omega*t)/U)

# ─── Wing builder ───
def make_wing(chord=1.0, half_span=5.0, nc=10, ns=6):
    return ps.geometry.wing.Wing(
        name="Wing", symmetric=True,
        symmetryNormal_G=(0,1,0), symmetryPoint_G_Cg=(0,0,0),
        num_chordwise_panels=nc, chordwise_spacing="uniform",
        wing_cross_sections=[
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=ns, spanwise_spacing="uniform", chord=chord,
                Lp_Wcsp_Lpp=(0,0,0), airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75, control_surface_deflection=0.0),
            ps.geometry.wing_cross_section.WingCrossSection(
                num_spanwise_panels=None, spanwise_spacing=None, chord=chord,
                Lp_Wcsp_Lpp=(0,half_span,0), airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                control_surface_symmetry_type="symmetric",
                control_surface_hinge_point=0.75, control_surface_deflection=0.0),
        ])

def make_plunge(wing, h0, period, V=10.0):
    ap = ps.geometry.airplane.Airplane(wings=[wing], name="P")
    op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=V, alpha=0, beta=0)
    wcs = [ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
        base_wing_cross_section=w) for w in wing.wing_cross_sections]
    wm = ps.movements.wing_movement.WingMovement(
        base_wing=wing, wing_cross_section_movements=wcs,
        ampLer_Gs_Cgs=(0,0,h0), periodLer_Gs_Cgs=(0,0,period),
        spacingLer_Gs_Cgs=("sine","sine","sine"), phaseLer_Gs_Cgs=(0,0,0))
    am = ps.movements.airplane_movement.AirplaneMovement(base_airplane=ap, wing_movements=[wm])
    opm = ps.movements.operating_point_movement.OperatingPointMovement(base_operating_point=op)
    return ap, ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        num_cycles=1, delta_time=period/50)

def extract_cl(sol, mv):
    first = sol.unsteady_problem.first_results_step
    dt = mv.delta_time
    ts, cls = [], []
    for step in range(first, sol.num_steps):
        for ap in sol.steady_problems[step].airplanes:
            c = ap.forceCoefficients_W
            if c is not None:
                ts.append(step*dt); cls.append(-c[2])
    return np.array(ts), np.array(cls)

# ─── Line-vortex kernel (replaces Gaussian-erf for strategy D) ───
def velocity_from_line_segments(targets, starts, ends, strengths, rc=1e-10):
    """Biot-Savart for finite line vortex segments (PteraSoftware-style)."""
    N = targets.shape[0]
    M = starts.shape[0]
    out = np.zeros((N, 3))
    for i in range(N):
        p = targets[i]
        for j in range(M):
            s, e, g = starts[j], ends[j], strengths[j]
            r0 = e - s
            r0_sq = np.dot(r0, r0)
            r0_len = np.sqrt(r0_sq)
            if r0_len < 1e-15: continue
            r1 = s - p; r2 = e - p
            r3 = np.cross(r1, r2)
            r1_len = np.linalg.norm(r1)
            r2_len = np.linalg.norm(r2)
            if r1_len < r0_len*1e-10 or r2_len < r0_len*1e-10: continue
            r3_sq = np.dot(r3, r3)
            r3_len = np.sqrt(r3_sq)
            if r3_len < 1e-10 * r1_len * r2_len: continue
            rc_sq = rc*rc
            c1 = g / (4*np.pi)
            c2 = r0_sq * rc_sq
            c3 = np.dot(r1, r2)
            c4 = c1 * (r1_len+r2_len) * (r1_len*r2_len-c3) / (r1_len*r2_len*(r3_sq+c2))
            out[i] += c4 * r3
    return out


# ─── Strategy A: 4-particle ring with adaptive sigma ───
class StrategyA_Solver(ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver):
    """4 particles per TE panel: 4 legs of the wake ring, with adaptive sigma."""
    def __init__(self, prob, sigma_factor=2.0, **kw):
        super().__init__(prob)
        self._vpm = VortexParticleField(max_particles=100000, nu=0, rlxf=0.3)
        self._sigma_factor = sigma_factor  # multiply sigma to reduce cancellation

    def _calculate_wake_wing_influences(self):
        if self._current_step == 0 or self._vpm.np == 0:
            self._currentStackWakeWingInfluences__E = np.zeros(self.num_panels); return
        cp = self.stackCpp_GP1_CgP1
        U = self._vpm.induce_velocity_at(cp)
        self._currentStackWakeWingInfluences__E = np.einsum("ij,ij->i", U, self.stackUnitNormals_GP1)

    def _populate_next_airplanes_wake(self):
        if self._current_step > 0:
            self._shed_4ring()
            self._advect()
        self._prescribed_wake = True
        self._populate_next_airplanes_wake_vortex_points()
        self._populate_next_airplanes_wake_vortices()

    def _shed_4ring(self):
        strength = self._current_bound_vortex_strengths
        if strength is None: return
        op = self.current_operating_point
        Vvec = op.vCg__E * np.array([np.cos(np.radians(op.beta))*np.cos(np.radians(op.alpha)),
                                      np.sin(np.radians(op.beta)),
                                      -np.cos(np.radians(op.beta))*np.sin(np.radians(op.alpha))])
        V = np.linalg.norm(Vvec)
        if V < 1e-10: return
        infD = Vvec / V
        dt = self.delta_time; dl = V * dt
        sigma = dl * self._sigma_factor  # adaptive sigma
        off = infD * dl * 0.5

        pos_list, gam_list, sig_list = [], [], []
        idx = 0
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                panels = wing.panels
                nc, ns = wing.num_chordwise_panels, wing.num_spanwise_panels
                for i in range(nc):
                    for j in range(ns):
                        p = panels[i,j]; g = strength[idx]; idx += 1
                        if not p.is_trailing_edge or abs(g) < 1e-15: continue

                        bl = p.Blpp_GP1_CgP1; br = p.Brpp_GP1_CgP1
                        fl = p.Flpp_GP1_CgP1; fr = p.Frpp_GP1_CgP1
                        # Back leg: BL→BR (opposes bound back leg)
                        pos_list.append(0.5*(bl+br) + off)
                        gam_list.append(-(br-bl)*g)
                        sig_list.append(sigma)
                        # Front leg: FR→FL (wake front, opposes bound front)
                        pos_list.append(0.5*(fl+fr) + off)
                        gam_list.append(-(fl-fr)*g)
                        sig_list.append(sigma)
                        # Left leg: FL→BL
                        pos_list.append(0.5*(fl+bl) + off)
                        gam_list.append(-(bl-fl)*g)
                        sig_list.append(sigma)
                        # Right leg: BR→FR
                        pos_list.append(0.5*(br+fr) + off)
                        gam_list.append(-(fr-br)*g)
                        sig_list.append(sigma)
        if pos_list:
            self._vpm.add_particles_batch(np.array(pos_list), np.array(gam_list), np.array(sig_list))

    def _advect(self):
        dt = self.delta_time
        op = self.current_operating_point
        V = op.vCg__E * np.array([np.cos(np.radians(op.beta))*np.cos(np.radians(op.alpha)),
                                   np.sin(np.radians(op.beta)),
                                   -np.cos(np.radians(op.beta))*np.sin(np.radians(op.alpha))])
        self._vpm.advect_rk3(dt, lambda X: np.broadcast_to(V,X.shape).copy(),
                              stretch=False, free_wake=False)


# ─── Strategy D: Line-vortex segment particles ───
class StrategyD_Solver(ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver):
    """Single front-leg particle using LINE-VORTEX kernel (not Gaussian)."""
    def __init__(self, prob, kf=1.0, **kw):
        super().__init__(prob)
        self._kf = kf
        # Store segments as (start, end, strength) arrays
        self._seg_starts = []
        self._seg_ends = []
        self._seg_strengths = []
        self._n_segs = 0

    def _calculate_wake_wing_influences(self):
        if self._current_step == 0 or self._n_segs == 0:
            self._currentStackWakeWingInfluences__E = np.zeros(self.num_panels); return
        cp = self.stackCpp_GP1_CgP1
        starts = np.array(self._seg_starts)
        ends = np.array(self._seg_ends)
        strengths = np.array(self._seg_strengths)
        U = velocity_from_line_segments(cp, starts, ends, strengths)
        self._currentStackWakeWingInfluences__E = np.einsum("ij,ij->i", U, self.stackUnitNormals_GP1)

    def _populate_next_airplanes_wake(self):
        if self._current_step > 0:
            self._shed()
            self._advect()
        self._prescribed_wake = True
        self._populate_next_airplanes_wake_vortex_points()
        self._populate_next_airplanes_wake_vortices()

    def _shed(self):
        strength = self._current_bound_vortex_strengths
        if strength is None: return
        op = self.current_operating_point
        Vvec = op.vCg__E * np.array([np.cos(np.radians(op.beta))*np.cos(np.radians(op.alpha)),
                                      np.sin(np.radians(op.beta)),
                                      -np.cos(np.radians(op.beta))*np.sin(np.radians(op.alpha))])
        V = np.linalg.norm(Vvec)
        if V < 1e-10: return
        infD = Vvec / V
        dt = self.delta_time; dl = V * dt
        off = infD * dl * 0.5

        idx = 0
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                panels = wing.panels
                nc = wing.num_chordwise_panels
                ns = wing.num_spanwise_panels
                for i in range(nc):
                    for j in range(ns):
                        p = panels[i,j]; g = strength[idx]; idx += 1
                        if not p.is_trailing_edge or abs(g) < 1e-15: continue
                        bl = p.Blpp_GP1_CgP1; br = p.Brpp_GP1_CgP1
                        back = br - bl
                        # Line segment from BL+offset to BR+offset
                        self._seg_starts.append(bl + off)
                        self._seg_ends.append(br + off)
                        self._seg_strengths.append(-g * self._kf)
                        self._n_segs += 1

    def _advect(self):
        dt = self.delta_time
        op = self.current_operating_point
        Vvec = op.vCg__E * np.array([np.cos(np.radians(op.beta))*np.cos(np.radians(op.alpha)),
                                      np.sin(np.radians(op.beta)),
                                      -np.cos(np.radians(op.beta))*np.sin(np.radians(op.alpha))])
        V = np.linalg.norm(Vvec)
        if V < 1e-10: return
        # Advect all segments with freestream
        shift = Vvec * dt
        for i in range(self._n_segs):
            self._seg_starts[i] = self._seg_starts[i] + shift
            self._seg_ends[i] = self._seg_ends[i] + shift


# ─── Pilot runner ───
def pilot(label, solver_cls, solver_kw, k=0.5, h0c=0.1, V=10.0):
    omega = 2*k*V/1.0; period = 2*np.pi/omega; h0 = h0c*1.0
    w = make_wing()
    _, mv = make_plunge(w, h0, period, V)
    prob = ps.problems.UnsteadyProblem(movement=mv, only_final_results=False)
    sol = solver_cls(prob, **solver_kw)
    t0 = time.perf_counter()
    try:
        sol.run(prescribed_wake=False, calculate_streamlines=False, show_progress=False)
        dt_sec = time.perf_counter() - t0
        t, cl = extract_cl(sol, mv)
        cl_max = np.max(np.abs(cl))
        if cl_max > 100:
            print(f"  {label:<35s} BLOWUP (|CL|={cl_max:.1e})")
            return None
        cl_amp = (np.max(cl) - np.min(cl))/2
        t_th = t; cl_th = theo_cl(k, h0c, omega, t_th)
        theo_amp = (np.max(cl_th)-np.min(cl_th))/2
        ratio = cl_amp/theo_amp if theo_amp > 1e-10 else 0
        corr = np.corrcoef(cl, cl_th)[0,1]
        print(f"  {label:<35s} amp={cl_amp:.4f} Theo={theo_amp:.4f} ratio={ratio:.3f} corr={corr:.3f} t={dt_sec:.1f}s")
        return {'amp': cl_amp, 'theo': theo_amp, 'ratio': ratio, 'corr': corr}
    except Exception as e:
        print(f"  {label:<35s} ERROR: {e}")
        return None


# ─── Main ───
if __name__ == '__main__':
    print("="*70)
    print("Pilot: Multi-particle shedding strategies")
    print("k=0.5, h0/c=0.1, nc=10, ns=6, 1 cycle (50 steps)")
    print("="*70)

    # Ring baseline
    print("\n--- Baselines ---")
    w1 = make_wing()
    _, mv1 = make_plunge(w1, 0.1*1.0, 2*np.pi/(2*0.5*10))
    prob1 = ps.problems.UnsteadyProblem(movement=mv1, only_final_results=False)
    sol1 = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(prob1)
    sol1.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    t1, cl1 = extract_cl(sol1, mv1)
    k=0.5; omega=10.0
    cl_th1 = theo_cl(k, 0.1, omega, t1)
    ring_amp = (np.max(cl1)-np.min(cl1))/2
    theo_amp = (np.max(cl_th1)-np.min(cl_th1))/2
    print(f"  {'Ring-wake baseline':<35s} amp={ring_amp:.4f} Theo={theo_amp:.4f} ratio={ring_amp/theo_amp:.3f}")

    # Strategy A: 4-particle ring, sigma_factor sweep
    print("\n--- Strategy A: 4-particle ring (adaptive sigma) ---")
    for sf in [1.0, 2.0, 3.0, 5.0]:
        pilot(f"A: 4-ring sigma={sf}x", StrategyA_Solver, {'sigma_factor': sf})

    # Strategy B: Single front-leg, kf sweep (already known data, include for reference)
    print("\n--- Strategy B: Single front-leg (kf sweep) ---")

    class StrategyB_Solver(ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver):
        def __init__(self, prob, kf=1.0, **kw):
            super().__init__(prob)
            self._vpm = VortexParticleField(max_particles=50000, nu=0, rlxf=0.3)
            self._kf = kf
        def _calculate_wake_wing_influences(self):
            if self._current_step==0 or self._vpm.np==0:
                self._currentStackWakeWingInfluences__E=np.zeros(self.num_panels); return
            cp = self.stackCpp_GP1_CgP1
            U = self._vpm.induce_velocity_at(cp)
            self._currentStackWakeWingInfluences__E = np.einsum("ij,ij->i",U,self.stackUnitNormals_GP1)
        def _populate_next_airplanes_wake(self):
            if self._current_step>0: self._shed(); self._advect()
            self._prescribed_wake=True
            self._populate_next_airplanes_wake_vortex_points()
            self._populate_next_airplanes_wake_vortices()
        def _shed(self):
            strength=self._current_bound_vortex_strengths
            if strength is None: return
            op=self.current_operating_point
            Vvec=op.vCg__E*np.array([np.cos(np.radians(op.beta))*np.cos(np.radians(op.alpha)),
                                     np.sin(np.radians(op.beta)),-np.cos(np.radians(op.beta))*np.sin(np.radians(op.alpha))])
            V=np.linalg.norm(Vvec);
            if V<1e-10: return
            infD=Vvec/V; dt=self.delta_time; dl=V*dt; sigma=dl*0.5; off=infD*dl*0.25
            pos,gam,sig=[],[],[]; idx=0
            for ap in self.current_airplanes:
                for w in ap.wings:
                    ps2=w.panels; nc=w.num_chordwise_panels; ns=w.num_spanwise_panels
                    for i in range(nc):
                        for j in range(ns):
                            p=ps2[i,j]; g=strength[idx]; idx+=1
                            if not p.is_trailing_edge or abs(g)<1e-15: continue
                            bl=p.Blpp_GP1_CgP1; br=p.Brpp_GP1_CgP1
                            pos.append(0.5*(bl+br)+off)
                            gam.append(-(br-bl)*g*self._kf)
                            sig.append(sigma)
            if pos: self._vpm.add_particles_batch(np.array(pos),np.array(gam),np.array(sig))
        def _advect(self):
            dt=self.delta_time; op=self.current_operating_point
            V=op.vCg__E*np.array([np.cos(np.radians(op.beta))*np.cos(np.radians(op.alpha)),
                                  np.sin(np.radians(op.beta)),-np.cos(np.radians(op.beta))*np.sin(np.radians(op.alpha))])
            self._vpm.advect_rk3(dt, lambda X: np.broadcast_to(V,X.shape).copy(), stretch=False, free_wake=False)

    for kf in [0.8, 1.0, 1.2]:
        pilot(f"B: 1-front kf={kf}", StrategyB_Solver, {'kf': kf})

    # Strategy D: Line-vortex segment (no Gaussian)
    print("\n--- Strategy D: Line-vortex segment particles ---")
    for kf in [1.0]:
        pilot(f"D: line-seg kf={kf}", StrategyD_Solver, {'kf': kf})

    print("\n" + "="*70)
    print("Done. Compare ratio (closer to ring ratio 0.929 = best).")
