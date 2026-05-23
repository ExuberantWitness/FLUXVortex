"""
Full comparison experiment: Ring vs VPM strategies vs Theodorsen
3 cycles, k = 0.5 / 0.2 / 0.1

Strategies:
  A-fixed: 4-particle ring with PROPER geometry
           (front at TE, back at TE+dl, side legs connecting)
  B: Single front-leg, kf=1.0 (best from pilot)
  D-fixed: Line-vortex segments with regularized core radius
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
    return hankel2(1,k)/(hankel2(1,k)+1j*hankel2(0,k))

def theo_cl(k, h0c, omega, t, c=1.0):
    U = omega*c/(2*k) if k>1e-10 else 1e10
    h0=h0c*c
    return np.real(np.pi*(-h0*omega**2*np.sin(omega*t))*c/(2*U**2)
                   -2*np.pi*theo_Ck(k)*h0*omega*np.cos(omega*t)/U)

# ─── Wing builders ───
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
                control_surface_hinge_point=0.75, control_surface_deflection=0.0)])

def make_plunge(wing, h0, period, V=10.0):
    ap = ps.geometry.airplane.Airplane(wings=[wing], name="P")
    op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=V, alpha=0, beta=0)
    wcs=[ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
        base_wing_cross_section=w) for w in wing.wing_cross_sections]
    wm=ps.movements.wing_movement.WingMovement(
        base_wing=wing, wing_cross_section_movements=wcs,
        ampLer_Gs_Cgs=(0,0,h0), periodLer_Gs_Cgs=(0,0,period),
        spacingLer_Gs_Cgs=("sine","sine","sine"), phaseLer_Gs_Cgs=(0,0,0))
    am=ps.movements.airplane_movement.AirplaneMovement(base_airplane=ap, wing_movements=[wm])
    opm=ps.movements.operating_point_movement.OperatingPointMovement(base_operating_point=op)
    return ap, ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        num_cycles=3, delta_time=period/50)

def extract_cl(sol, mv):
    first=sol.unsteady_problem.first_results_step; dt=mv.delta_time
    ts,cls=[],[]
    for step in range(first, sol.num_steps):
        for ap in sol.steady_problems[step].airplanes:
            c=ap.forceCoefficients_W
            if c is not None: ts.append(step*dt); cls.append(-c[2])
    return np.array(ts), np.array(cls)

# ─── Base VPM solver class ───
class BaseVPMSolver(ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver):
    def __init__(self, prob, **kw):
        super().__init__(prob)
        self._vpm = VortexParticleField(max_particles=200000, nu=0, rlxf=0.3)

    def _calculate_wake_wing_influences(self):
        if self._current_step==0 or self._vpm.np==0:
            self._currentStackWakeWingInfluences__E=np.zeros(self.num_panels); return
        cp=self.stackCpp_GP1_CgP1
        U=self._vpm.induce_velocity_at(cp)
        self._currentStackWakeWingInfluences__E=np.einsum("ij,ij->i",U,self.stackUnitNormals_GP1)

    def _populate_next_airplanes_wake(self):
        if self._current_step>0: self._shed(); self._advect()
        self._prescribed_wake=True
        self._populate_next_airplanes_wake_vortex_points()
        self._populate_next_airplanes_wake_vortices()

    def _advect(self):
        dt=self.delta_time; op=self.current_operating_point
        V=op.vCg__E*np.array([np.cos(np.radians(op.beta))*np.cos(np.radians(op.alpha)),
                               np.sin(np.radians(op.beta)),-np.cos(np.radians(op.beta))*np.sin(np.radians(op.alpha))])
        self._vpm.advect_rk3(dt, lambda X: np.broadcast_to(V,X.shape).copy(),
                              stretch=False, free_wake=False)

    def _get_inf(self):
        op=self.current_operating_point
        Vvec=op.vCg__E*np.array([np.cos(np.radians(op.beta))*np.cos(np.radians(op.alpha)),
                                  np.sin(np.radians(op.beta)),-np.cos(np.radians(op.beta))*np.sin(np.radians(op.alpha))])
        V=np.linalg.norm(Vvec)
        return Vvec, V, Vvec/max(V,1e-10), self.delta_time, V*self.delta_time


# ─── Strategy A-fixed: 4-particle ring with PROPER geometry ───
class StrategyA_Fixed(BaseVPMSolver):
    """
    4 particles per TE panel forming a proper wake ring:
      - Front leg at TE, direction = back_vec, strength = gamma
      - Back leg at TE+dl, direction = -back_vec, strength = -gamma
      - Left side leg connecting front-left to back-left
      - Right side leg connecting front-right to back-right
    sigma = dl for all particles.
    """
    def __init__(self, prob, **kw):
        super().__init__(prob)

    def _shed(self):
        strength=self._current_bound_vortex_strengths
        if strength is None: return
        Vvec,V,infD,dt,dl = self._get_inf()
        if V<1e-10: return
        sigma = dl  # same as FLOWVLM
        offset_back = infD * dl  # back leg position offset

        pos_list,gam_list,sig_list=[],[],[]
        idx=0
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                panels=wing.panels; nc=wing.num_chordwise_panels; ns=wing.num_spanwise_panels
                for i in range(nc):
                    for j in range(ns):
                        p=panels[i,j]; g=strength[idx]; idx+=1
                        if not p.is_trailing_edge or abs(g)<1e-15: continue

                        bl=p.Blpp_GP1_CgP1; br=p.Brpp_GP1_CgP1
                        fl=p.Flpp_GP1_CgP1; fr=p.Frpp_GP1_CgP1
                        back_vec=br-bl  # spanwise

                        # Back corners shifted downstream
                        bl_b = bl + offset_back
                        br_b = br + offset_back

                        # Front leg (at TE): BL → BR, Gamma = +back_vec * gamma
                        pos_list.append(0.5*(bl+br))
                        gam_list.append(back_vec * g)
                        sig_list.append(sigma)

                        # Back leg (downstream): BR_b → BL_b, Gamma = -back_vec * gamma
                        pos_list.append(0.5*(bl_b+br_b))
                        gam_list.append(-back_vec * g)
                        sig_list.append(sigma)

                        # Left side leg: BL → BL_b (streamwise), Gamma = infD*dl * gamma
                        pos_list.append(0.5*(bl+bl_b))
                        gam_list.append(offset_back * g)
                        sig_list.append(sigma)

                        # Right side leg: BR_b → BR (upstream), Gamma = -infD*dl * gamma
                        pos_list.append(0.5*(br+br_b))
                        gam_list.append(-offset_back * g)
                        sig_list.append(sigma)

        if pos_list:
            self._vpm.add_particles_batch(np.array(pos_list),np.array(gam_list),np.array(sig_list))


# ─── Strategy B: Single front-leg, kf=1.0 ───
class StrategyB(BaseVPMSolver):
    def __init__(self, prob, kf=1.0, **kw):
        super().__init__(prob)
        self._kf = kf

    def _shed(self):
        strength=self._current_bound_vortex_strengths
        if strength is None: return
        Vvec,V,infD,dt,dl = self._get_inf()
        if V<1e-10: return
        sigma=dl*0.5; off=infD*dl*0.25

        pos_list,gam_list,sig_list=[],[],[]
        idx=0
        for ap in self.current_airplanes:
            for w in ap.wings:
                panels=w.panels; nc=w.num_chordwise_panels; ns=w.num_spanwise_panels
                for i in range(nc):
                    for j in range(ns):
                        p=panels[i,j]; g=strength[idx]; idx+=1
                        if not p.is_trailing_edge or abs(g)<1e-15: continue
                        bl=p.Blpp_GP1_CgP1; br=p.Brpp_GP1_CgP1
                        pos_list.append(0.5*(bl+br)+off)
                        gam_list.append(-(br-bl)*g*self._kf)
                        sig_list.append(sigma)
        if pos_list:
            self._vpm.add_particles_batch(np.array(pos_list),np.array(gam_list),np.array(sig_list))


# ─── Experiment runner ───
def run_k(k, h0c=0.1, V=10.0, chord=1.0):
    omega=2*k*V/chord; period=2*np.pi/omega; h0=h0c*chord

    print(f"\n{'='*70}")
    print(f"  k={k:.2f}  period={period:.4f}s  C(k)=|{abs(theo_Ck(k)):.4f}|∠{np.degrees(np.angle(theo_Ck(k))):.1f}°")
    print(f"{'='*70}")

    # Ring baseline
    print(f"  Ring-wake ... ", end="", flush=True)
    w1=make_wing(chord); _,mv1=make_plunge(w1,h0,period,V)
    prob1=ps.problems.UnsteadyProblem(movement=mv1,only_final_results=False)
    sol1=ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(prob1)
    sol1.run(prescribed_wake=True,calculate_streamlines=False,show_progress=False)
    t_r,cl_r=extract_cl(sol1,mv1)
    print(f"done")

    # VPM strategies
    strategies = [
        ("A-fixed: 4-ring proper geometry", StrategyA_Fixed, {}),
        ("B: 1-front kf=1.0", StrategyB, {'kf': 1.0}),
        ("B: 1-front kf=0.8", StrategyB, {'kf': 0.8}),
    ]

    results = {}
    for label, cls, kw in strategies:
        print(f"  {label:<35s} ... ", end="", flush=True)
        w2=make_wing(chord); _,mv2=make_plunge(w2,h0,period,V)
        prob2=ps.problems.UnsteadyProblem(movement=mv2,only_final_results=False)
        try:
            sol2=cls(prob2, **kw)
            sol2.run(prescribed_wake=False,calculate_streamlines=False,show_progress=False)
            t_v,cl_v=extract_cl(sol2,mv2)
            np_count=sol2._vpm.np
            cl_max=np.max(np.abs(cl_v))

            if cl_max > 100:
                print(f"BLOWUP (|CL|={cl_max:.1e})")
                results[label]={'status':'BLOWUP','cl_max':cl_max}
                continue

            # Discard first 2 cycles
            t_trans=2*period
            mr=t_r>t_trans; mv_=t_v>t_trans
            n=min(np.sum(mr),np.sum(mv_))
            ring_amp=(np.max(cl_r[mr])-np.min(cl_r[mr]))/2
            vpm_amp=(np.max(cl_v[mv_])-np.min(cl_v[mv_]))/2
            corr=np.corrcoef(cl_r[mr][:n],cl_v[mv_][:n])[0,1]
            rmse=np.sqrt(np.mean((cl_r[mr][:n]-cl_v[mv_][:n])**2))
            ratio=vpm_amp/ring_amp if ring_amp>1e-10 else 0
            print(f"amp={vpm_amp:.4f} ({ratio:.1%}) corr={corr:.4f} np={np_count}")
            results[label]={'status':'OK','amp':vpm_amp,'ratio':ratio,'corr':corr,'rmse':rmse,'np':np_count}
        except Exception as e:
            print(f"ERROR: {e}")
            results[label]={'status':'ERROR','msg':str(e)}

    # Theodorsen
    t_trans=2*period; mr=t_r>t_trans
    cl_th=theo_cl(k,h0c,omega,t_r,chord)
    ring_amp=(np.max(cl_r[mr])-np.min(cl_r[mr]))/2
    theo_amp=(np.max(cl_th[mr])-np.min(cl_th[mr]))/2
    ring_theo=ring_amp/theo_amp if theo_amp>1e-10 else 0
    ring_corr=np.corrcoef(cl_r[mr],cl_th[mr])[0,1]

    print(f"\n  --- k={k:.2f} Summary ---")
    print(f"  {'Method':<35s} {'Amp':>8s} {'vs Ring':>8s} {'vs Theo':>8s} {'Corr':>6s}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    print(f"  {'Theodorsen':<35s} {theo_amp:8.4f}")
    print(f"  {'Ring-wake':<35s} {ring_amp:8.4f} {'100.0%':>8s} {ring_theo:8.3f} {ring_corr:6.3f}")
    for label in [l for l,_ ,_ in strategies]:
        r=results[label]
        if r['status']=='OK':
            vt=r['amp']/theo_amp if theo_amp>1e-10 else 0
            print(f"  {label:<35s} {r['amp']:8.4f} {r['ratio']:8.1%} {vt:8.3f} {r['corr']:6.3f}")
        else:
            print(f"  {label:<35s} {r['status']:>8s}")

    return {'k':k,'t_r':t_r,'cl_r':cl_r,'cl_th':cl_th,'ring_amp':ring_amp,'theo_amp':theo_amp,'vpm':results}


# ─── Main ───
if __name__=='__main__':
    print("="*70)
    print("Full comparison: Ring vs VPM strategies vs Theodorsen")
    print("NACA 0012, AR=10, h0/c=0.1, nc=10, ns=6, 3 cycles")
    print("="*70)

    all_res=[]
    for k in [0.5, 0.2, 0.1]:
        r=run_k(k)
        all_res.append(r)

    # Grand summary
    print(f"\n{'='*70}")
    print("  Grand Summary (3 cycles, discard first 2)")
    print(f"{'='*70}")
    print(f"  {'k':>5s} | {'Ring/Theo':>9s} | {'A-fixed':>12s} | {'B kf=1.0':>12s} | {'B kf=0.8':>12s}")
    print(f"  {'─'*5}─┼─{'─'*9}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*12}")

    strategies_labels = ["A-fixed: 4-ring proper geometry", "B: 1-front kf=1.0", "B: 1-front kf=0.8"]
    for r in all_res:
        k=r['k']
        rt=r['ring_amp']/r['theo_amp'] if r['theo_amp']>1e-10 else 0
        line=f"  {k:5.2f} | {rt:9.3f} |"
        for label in strategies_labels:
            v=r['vpm'].get(label,{})
            if v.get('status')=='OK':
                vt=v['amp']/r['theo_amp'] if r['theo_amp']>1e-10 else 0
                line+=f" {vt:6.3f} c={v['corr']:.3f} |"
            else:
                line+=f" {'['+v.get('status','?')+']':>12s} |"
        print(line)
    print(f"{'='*70}")
