import logging, math, json, sys
import torch
from fluxvortex.aero.v5m.stepper import RigidV5MStepper
from fluxvortex.cases.izraelevitz_kinematics import build_izra_reference_grid
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig
from fluxvortex.warp_fsi.rigid_flux_v5m_native import RigidNativeV5MSolver, RigidV5MSurface

D = "cuda:0"; C = 0.0688; B = 0.1375; DT = 0.01*C/5.0; QS = 0.5*1.208*25*C*B
def law(alpha):
    th = math.radians(alpha); h = th/2
    q = torch.tensor([math.cos(h),0.0,math.sin(h),0.0],device=D,dtype=torch.float64)
    z3 = torch.zeros(3,device=D,dtype=torch.float64); p = torch.zeros(3,device=D,dtype=torch.float64)
    return lambda t: (p,q,z3,z3.clone())
def run(alpha, steps=150):
    ref = build_izra_reference_grid(chordwise_panels=15,spanwise_panels=30,chord_m=C,span_m=B,pivot_fraction_chord=0.25,device=D)
    kin = PrescribedRigidSurfaceKinematics(ref, law(alpha), surface_id="w", body_id="b")
    surf = RigidV5MSurface(chordwise_panels=15,spanwise_panels=30,device=D)
    cfg = NativeV5MConfig(chordwise_panels=15,spanwise_panels=30,density=1.208,freestream=5.0,aerodynamic_dt=DT,lesp_crit=0.11,
        wake_max_rows=300,particle_capacity=32768,particle_max_age_steps=100,wake_history_mode="bound_rate",
        wake_free_rows=100,dvm_target_spacing_chord=0.018,device=D,joint_separation_solve=True)
    solver = RigidNativeV5MSolver(surf,cfg); stepper = RigidV5MStepper(solver,device=D)
    owner = stepper.initialize((kin.evaluate(0.0),))
    n = torch.tensor([math.sin(math.radians(alpha)),0,math.cos(math.radians(alpha))],device=D,dtype=torch.float64)
    cn_last=0.0; lev_total=0; particles=0; full_res=0.0; newborn=0.0; kelvin=0.0
    for s in range(1,steps+1):
        prop = stepper.propose(owner,(kin.evaluate(s*DT),),DT)
        f = prop.load.total_force
        cn_last = float(torch.dot(f,n).item())/QS
        stepper.commit(owner,prop)
        d = owner.state.diagnostics[-1]
        lev_total += int(d["lev_release_count"]); particles = int(d["particle_count"])
        full_res = max(full_res, float(d.get("full_surface_neumann_max_abs",0.0)))
        newborn = max(newborn, abs(float(d.get("newborn_lev_circulation_max_abs",0.0))))
        kelvin = max(kelvin, float(d["kelvin_max_abs"]))
    return dict(alpha=alpha, cn_last=cn_last, lev_release_total=lev_total, particles=particles,
                full_surface_neumann_max=full_res, newborn_lev_max=newborn, kelvin_max=kelvin)
out = [run(a) for a in (5.0,15.0,19.0)]
print(json.dumps(out, indent=1))
