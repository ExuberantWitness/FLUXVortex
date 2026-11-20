"""Is the deposited LE ribbon a dipole (1/r^3) or a monopole (1/r)?"""
import math, json
import torch
from fluxvortex.aero.v5m.stepper import RigidV5MStepper
from fluxvortex.cases.izraelevitz_kinematics import build_izra_reference_grid
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig
from fluxvortex.warp_fsi.rigid_flux_v5m_native import RigidNativeV5MSolver, RigidV5MSurface
from pfield_torch_gpu import CudaParticleField

D="cuda:0"; C=0.0688; B=0.1375; U=5.0; DT=0.01*C/U; NC,NS=15,30
def law(a):
    t=math.radians(a); h=0.5*t
    q=torch.tensor([math.cos(h),0,math.sin(h),0.0],device=D,dtype=torch.float64)
    z=torch.zeros(3,device=D,dtype=torch.float64)
    return lambda _t:(z,q,z,z.clone())
ref=build_izra_reference_grid(chordwise_panels=NC,spanwise_panels=NS,chord_m=C,span_m=B,pivot_fraction_chord=0.25,device=D)
kin=PrescribedRigidSurfaceKinematics(ref,law(15.0),surface_id="w",body_id="b")
cfg=NativeV5MConfig(chordwise_panels=NC,spanwise_panels=NS,density=1.208,freestream=U,aerodynamic_dt=DT,
    lesp_crit=0.11,wake_max_rows=300,particle_capacity=32768,particle_max_age_steps=100,
    wake_history_mode="bound_rate",wake_free_rows=100,dvm_target_spacing_chord=0.018,
    device=D,joint_separation_solve=True)
solver=RigidNativeV5MSolver(RigidV5MSurface(chordwise_panels=NC,spanwise_panels=NS,device=D),cfg)
stepper=RigidV5MStepper(solver,device=D)
owner=stepper.initialize((kin.evaluate(0.0),))
for s in range(1,4):
    p=stepper.propose(owner,(kin.evaluate(s*DT),),DT); stepper.commit(owner,p)
f=owner.state.particle_field
newest=int(f.birth_step[:f.n].max().item())
mask=f.birth_step[:f.n]==newest
n=int(torch.count_nonzero(mask).item())
scratch=CudaParticleField(max(n,1),device=D)
scratch.pos[:n].copy_(f.pos[:f.n][mask]); scratch.gamma[:n].copy_(f.gamma[:f.n][mask])
scratch.sigma[:n].copy_(f.sigma[:f.n][mask]); scratch.circul[:n].copy_(f.circul[:f.n][mask])
scratch.n=n
geo=solver.surface.evaluate(None,None)
le_mid=geo.leading_edge[:NS].mean(dim=0)
print(f"newest ribbon: {n} particles | sum(circul) = {float(scratch.circul[:n].sum()):+.3e} "
      f"| sum|gamma| = {float(torch.linalg.vector_norm(scratch.gamma[:n],dim=1).sum()):.3e}")
print(f"| vector sum(gamma) | = {float(torch.linalg.vector_norm(scratch.gamma[:n].sum(dim=0))):.3e}")
# far-field decay along +z (normal to the plate) from the LE mid
zs=[0.2,0.4,0.8,1.6,3.2]
pts=torch.stack([le_mid+torch.tensor([0.0,0.0,z*C],device=D,dtype=torch.float64) for z in zs])
v=scratch.velocity_at_cuda(pts)
mag=torch.linalg.vector_norm(v,dim=1)
print("  z/c     |v| (m/s)")
for z,m in zip(zs,mag.tolist()): print(f"  {z:>4.1f}  {m:.6e}")
# fit exponent between successive pairs
import math as m_
exps=[m_.log(mag[i].item()/mag[i+1].item())/m_.log(zs[i+1]/zs[i]) for i in range(len(zs)-1)]
print("  decay exponent p (|v| ~ r^-p) between successive pairs:", [round(e,3) for e in exps])
