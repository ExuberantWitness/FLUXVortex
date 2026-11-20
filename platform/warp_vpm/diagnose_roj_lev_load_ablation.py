"""Ablation: does the deposited LEV particle cloud influence the loads at all?

Runs the production joint path at alpha=15 for N steps twice: once normally, once
with the particle field emptied after every commit (so the next step's RHS and
loads see no LEV induction, while the pinned bound circulation, the source bank
and the TEV chain are untouched).

If Cn is unchanged, the LEV is decorative in the load channel and the lift comes
from the pinned bound circulation, not from vortex dynamics.
"""
import sys, math, torch
from dataclasses import replace
from fluxvortex.aero.v5m.stepper import RigidV5MStepper
from fluxvortex.cases.izraelevitz_kinematics import build_izra_reference_grid
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig
from fluxvortex.warp_fsi.rigid_flux_v5m_native import RigidNativeV5MSolver, RigidV5MSurface

D="cuda:0"; C=0.0688; B=0.1375; U=5.0; NC,NS=15,30; DT=0.01*C/U
A=float(sys.argv[1]) if len(sys.argv)>1 else 15.0
N=int(sys.argv[2]) if len(sys.argv)>2 else 30

def build():
    ref=build_izra_reference_grid(chordwise_panels=NC,spanwise_panels=NS,chord_m=C,span_m=B,
        pivot_fraction_chord=0.25,device=D)
    th=math.radians(A)*0.5
    q=torch.tensor([math.cos(th),0,math.sin(th),0.0],device=D,dtype=torch.float64)
    z=torch.zeros(3,device=D,dtype=torch.float64)
    kin=PrescribedRigidSurfaceKinematics(ref,lambda _t:(z,q,z,z.clone()),surface_id="w",body_id="b")
    cfg=NativeV5MConfig(chordwise_panels=NC,spanwise_panels=NS,density=1.208,freestream=U,
        aerodynamic_dt=DT,lesp_crit=0.11,wake_max_rows=300,particle_capacity=32768,
        particle_max_age_steps=100,wake_history_mode="bound_rate",wake_free_rows=100,
        dvm_target_spacing_chord=0.018,device=D,joint_separation_solve=True)
    solver=RigidNativeV5MSolver(RigidV5MSurface(chordwise_panels=NC,spanwise_panels=NS,device=D),cfg)
    st=RigidV5MStepper(solver,device=D)
    return solver,st,st.initialize((kin.evaluate(0.0),)),kin

def run(empty_field):
    solver,st,owner,kin=build()
    cn=[]
    for s in range(1,N+1):
        p=st.propose(owner,(kin.evaluate(s*DT),),DT); st.commit(owner,p)
        nrm=torch.tensor([math.sin(math.radians(A)),0.0,math.cos(math.radians(A))],device=D,dtype=torch.float64)
        q=0.5*1.208*U*U*(C*B)
        cn.append(float((p.load.total_force@nrm).item()/q))
        if empty_field:
            owner.state.particle_field.n=0
    return cn

for tag,empty in (("with LEV field",False),("LEV field emptied each step",True)):
    cs=run(empty)
    print(f"{tag:32s} n={N}  cn[3]={cs[2]:+8.3f}  cn[{N//2}]={cs[N//2-1]:+8.3f}  cn[-1]={cs[-1]:+8.3f}  mean[3:]={sum(cs[2:])/len(cs[2:]):+8.3f}")
