"""Analytic symmetry and order witnesses for the material integrator."""
from dataclasses import replace
from types import SimpleNamespace
import math

import pytest
import torch

from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig, Q16NativeV5MSolver
from pfield_torch_gpu import CudaParticleField

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def system(dt):
    device=torch.device("cuda:0")
    solver=Q16NativeV5MSolver.__new__(Q16NativeV5MSolver)
    solver.device=device
    solver.settings=NativeV5MConfig(chordwise_panels=2,spanwise_panels=1,
        aerodynamic_dt=dt,joint_separation_solve=True,device="cuda:0")
    solver.v_inf=torch.zeros(3,device=device,dtype=torch.float64)
    empty=torch.empty((0,4,3),device=device,dtype=torch.float64)
    geometry=SimpleNamespace(rings=empty)
    state=SimpleNamespace(wake_rings=empty.clone(),wake_gamma=torch.empty(0,device=device,dtype=torch.float64),
        gamma_bound=torch.empty(0,device=device,dtype=torch.float64),particle_field=CudaParticleField(16,device=device),step=0)
    return solver,geometry,state


@pytest.mark.parametrize("dt",[.1,.05,.025])
def test_square_ring_self_advection_is_pure_translation(dt):
    solver,geometry,state=system(dt)
    initial=torch.tensor([[[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]]],device="cuda:0",dtype=torch.float64)
    state.wake_rings=initial.clone()
    state.wake_gamma=torch.ones(1,device="cuda:0",dtype=torch.float64)
    velocity=solver._external_velocity(initial.reshape(-1,3),geometry,state).reshape_as(initial)
    assert velocity.norm().item()>0.
    torch.testing.assert_close(velocity,velocity[:,:1].expand_as(velocity),atol=1.e-15,rtol=0.)
    solver._advance_material_state(geometry,state)
    torch.testing.assert_close(state.wake_rings,initial+dt*velocity,atol=1.e-14,rtol=0.)


def test_particle_pair_rotation_has_third_order_global_accuracy():
    errors=[]
    for substeps in (1,2,4):
        solver,geometry,state=system(.1)
        initial=torch.tensor([[1.,0.,0.],[-1.,0.,0.]],device="cuda:0",dtype=torch.float64)
        strengths=torch.tensor([[0.,0.,1.],[0.,0.,1.]],device="cuda:0",dtype=torch.float64)
        sigma=torch.full((2,),.4,device="cuda:0",dtype=torch.float64)
        state.particle_field.add_particles(initial,strengths,sigma)
        omega=state.particle_field.velocity_self_cuda()[0,1].item()
        duration=.4/omega
        solver.settings=replace(solver.settings,aerodynamic_dt=duration/substeps)
        for _ in range(substeps):
            solver._advance_material_state(geometry,state)
        exact=torch.tensor([[math.cos(.4),math.sin(.4),0.],[-math.cos(.4),-math.sin(.4),0.]],device="cuda:0",dtype=torch.float64)
        errors.append((state.particle_field.pos[:2]-exact).norm().item())
        assert torch.equal(state.particle_field.gamma[:2],strengths)
    assert errors[0]/errors[1] > 6., errors
    assert errors[1]/errors[2] > 6., errors
    assert errors[-1] < 1.e-4, errors


def test_far_wake_keeps_freestream_policy_and_particle_retention():
    solver,geometry,state=system(.01)
    solver.settings=replace(solver.settings,wake_free_rows=1,particle_max_age_steps=1)
    solver.v_inf[0]=1.25
    ring=torch.tensor([[[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]]],device="cuda:0",dtype=torch.float64)
    state.wake_rings=torch.cat((ring,ring+torch.tensor([0.,0.,3.],device="cuda:0")))
    state.wake_gamma=torch.tensor([1.,.5],device="cuda:0",dtype=torch.float64)
    far=state.wake_rings[1].clone()
    field=state.particle_field
    field.add_particles(torch.tensor([[8.,0.,0.]],device="cuda:0",dtype=torch.float64),
                        torch.tensor([[0.,0.,.01]],device="cuda:0",dtype=torch.float64),
                        torch.tensor([.4],device="cuda:0",dtype=torch.float64),birth_step=0)
    state.step=3
    retention=solver._advance_material_state(geometry,state)
    torch.testing.assert_close(state.wake_rings[1],far+.01*solver.v_inf,atol=1.e-14,rtol=0.)
    assert field.n==0
    assert retention["particle_cull_count"]==1
