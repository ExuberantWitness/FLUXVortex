"""RoboEagle flapping driven through the VALIDATED StandaloneUVLM (unsteady-Bernoulli surface-pressure
force, MATLAB-validated) + VPM free particles. Wing FIXED + freestream; shed wake -> free particles that
convect (the user's directive). Baseline first (attached, twist0, no LEV); LEV particle shedding added next.

Force = circulation + added-mass via the Bernoulli surface pressure with V_colloc INCLUDING the particle
(LEV+wake) induced velocity -> particles' induced flow on the wing surface enters the force (captures LEV).
"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import _v2_robogeom as rg
from _v2_robo import twisted_state
from fluxvortex.standalone_uvlm import StandaloneUVLM, ring_vortex_velocity


def flap_hybrid(U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0, twist_phase_deg=270.0,
                freq=2.3, nc=4, ns=12, n_cycle=3, steps_per_cycle=120, half_span=0.80, chord=0.287,
                n_keep=10, rho=1.225, core=1e-6, use_vpm=True, free_wake=True, verbose=False):
    # n_keep: keep the NEAR wake as UVLM rings; convert to free particles only after it convects ~n_keep
    # rows downstream (else particles form next to the wing -> near-singular bound induction -> blow-up).
    from fluxvortex.particles import VortexParticleField
    C0 = rg.robowing_real(nc, ns, half_span)
    A_f = np.radians(flap_amp_deg); A_t = np.radians(twist_amp_deg); phi = np.radians(twist_phase_deg)
    Om = 2 * np.pi * freq; x_ea = 0.25 * chord
    V_inf = np.array([U, 0.0, U * np.tan(np.radians(aoa_deg))])
    dt = (1.0 / freq) / steps_per_cycle; N = n_cycle * steps_per_cycle
    uvlm = StandaloneUVLM(C0, V_inf, rho, core_radius=core)
    vpm = VortexParticleField(max_particles=200000, kernel='gaussianerf') if use_vpm else None

    def bound_vel(pts):   # wing bound + near-wake ring induced velocity at particle positions
        V = np.zeros_like(pts)
        for i in range(nc):
            for j in range(ns):
                if abs(uvlm.gamma[i, j]) > 1e-15:
                    V += ring_vortex_velocity(pts, uvlm._corners[i, j], uvlm.gamma[i, j], core)
        for w in range(len(uvlm.wake_vertices)):
            for js in range(ns):
                if abs(uvlm.wake_gamma[w][js]) > 1e-15:
                    V += ring_vortex_velocity(pts, uvlm.wake_vertices[w][js], uvlm.wake_gamma[w][js], core)
        return V

    Lh = np.zeros(N); Xh = np.zeros(N)
    for t in range(N):
        verts, vel = twisted_state(C0, t * dt, A_f, A_t, Om, phi, x_ea, half_span, swept_axis=True)
        uvlm._verts = verts; uvlm._compute_geometry()
        LE = 0.5 * (vel[:-1, :-1] + vel[:-1, 1:]); TE = 0.5 * (vel[1:, :-1] + vel[1:, 1:])
        vstruct = 0.25 * LE + 0.75 * TE                       # body velocity at 3/4-chord collocations
        V_ext = None
        if vpm is not None and vpm.np > 0:
            V_ext = vpm.induce_velocity_at(uvlm._colloc.reshape(-1, 3)).reshape(nc, ns, 3)
        uvlm.solve(V_ext_colloc=V_ext, V_struct_colloc=vstruct)
        uvlm.compute_forces(dt, V_ext_colloc=V_ext, V_struct_colloc=vstruct)
        Lh[t] = float(np.sum(uvlm.forces[:, :, 2])); Xh[t] = float(np.sum(uvlm.forces[:, :, 0]))
        uvlm.shed_wake(dt)
        if vpm is not None and len(uvlm.wake_vertices) > n_keep:
            pos, gam, sig = uvlm.get_wake_particle_sources(dt)
            if pos is not None:
                uvlm.wake_vertices.pop(0); uvlm.wake_gamma.pop(0); uvlm.wake_ages.pop(0)
                vpm.add_particles_batch(pos, gam, sig)
        if vpm is not None and vpm.np > 0:
            vpm.advect_rk3(dt, lambda p: np.broadcast_to(V_inf, p.shape).copy(),
                           bound_velocity_func=bound_vel, stretch=free_wake, free_wake=free_wake)
        if verbose and t % 20 == 0:
            print(f"  t={t} L={2*Lh[t]:.2f} np={vpm.np if vpm else 0}", flush=True)
    last = slice((n_cycle - 1) * steps_per_cycle, N)
    return dict(L=2.0 * np.mean(Lh[last]), T=-2.0 * np.mean(Xh[last]), nparticles=(vpm.np if vpm else 0))


if __name__ == "__main__":
    import time
    out = open("/tmp/hybrid_base.txt", "w")
    out.write("RoboEagle StandaloneUVLM(Bernoulli)+VPM, n_keep=10. 8m/s 5deg 2.3Hz twist0. data~7.8N\n"); out.flush()
    t0 = time.time()
    r = flap_hybrid(U=8.0, aoa_deg=5.0, twist_amp_deg=0.0, freq=2.3, n_cycle=2, steps_per_cycle=60, verbose=False)
    out.write(f"BASELINE (attached, no LEV): L={r['L']:.2f}N  T={r['T']:.2f}N  particles={r['nparticles']}  "
              f"({time.time()-t0:.0f}s)\n"); out.flush(); out.close()
