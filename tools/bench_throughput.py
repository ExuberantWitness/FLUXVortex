"""Phase 5 — multi-env throughput benchmark of the GPU coupled FSI loop on the 4090.
Sweeps B (environments), measures env-steps/sec, finds saturation."""
import os, sys, time
os.chdir('/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV')
sys.path.insert(0,'src'); sys.path.insert(0,'tests')
import numpy as np
import warp as wp
from fluxvortex.standalone_hybrid_solver import StandaloneHybridSolver
from run_standalone_yamano import yamano_params, build_yamano_shell
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.kernels_ancf import ANCFConstants
from fluxvortex.warp_fsi.kernels_coupling import CSR
from fluxvortex.warp_fsi.coupled import GpuFluidSolve, gpu_coupled_trajectory

NP = config.NP_DTYPE
p = yamano_params(); V=p['V_inf']; L=p['Length']; dt=2e-4
sh,_,_,_ = build_yamano_shell(p, nx=15, ny=10)
s = StandaloneHybridSolver(sh, np.array([V,0,0.]), rho_fluid=p['rho_fluid'], structural_dt=dt,
    uvlm_dt_ratio=34, integrator='implicit', relaxation=1.0, newton_tol=1e-4, max_newton=20,
    max_particles=5000, wake_truncation=5.5, core_radius=1e-6, coupling='strong')
s.uvlm.disable_wake=True; s.enable_sc_geometry()
C = ANCFConstants(sh)
q0 = sh.q.copy(); ndof=len(q0)
T=0.2*L/V; fd=p['rho_fluid']*V**2/p['thickness']
pulse_shape = sh.distributed_load(np.array([0,0,0.5*fd]))
def profile(t):
    ts=t*V/L
    return np.sin(np.pi*ts/0.2) if ts<0.2 else 0.0
Madd=s._M_added_full.tocsc()

WARM=4; TIMED=10
def bench(B, vel):
    gfs = GpuFluidSolve(s)
    madd=CSR(Madd); madd_diag=wp.array(np.asarray(Madd.diagonal()).astype(NP),dtype=config.DTYPE,device=config.DEVICE)
    q0w = wp.array(np.broadcast_to(q0,(B,ndof)).astype(NP),dtype=config.DTYPE,device=config.DEVICE)
    dq0w = wp.zeros((B,ndof),dtype=config.DTYPE,device=config.DEVICE)
    gpu_coupled_trajectory(C,gfs,q0w,dq0w,pulse_shape,profile,dt,WARM,madd=madd,madd_diag=madd_diag,velocity_coupling=vel,tip_dof=None)
    wp.synchronize()
    t0=time.time()
    gpu_coupled_trajectory(C,gfs,q0w,dq0w,pulse_shape,profile,dt,TIMED,madd=madd,madd_diag=madd_diag,velocity_coupling=vel,tip_dof=None)
    wp.synchronize()
    el=time.time()-t0
    return el, B*TIMED/el, el/TIMED*1000

print(f"=== GPU coupled FSI throughput (4090 D, fp64={config.NP_DTYPE is np.float64}, {TIMED} steps timed) ===", flush=True)
print(f"{'B':>6} {'time(s)':>9} {'ms/step':>9} {'env-steps/s':>13} {'speedup':>8}", flush=True)
base=None
for B in [1,4,16,64,256,1024]:
    try:
        el, thr, msps = bench(B, False)
        if base is None: base=thr
        print(f"{B:>6} {el:>9.3f} {msps:>9.1f} {thr:>13.0f} {thr/base:>7.1f}x", flush=True)
    except Exception as e:
        print(f"{B:>6}  FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)
        break
