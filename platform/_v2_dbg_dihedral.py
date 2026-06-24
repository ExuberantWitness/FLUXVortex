import numpy as np, warp as wp
wp.init()
from fluxvortex.warp_fsi import config as cfg
from fluxvortex.warp_fsi.config import DTYPE
from fluxvortex.warp_fsi.batched_solver import batched_dense_solve
import diff_uvlm_unsteady_gpu as ug
import robowing as rw
V3=wp.vec3d; dev=cfg.DEVICE; NP=cfg.NP_DTYPE
def steady_lift(theta_deg, U=8.0, aoa=5.0, nc=6, ns=12):
    C0 = rw.robowing(nc, ns, 0.287, 0.80)            # (nc+1,ns+1,3) flat wing w/ camber+round tip
    th = np.radians(theta_deg)
    x,y,z = C0[...,0], C0[...,1], C0[...,2]           # rotate by dihedral theta about x-axis
    C = np.stack([x, y*np.cos(th)-z*np.sin(th), y*np.sin(th)+z*np.cos(th)], -1)
    npan=nc*ns; ncv=(nc+1)*(ns+1)
    Vinf=np.array([U,0.0,U*np.tan(np.radians(aoa))]); Vw=V3(*[float(v) for v in Vinf])
    cw=wp.array(C.reshape(ncv,3).astype(NP),dtype=V3,device=dev)
    rings=wp.zeros((npan,4),dtype=V3,device=dev); col=wp.zeros(npan,dtype=V3,device=dev); nrm=wp.zeros(npan,dtype=V3,device=dev)
    wp.launch(ug.bound_rings_kernel,dim=npan,inputs=[cw,nc,ns],outputs=[rings,col,nrm],device=dev)
    AIC=wp.zeros((1,npan,npan),dtype=DTYPE,device=dev)
    wp.launch(ug.aic_kernel,dim=(npan,npan),inputs=[rings,col,nrm],outputs=[AIC],device=dev)
    wr0=wp.zeros((1,4),dtype=V3,device=dev); wg0=wp.zeros(1,dtype=DTYPE,device=dev)
    rhs=wp.zeros((1,npan),dtype=DTYPE,device=dev)
    wp.launch(ug.rhs_kernel,dim=npan,inputs=[col,nrm,Vw,wr0,wg0,0],outputs=[rhs],device=dev)
    gamma=batched_dense_solve(AIC,rhs,dev)
    lift=wp.zeros(1,dtype=DTYPE,device=dev)
    wp.launch(ug.lift_kernel,dim=npan,inputs=[rings,nrm,gamma,gamma,Vw,DTYPE(1.0),DTYPE(ug.RHO),ns],outputs=[lift],device=dev)
    return 2.0*float(lift.numpy()[0])   # both wings
L0=steady_lift(0.0)
print(f"3D UVLM STEADY lift at FIXED dihedral (5deg AoA, 8m/s) — is it cos^2(theta)*L0?  L0={L0:.2f}N")
for th in (0,15,30,45):
    L=steady_lift(th); exp=L0*np.cos(np.radians(th))**2
    print(f"  dihedral={th:2d}deg: L={L:+6.2f}N   cos^2*L0={exp:+6.2f}N   ratio={L/exp:.2f}", flush=True)
print("DONE")
