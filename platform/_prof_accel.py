"""Scratch profiler: confirm the GPU-accel bottleneck (launch overhead) + CUDA-graph capability."""
import time
import numpy as np
import warp as wp
from fluxvortex.warp_fsi import config as cfg
from fluxvortex.warp_fsi.config import DTYPE
from fluxvortex.warp_fsi.batched_solver import structural_cg
from fluxvortex.warp_fsi.kernels_ancf import ANCFConstants, assemble_kmem_blocks
from diff_struct_design import _build_shell
import diff_struct_design_gpu as dsg

dev = cfg.DEVICE; NP = cfg.NP_DTYPE


@wp.kernel
def addk(a: wp.array(dtype=DTYPE), b: wp.array(dtype=DTYPE)):
    i = wp.tid(); b[i] = a[i] + DTYPE(1.0)


def main():
    wp.init()
    print("warp", wp.config.version)
    a = wp.zeros(1000, dtype=DTYPE, device=dev); b = wp.zeros(1000, dtype=DTYPE, device=dev)
    try:
        with wp.ScopedCapture(device=dev) as cap:
            for _ in range(50):
                wp.launch(addk, dim=1000, inputs=[a], outputs=[b], device=dev)
        g = cap.graph
        wp.capture_launch(g); wp.synchronize()
        t0 = time.time()
        for _ in range(500): wp.capture_launch(g)
        wp.synchronize(); tg = time.time() - t0
        t0 = time.time()
        for _ in range(500):
            for _ in range(50): wp.launch(addk, dim=1000, inputs=[a], outputs=[b], device=dev)
        wp.synchronize(); tl = time.time() - t0
        print(f"CUDA graph WORKS: 500x(50 launches) graph={tg*1e3:.0f}ms plain={tl*1e3:.0f}ms speedup {tl/tg:.1f}x")
    except Exception as e:
        print("CUDA graph FAILED:", repr(e)[:300])
    for (nx, ny) in [(6, 4), (10, 6), (15, 10)]:
        sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)])
        C = ANCFConstants(sh, device=dev); ne = sh.ne; ndof = sh.ndof
        rng = np.random.default_rng(0)
        Es = np.exp(0.1 * rng.standard_normal(ne)); Rs = np.exp(0.1 * rng.standard_normal(ne))
        sh.set_distribution(E_scale=Es, rho_scale=Rs)
        Esw = wp.array(Es.astype(NP), dtype=DTYPE, device=dev)
        Mscaled = wp.zeros((ne, 36, 36), dtype=DTYPE, device=dev)
        wp.launch(dsg._scaled_mass, dim=(ne, 36, 36),
                  inputs=[C.Me, wp.array(Rs.astype(NP), dtype=DTYPE, device=dev)], outputs=[Mscaled], device=dev)
        q = sh.q.copy()
        Kblk = assemble_kmem_blocks(wp.array(q[None].astype(NP), dtype=DTYPE, device=dev), C, dev)
        wp.launch(dsg._scale_kblk, dim=(1, ne, 36, 36), inputs=[Kblk, Esw], device=dev)
        rhs = wp.array((np.random.standard_normal(ndof) * C.free_np)[None].astype(NP), dtype=DTYPE, device=dev)
        coef = 0.25 * (2e-4) ** 2
        msg = f"{nx}x{ny}={ne}elem ndof={ndof}:"
        for ce in [1, 10, 25, 100]:
            x, it = structural_cg(rhs, Mscaled, Kblk, C.edofs, C.free, coef, ndof, tol=1e-6, device=dev, check_every=ce)
            wp.synchronize()
            t0 = time.time()
            for _ in range(50):
                x, it = structural_cg(rhs, Mscaled, Kblk, C.edofs, C.free, coef, ndof, tol=1e-6, device=dev, check_every=ce)
            wp.synchronize(); dtcg = (time.time() - t0) / 50
            msg += f"  ce={ce}:{dtcg*1e3:.1f}ms(it{int(it)})"
        print(msg)


if __name__ == "__main__":
    main()
