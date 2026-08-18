"""Minimal Warp vec3d kernel test (ALL scalars as wp.constant float64)."""
import numpy as np
import warp as wp

_C2 = wp.constant(wp.float64(2.0))
_C0 = wp.constant(wp.float64(0.0))
_C_NEG_HALF = wp.constant(wp.float64(-0.5))
_C_SQRT2 = wp.constant(wp.float64(1.4142135623730951))
_C_SQRT2_OVER_PI = wp.constant(wp.float64(0.7978845608028654))
_C_NEG_1_OVER_4PI = wp.constant(wp.float64(-0.07957747154594767))


@wp.kernel
def double_kernel(
    pos: wp.array1d(dtype=wp.vec3d),
    out: wp.array1d(dtype=wp.vec3d),
):
    i = wp.tid()
    p = pos[i]
    out[i] = wp.vec3d(p[0] * _C2, p[1] * _C2, p[2] * _C2)


@wp.kernel
def blob_kernel(
    src_pos: wp.array1d(dtype=wp.vec3d),
    src_gamma: wp.array1d(dtype=wp.vec3d),
    src_sigma: wp.array1d(dtype=wp.float64),
    tgt_pos: wp.array1d(dtype=wp.vec3d),
    vel_out: wp.array1d(dtype=wp.vec3d),
    n_src: int,
):
    tid = wp.tid()
    t = tgt_pos[tid]
    vx = _C0
    vy = _C0
    vz = _C0

    for j in range(n_src):
        s = src_pos[j]
        dx = t - s
        r2 = wp.dot(dx, dx)
        if r2 > _C0:
            r = wp.sqrt(r2)
            rho = r / src_sigma[j]
            expo = wp.exp(_C_NEG_HALF * rho * rho)
            erf_val = wp.erf(rho / _C_SQRT2)
            aux = _C_SQRT2_OVER_PI * rho * expo
            q = erf_val - aux
            w = _C_NEG_1_OVER_4PI * q / (r2 * r)
            g = src_gamma[j]
            cx = dx[1] * g[2] - dx[2] * g[1]
            cy = dx[2] * g[0] - dx[0] * g[2]
            cz = dx[0] * g[1] - dx[1] * g[0]
            vx = vx + w * cx
            vy = vy + w * cy
            vz = vz + w * cz

    vel_out[tid] = wp.vec3d(vx, vy, vz)


if __name__ == "__main__":
    n = 10
    data = np.random.randn(n, 3).astype(np.float64)
    arr = wp.zeros(n, dtype=wp.vec3d, device="cuda"); arr.numpy()[:] = data
    out = wp.zeros(n, dtype=wp.vec3d, device="cuda")
    wp.launch(double_kernel, dim=[n], inputs=[arr, out], device="cuda")
    print(f"double kernel: max_diff = {np.abs(out.numpy() - data*2).max():.2e}")

    ns, nt = 100, 5
    pos = np.random.randn(ns, 3).astype(np.float64)
    gam = (np.random.randn(ns, 3) * 1e-4).astype(np.float64)
    sig = np.random.uniform(0.005, 0.05, ns).astype(np.float64)
    tgt = np.random.randn(nt, 3).astype(np.float64)

    sp = wp.zeros(ns, dtype=wp.vec3d, device="cuda"); sp.numpy()[:] = pos
    sg = wp.zeros(ns, dtype=wp.vec3d, device="cuda"); sg.numpy()[:] = gam
    ss = wp.zeros(ns, dtype=wp.float64, device="cuda"); ss.numpy()[:] = sig
    tp = wp.zeros(nt, dtype=wp.vec3d, device="cuda"); tp.numpy()[:] = tgt
    vo = wp.zeros(nt, dtype=wp.vec3d, device="cuda")

    wp.launch(blob_kernel, dim=[nt], inputs=[sp, sg, ss, tp, vo, ns], device="cuda")
    print(f"blob kernel: |v|max={np.abs(vo.numpy()).max():.3e}")

    import sys; sys.path.insert(0, "src")
    from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian
    v_ref = np.asarray(direct_gaussian_erf_velocity_jacobian(pos, gam, sig, target_positions=tgt).velocity)
    rel = np.abs(vo.numpy() - v_ref).max() / np.abs(v_ref).max()
    print(f"validation: rel_diff = {rel:.3e} ({'PASS' if rel < 1e-10 else 'FAIL'})")

    # Benchmark
    import time
    nb = 40000
    pb = np.random.randn(nb, 3).astype(np.float64)
    gb = (np.random.randn(nb, 3) * 1e-4).astype(np.float64)
    sb = np.full(nb, 0.03, dtype=np.float64)
    spb = wp.zeros(nb, dtype=wp.vec3d, device="cuda"); spb.numpy()[:] = pb
    sgb = wp.zeros(nb, dtype=wp.vec3d, device="cuda"); sgb.numpy()[:] = gb
    ssb = wp.zeros(nb, dtype=wp.float64, device="cuda"); ssb.numpy()[:] = sb
    tpb = wp.zeros(nb, dtype=wp.vec3d, device="cuda"); tpb.numpy()[:] = pb
    vob = wp.zeros(nb, dtype=wp.vec3d, device="cuda")

    wp.launch(blob_kernel, dim=[nb], inputs=[spb, sgb, ssb, tpb, vob, nb], device="cuda")  # warmup
    wp.synchronize()
    t0 = time.perf_counter()
    wp.launch(blob_kernel, dim=[nb], inputs=[spb, sgb, ssb, tpb, vob, nb], device="cuda")
    wp.synchronize()
    t1 = time.perf_counter()
    print(f"benchmark: {nb} particles self-eval = {t1-t0:.2f}s")
