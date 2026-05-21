"""
NVIDIA Warp GPU kernels for vortex particle Biot-Savart and Jacobian.

Replaces the NumPy chunked computations in kernel.py with GPU-accelerated
Warp kernels. Each (target, source) pair maps to one GPU thread.
"""
import numpy as np
import warp as wp


# ── Float64 constants ─────────────────────────────────────────────────
_F64_INV_4PI = wp.float64(0.07957747154594767)   # 1/(4*pi)
_F64_SQRT2_INV = wp.float64(0.7071067811865476)   # 1/sqrt(2)
_F64_SQRT2_OVER_PI = wp.float64(0.7978845608028654)  # sqrt(2/pi)
_F64_HALF = wp.float64(0.5)


# ── Particle Biot-Savart (Gaussian-erf) ───────────────────────────────
@wp.kernel
def particle_bs_kernel(
    targets: wp.array(dtype=wp.vec3d),
    sources: wp.array(dtype=wp.vec3d),
    gammas: wp.array(dtype=wp.vec3d),
    sigmas: wp.array(dtype=wp.float64),
    num_sources: wp.int32,
    output: wp.array(dtype=wp.vec3d),
):
    """Velocity induced by N vortex particles at M target points.
    Gaussian-erf regularization: g(r/sigma) = erf(r/sigma/sqrt2) - sqrt(2/pi)*(r/sigma)*exp(-(r/sigma)^2/2)"""
    tid = wp.tid()
    tgt_id = wp.int32(tid // num_sources)
    src_id = wp.int32(tid % num_sources)

    dx = targets[tgt_id] - sources[src_id]
    r_sq = wp.dot(dx, dx)
    r = wp.sqrt(r_sq)
    if r < wp.float64(1.0e-12):
        return

    sigma = sigmas[src_id]
    r_bar = r / sigma

    g = wp.erf(r_bar * wp.float64(0.7071067811865476)) - wp.float64(0.7978845608028654) * r_bar * wp.exp(wp.float64(-0.5) * r_bar * r_bar)

    cross = wp.cross(dx, gammas[src_id])
    coeff = wp.float64(-0.07957747154594767) * g / (r * r * r)

    vel = coeff * cross
    wp.atomic_add(output, tgt_id, vel)


def velocity_from_particles_gpu(target_points, src_pos, src_gamma, src_sigma):
    """GPU-accelerated particle Biot-Savart. Same API as kernel.velocity_from_particles."""
    M = target_points.shape[0]
    N = src_pos.shape[0]

    if N == 0 or M == 0:
        return np.zeros((M, 3))

    wp_targets = wp.array(target_points, dtype=wp.vec3d)
    wp_sources = wp.array(src_pos, dtype=wp.vec3d)
    wp_gammas = wp.array(src_gamma, dtype=wp.vec3d)
    wp_sigmas = wp.array(src_sigma, dtype=wp.float64)
    wp_output = wp.zeros(M, dtype=wp.vec3d)

    wp.launch(
        particle_bs_kernel,
        dim=M * N,
        inputs=[wp_targets, wp_sources, wp_gammas, wp_sigmas, N, wp_output],
    )

    result = wp_output.numpy()
    return result.view(np.float64).reshape(M, 3)


# ── Particle Jacobian (for vortex stretching) ─────────────────────────
@wp.kernel
def particle_jacobian_kernel(
    targets: wp.array(dtype=wp.vec3d),
    sources: wp.array(dtype=wp.vec3d),
    gammas: wp.array(dtype=wp.vec3d),
    sigmas: wp.array(dtype=wp.float64),
    num_sources: wp.int32,
    out_00: wp.array(dtype=wp.float64),
    out_01: wp.array(dtype=wp.float64),
    out_02: wp.array(dtype=wp.float64),
    out_10: wp.array(dtype=wp.float64),
    out_11: wp.array(dtype=wp.float64),
    out_12: wp.array(dtype=wp.float64),
    out_20: wp.array(dtype=wp.float64),
    out_21: wp.array(dtype=wp.float64),
    out_22: wp.array(dtype=wp.float64),
):
    """Compute velocity gradient tensor J = dU/dx at each target particle.
    J[target, i, j] = dU_i/dx_j using Gaussian-erf kernel derivatives."""
    tid = wp.tid()
    tgt_id = wp.int32(tid // num_sources)
    src_id = wp.int32(tid % num_sources)

    dx = targets[tgt_id] - sources[src_id]
    r_sq = wp.dot(dx, dx)
    r = wp.sqrt(r_sq)
    if r < wp.float64(1.0e-12):
        return

    sigma = sigmas[src_id]
    r_bar = r / sigma

    g = wp.erf(r_bar * wp.float64(0.7071067811865476)) - wp.float64(0.7978845608028654) * r_bar * wp.exp(wp.float64(-0.5) * r_bar * r_bar)
    dgdr = wp.float64(0.7978845608028654) * r_bar * r_bar * wp.exp(wp.float64(-0.5) * r_bar * r_bar)

    gamma_src = gammas[src_id]

    cross_KG = wp.cross(dx, gamma_src)
    r3inv = wp.float64(1.0) / (r * r * r)
    coeff_K = wp.float64(-0.07957747154594767) * r3inv
    KxG0 = coeff_K * cross_KG[0]
    KxG1 = coeff_K * cross_KG[1]
    KxG2 = coeff_K * cross_KG[2]

    aux = dgdr / (sigma * r) - wp.float64(3.0) * g / (r * r)
    aux2 = wp.float64(-0.07957747154594767) * g * r3inv

    dx0 = dx[0]
    dx1 = dx[1]
    dx2 = dx[2]
    g0 = gamma_src[0]
    g1 = gamma_src[1]
    g2 = gamma_src[2]

    wp.atomic_add(out_00, tgt_id, aux * KxG0 * dx0)
    wp.atomic_add(out_01, tgt_id, aux * KxG0 * dx1 + aux2 * g2)
    wp.atomic_add(out_02, tgt_id, aux * KxG0 * dx2 - aux2 * g1)

    wp.atomic_add(out_10, tgt_id, aux * KxG1 * dx0 - aux2 * g2)
    wp.atomic_add(out_11, tgt_id, aux * KxG1 * dx1)
    wp.atomic_add(out_12, tgt_id, aux * KxG1 * dx2 + aux2 * g0)

    wp.atomic_add(out_20, tgt_id, aux * KxG2 * dx0 + aux2 * g1)
    wp.atomic_add(out_21, tgt_id, aux * KxG2 * dx1 - aux2 * g0)
    wp.atomic_add(out_22, tgt_id, aux * KxG2 * dx2)


def jacobian_from_particles_gpu(tgt_pos, tgt_gamma, src_pos, src_gamma, src_sigma):
    """GPU-accelerated particle Jacobian. Same API as kernel.jacobian_from_particles."""
    M = tgt_pos.shape[0]
    N = src_pos.shape[0]

    if N == 0 or M == 0:
        return np.zeros((M, 3, 3))

    wp_targets = wp.array(tgt_pos, dtype=wp.vec3d)
    wp_sources = wp.array(src_pos, dtype=wp.vec3d)
    wp_gammas = wp.array(src_gamma, dtype=wp.vec3d)
    wp_sigmas = wp.array(src_sigma, dtype=wp.float64)

    wp_J = [wp.zeros(M, dtype=wp.float64) for _ in range(9)]

    wp.launch(
        particle_jacobian_kernel,
        dim=M * N,
        inputs=[
            wp_targets, wp_sources, wp_gammas, wp_sigmas, N,
            *wp_J,
        ],
    )

    J = np.zeros((M, 3, 3))
    for idx in range(9):
        i, j = divmod(idx, 3)
        J[:, i, j] = wp_J[idx].numpy()

    return J
