"""
NVIDIA Warp GPU kernels for PteraSoftware Biot-Savart calculations.

Replaces all @njit Numba functions in _aerodynamics_functions.py with
GPU-accelerated equivalents. Thread mapping: each (point, vortex) pair -> 1 GPU thread.

IMPORTANT: Warp treats Python float literals as float32. Use wp.float64() for constants
when operating on vec3d (which uses float64 components).
"""
import numpy as np
import warp as wp


# ── Float64 constants for Warp kernels ────────────────────────────────
# Warp float literals default to float32, so we must wrap all constants.
_F64_EPS = wp.float64(2.22e-16)
_F64_TOL = wp.float64(1.0e-10)
_F64_FOUR_LAMB = wp.float64(5.02572)        # 4 * 1.25643
_F64_SQUIRE = wp.float64(1.0e-4)
_F64_INV_4PI = wp.float64(0.07957747154594767)  # 1/(4*pi)
_F64_ZERO = wp.float64(0.0)


# ── Warp kernel: collapsed (summed) line vortex Biot-Savart ───────────
@wp.kernel
def _collapsed_line_vortex_bs(
    points: wp.array(dtype=wp.vec3d),
    starts: wp.array(dtype=wp.vec3d),
    ends: wp.array(dtype=wp.vec3d),
    strengths: wp.array(dtype=wp.float64),
    rc0s: wp.array(dtype=wp.float64),
    ages: wp.array(dtype=wp.float64),
    nu: wp.float64,
    num_vortices: wp.int32,
    output: wp.array(dtype=wp.vec3d),
):
    """Each GPU thread computes the Biot-Savart contribution of one vortex at one point."""
    tid = wp.tid()
    point_id = wp.int32(tid // num_vortices)
    vortex_id = wp.int32(tid % num_vortices)

    p = points[point_id]
    s = starts[vortex_id]
    e = ends[vortex_id]

    r0x = e[0] - s[0]
    r0y = e[1] - s[1]
    r0z = e[2] - s[2]
    r0_sq = r0x * r0x + r0y * r0y + r0z * r0z
    r0_len = wp.sqrt(r0_sq)

    if r0_len < wp.float64(2.22e-16):
        return

    r1x = s[0] - p[0]
    r1y = s[1] - p[1]
    r1z = s[2] - p[2]

    r2x = e[0] - p[0]
    r2y = e[1] - p[1]
    r2z = e[2] - p[2]

    r3x = r1y * r2z - r1z * r2y
    r3y = r1z * r2x - r1x * r2z
    r3z = r1x * r2y - r1y * r2x

    r1_len = wp.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
    r2_len = wp.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)

    r0_tol = r0_len * wp.float64(1.0e-10)

    if r1_len < r0_tol:
        return
    if r2_len < r0_tol:
        return

    r3_sq = r3x * r3x + r3y * r3y + r3z * r3z
    r3_len = wp.sqrt(r3_sq)
    r1r2 = r1_len * r2_len

    if r3_len < wp.float64(1.0e-10) * r1r2:
        return

    strength = strengths[vortex_id]
    age = ages[vortex_id]
    rc0 = rc0s[vortex_id]

    rc_sq = rc0 * rc0 + wp.float64(5.02572) * (nu + wp.float64(1.0e-4) * wp.abs(strength)) * age
    c1 = strength * wp.float64(0.07957747154594767)
    c2 = r0_sq * rc_sq

    r1pr2 = r1_len + r2_len
    c3 = r1x * r2x + r1y * r2y + r1z * r2z
    c4 = c1 * r1pr2 * (r1r2 - c3) / (r1r2 * (r3_sq + c2))

    vel = wp.vec3d(c4 * r3x, c4 * r3y, c4 * r3z)
    wp.atomic_add(output, point_id, vel)


# ── Warp kernel: expanded (per-pair) line vortex Biot-Savart ──────────
@wp.kernel
def _expanded_line_vortex_bs(
    points: wp.array(dtype=wp.vec3d),
    starts: wp.array(dtype=wp.vec3d),
    ends: wp.array(dtype=wp.vec3d),
    strengths: wp.array(dtype=wp.float64),
    rc0s: wp.array(dtype=wp.float64),
    ages: wp.array(dtype=wp.float64),
    nu: wp.float64,
    num_vortices: wp.int32,
    output: wp.array(dtype=wp.vec3d),
):
    """Each GPU thread computes the Biot-Savart contribution of one vortex at one point.
    Writes to output[point_id * num_vortices + vortex_id]."""
    tid = wp.tid()
    point_id = wp.int32(tid // num_vortices)
    vortex_id = wp.int32(tid % num_vortices)
    flat_idx = point_id * num_vortices + vortex_id

    p = points[point_id]
    s = starts[vortex_id]
    e = ends[vortex_id]

    r0x = e[0] - s[0]
    r0y = e[1] - s[1]
    r0z = e[2] - s[2]
    r0_sq = r0x * r0x + r0y * r0y + r0z * r0z
    r0_len = wp.sqrt(r0_sq)

    if r0_len < wp.float64(2.22e-16):
        return

    r1x = s[0] - p[0]
    r1y = s[1] - p[1]
    r1z = s[2] - p[2]

    r2x = e[0] - p[0]
    r2y = e[1] - p[1]
    r2z = e[2] - p[2]

    r3x = r1y * r2z - r1z * r2y
    r3y = r1z * r2x - r1x * r2z
    r3z = r1x * r2y - r1y * r2x

    r1_len = wp.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
    r2_len = wp.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)

    r0_tol = r0_len * wp.float64(1.0e-10)

    if r1_len < r0_tol or r2_len < r0_tol:
        return

    r3_sq = r3x * r3x + r3y * r3y + r3z * r3z
    r3_len = wp.sqrt(r3_sq)
    r1r2 = r1_len * r2_len

    if r3_len < wp.float64(1.0e-10) * r1r2:
        return

    strength = strengths[vortex_id]
    age = ages[vortex_id]
    rc0 = rc0s[vortex_id]

    rc_sq = rc0 * rc0 + wp.float64(5.02572) * (nu + wp.float64(1.0e-4) * wp.abs(strength)) * age
    c1 = strength * wp.float64(0.07957747154594767)
    c2 = r0_sq * rc_sq

    r1pr2 = r1_len + r2_len
    c3 = r1x * r2x + r1y * r2y + r1z * r2z
    c4 = c1 * r1pr2 * (r1r2 - c3) / (r1r2 * (r3_sq + c2))

    vel = wp.vec3d(c4 * r3x, c4 * r3y, c4 * r3z)
    wp.atomic_add(output, flat_idx, vel)


# ── Helper: numpy -> wp.array conversion ──────────────────────────────
def _to_wp_vec3d(arr):
    """Convert (N, 3) float64 numpy array to wp.array(dtype=wp.vec3d)."""
    return wp.array(arr, dtype=wp.vec3d)


def _to_wp_1d(arr):
    """Convert 1D float64 numpy array to wp.array(dtype=wp.float64)."""
    return wp.array(arr, dtype=wp.float64)


# ── Collapsed wrapper: multiple leg launches with accumulation ────────
def _collapsed_from_legs(points_np, leg_pairs, strengths_np, rc0s_np, ages_np, nu):
    """Launch collapsed kernel for each leg pair, accumulate into single output."""
    N = points_np.shape[0]
    M = strengths_np.shape[0]

    if M == 0 or N == 0:
        return np.zeros((N, 3))

    wp_points = _to_wp_vec3d(points_np)
    wp_strengths = _to_wp_1d(strengths_np)
    wp_rc0s = _to_wp_1d(rc0s_np)
    wp_ages = _to_wp_1d(ages_np)
    wp_output = wp.zeros(N, dtype=wp.vec3d)

    for starts_np, ends_np in leg_pairs:
        wp_starts = _to_wp_vec3d(starts_np)
        wp_ends = _to_wp_vec3d(ends_np)
        wp.launch(
            _collapsed_line_vortex_bs,
            dim=N * M,
            inputs=[
                wp_points, wp_starts, wp_ends, wp_strengths,
                wp_rc0s, wp_ages, float(nu), M, wp_output,
            ],
        )

    result = wp_output.numpy()
    return result.view(np.float64).reshape(N, 3)


# ── Expanded wrapper: multiple leg launches with accumulation ─────────
def _expanded_from_legs(points_np, leg_pairs, strengths_np, rc0s_np, ages_np, nu):
    """Launch expanded kernel for each leg pair, accumulate into (N, M, 3) output."""
    N = points_np.shape[0]
    M = strengths_np.shape[0]

    if M == 0 or N == 0:
        return np.zeros((N, M, 3))

    wp_points = _to_wp_vec3d(points_np)
    wp_strengths = _to_wp_1d(strengths_np)
    wp_rc0s = _to_wp_1d(rc0s_np)
    wp_ages = _to_wp_1d(ages_np)
    wp_output = wp.zeros(N * M, dtype=wp.vec3d)

    for starts_np, ends_np in leg_pairs:
        wp_starts = _to_wp_vec3d(starts_np)
        wp_ends = _to_wp_vec3d(ends_np)
        wp.launch(
            _expanded_line_vortex_bs,
            dim=N * M,
            inputs=[
                wp_points, wp_starts, wp_ends, wp_strengths,
                wp_rc0s, wp_ages, float(nu), M, wp_output,
            ],
        )

    result = wp_output.numpy()
    return result.view(np.float64).reshape(N, M, 3)


# ── Public API: Ring Vortex functions ─────────────────────────────────

def collapsed_velocities_from_ring_vortices(
    stackP_GP1_CgP1,
    stackBrrvp_GP1_CgP1,
    stackFrrvp_GP1_CgP1,
    stackFlrvp_GP1_CgP1,
    stackBlrvp_GP1_CgP1,
    strengths,
    r_c0s,
    singularity_counts,
    ages=None,
    nu=0.0,
):
    """GPU replacement for PteraSoftware collapsed_velocities_from_ring_vortices."""
    M = strengths.shape[0]
    if ages is None:
        ages = np.zeros(M)

    leg_pairs = [
        (stackBrrvp_GP1_CgP1, stackFrrvp_GP1_CgP1),  # back-right -> front-right
        (stackFrrvp_GP1_CgP1, stackFlrvp_GP1_CgP1),  # front-right -> front-left
        (stackFlrvp_GP1_CgP1, stackBlrvp_GP1_CgP1),  # front-left -> back-left
        (stackBlrvp_GP1_CgP1, stackBrrvp_GP1_CgP1),  # back-left -> back-right
    ]
    return _collapsed_from_legs(
        stackP_GP1_CgP1, leg_pairs, strengths, r_c0s, ages, nu,
    )


def collapsed_velocities_from_ring_vortices_chordwise_segments(
    stackP_GP1_CgP1,
    stackBrrvp_GP1_CgP1,
    stackFrrvp_GP1_CgP1,
    stackFlrvp_GP1_CgP1,
    stackBlrvp_GP1_CgP1,
    strengths,
    r_c0s,
    singularity_counts,
    ages=None,
    nu=0.0,
):
    """GPU replacement -- only left and right legs (chordwise segments)."""
    M = strengths.shape[0]
    if ages is None:
        ages = np.zeros(M)

    leg_pairs = [
        (stackBrrvp_GP1_CgP1, stackFrrvp_GP1_CgP1),  # right leg
        (stackFlrvp_GP1_CgP1, stackBlrvp_GP1_CgP1),  # left leg
    ]
    return _collapsed_from_legs(
        stackP_GP1_CgP1, leg_pairs, strengths, r_c0s, ages, nu,
    )


def expanded_velocities_from_ring_vortices(
    stackP_GP1_CgP1,
    stackBrrvp_GP1_CgP1,
    stackFrrvp_GP1_CgP1,
    stackFlrvp_GP1_CgP1,
    stackBlrvp_GP1_CgP1,
    strengths,
    r_c0s,
    singularity_counts,
    ages=None,
    nu=0.0,
):
    """GPU replacement for PteraSoftware expanded_velocities_from_ring_vortices."""
    M = strengths.shape[0]
    if ages is None:
        ages = np.zeros(M)

    leg_pairs = [
        (stackBrrvp_GP1_CgP1, stackFrrvp_GP1_CgP1),
        (stackFrrvp_GP1_CgP1, stackFlrvp_GP1_CgP1),
        (stackFlrvp_GP1_CgP1, stackBlrvp_GP1_CgP1),
        (stackBlrvp_GP1_CgP1, stackBrrvp_GP1_CgP1),
    ]
    return _expanded_from_legs(
        stackP_GP1_CgP1, leg_pairs, strengths, r_c0s, ages, nu,
    )


# ── Public API: Horseshoe Vortex functions ────────────────────────────

def collapsed_velocities_from_horseshoe_vortices(
    stackP_GP1_CgP1,
    stackBrhvp_GP1_CgP1,
    stackFrhvp_GP1_CgP1,
    stackFlhvp_GP1_CgP1,
    stackBlhvp_GP1_CgP1,
    strengths,
    r_c0s,
    singularity_counts,
    nu=0.0,
):
    """GPU replacement for PteraSoftware collapsed_velocities_from_horseshoe_vortices."""
    M = strengths.shape[0]
    ages = np.zeros(M)

    leg_pairs = [
        (stackBrhvp_GP1_CgP1, stackFrhvp_GP1_CgP1),
        (stackFrhvp_GP1_CgP1, stackFlhvp_GP1_CgP1),
        (stackFlhvp_GP1_CgP1, stackBlhvp_GP1_CgP1),
    ]
    return _collapsed_from_legs(
        stackP_GP1_CgP1, leg_pairs, strengths, r_c0s, ages, nu,
    )


def expanded_velocities_from_horseshoe_vortices(
    stackP_GP1_CgP1,
    stackBrhvp_GP1_CgP1,
    stackFrhvp_GP1_CgP1,
    stackFlhvp_GP1_CgP1,
    stackBlhvp_GP1_CgP1,
    strengths,
    r_c0s,
    singularity_counts,
    nu=0.0,
):
    """GPU replacement for PteraSoftware expanded_velocities_from_horseshoe_vortices."""
    M = strengths.shape[0]
    ages = np.zeros(M)

    leg_pairs = [
        (stackBrhvp_GP1_CgP1, stackFrhvp_GP1_CgP1),
        (stackFrhvp_GP1_CgP1, stackFlhvp_GP1_CgP1),
        (stackFlhvp_GP1_CgP1, stackBlhvp_GP1_CgP1),
    ]
    return _expanded_from_legs(
        stackP_GP1_CgP1, leg_pairs, strengths, r_c0s, ages, nu,
    )
