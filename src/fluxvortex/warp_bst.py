"""NVIDIA Warp GPU kernels for BST shell force computation and integration.

Three kernels:
1. bst_membrane_forces — per-triangle CST membrane (orthotropic)
2. bst_bending_forces — per-interior-edge dihedral bending (per-edge D)
3. bst_verlet_step — per-node Velocity-Verlet integration

Plus BSTGPUContext for persistent GPU state and bst_subcycle_gpu for subcycling.
"""
import numpy as np
import warp as wp


# ── Float64 constants ─────────────────────────────────────────────────
_F64_ZERO = wp.float64(0.0)
_F64_HALF = wp.float64(0.5)
_F64_ONE = wp.float64(1.0)
_F64_TWO = wp.float64(2.0)
_F64_NEG_ONE = wp.float64(-1.0)
_F64_EPS = wp.float64(1e-30)
_F64_ANG_EPS = wp.float64(1e-15)


# ── Kernel 1: CST membrane forces (per triangle) ──────────────────────
@wp.kernel
def bst_membrane_forces(
    u: wp.array(dtype=wp.vec3d),
    x0: wp.array(dtype=wp.vec3d),
    tri_flat: wp.array(dtype=wp.int32),
    dNdx_flat: wp.array(dtype=wp.float64),
    dNdy_flat: wp.array(dtype=wp.float64),
    ref_area: wp.array(dtype=wp.float64),
    h: wp.float64,
    D00: wp.float64,
    D01: wp.float64,
    D11: wp.float64,
    D22: wp.float64,
    F_out: wp.array(dtype=wp.vec3d),
):
    tid = wp.tid()
    base = tid * 3

    i0 = tri_flat[base]
    i1 = tri_flat[base + 1]
    i2 = tri_flat[base + 2]

    u0 = u[i0]
    u1 = u[i1]
    u2 = u[i2]

    # Strain
    eps_xx = (dNdx_flat[base] * u0[0] + dNdx_flat[base + 1] * u1[0]
              + dNdx_flat[base + 2] * u2[0])
    eps_yy = (dNdy_flat[base] * u0[1] + dNdy_flat[base + 1] * u1[1]
              + dNdy_flat[base + 2] * u2[1])
    eps_xy = (dNdx_flat[base] * u0[1] + dNdx_flat[base + 1] * u1[1]
              + dNdx_flat[base + 2] * u2[1]
              + dNdy_flat[base] * u0[0] + dNdy_flat[base + 1] * u1[0]
              + dNdy_flat[base + 2] * u2[0])

    # Orthotropic plane-stress
    sig_xx = D00 * eps_xx + D01 * eps_yy
    sig_yy = D01 * eps_xx + D11 * eps_yy
    sig_xy = D22 * eps_xy

    coeff = wp.float64(-1.0) * ref_area[tid] * h

    # Node 0
    dn0 = dNdx_flat[base]
    dm0 = dNdy_flat[base]
    wp.atomic_add(F_out, i0, wp.vec3d(
        coeff * (dn0 * sig_xx + dm0 * sig_xy),
        coeff * (dn0 * sig_xy + dm0 * sig_yy),
        _F64_ZERO))

    # Node 1
    dn1 = dNdx_flat[base + 1]
    dm1 = dNdy_flat[base + 1]
    wp.atomic_add(F_out, i1, wp.vec3d(
        coeff * (dn1 * sig_xx + dm1 * sig_xy),
        coeff * (dn1 * sig_xy + dm1 * sig_yy),
        _F64_ZERO))

    # Node 2
    dn2 = dNdx_flat[base + 2]
    dm2 = dNdy_flat[base + 2]
    wp.atomic_add(F_out, i2, wp.vec3d(
        coeff * (dn2 * sig_xx + dm2 * sig_xy),
        coeff * (dn2 * sig_xy + dm2 * sig_yy),
        _F64_ZERO))


# ── Kernel 2: dihedral bending forces (per interior edge) ─────────────
@wp.kernel
def bst_bending_forces(
    u: wp.array(dtype=wp.vec3d),
    x0: wp.array(dtype=wp.vec3d),
    edge_ea: wp.array(dtype=wp.int32),
    edge_eb: wp.array(dtype=wp.int32),
    edge_ec: wp.array(dtype=wp.int32),
    edge_ed: wp.array(dtype=wp.int32),
    edge_L: wp.array(dtype=wp.float64),
    edge_theta_ref: wp.array(dtype=wp.float64),
    edge_A0: wp.array(dtype=wp.float64),
    edge_A1: wp.array(dtype=wp.float64),
    edge_D: wp.array(dtype=wp.float64),
    F_out: wp.array(dtype=wp.vec3d),
):
    tid = wp.tid()

    ea = edge_ea[tid]
    eb = edge_eb[tid]
    ec = edge_ec[tid]
    ed = edge_ed[tid]
    L_ref = edge_L[tid]
    theta_ref = edge_theta_ref[tid]
    A0 = edge_A0[tid]
    A1 = edge_A1[tid]
    D_edge = edge_D[tid]

    # Current positions
    pa = x0[ea] + u[ea]
    pb = x0[eb] + u[eb]
    pc = x0[ec] + u[ec]
    pd = x0[ed] + u[ed]

    # Edge vector
    e_vec = pb - pa
    L_cur_sq = wp.dot(e_vec, e_vec)
    L_cur = wp.sqrt(L_cur_sq)
    if L_cur < _F64_EPS:
        return
    inv_L = _F64_ONE / L_cur

    # Face normals (same orientation convention as reference)
    n0 = wp.cross(pb - pa, pc - pa)
    n1 = wp.cross(pd - pa, pb - pa)
    n0_len = wp.sqrt(wp.dot(n0, n0))
    n1_len = wp.sqrt(wp.dot(n1, n1))
    if n0_len < _F64_EPS or n1_len < _F64_EPS:
        return

    n0_hat = n0 * (_F64_ONE / n0_len)
    n1_hat = n1 * (_F64_ONE / n1_len)

    # Dihedral angle
    cos_th = wp.dot(n0_hat, n1_hat)
    if cos_th > _F64_ONE:
        cos_th = _F64_ONE
    if cos_th < _F64_NEG_ONE:
        cos_th = _F64_NEG_ONE
    sin_th = wp.dot(wp.cross(n0_hat, n1_hat), e_vec * inv_L)
    theta = wp.atan2(sin_th, cos_th)

    dtheta = theta - theta_ref
    if dtheta > _F64_ANG_EPS or dtheta < wp.float64(-1e-15):
        pass
    else:
        return

    # Gradients w.r.t. opposite vertices
    grad_c = wp.float64(-1.0) * (L_cur * _F64_HALF / A0) * n0_hat
    grad_d = wp.float64(-1.0) * (L_cur * _F64_HALF / A1) * n1_hat

    # Gradients w.r.t. edge vertices (translational invariance)
    inv_L2 = _F64_ONE / L_cur_sq
    dot_ca = wp.dot(pc - pa, e_vec)
    dot_da = wp.dot(pd - pa, e_vec)
    dot_cb = wp.dot(pc - pb, e_vec)
    dot_db = wp.dot(pd - pb, e_vec)

    grad_a = wp.float64(-1.0) * dot_ca * inv_L2 * grad_c \
             - dot_da * inv_L2 * grad_d
    grad_b = dot_cb * inv_L2 * grad_c \
             + dot_db * inv_L2 * grad_d

    coeff = wp.float64(-1.0) * D_edge * L_ref * dtheta

    wp.atomic_add(F_out, ea, coeff * grad_a)
    wp.atomic_add(F_out, eb, coeff * grad_b)
    wp.atomic_add(F_out, ec, coeff * grad_c)
    wp.atomic_add(F_out, ed, coeff * grad_d)


# ── Kernel 3: Velocity-Verlet step (per node) ─────────────────────────
@wp.kernel
def bst_position_update(
    dt: wp.float64,
    u: wp.array(dtype=wp.vec3d),
    v: wp.array(dtype=wp.vec3d),
    a: wp.array(dtype=wp.vec3d),
    mass_inv: wp.array(dtype=wp.float64),
):
    """Velocity-Verlet position update: u += v*dt + 0.5*a*dt^2."""
    tid = wp.tid()
    if mass_inv[tid] < _F64_EPS:
        u[tid] = wp.vec3d(_F64_ZERO, _F64_ZERO, _F64_ZERO)
        v[tid] = wp.vec3d(_F64_ZERO, _F64_ZERO, _F64_ZERO)
        a[tid] = wp.vec3d(_F64_ZERO, _F64_ZERO, _F64_ZERO)
        return
    dt2_half = _F64_HALF * dt * dt
    ui = u[tid]
    vi = v[tid]
    ai = a[tid]
    u[tid] = wp.vec3d(ui[0] + vi[0] * dt + ai[0] * dt2_half,
                       ui[1] + vi[1] * dt + ai[1] * dt2_half,
                       ui[2] + vi[2] * dt + ai[2] * dt2_half)


@wp.kernel
def bst_velocity_update(
    F_int: wp.array(dtype=wp.vec3d),
    F_ext: wp.array(dtype=wp.vec3d),
    mass: wp.array(dtype=wp.float64),
    mass_inv: wp.array(dtype=wp.float64),
    damping: wp.float64,
    dt: wp.float64,
    v: wp.array(dtype=wp.vec3d),
    a: wp.array(dtype=wp.vec3d),
):
    """Velocity-Verlet velocity update: v += 0.5*(a_old + a_new)*dt."""
    tid = wp.tid()
    mi = mass_inv[tid]
    if mi < _F64_EPS:
        return

    m = mass[tid]
    vi = v[tid]
    ai = a[tid]

    F_damp = wp.float64(-1.0) * damping * m * vi
    a_new = (F_int[tid] + F_ext[tid] + F_damp) * mi

    half_dt = _F64_HALF * dt
    v[tid] = wp.vec3d(vi[0] + half_dt * (ai[0] + a_new[0]),
                       vi[1] + half_dt * (ai[1] + a_new[1]),
                       vi[2] + half_dt * (ai[2] + a_new[2]))
    a[tid] = a_new


# ── GPU context ────────────────────────────────────────────────────────
class BSTGPUContext:
    """Persistent GPU arrays for BST shell."""

    def __init__(self, shell):
        self.nv = shell.nv
        self.nt = shell.nt
        self.ne = shell.n_interior_edges

        # Reference positions
        self.wp_x0 = wp.array(shell.vertices0, dtype=wp.vec3d)

        # Triangle connectivity (flat: nt*3)
        self.wp_tri = wp.array(
            shell.triangles.flatten().astype(np.int32), dtype=wp.int32)

        # Shape function derivatives (flat: nt*3)
        self.wp_dNdx = wp.array(
            np.ascontiguousarray(shell._dNdx.flatten()), dtype=wp.float64)
        self.wp_dNdy = wp.array(
            np.ascontiguousarray(shell._dNdy.flatten()), dtype=wp.float64)

        # Reference area
        self.wp_ref_area = wp.array(shell._ref_area, dtype=wp.float64)

        # Interior edge arrays
        self.wp_edge_ea = wp.array(shell._edge_ea, dtype=wp.int32)
        self.wp_edge_eb = wp.array(shell._edge_eb, dtype=wp.int32)
        self.wp_edge_ec = wp.array(shell._edge_ec, dtype=wp.int32)
        self.wp_edge_ed = wp.array(shell._edge_ed, dtype=wp.int32)
        self.wp_edge_L = wp.array(shell._edge_L, dtype=wp.float64)
        self.wp_edge_theta_ref = wp.array(
            shell._edge_theta_ref, dtype=wp.float64)
        self.wp_edge_A0 = wp.array(shell._edge_A0, dtype=wp.float64)
        self.wp_edge_A1 = wp.array(shell._edge_A1, dtype=wp.float64)
        self.wp_edge_D = wp.array(shell._edge_D, dtype=wp.float64)

        # Mass
        self.wp_mass = wp.array(shell.mass, dtype=wp.float64)
        self.wp_mass_inv = wp.array(shell.mass_inv, dtype=wp.float64)

        # Material scalars
        self.h = float(shell.h)
        self.D00 = float(shell._D00)
        self.D01 = float(shell._D01)
        self.D11 = float(shell._D11)
        self.D22 = float(shell._D22)
        self.damping = float(shell.damping)

        # State on GPU
        self.wp_u = wp.array(shell.u, dtype=wp.vec3d)
        self.wp_v = wp.array(shell.v, dtype=wp.vec3d)
        self.wp_a = wp.array(shell.a, dtype=wp.vec3d)

    def upload_state(self, shell):
        self.wp_u = wp.array(shell.u, dtype=wp.vec3d)
        self.wp_v = wp.array(shell.v, dtype=wp.vec3d)
        self.wp_a = wp.array(shell.a, dtype=wp.vec3d)

    def download_state(self, shell):
        shell.u[:] = self.wp_u.numpy().view(np.float64).reshape(self.nv, 3)
        shell.v[:] = self.wp_v.numpy().view(np.float64).reshape(self.nv, 3)
        shell.a[:] = self.wp_a.numpy().view(np.float64).reshape(self.nv, 3)


# ── Subcycling wrapper ─────────────────────────────────────────────────
def bst_subcycle_gpu(ctx, F_ext_np, dt_sub, n_sub):
    """Run n_sub substeps on GPU. Single upload, single download."""
    wp_F_ext = wp.array(F_ext_np, dtype=wp.vec3d)

    for _ in range(n_sub):
        # 1. Position update: u += v*dt + 0.5*a*dt^2
        wp.launch(
            bst_position_update,
            dim=ctx.nv,
            inputs=[
                dt_sub,
                ctx.wp_u, ctx.wp_v, ctx.wp_a,
                ctx.wp_mass_inv,
            ],
        )

        # 2. Compute forces at new position
        wp_F_int = wp.zeros(ctx.nv, dtype=wp.vec3d)

        # Membrane forces (per triangle)
        wp.launch(
            bst_membrane_forces,
            dim=ctx.nt,
            inputs=[
                ctx.wp_u, ctx.wp_x0, ctx.wp_tri,
                ctx.wp_dNdx, ctx.wp_dNdy, ctx.wp_ref_area,
                ctx.h, ctx.D00, ctx.D01, ctx.D11, ctx.D22,
                wp_F_int,
            ],
        )

        # Bending forces (per interior edge)
        wp.launch(
            bst_bending_forces,
            dim=ctx.ne,
            inputs=[
                ctx.wp_u, ctx.wp_x0,
                ctx.wp_edge_ea, ctx.wp_edge_eb,
                ctx.wp_edge_ec, ctx.wp_edge_ed,
                ctx.wp_edge_L, ctx.wp_edge_theta_ref,
                ctx.wp_edge_A0, ctx.wp_edge_A1,
                ctx.wp_edge_D,
                wp_F_int,
            ],
        )

        # 3. Velocity update: v += 0.5*(a_old + a_new)*dt
        wp.launch(
            bst_velocity_update,
            dim=ctx.nv,
            inputs=[
                wp_F_int, wp_F_ext,
                ctx.wp_mass, ctx.wp_mass_inv,
                ctx.damping, dt_sub,
                ctx.wp_v, ctx.wp_a,
            ],
        )
