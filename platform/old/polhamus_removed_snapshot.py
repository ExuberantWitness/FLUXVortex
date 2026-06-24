"""ISOLATED EMPIRICAL CODE (Polhamus LEV + separation cap + induced-drag attempt) — REMOVED from the
active first-principles model on 2026-06-24 per directive. Kept for reference only. The production
gpu_run_twist now uses ONLY: standard unsteady UVLM (circulation KJ + added-mass dGamma/dt) + REAL
discrete leading-edge vortex shedding (real_lev) + (coming) first-principles viscous term."""

# ---- _lev_kernel (Polhamus C_Nv=K_v sin^2 a, empirical K_v/onset/ds) from _v2_robo.py ----
@wp.kernel
def _lev_kernel(rings: wp.array(dtype=V3, ndim=2), nrm: wp.array(dtype=V3), vcol: wp.array(dtype=V3),
                Vinf: V3, rho: DTYPE, K_v: DTYPE, sin_onset: DTYPE, sin_ds: DTYPE,
                out: wp.array(dtype=DTYPE)):
    """Dynamic-stall LEV (Polhamus): per-panel vortex normal force C_Nv=K_v*sin^2(a_eff) (signed) at the
    LOCAL dynamic pressure when |sin a_eff|>sin_onset, SATURATED beyond the deep-stall angle (sin_ds): the
    LEV detaches so its contribution caps (matches the measured lift plateau at high AoA, not unbounded)."""
    p = wp.tid()
    vr = Vinf - vcol[p]
    vmag = wp.length(vr) + wp.float64(1.0e-9)
    n = nrm[p]
    sina = wp.dot(vr, n) / vmag
    sa = wp.abs(sina)
    if sa > sin_onset:
        sc = wp.min(sa, sin_ds)              # cap magnitude at deep stall (LEV detaches beyond)
        sgn = wp.float64(1.0)
        if sina < wp.float64(0.0):
            sgn = wp.float64(-1.0)
        cr = wp.cross(rings[p, 2] - rings[p, 0], rings[p, 3] - rings[p, 1])
        area = wp.float64(0.5) * wp.length(cr)
        dN = K_v * sgn * sc * sc * wp.float64(0.5) * rho * vmag * vmag * area
        wp.atomic_add(out, 0, dN * n[2])     # vertical (lift), dihedral-projected
        wp.atomic_add(out, 1, dN * n[0])     # streamwise (LE suction -> thrust)


# ---- panel_force_ind_kernel + thrust_sep_cap_kernel from diff_uvlm_unsteady_gpu.py ----
@wp.kernel
def panel_force_ind_kernel(rings: wp.array(dtype=V3, ndim=2), nrm: wp.array(dtype=V3),
                           gamma: wp.array(dtype=DTYPE, ndim=2), gprev: wp.array(dtype=DTYPE, ndim=2),
                           vcol: wp.array(dtype=V3), Vinf: V3, dt: DTYPE, rho: DTYPE, ns: int,
                           wr: wp.array(dtype=V3, ndim=2), wg: wp.array(dtype=DTYPE), nw: int,
                           Fp: wp.array(dtype=V3)):
    """panel_force_kernel + the WAKE-induced velocity at the leading bound segment, so the KJ force
    tilts back into INDUCED DRAG. V_local = V∞ − V_body + Σ_wake induced; the streamwise component is
    the induced drag the V_rel-only kernel omits (which otherwise leaves the inviscid Knoller-Betz
    flapping thrust uncancelled -> grossly over-positive thrust)."""
    p = wp.tid()
    gnet = gamma[0, p]
    if p // ns > 0:
        gnet = gamma[0, p] - gamma[0, p - ns]
    lb = rings[p, 1] - rings[p, 0]
    mid = wp.float64(0.5) * (rings[p, 0] + rings[p, 1])
    vx = wp.float64(0.0); vy = wp.float64(0.0); vz = wp.float64(0.0)   # scalar accum (clean adjoint)
    for k in range(nw):
        vv = wg[k] * ring_vel(mid, wr[k, 0], wr[k, 1], wr[k, 2], wr[k, 3])
        vx = vx + vv[0]; vy = vy + vv[1]; vz = vz + vv[2]
    vrel = Vinf - vcol[p] + V3(vx, vy, vz)
    Fkj = rho * gnet * wp.cross(vrel, lb)
    cr = wp.cross(rings[p, 2] - rings[p, 0], rings[p, 3] - rings[p, 1])
    area = wp.float64(0.5) * wp.sqrt(wp.dot(cr, cr) + wp.float64(1.0e-30))
    dGdt = (gamma[0, p] - gprev[0, p]) / dt
    Fp[p] = Fkj + rho * dGdt * area * nrm[p]


@wp.kernel
def thrust_sep_cap_kernel(Fp: wp.array(dtype=V3), nrm: wp.array(dtype=V3), vcol: wp.array(dtype=V3),
                          Vinf: V3, sin_onset: DTYPE):
    """Separation cap on the streamwise THRUST: where the panel is separated (|sin a_eff|>sin_onset) the
    LE suction saturates (2D-validated A0->A0_crit), so the inviscid KJ thrust (~sin^2) must not keep
    growing. Scale the FORWARD (-x) part of the panel force by (sin_onset/|sin a_eff|)^2 -> capped at the
    onset (A0_crit) value. Leaves drag (+x) and lift (z) untouched."""
    p = wp.tid()
    vr = Vinf - vcol[p]
    vmag = wp.length(vr) + wp.float64(1.0e-9)
    sa = wp.abs(wp.dot(vr, nrm[p]) / vmag)
    if sa > sin_onset:
        fac = (sin_onset / sa) * (sin_onset / sa)
        f = Fp[p]
        fx = f[0]
        if fx < wp.float64(0.0):        # -x = forward = thrust (T = -Fx); cap only the thrust part
            fx = fx * fac
        Fp[p] = V3(fx, f[1], f[2])
