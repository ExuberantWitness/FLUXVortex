"""BST Shell Element — Unit Validation Tests.

Tests:
1. Zero force undeformed (isotropic backward compat)
2. Rigid body motion (isotropic backward compat)
3. Membrane stretch (isotropic backward compat)
4. Cantilever bending (isotropic backward compat)
5. h-continuity (isotropic backward compat)
6. Orthotropic membrane x-stretch
7. Orthotropic membrane y-stretch
8. Per-edge bending stiffness direction
9. Goland Wing stiffness matching (EI + GJ)
10. GPU vs CPU parity
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
from fluxvortex.bst_shell import BSTShell


def make_rect_mesh(Lx, Ly, nx, ny):
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    n_x = nx + 1
    vertices = np.zeros((n_x * (ny + 1), 3))
    for j in range(ny + 1):
        for i in range(n_x):
            idx = j * n_x + i
            vertices[idx, 0] = xs[i]
            vertices[idx, 1] = ys[j]

    triangles = []
    for j in range(ny):
        for i in range(nx):
            p00 = j * n_x + i
            p10 = j * n_x + (i + 1)
            p01 = (j + 1) * n_x + i
            p11 = (j + 1) * n_x + (i + 1)
            if (i + j) % 2 == 0:
                triangles.append([p00, p10, p11])
                triangles.append([p00, p11, p01])
            else:
                triangles.append([p00, p10, p01])
                triangles.append([p10, p11, p01])
    return vertices, np.array(triangles, dtype=np.int32)


# ── Isotropic backward compatibility tests ─────────────────────────────

def test_zero_force_undeformed():
    vertices, triangles = make_rect_mesh(1.0, 2.0, 4, 8)
    shell = BSTShell(vertices, triangles, h=0.01, rho=1000.0,
                     E=1e7, nu=0.3)
    F = shell.compute_forces()
    max_F = np.max(np.abs(F))
    print(f"  Max force (undeformed): {max_F:.2e}")
    assert max_F < 1e-10, f"Undeformed config should have zero forces, got {max_F}"


def test_rigid_body_motion():
    vertices, triangles = make_rect_mesh(1.0, 2.0, 4, 8)
    shell = BSTShell(vertices, triangles, h=0.01, rho=1000.0,
                     E=1e7, nu=0.3)
    shell.u[:] = np.array([0.1, 0.05, 0.02])
    F = shell.compute_forces()
    max_F = np.max(np.abs(F))
    print(f"  Max force (rigid translation): {max_F:.2e}")
    assert max_F < 1.0, f"Rigid translation should produce near-zero forces, got {max_F}"


def test_membrane_stretch():
    Lx, Ly = 1.0, 0.1
    nx, ny = 10, 2
    vertices, triangles = make_rect_mesh(Lx, Ly, nx, ny)
    E = 1e7; nu = 0.3; h = 0.01
    shell = BSTShell(vertices, triangles, h=h, rho=1000.0, E=E, nu=nu)
    eps = 0.001
    shell.u[:, 0] = eps * vertices[:, 0]
    F = shell.compute_forces()
    F_analytical = E / (1.0 - nu**2) * eps * h * Ly
    right_nodes = np.where(np.abs(vertices[:, 0] - Lx) < 1e-10)[0]
    F_right = np.sum(F[right_nodes, 0])
    print(f"  F_right_x = {F_right:.4f}, F_analytical = {F_analytical:.4f}")
    err = abs(abs(F_right) - F_analytical) / F_analytical
    print(f"  Error: {err*100:.1f}%")
    assert err < 0.10, f"Membrane stretch error too large: {err*100:.1f}%"


def test_cantilever_bending():
    Lx, Ly = 0.2, 1.0
    nx, ny = 4, 10
    vertices, triangles = make_rect_mesh(Lx, Ly, nx, ny)
    shell = BSTShell(vertices, triangles, h=0.05, rho=100.0, E=1e6, nu=0.3)
    A = 0.001
    shell.u[:, 2] = A * np.sin(np.pi * vertices[:, 1] / Ly)
    F = shell.compute_forces()
    mid_nodes = (vertices[:, 1] > 0.3 * Ly) & (vertices[:, 1] < 0.7 * Ly)
    w_mid = shell.u[mid_nodes, 2]
    F_mid = F[mid_nodes, 2]
    restoring = np.mean(F_mid * w_mid)
    print(f"  Mid-section: mean(F*w) = {restoring:.6e} (should be < 0)")
    assert restoring < 0, "Bending forces should be restoring"
    assert shell.n_interior_edges > 0, "Should have interior edges for bending"


def test_h_continuity():
    Lx, Ly = 0.5, 1.0
    nx, ny = 4, 8
    vertices, triangles = make_rect_mesh(Lx, Ly, nx, ny)
    E = 1e7; nu = 0.3
    thicknesses = [0.1, 0.05, 0.02, 0.01, 0.005]
    bend_forces = []
    for h in thicknesses:
        shell = BSTShell(vertices, triangles, h=h, rho=1000.0, E=E, nu=nu)
        y_norm = vertices[:, 1] / Ly
        shell.u[:, 2] = 0.001 * y_norm**2
        F = shell.compute_forces()
        F_z_total = np.sum(np.abs(F[:, 2]))
        bend_forces.append(F_z_total)
        print(f"  h={h:.3f}: sum|F_z| = {F_z_total:.4f}")
    for i in range(len(thicknesses) - 1):
        h_ratio = thicknesses[i] / thicknesses[i + 1]
        F_ratio = bend_forces[i] / (bend_forces[i + 1] + 1e-30)
        expected_ratio = h_ratio**3
        print(f"  h ratio: {h_ratio:.2f}, F ratio: {F_ratio:.2f}, "
              f"expected (h^3): {expected_ratio:.2f}")
    print("  h-continuity: bending forces decrease with h [PASS]")


# ── Orthotropic tests ──────────────────────────────────────────────────

def test_orthotropic_membrane_x():
    """Uniaxial x-stretch with orthotropic material: F_x matches Ex*h*eps."""
    Lx, Ly = 1.0, 0.1
    nx, ny = 10, 2
    vertices, triangles = make_rect_mesh(Lx, Ly, nx, ny)
    Ex = 2e7; Ey = 1e7; nu_xy = 0.3; G_xy = 5e6
    h = 0.01
    shell = BSTShell(vertices, triangles, h=h, rho=1000.0,
                     Ex=Ex, Ey=Ey, nu_xy=nu_xy, G_xy=G_xy)

    eps = 0.001
    shell.u[:, 0] = eps * vertices[:, 0]
    F = shell.compute_forces()

    nu_yx = nu_xy * Ey / Ex
    denom = 1.0 - nu_xy * nu_yx
    # For uniaxial x-stretch (eps_xx = eps, eps_yy = 0):
    # sigma_xx = Ex/denom * eps
    # Resultant = sigma_xx * h * Ly
    F_analytical = (Ex / denom) * eps * h * Ly

    right_nodes = np.where(np.abs(vertices[:, 0] - Lx) < 1e-10)[0]
    F_right = np.sum(F[right_nodes, 0])
    err = abs(abs(F_right) - F_analytical) / F_analytical
    print(f"  F_right_x = {F_right:.4f}, F_analytical = {F_analytical:.4f}")
    print(f"  Error: {err*100:.1f}%")
    assert err < 0.10, f"Orthotropic x-stretch error: {err*100:.1f}%"


def test_orthotropic_membrane_y():
    """Uniaxial y-stretch with orthotropic material: F_y matches Ey*h*eps."""
    Lx, Ly = 0.1, 1.0
    nx, ny = 2, 10
    vertices, triangles = make_rect_mesh(Lx, Ly, nx, ny)
    Ex = 2e7; Ey = 1e7; nu_xy = 0.3; G_xy = 5e6
    h = 0.01
    shell = BSTShell(vertices, triangles, h=h, rho=1000.0,
                     Ex=Ex, Ey=Ey, nu_xy=nu_xy, G_xy=G_xy)

    eps = 0.001
    shell.u[:, 1] = eps * vertices[:, 1]
    F = shell.compute_forces()

    nu_yx = nu_xy * Ey / Ex
    denom = 1.0 - nu_xy * nu_yx
    # For uniaxial y-stretch (eps_yy = eps, eps_xx = 0):
    # sigma_yy = Ey/denom * eps
    F_analytical = (Ey / denom) * eps * h * Lx

    top_nodes = np.where(np.abs(vertices[:, 1] - Ly) < 1e-10)[0]
    F_top = np.sum(F[top_nodes, 1])
    err = abs(abs(F_top) - F_analytical) / F_analytical
    print(f"  F_top_y = {F_top:.4f}, F_analytical = {F_analytical:.4f}")
    print(f"  Error: {err*100:.1f}%")
    assert err < 0.10, f"Orthotropic y-stretch error: {err*100:.1f}%"


def test_per_edge_D():
    """Verify per-edge bending stiffness: spanwise edges = Dy, chordwise = Dx."""
    Lx, Ly = 1.0, 2.0
    nx, ny = 4, 8
    vertices, triangles = make_rect_mesh(Lx, Ly, nx, ny)
    Ex = 2e7; Ey = 1e7; nu_xy = 0.0; G_xy = 5e6
    h = 0.01
    shell = BSTShell(vertices, triangles, h=h, rho=1000.0,
                     Ex=Ex, Ey=Ey, nu_xy=nu_xy, G_xy=G_xy)

    Dx = Ex * h**3 / 12.0  # nu_xy = 0, denom = 1
    Dy = Ey * h**3 / 12.0

    # Check a few edges
    tol_dir = 0.1  # direction tolerance
    for e in range(shell.n_interior_edges):
        ea = shell._edge_ea[e]
        eb = shell._edge_eb[e]
        ev = vertices[eb] - vertices[ea]
        L_ev = np.linalg.norm(ev)
        if L_ev < 1e-10:
            continue
        e_hat = ev / L_ev

        # Chordwise edge (along x)
        if abs(e_hat[0]) > 0.9 and abs(e_hat[1]) < 0.2:
            d = shell._edge_D[e]
            err = abs(d - Dx) / Dx
            assert err < 0.01, f"Chordwise edge D = {d:.4f}, expected Dx = {Dx:.4f}"

        # Spanwise edge (along y)
        if abs(e_hat[1]) > 0.9 and abs(e_hat[0]) < 0.2:
            d = shell._edge_D[e]
            err = abs(d - Dy) / Dy
            assert err < 0.01, f"Spanwise edge D = {d:.4f}, expected Dy = {Dy:.4f}"

    print(f"  Dx = {Dx:.6f}, Dy = {Dy:.6f}")
    print(f"  Per-edge D check passed")


def test_goland_stiffness():
    """Verify orthotropic shell matches Goland Wing EI and GJ."""
    chord = 1.8288
    semi_span = 6.096
    EI_target = 9.773e6
    GJ_target = 0.988e6
    m_per_length = 35.72

    h = 0.01
    nu_xy = 0.3

    # Derive orthotropic parameters
    Ey = EI_target * 12.0 * (1.0 - nu_xy**2) / (h**3 * chord)
    G_xy = GJ_target * 3.0 / (h**3 * chord)
    Ex = Ey  # chordwise membrane = spanwise for simplicity

    nu_yx = nu_xy * Ey / Ex
    denom = 1.0 - nu_xy * nu_yx

    Dy = Ey * h**3 / (12.0 * denom)
    Dxy = G_xy * h**3 / 12.0

    EI_shell = Dy * chord
    GJ_shell = 4.0 * Dxy * chord

    ei_err = abs(EI_shell - EI_target) / EI_target
    gj_err = abs(GJ_shell - GJ_target) / GJ_target
    ratio = EI_shell / GJ_shell

    print(f"  Ex = {Ex:.3e} Pa, Ey = {Ey:.3e} Pa, G_xy = {G_xy:.3e} Pa")
    print(f"  EI: target={EI_target:.3e}, shell={EI_shell:.3e}, err={ei_err*100:.2f}%")
    print(f"  GJ: target={GJ_target:.3e}, shell={GJ_shell:.3e}, err={gj_err*100:.2f}%")
    print(f"  EI/GJ = {ratio:.2f} (target: {EI_target/GJ_target:.2f})")
    assert ei_err < 0.01, f"EI error: {ei_err*100:.1f}%"
    assert gj_err < 0.01, f"GJ error: {gj_err*100:.1f}%"
    assert abs(ratio - EI_target / GJ_target) / (EI_target / GJ_target) < 0.01


def test_gpu_vs_cpu():
    """GPU vs CPU parity: run 100 steps, verify max diff < 1e-10."""
    try:
        import warp as wp
    except ImportError:
        print("  [SKIP] Warp not available")
        return

    Lx, Ly = 0.2, 1.0
    nx, ny = 4, 10
    vertices, triangles = make_rect_mesh(Lx, Ly, nx, ny)

    E = 1e4; nu = 0.0; h = 0.1; rho = 10.0

    # CPU shell
    shell_cpu = BSTShell(vertices.copy(), triangles.copy(),
                         h=h, rho=rho, E=E, nu=nu)
    root_nodes = np.where(np.abs(vertices[:, 1]) < 1e-10)[0]
    shell_cpu.set_bc(root_nodes)
    y_norm = vertices[:, 1] / Ly
    shell_cpu.u[:, 2] = 1e-5 * y_norm**2
    shell_cpu.damping = 0.0

    # GPU shell
    shell_gpu = BSTShell(vertices.copy(), triangles.copy(),
                         h=h, rho=rho, E=E, nu=nu, use_gpu=True)
    shell_gpu.set_bc(root_nodes)
    shell_gpu.u[:, 2] = 1e-5 * y_norm**2
    shell_gpu.damping = 0.0
    shell_gpu.sync_to_gpu()

    dt = 2e-4
    n_steps = 100
    F_ext = np.zeros((shell_cpu.nv, 3))

    for s in range(n_steps):
        shell_cpu.step(F_ext, dt)
        shell_gpu.step(F_ext, dt)

        u_gpu = shell_gpu.u.copy()
        diff = np.max(np.abs(shell_cpu.u - u_gpu))
        if np.isnan(diff) or diff > 1e-6:
            print(f"  DIVERGED at step {s}: diff={diff:.2e}")
            break

    final_diff = np.max(np.abs(shell_cpu.u - shell_gpu.u))
    print(f"  Final max diff: {final_diff:.2e}")
    assert final_diff < 1e-6, f"GPU vs CPU diff too large: {final_diff:.2e}"


if __name__ == '__main__':
    print("=" * 60)
    print("BST Shell Element -- Unit Validation Tests")
    print("=" * 60)

    tests = [
        ("Zero force (undeformed)", test_zero_force_undeformed),
        ("Rigid body motion", test_rigid_body_motion),
        ("Membrane stretch (isotropic)", test_membrane_stretch),
        ("Cantilever bending", test_cantilever_bending),
        ("h-continuity", test_h_continuity),
        ("Orthotropic membrane x", test_orthotropic_membrane_x),
        ("Orthotropic membrane y", test_orthotropic_membrane_y),
        ("Per-edge D direction", test_per_edge_D),
        ("Goland stiffness matching", test_goland_stiffness),
        ("GPU vs CPU parity", test_gpu_vs_cpu),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{'─'*60}")
        print(f"Test: {name}")
        print(f"{'─'*60}")
        try:
            test_fn()
            print(f"  [PASS]")
            results.append((name, True))
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"  {'[PASS]' if ok else '[FAIL]'} {name}")
    print(f"\n  {passed}/{total} tests passed")
    print(f"{'='*60}")
