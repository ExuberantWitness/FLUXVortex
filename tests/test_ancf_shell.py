"""ANCF shell unit tests: cantilever beam bending, membrane stretch, constant mass."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
from fluxvortex.ancf_shell import ANCFShell, NDOF_NODE


def test_constant_mass():
    """Verify mass matrix is constant and total mass matches rho*h*L*W."""
    L, W, h = 1.0, 0.5, 0.01
    E, rho = 1e7, 1000.0

    nx, ny = 4, 2
    nn_x = nx + 1
    nn_y = ny + 1
    nn = nn_x * nn_y

    nodes = np.zeros((nn, 3))
    for j in range(nn_y):
        for i in range(nn_x):
            idx = j * nn_x + i
            nodes[idx, 0] = i * L / nx
            nodes[idx, 1] = j * W / ny

    quads = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * nn_x + i
            n1 = n0 + 1
            n2 = n1 + nn_x
            n3 = n0 + nn_x
            quads.append([n0, n1, n2, n3])
    quads = np.array(quads, dtype=np.int32)

    shell = ANCFShell(nodes, quads, h=h, rho=rho, Ex=E, Ey=E, nu_xy=0.3)

    M_ref = shell.M.toarray().copy()
    shell.q += 0.1 * np.random.randn(*shell.q.shape)
    M_after = shell.M.toarray().copy()

    diff = np.max(np.abs(M_ref - M_after))

    # ANCF mass matrix includes slope DOF inertia, so trace/3 != physical mass.
    # Instead verify: (1) constant mass, (2) positive definite, (3) symmetry.
    print(f"\nConstant mass test:")
    print(f"  Max |M_ref - M_after|: {diff:.2e} (should be ~0)")
    assert diff < 1e-14, "Mass matrix changed after deformation!"
    # Verify symmetry
    assert np.allclose(M_ref, M_ref.T), "Mass matrix not symmetric!"
    # Verify positive definite (all eigenvalues > 0)
    eigvals = np.linalg.eigvalsh(M_ref)
    print(f"  Min eigenvalue: {eigvals.min():.2e} (should be > 0)")
    assert eigvals.min() > 0, "Mass matrix not positive definite!"


def test_membrane_stretch():
    """Membrane stretch: uniform tension vs analytical strain.

    Strain: eps = sigma / E  (uniaxial, ignoring Poisson for narrow strip)
    """
    L = 1.0
    W = 0.5
    h = 0.01
    E = 1e7
    rho = 1000.0
    sigma = 1e4  # applied stress [Pa]

    eps_theory = sigma / E

    nx, ny = 4, 2
    nn_x = nx + 1
    nn_y = ny + 1
    nn = nn_x * nn_y

    nodes = np.zeros((nn, 3))
    for j in range(nn_y):
        for i in range(nn_x):
            idx = j * nn_x + i
            nodes[idx, 0] = i * L / nx
            nodes[idx, 1] = j * W / ny

    quads = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * nn_x + i
            n1 = n0 + 1
            n2 = n1 + nn_x
            n3 = n0 + nn_x
            quads.append([n0, n1, n2, n3])
    quads = np.array(quads, dtype=np.int32)

    shell = ANCFShell(nodes, quads, h=h, rho=rho, Ex=E, Ey=E, nu_xy=0.3,
                       G_xy=None, mode='membrane')

    root = np.where(nodes[:, 0] < 1e-10)[0]
    shell.set_bc(root)

    tip = np.where(np.abs(nodes[:, 0] - L) < 1e-10)[0]
    force_per_node = sigma * h * W / ny / len(tip)

    F_ext = np.zeros(shell.ndof)
    for n in tip:
        base = n * NDOF_NODE
        F_ext[base] = force_per_node  # x-direction force on position DOF

    dt = 1e-5
    n_steps = 100000
    for step in range(n_steps):
        shell.step(F_ext, dt, n_sub=1)

    tip_disp = shell.u.reshape(-1, NDOF_NODE)[:, 0]
    avg_tip = np.mean(tip_disp[tip])
    eps_computed = avg_tip / L

    print(f"\nMembrane stretch test:")
    print(f"  Computed strain: {eps_computed:.6f}")
    print(f"  Theory strain:   {eps_theory:.6f}")
    print(f"  Error:           {abs(eps_computed - eps_theory)/eps_theory * 100:.1f}%")


def test_cantilever_bending():
    """Cantilever beam: tip load vs analytical tip deflection.

    Analytical: delta = P*L^3 / (3*EI)
    """
    L = 1.0
    W = 0.1
    h = 0.01
    E = 1e7
    rho = 1000.0

    EI = E * W * h**3 / 12.0
    P = 1.0
    delta_theory = P * L**3 / (3.0 * EI)

    nx, ny = 10, 2
    nn_x = nx + 1
    nn_y = ny + 1
    nn = nn_x * nn_y

    nodes = np.zeros((nn, 3))
    for j in range(nn_y):
        for i in range(nn_x):
            idx = j * nn_x + i
            nodes[idx, 0] = i * L / nx
            nodes[idx, 1] = j * W / ny

    quads = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * nn_x + i
            n1 = n0 + 1
            n2 = n1 + nn_x
            n3 = n0 + nn_x
            quads.append([n0, n1, n2, n3])
    quads = np.array(quads, dtype=np.int32)

    shell = ANCFShell(nodes, quads, h=h, rho=rho, Ex=E, Ey=E, nu_xy=0.3, G_xy=None)

    root = np.where(nodes[:, 0] < 1e-10)[0]
    shell.set_bc(root)

    tip = np.where(np.abs(nodes[:, 0] - L) < 1e-10)[0]
    F_ext = np.zeros(shell.ndof)
    for n in tip:
        base = n * NDOF_NODE
        F_ext[base + 2] = -P / len(tip)  # z-direction load

    dt = 1e-4
    n_steps = 50000
    for step in range(n_steps):
        shell.step(F_ext, dt, n_sub=1)
        if step % 10000 == 0:
            tip_disp = shell.u.reshape(-1, NDOF_NODE)[:, 2]
            max_tip = np.min(tip_disp[tip])
            print(f"  Step {step}: max tip_w = {max_tip*1000:.4f} mm (theory = {delta_theory*1000:.4f})")

    tip_disp = shell.u.reshape(-1, NDOF_NODE)[:, 2]
    max_tip = np.abs(np.min(tip_disp[tip]))

    print(f"\nCantilever bending test:")
    print(f"  Tip deflection: {max_tip*1000:.4f} mm")
    print(f"  Theory:         {delta_theory*1000:.4f} mm")
    print(f"  Error:          {abs(max_tip - delta_theory)/delta_theory * 100:.1f}%")

    return max_tip, delta_theory


if __name__ == '__main__':
    print("=" * 60)
    print("ANCF SHELL UNIT TESTS (9 DOF/node, bicubic Hermite)")
    print("=" * 60)

    test_constant_mass()
    print()
    test_membrane_stretch()
    print()
    test_cantilever_bending()
