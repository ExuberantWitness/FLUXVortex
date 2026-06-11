"""W2 unit tests: prescribed root motion in ANCFShell.

T1  static red line: with no prescribed callback the step_newmark results are
    bit-identical to before the extension (guarded code path).
T2  rigid-follow: a very stiff cantilever plate whose root edge is driven by a
    slow rotation about the x-axis follows the rotation rigidly (tip lag -> 0).

Run: pytest tests/test_prescribed_bc.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fluxvortex.ancf_shell import ANCFShell  # noqa: E402


def _build_plate(nx=4, ny=3, L=1.0, W=0.6, stiff=1.0):
    x = np.arange(nx + 1) / nx * L
    y = np.arange(ny + 1) / ny * W
    nn = (nx + 1) * (ny + 1)
    nodes = np.zeros((nn, 3))
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes[j * (nx + 1) + i, 0] = x[i]
            nodes[j * (nx + 1) + i, 1] = y[j]
    quads = np.zeros((nx * ny, 4), dtype=np.int32)
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            quads[j * nx + i] = (n1, n1 + 1, n1 + nx + 2, n1 + nx + 1)
    E = 2e9 * stiff
    return ANCFShell(nodes, quads, h=1e-3, rho=1200.0, Ex=E, Ey=E, nu_xy=0.3)


def _root_nodes(shell, nx, ny):
    # nodes are laid out j-outer (spanwise y) — pick the y=0 chord edge
    nodes = [n for n in range(shell.nn) if abs(shell.nodes[n, 1]) < 1e-12]
    assert len(nodes) == nx + 1
    return nodes


def _rotation_callback(shell, nodes, A, Omega):
    """Smooth-start rigid rotation about x: theta(t) = A*(1 - cos(Omega*t)).
    theta_dot(0) = 0 -> no impulsive start (no spurious elastic ringing)."""
    pd = np.array([9 * n + d for n in sorted(nodes) for d in range(9)])
    q0 = shell.q[pd].reshape(-1, 3, 3)   # per node: [r, dx_r, dy_r]

    def cb(t):
        th = A * (1.0 - np.cos(Omega * t))
        thd = A * Omega * np.sin(Omega * t)
        thdd = A * Omega ** 2 * np.cos(Omega * t)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        Rp = np.array([[0, 0, 0], [0, -s, -c], [0, c, -s]])      # dR/dth
        Rpp = np.array([[0, 0, 0], [0, -c, s], [0, -s, -c]])     # d2R/dth2
        dR = thd * Rp
        ddR = thdd * Rp + thd ** 2 * Rpp
        return ((q0 @ R.T).reshape(-1), (q0 @ dR.T).reshape(-1),
                (q0 @ ddR.T).reshape(-1))

    return cb, pd


def test_static_red_line():
    """No callback => identical trajectory to plain clamped stepping."""
    nx, ny = 4, 3
    a = _build_plate(nx, ny)
    b = _build_plate(nx, ny)
    root_a = _root_nodes(a, nx, ny)
    a.set_bc(root_a, fix_slopes=True)
    b.set_prescribed_motion(_root_nodes(b, nx, ny), callback=None)
    F = np.zeros(a.ndof)
    F[2::9] = 0.05  # small z load on all nodes
    for k in range(10):
        a.step_newmark(F, 1e-3)
        b.step_newmark(F, 1e-3, t_end=(k + 1) * 1e-3)
    assert np.array_equal(a.q, b.q), "static red line broken (q differs)"
    assert np.array_equal(a.dq, b.dq), "static red line broken (dq differs)"


def test_rigid_follow():
    """Stiff plate driven at the root follows the rotation; tip lag small."""
    nx, ny = 4, 3
    shell = _build_plate(nx, ny, stiff=50.0)
    nodes = _root_nodes(shell, nx, ny)
    A, Omega = 0.15, 2.0   # theta_max ~ 0.3 rad, slow vs structure freqs
    cb, pd = _rotation_callback(shell, nodes, A, Omega)
    shell.set_prescribed_motion(nodes, cb)
    dt = 2e-4
    n_steps = 500     # t_end = 0.1 s
    F = np.zeros(shell.ndof)
    for k in range(n_steps):
        shell.step_newmark(F, dt, t_end=(k + 1) * dt)
    t = n_steps * dt
    th = A * (1.0 - np.cos(Omega * t))
    # tip node = far corner; rigid prediction of its position
    tip = int(np.argmax(shell.nodes[:, 0] + shell.nodes[:, 1]))
    r0 = shell.nodes[tip].copy()
    c, s = np.cos(th), np.sin(th)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    r_rigid = R @ r0
    r_now = shell.q[9 * tip:9 * tip + 3]
    err = np.linalg.norm(r_now - r_rigid) / np.linalg.norm(r0)
    assert err < 5e-3, f"stiff plate should follow root rotation, lag={err:.2e}"
    # and the root itself is exactly on the prescribed path
    rootn = nodes[0]
    r_root = shell.q[9 * rootn:9 * rootn + 3]
    assert np.linalg.norm(r_root - R @ shell.nodes[rootn]) < 1e-12
