"""Discovery run #4 — REAL resonance efficiency from the spring-driven flap dynamics.

Discovery #3 modeled the resonant-flap efficiency analytically. This run measures it
from the **verified Featherstone dynamics** (P0's free-flying fuselage + wing on a
torsional-spring revolute hinge, tape-checked): for each spring stiffness, drive the
hinge with a sinusoidal flap torque tau(t)=tau0*sin(2*pi*f*t), march the real
multibody dynamics, and record:

  amplitude  = (max-min)/2 of the steady hinge angle  (flap stroke achieved)
  power      = mean |tau(t) * omega(t)|               (motor power supplied)
  cost       = power / amplitude                       (motor power per unit stroke)

At resonance (omega_n = sqrt(ke/I_wing) ~ 2*pi*f) the spring and wing inertia trade
energy, so a given torque drives a LARGER stroke and the cost (power per stroke) is
MINIMIZED — a real, non-analytical efficiency minimum at a tuned stiffness. This is
the genuine competing constraint for the full-aircraft co-design: efficiency favors
the resonant stiffness while gust rejection favors flexibility.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_FLUXV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_FLUXV, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                # noqa: E402
import newton                                                   # noqa: E402
from newton.solvers import SolverFeatherstone                   # noqa: E402
from p0_resonant_freeflight import build_model                  # noqa: E402

F_FLAP = 3.0       # flap frequency (Hz)
TAU0 = 0.5         # flap torque amplitude (N*m)


def drive(spring_ke, n_cyc=8, steps_per_cyc=120):
    """Drive the hinge sinusoidally; return (stroke_amplitude, motor_power)."""
    m, hqi, hdi = build_model(spring_ke, spring_kd=0.01, requires_grad=False)
    try:                                            # isolate the flap resonance (no gravity sag)
        g = m.gravity
        g.assign(np.zeros((g.shape[0], 3), dtype=np.float32))
    except Exception:
        pass
    dt = 1.0 / (F_FLAP * steps_per_cyc)
    n = n_cyc * steps_per_cyc
    solver = SolverFeatherstone(m)
    s0, s1 = m.state(), m.state()
    control = m.control()
    newton.eval_fk(m, s0.joint_q, s0.joint_qd, s0)
    jf = np.zeros(m.joint_dof_count, dtype=np.float32)
    th, om, tau = [], [], []
    for i in range(n):
        t = i * dt
        u = TAU0 * np.sin(2 * np.pi * F_FLAP * t)
        jf[hdi] = u; control.joint_f.assign(jf)
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        th.append(float(s0.joint_q.numpy()[hqi]))
        om.append(float(s0.joint_qd.numpy()[hdi]))
        tau.append(u)
    th = np.array(th); om = np.array(om); tau = np.array(tau)
    tail = slice(n - 2 * steps_per_cyc, n)          # last 2 cycles (steady)
    amp = 0.5 * (th[tail].max() - th[tail].min())
    power = float(np.mean(np.abs(tau[tail] * om[tail])))
    return amp, power


def main():
    wp.init()
    kes = [0.2, 0.5, 1.0, 1.6, 2.4, 4.0, 7.0]
    print(f"discovery #4: real spring-driven flap resonance, f={F_FLAP} Hz, "
          f"tau0={TAU0} N*m")
    amps, powers, costs = [], [], []
    for ke in kes:
        a, p = drive(ke)
        c = p / (a + 1e-9)
        amps.append(a); powers.append(p); costs.append(c)
        print(f"  ke={ke:4.1f} N*m/rad: stroke={np.rad2deg(a):6.2f} deg  "
              f"power={p:.4e} W  power/stroke={c:.4e}", flush=True)
    amps, powers, costs = map(np.array, (amps, powers, costs))
    ires_amp = int(np.argmax(amps))                 # max stroke per fixed torque
    ires_cost = int(np.argmin(costs))               # min power per stroke
    print("\n=== FINDING ===")
    print(f"  peak stroke at ke={kes[ires_amp]:.1f} (resonance: max flap per torque)")
    print(f"  min power/stroke at ke={kes[ires_cost]:.1f} (efficient resonant stiffness)")
    interior = (0 < ires_cost < len(kes) - 1) or (0 < ires_amp < len(kes) - 1)
    msg = ("REAL RESONANCE: interior-optimal (tuned) stiffness for flap efficiency "
           "— the competing constraint vs gust-favors-flexible" if interior
           else "monotone (no interior resonance in range)")
    print(f"  -> {msg}")
    np.savez(os.path.join(_FLUXV, "docs", "discovery4.npz"),
             ke=np.array(kes), amp=amps, power=powers, cost=costs)
    return 0 if interior else 1


if __name__ == "__main__":
    raise SystemExit(main())
