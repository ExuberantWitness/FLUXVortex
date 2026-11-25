"""S0-bench (plan V3, stage 3): what one step of the rigid plate costs, phase by phase.

Times the same production loop as diagnose_rigid_plate_coefficients.py (frozen rigid queue,
half-cosine start-up ramp) and splits each step into its phases by wrapping the solver's own
entry points.  Every ring-filament evaluation is attributed to the phase it happens in:

  convection   `_advance_material_state`: the Runge-Kutta convection of the wake and particles
  matrix       `native_aic`: the bound influence matrix
  wake+solve   the rest of the circulation solve: wake and nascent row at the collocation
               points, the linear solves, the gates
  loads        the author load assembly (`author_load_assembler.assemble`)
  packet       the resolved load packet (`assemble_packet`), whichever caller asks for it
  refine       the rigid Kutta-Joukowski refinement, its packet aside
  rest         everything else (kinematics, ledgers, the wake insertion, the commit)

The filament evaluation count (points x rings x legs) and its rate are the numbers a precision
or tiling change has to move.  A device synchronisation around every wrapped call makes the
phases a little more expensive than the bare loop, so the bare step total is measured first,
without wrappers.

Usage
-----
    PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 \\
        python3 platform/warp_vpm/bench_rigid_step_v5m.py --alpha 5 --steps 60
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for sub in ("src", "platform", "platform/warp_vpm"):
    sys.path.insert(0, str(REPO_ROOT / sub))

PHASES = ("convection", "matrix", "wake+solve", "loads", "packet", "refine", "rest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=20, help="steps run before timing starts")
    parser.add_argument("--steps", type=int, default=60, help="timed steps")
    parser.add_argument("--ramp-steps", type=int, default=100)
    parser.add_argument("--dt-star", type=float, default=None,
                        help="gate-style override: dt* with the wake held at 3 chords")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only")

    import queue_roj_rigid_fig9_12_13_15 as frozen
    import diagnose_roj_rigid_loads as loads_module
    import fluxvortex.warp_fsi.q16_flux_v5m_native as native_module
    from fluxvortex.warp_fsi import kernels_ring_velocity as ring_kernel

    if arguments.dt_star is not None:
        frozen.DT_STAR = arguments.dt_star
        frozen.WAKE_MAX_ROWS = int(round(3.0 / arguments.dt_star))
        frozen.WAKE_FREE_ROWS = int(round(1.0 / arguments.dt_star))
        frozen.PARTICLE_MAX_AGE_STEPS = int(round(1.0 / arguments.dt_star))

    solver, stepper, owner, kin = loads_module.build(arguments.alpha, observed=False)
    solver.v_inf = solver.v_inf.to("cuda:0")
    dt = float(solver.settings.aerodynamic_dt)
    speed = 5.0
    base_v_inf = solver.v_inf.clone()
    ramp = max(arguments.ramp_steps, 1)

    def step_once(step: int):
        factor = 1.0 if step >= ramp else 0.5 * (1.0 - math.cos(math.pi * step / ramp))
        solver.v_inf = base_v_inf * factor
        object.__setattr__(solver.settings, "freestream", speed * factor)
        return stepper.propose(owner, (kin.evaluate(step * dt),), dt)

    def run(count: int, first_step: int):
        proposal = None
        for index in range(count):
            proposal = step_once(first_step + index)
            stepper.commit(owner, proposal)
        return proposal

    # ---- bare loop ------------------------------------------------------------
    run(arguments.warmup, 1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    proposal = run(arguments.steps, arguments.warmup + 1)
    torch.cuda.synchronize()
    bare_total = time.perf_counter() - started
    wake_size = int(proposal.trial_state.wake_rings.shape[0])

    # ---- wrapped loop ---------------------------------------------------------
    seconds = {phase: 0.0 for phase in PHASES}
    calls = {phase: 0 for phase in PHASES}
    evaluations = {phase: 0 for phase in PHASES}
    current = ["rest"]

    def timed(phase: str, function, after: str | None = None):
        def wrapper(*args, **keywords):
            previous = current[0]
            current[0] = phase
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = function(*args, **keywords)
            torch.cuda.synchronize()
            seconds[phase] += time.perf_counter() - begin
            calls[phase] += 1
            current[0] = previous if after is None else after
            return result
        return wrapper

    original_ring_velocity = solver._ring_velocity

    def counting_ring_velocity(points, rings, gamma, **keywords):
        result = original_ring_velocity(points, rings, gamma, **keywords)
        if rings.shape[0]:
            evaluations[current[0]] += int(points.shape[0]) * int(rings.shape[0]) * 4
        return result

    solver._ring_velocity = counting_ring_velocity
    # what follows each of these inside propose() is the circulation solve up to the loads
    solver._advance_material_state = timed("convection", solver._advance_material_state, after="wake+solve")
    native_module.native_aic = timed("matrix", native_module.native_aic, after="wake+solve")
    solver.author_load_assembler.assemble = timed("loads", solver.author_load_assembler.assemble, after="wake+solve")
    packet_targets = []
    resolved = getattr(solver, "_resolved_load_assembler", None)
    if resolved is not None:
        packet_targets.append(resolved)
    rigid_resolved = getattr(solver.author_load_assembler, "_resolved", None)
    if rigid_resolved is not None and all(rigid_resolved is not item for item in packet_targets):
        packet_targets.append(rigid_resolved)
    for target in packet_targets:
        target.assemble_packet = timed("packet", target.assemble_packet)
    solver._refine_load_with_filament_velocity = timed(
        "refine", solver._refine_load_with_filament_velocity)

    first = arguments.warmup + arguments.steps + 1
    torch.cuda.synchronize()
    started = time.perf_counter()
    run(arguments.steps, first)
    torch.cuda.synchronize()
    wrapped_total = time.perf_counter() - started

    covered_per_step = sum(seconds[phase] for phase in PHASES if phase != "rest") / arguments.steps
    per_step = {phase: seconds[phase] / arguments.steps for phase in PHASES}
    per_step["rest"] = max(wrapped_total / arguments.steps - covered_per_step, 0.0)
    per_calls = {phase: calls[phase] / arguments.steps for phase in PHASES}
    per_step_evaluations = {phase: evaluations[phase] / arguments.steps for phase in PHASES}
    total_evaluations = sum(per_step_evaluations.values())
    convection_rate = (
        evaluations["convection"] / seconds["convection"] if seconds["convection"] > 0 else 0.0)
    result = {
        "alpha_deg": arguments.alpha, "warmup_steps": arguments.warmup, "timed_steps": arguments.steps,
        "dt_star": frozen.DT_STAR, "wake_max_rows": frozen.WAKE_MAX_ROWS,
        "wake_free_rows": frozen.WAKE_FREE_ROWS, "wake_rings_during_timing": wake_size,
        "kernel": "v3" if ring_kernel._KERNEL_V3 else "legacy",
        "te_wake": "v3" if native_module._TE_WAKE_V3 else "legacy",
        "wake_core_panels": native_module._WAKE_CORE_PANELS,
        "dtype": str(solver.v_inf.dtype), "device": torch.cuda.get_device_name(0),
        "bare_seconds_per_step": bare_total / arguments.steps,
        "wrapped_seconds_per_step": wrapped_total / arguments.steps,
        "phase_seconds_per_step": per_step, "phase_calls_per_step": per_calls,
        "filament_evaluations_per_step": {phase: int(value) for phase, value in per_step_evaluations.items()},
        "convection_filament_evaluations_per_second": convection_rate,
    }
    share_denominator = max(result["wrapped_seconds_per_step"], 1e-300)
    print(f"rigid plate alpha={arguments.alpha:g} deg, dt* {frozen.DT_STAR:g}, wake {frozen.WAKE_MAX_ROWS} rows "
          f"({frozen.WAKE_FREE_ROWS} free; {wake_size} rings during timing), kernel {result['kernel']}, "
          f"trailing-edge wake {result['te_wake']}, dtype {result['dtype']}, {result['device']}")
    print(f"warm-up {arguments.warmup} steps, then {arguments.steps} timed steps")
    print(f"seconds per step: bare loop {result['bare_seconds_per_step']:.4f}; with the phase timers "
          f"{result['wrapped_seconds_per_step']:.4f}")
    print(f"{'phase':12s} {'s / step':>9} {'share':>7} {'calls / step':>13} {'filament evaluations / step':>26}")
    for phase in PHASES:
        print(f"{phase:12s} {per_step[phase]:9.4f} {100.0 * per_step[phase] / share_denominator:6.1f}% "
              f"{per_calls[phase]:13.1f} {int(per_step_evaluations[phase]):26d}")
    print(f"filament evaluations per step {int(total_evaluations):,}; in the convection "
          f"{int(per_step_evaluations['convection']):,} at {convection_rate / 1e9:.2f} billion per second")
    if arguments.out:
        Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
