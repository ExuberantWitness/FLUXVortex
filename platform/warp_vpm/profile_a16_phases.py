"""S2: phase profiling for ROJ11-A16 (measurement only -- no production edits).

Why this file exists
--------------------
The pre-S1 speed ledger mixed two kinds of numbers: measured (factorizations, pairs,
wall clock) and derived (``propose`` cost extrapolated by N^2 from a 30-step window).
It also had no ring-versus-particle split, even though ~18 of the ~28 heavy kernel
launches per propose are ring sums while the 9.63e9 pair ledger only covers particle
Biot-Savart.  This script replaces the derived numbers with CUDA-event measurements
and is intended to be the single authoritative baseline table for S3/S4/S5.

Design notes that matter for the numbers being trustworthy
----------------------------------------------------------
* Timing is ``torch.cuda.Event``: a plain ``perf_counter`` around an async torch call
  measures ENQUEUE, not execution (that mistake made "LU is 1% of the run" look true
  when it was 21.65 ms/call).  Here every wrapped call records a start/end event pair
  on the current stream; the pairs of one step are drained after a single
  ``torch.cuda.synchronize`` at the end of that step.
* Syncs are counted and timed separately because both kinds block: ``Tensor.item``
  (fp64 host readback) and ``warp.synchronize_device`` (Warp's own explicit sync).
  ``perf_counter`` IS valid around those, because the call returns only after the
  queue drains -- the wait is real host wall time.
* Inclusive/nested labels are reported as inclusive; the exclusive bucket is derived
  from the known containment tree so that summing the leaves does not double count.

Usage
-----
    PYTHONPATH=src:platform:platform/warp_vpm PFIELD_DEVICE=cuda:0 \
    FLUXV_GPU_ONLY=1 python3 platform/warp_vpm/profile_a16_phases.py --steps 150
    ... --steps 30 --count-only      # factorization accounting only (no event timers)
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

import torch
import warp as wp


#: (module path, class name, method name, label).  Labels containing ">" are children
#: of the label before the ">"; the table prints inclusive times and the roll-up
#: subtracts children so the leaves sum to the top-level wall.
TARGETS = (
    ("fluxvortex.warp_fsi.q16_flux_v5m_native", "Q16NativeV5MOwner", "propose", "propose"),
    (
        "fluxvortex.warp_fsi.q16_flux_v5m_native",
        "Q16NativeV5MSolver",
        "_ring_velocity",
        "ring_BS",
    ),
    (
        "fluxvortex.warp_fsi.q16_flux_v5m_native",
        "Q16NativeV5MSurface",
        "evaluate",
        "surface_evaluate",
    ),
    (
        "fluxvortex.warp_fsi.q16_flux_v5m_native_fsi",
        "Q16NativeV5MFSIStepper",
        "_integrate_structure",
        "integrate_structure",
    ),
    (
        "fluxvortex.warp_fsi.q16_author_loads",
        "Q16NativeAuthorEndpointLoad",
        "velocity_force",
        "velocity_force",
    ),
    (
        "fluxvortex.warp_fsi.q16_author_loads",
        "Q16NativeAuthorEndpointLoad",
        "assemble",
        "load_assemble",
    ),
    (
        "fluxvortex.warp_fsi.q16_structural_solver",
        "Q16CudaNewmarkStepper",
        "step",
        "struct_step",
    ),
    (
        "fluxvortex.warp_fsi.q16_structural_solver",
        "Q16CudaNewmarkStepper",
        "_reference_dense_nonsymmetric_solve",
        "dense_solve",
    ),
    (
        "fluxvortex.warp_fsi.q16_structural_solver",
        "Q16CudaNewmarkStepper",
        "_acceleration_load_matrix",
        "accel_matrix_validate",
    ),
    (
        "fluxvortex.warp_fsi.q16_structural_solver",
        "Q16CudaNewmarkStepper",
        "_refresh_reference_tangent_body",
        "tangent_refresh",
    ),
    ("pfield_torch_gpu", "CudaParticleField", "velocity_at_cuda", "particle_BS"),
    (
        "pfield_torch_gpu",
        "CudaParticleField",
        "velocity_and_jacobian_self_cuda",
        "jacobian_BS",
    ),
    ("pfield_torch_gpu", "CudaParticleField", "advance_wrk3", "advance_wrk3"),
    # -- S2b: the non-dense structural interior -------------------------------
    (
        "fluxvortex.warp_fsi.q16_structural_solver",
        "Q16CudaNewmarkStepper",
        "predict_kinematics",
        "predict_kinematics",
    ),
    (
        "fluxvortex.warp_fsi.q16_structural_solver",
        "Q16CudaNewmarkStepper",
        "_kinematics",
        "kinematics",
    ),
    (
        "fluxvortex.warp_fsi.q16_structural_solver",
        "Q16CudaNewmarkStepper",
        "_residual",
        "residual",
    ),
    (
        "fluxvortex.warp_fsi.q16_structural_solver",
        "Q16CudaNewmarkStepper",
        "_norms",
        "norms",
    ),
    (
        "fluxvortex.warp_fsi.q16_structural_solver",
        "Q16CudaNewmarkStepper",
        "_dot",
        "dot",
    ),
    (
        "fluxvortex.warp_fsi.q16_structural_solver",
        "Q16CudaNewmarkStepper",
        "_apply_preconditioner",
        "preconditioner",
    ),
    (
        "fluxvortex.warp_fsi.kernels_q16_ans_eas",
        "Q16CudaMITC16EASOperator",
        "_linearization_prechecked",
        "linearization",
    ),
    (
        "fluxvortex.warp_fsi.kernels_q16_ans_eas",
        "Q16CudaMITC16EASOperator",
        "_tangent_action_linearized_prechecked",
        "tangent_action",
    ),
    (
        "fluxvortex.warp_fsi.kernels_q16_constraints",
        "Q16CudaBoundaryConstraints",
        "_project_free_prechecked",
        "project_free",
    ),
    # -- S3b: what `surface.evaluate` actually spends (state side vs velocity side) --
    (
        "fluxvortex.warp_fsi.kernels_q16_transfer",
        "Q16CudaSurfaceTransfer",
        "interpolate",
        "surface_interpolate",
    ),
    (
        "fluxvortex.warp_fsi.q16_flux_v5m_native",
        None,
        "assemble_native_ring_geometry",
        "ring_geometry_assemble",
    ),
)

#: Labels that get an entry synchronisation so the measured span can be split into
#: "backlog wait" (work already queued by earlier calls) and "own work".  This is the
#: distinction that a bare inclusive timer cannot make: a phase containing a host sync
#: reports the wait for everything enqueued before it.
#: ONLY the three outermost phases are split (about 6 entry syncs per step).  Adding
#: ``struct_step`` here was measured to be a mistake: it is called ~58 times per step and
#: each entry sync collapses the CPU run-ahead, which stretched a 150-step run past 23
#: minutes, so its split is not worth the serialisation.
SPLIT_LABELS = ("propose", "integrate_structure", "tangent_refresh")

#: author_loads is imported under two possible module names in this tree.
TARGET_ALIASES = {"fluxvortex.warp_fsi.q16_author_loads": (
    "fluxvortex.warp_fsi.q16_flux_v5m_author_loads",
)}

#: Inclusive parent -> DIRECT children, used to derive exclusive buckets.  The tree is
#: only as fine as the labels actually nest:
#:   * ``velocity_self_cuda`` is NOT labelled separately -- its body is literally
#:     ``velocity_at_cuda(self.pos[:n])``, so a separate label measured the same GPU span
#:     twice (that made a child sum exceed its parent).
#:   * ``advance_wrk3`` is NOT a child of ``particle_BS``: its three RK3 stages call
#:     ``velocity_self_cuda`` (hence ``particle_BS``) but also do position copies and
#:     ring influences that ``particle_BS`` does not cover.  They partially OVERLAP, so
#:     neither can be subtracted from the other; both are reported as inclusive.
#:   * ``accel_matrix_validate`` is called from inside ``dense_solve`` (and a few times
#:     from the structural step), so it is a child of ``dense_solve``.
CHILDREN = {
    "propose": ("ring_BS", "particle_BS", "surface_evaluate"),
    "integrate_structure": ("velocity_force", "struct_step"),
    "struct_step": ("dense_solve", "predict_kinematics", "residual", "norms",
                    "preconditioner"),
    "dense_solve": ("accel_matrix_validate",),
    "norms": ("dot",),
}

#: Labels that OVERLAP other labels by construction; their "exclusive" column is not
#: meaningful and is suppressed in the printout.
OVERLAPPING = ("advance_wrk3", "particle_BS", "ring_BS", "accel_matrix_validate",
               "linearization", "tangent_action", "kinematics")

#: Labels whose completion drains the event bucket.  They are the outermost phases and
#: never nest inside one another, so draining here bounds the number of live CUDA events
#: and keeps ``elapsed_time`` reads correct without a sync per call.
DRAIN_LABELS = ("propose", "integrate_structure", "tangent_refresh")


class PhaseClock:
    """Per-step CUDA-event phase accumulator with correct async semantics."""

    def __init__(self) -> None:
        self.inclusive: dict[str, float] = collections.defaultdict(float)
        self.backlog: dict[str, float] = collections.defaultdict(float)
        self.own: dict[str, float] = collections.defaultdict(float)
        self.calls: dict[str, int] = collections.defaultdict(int)
        self._bucket: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event | None]]
        ] = collections.defaultdict(list)
        self._installed: list[str] = []

    # -- instrumentation -------------------------------------------------
    def wrap(self, owner: object, name: str, label: str) -> bool:
        original = getattr(owner, name, None)
        if original is None or not callable(original):
            return False
        clock = self
        split = label in SPLIT_LABELS

        def wrapped(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            mid = torch.cuda.Event(enable_timing=True) if split else None
            clock.calls[label] += 1
            start.record()
            if mid is not None:
                # Drain everything enqueued BEFORE this call: the span start->mid is
                # someone else's queued work, mid->end is this call's own GPU time.
                torch.cuda.synchronize()
                mid.record()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                clock._bucket[label].append((start, mid, end))
                if label in DRAIN_LABELS:
                    clock.flush_step()

        setattr(owner, name, wrapped)
        self._installed.append(label)
        return True

    def install(self) -> list[str]:
        import importlib

        missing: list[str] = []
        for module_name, class_name, method, label in TARGETS:
            names = (module_name,) + (
                TARGET_ALIASES.get(module_name, ())
            )
            resolved = None
            for candidate in names:
                try:
                    resolved = importlib.import_module(candidate)
                    break
                except ImportError:
                    continue
            if resolved is None:
                missing.append(f"{module_name} (module)")
                continue
            if class_name is None:
                owner = resolved
            else:
                owner = getattr(resolved, class_name, None)
                if owner is None:
                    missing.append(f"{module_name}.{class_name}")
                    continue
            if not self.wrap(owner, method, label):
                missing.append(f"{module_name}.{class_name}.{method}")
        return missing

    # -- per-step drain --------------------------------------------------
    def flush_step(self) -> None:
        if not self._bucket:
            return
        torch.cuda.synchronize()
        for label, triples in self._bucket.items():
            total = 0.0
            backlog = 0.0
            own = 0.0
            for start, mid, end in triples:
                total += start.elapsed_time(end)
                if mid is not None:
                    backlog += start.elapsed_time(mid)
                    own += mid.elapsed_time(end)
            self.inclusive[label] += total
            self.backlog[label] += backlog
            self.own[label] += own
        for triples in self._bucket.values():
            triples.clear()

    def exclusive(self) -> dict[str, float]:
        """Subtract direct children once.

        ``CHILDREN`` lists only DIRECT children, so a grandchild (e.g. ``dense_solve``
        under ``struct_step``) is removed from its own parent and never subtracted a
        second time from the grandparent.
        """

        result = {}
        for label, value in self.inclusive.items():
            child_sum = sum(
                self.inclusive.get(child, 0.0) for child in CHILDREN.get(label, ())
            )
            result[label] = max(0.0, value - child_sum)
        return result


class SyncCounter:
    """Counts and times the two blocking-read families separately."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = collections.Counter()
        self.wall: dict[str, float] = collections.Counter()
        self._originals: list[tuple[object, str, object]] = []

    def install(self) -> None:
        original_item = torch.Tensor.item
        counter = self

        def item(self, *args, **kwargs):
            start = time.perf_counter()
            try:
                return original_item(self, *args, **kwargs)
            finally:
                counter.wall["tensor_item"] += time.perf_counter() - start
                counter.counts["tensor_item"] += 1

        torch.Tensor.item = item
        self._originals.append((torch.Tensor, "item", original_item))

        original_sync = wp.synchronize_device

        def synchronize_device(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original_sync(*args, **kwargs)
            finally:
                counter.wall["warp_synchronize_device"] += time.perf_counter() - start
                counter.counts["warp_synchronize_device"] += 1

        wp.synchronize_device = synchronize_device
        self._originals.append((wp, "synchronize_device", original_sync))

    def remove(self) -> None:
        for owner, name, original in self._originals:
            setattr(owner, name, original)
        self._originals.clear()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--count-only", action="store_true",
                        help="only report factorization counters")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    from fluxvortex.cases.rojratsirikul2011 import ROJRATSIRIKUL2011_UNIFIED_CASES
    from fluxvortex.runtime.case_runner import RojratsirikulCaseRunner
    from fluxvortex.warp_fsi import q16_structural_solver as solver

    runner = RojratsirikulCaseRunner(
        ROJRATSIRIKUL2011_UNIFIED_CASES["ROJ11-A16"], device="cuda:0"
    )
    runner.build()

    solver.reset_lu_cache_stats()
    clock = PhaseClock()
    syncs = SyncCounter()
    missing = clock.install() if not arguments.count_only else []
    if not arguments.count_only:
        syncs.install()

    started = time.perf_counter()
    runner.run(max_aero_steps=arguments.steps)
    wall = time.perf_counter() - started
    steps = arguments.steps

    if not arguments.count_only:
        clock.flush_step()
        syncs.remove()
    stats = solver.lu_cache_stats()
    output_path = getattr(runner, "output", None)
    payload = (
        json.loads(Path(output_path).read_text())
        if output_path and Path(output_path).is_file()
        else {}
    )

    print(f"\n=== A16 phase profile: {steps} steps, wall {wall:.1f} s "
          f"({wall / steps:.3f} s/step) ===")
    if missing and not arguments.count_only:
        print("not instrumented (attribute not found):", ", ".join(missing))

    if not arguments.count_only:
        exclusive = clock.exclusive()
        print(f"\n{'phase':28s} {'calls/step':>10s} {'ms/step':>10s} {'exclusive ms/step':>18s}")
        for label, value in sorted(clock.inclusive.items(),
                                   key=lambda item: -item[1]):
            cell = (
                "    (overlaps)"
                if label in OVERLAPPING
                else f"{exclusive.get(label, 0.0) / steps:18.1f}"
            )
            print(f"{label:28s} {clock.calls[label] / steps:10.2f} "
                  f"{value / steps:10.1f} {cell}")

        if clock.backlog:
            print(f"\n{'split phase':28s} {'inclusive':>11s} {'backlog':>11s} "
                  f"{'own work':>11s}  (ms/step)")
            for label in SPLIT_LABELS:
                if label not in clock.inclusive:
                    continue
                print(f"{label:28s} {clock.inclusive[label] / steps:11.1f} "
                      f"{clock.backlog[label] / steps:11.1f} "
                      f"{clock.own[label] / steps:11.1f}")

        print(f"\n{'sync family':28s} {'calls/step':>10s} {'wall ms/step':>14s}")
        for family in ("tensor_item", "warp_synchronize_device"):
            print(f"{family:28s} {syncs.counts[family] / steps:10.2f} "
                  f"{1000.0 * syncs.wall[family] / steps:14.1f}")
        print("  NOTE: sync wall is INCLUSIVE of the GPU work it waits for, so it is a"
              "\n        decomposition of the phase times above, never an addition to them.")

        # Closure: the three outermost phases must cover the run's wall clock.
        top = sum(clock.inclusive.get(label, 0.0) / 1000.0
                  for label in DRAIN_LABELS)
        print(f"\nclosure: propose+integrate_structure+tangent_refresh = {top:.1f} s "
              f"of {wall:.1f} s wall ({100.0 * top / wall:.1f}%)")
        print(f"  (instrumented wall; compare with the --count-only run on the same "
              f"window to size the observer cost)")

    print(f"\nLU/factorization accounting (measured, not inferred):")
    for name, value in sorted(stats.items()):
        print(f"  {name:20s} {value:8d} total   {value / steps:8.2f} /step")
    if payload:
        print(f"  payload step_wall_seconds_mean = "
              f"{payload.get('step_wall_seconds_mean')}")

    if arguments.output is not None:
        arguments.output.write_text(
            json.dumps(
                {
                    "steps": steps,
                    "wall_seconds": wall,
                    "wall_seconds_per_step": wall / steps,
                    "phase_inclusive_ms": dict(clock.inclusive),
                    "phase_backlog_ms": dict(clock.backlog),
                    "phase_own_ms": dict(clock.own),
                    "phase_exclusive_ms": clock.exclusive() if not arguments.count_only else {},
                    "phase_calls": dict(clock.calls),
                    "sync_counts": dict(syncs.counts),
                    "sync_wall_seconds": dict(syncs.wall),
                    "lu_cache_stats": stats,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
