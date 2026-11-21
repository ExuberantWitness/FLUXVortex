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
    ("pfield_torch_gpu", "CudaParticleField", "velocity_self_cuda", "particle_self_BS"),
    (
        "pfield_torch_gpu",
        "CudaParticleField",
        "velocity_and_jacobian_self_cuda",
        "jacobian_BS",
    ),
    ("pfield_torch_gpu", "CudaParticleField", "advance_wrk3", "advance_wrk3"),
)

#: author_loads is imported under two possible module names in this tree.
TARGET_ALIASES = {"fluxvortex.warp_fsi.q16_author_loads": (
    "fluxvortex.warp_fsi.q16_flux_v5m_author_loads",
)}

#: Inclusive parent -> children, used to derive exclusive buckets.
CHILDREN = {
    "propose": ("ring_BS", "particle_BS", "particle_self_BS", "jacobian_BS",
                "advance_wrk3", "surface_evaluate"),
    "integrate_structure": ("velocity_force", "struct_step", "load_assemble",
                            "dense_solve", "accel_matrix_validate"),
    "struct_step": ("dense_solve", "accel_matrix_validate"),
    "dense_solve": ("accel_matrix_validate",),
    "advance_wrk3": ("particle_self_BS",),
}


class PhaseClock:
    """Per-step CUDA-event phase accumulator with correct async semantics."""

    def __init__(self) -> None:
        self.inclusive: dict[str, float] = collections.defaultdict(float)
        self.calls: dict[str, int] = collections.defaultdict(int)
        self._bucket: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            collections.defaultdict(list)
        )
        self._installed: list[str] = []

    # -- instrumentation -------------------------------------------------
    def wrap(self, owner: object, name: str, label: str) -> bool:
        original = getattr(owner, name, None)
        if original is None or not callable(original):
            return False
        clock = self

        def wrapped(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            clock.calls[label] += 1
            start.record()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                clock._bucket[label].append((start, end))

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
            klass = getattr(resolved, class_name, None)
            if klass is None:
                missing.append(f"{module_name}.{class_name}")
                continue
            if not self.wrap(klass, method, label):
                missing.append(f"{module_name}.{class_name}.{method}")
        return missing

    # -- per-step drain --------------------------------------------------
    def flush_step(self) -> None:
        if not self._bucket:
            return
        torch.cuda.synchronize()
        for label, pairs in self._bucket.items():
            total = 0.0
            for start, end in pairs:
                total += start.elapsed_time(end)
            self.inclusive[label] += total
        for pairs in self._bucket.values():
            pairs.clear()

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
            print(f"{label:28s} {clock.calls[label] / steps:10.2f} "
                  f"{value / steps:10.1f} {exclusive.get(label, 0.0) / steps:18.1f}")

        print(f"\n{'sync family':28s} {'calls/step':>10s} {'wall ms/step':>14s}")
        for family in ("tensor_item", "warp_synchronize_device"):
            print(f"{family:28s} {syncs.counts[family] / steps:10.2f} "
                  f"{1000.0 * syncs.wall[family] / steps:14.1f}")
        sync_wall = sum(syncs.wall.values())

        # Closure: top-level phases + syncs must account for the uninstrumented wall.
        top = sum(clock.inclusive.get(label, 0.0) / 1000.0
                  for label in ("propose", "integrate_structure", "tangent_refresh"))
        covered = top + sync_wall
        print(f"\nclosure: propose+integrate+tangent+syncs = {covered:.1f} s "
              f"of {wall:.1f} s wall ({100.0 * covered / wall:.1f}%)")

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
