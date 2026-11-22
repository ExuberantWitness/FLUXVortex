"""P0-3 feasibility gate: is the element tangent action LOCAL per element?

S3-v2A claims the 2,976-column global probe wastes ~96% of its work because each batch
feeds every element a mostly-zero direction, and that feeding 96 element-local unit
directions would produce all 50 element blocks in 2 batches instead of 47 (a ~31x
reduction).  That claim rests on one property: the element operator must act on each
element INDEPENDENTLY -- a direction nonzero in element e must produce an action nonzero
only in element e.

This script measures that directly instead of reading the code:
 1. feed a local direction that is nonzero in element 0 only, check the action's support;
 2. feed a direction nonzero in element 0 and element 31, check the support is the union;
 3. time one local batch of 64 directions vs one global batch of 64 columns, and report
    the per-column ratio -- the number the 31x claim is made of.
"""

from __future__ import annotations

import time

import torch
import warp as wp


def support(local_action: torch.Tensor) -> set[int]:
    """Element indices where the action has any nonzero entry."""

    per_element = local_action.reshape(local_action.shape[0], -1).abs().sum(dim=0)
    # local layout is (batch, element_count, 96)
    return {e for e, v in enumerate(per_element.tolist()) if v > 0.0}


def main() -> int:
    from fluxvortex.cases.rojratsirikul2011 import ROJRATSIRIKUL2011_UNIFIED_CASES
    from fluxvortex.runtime.case_runner import RojratsirikulCaseRunner
    from fluxvortex.warp_fsi.config import DTYPE

    runner = RojratsirikulCaseRunner(
        ROJRATSIRIKUL2011_UNIFIED_CASES["ROJ11-A16"], device="cuda:0"
    )
    runner.build()
    runner.run(max_aero_steps=1)
    stepper = runner.structural
    mesh_operator = stepper._operator
    element_operator = mesh_operator._element_operator
    dof = stepper.dof_count
    element_count = mesh_operator.element_count
    local_dof = mesh_operator._connectivity.shape[1] * 6 if hasattr(
        mesh_operator, "_connectivity"
    ) else None
    linearization = mesh_operator._linearization_prechecked(runner.owner.state)
    print(f"dof={dof} element_count={element_count} local_dof={local_dof}")

    def local_direction(elements: dict[int, int]) -> wp.array:
        """Local layout (1, element_count, local_dof) with unit entries at given elements."""

        direction = torch.zeros(
            (1, element_count, local_dof), device="cuda:0", dtype=torch.float64
        )
        for element, dof_index in elements.items():
            direction[0, element, dof_index] = 1.0
        return wp.from_torch(direction, dtype=DTYPE, requires_grad=False)

    def local_action_of(direction: wp.array) -> torch.Tensor:
        return wp.to_torch(
            element_operator._tangent_action_linearized_prechecked(
                direction, linearization
            )
        )

    one = local_action_of(local_direction({0: 7}))
    print("element 0 only  -> support:", sorted(support(one)), f"(nonzero values {int((one != 0).sum())})")

    pair = local_action_of(local_direction({0: 7, 31: 13}))
    print("elements 0+31   -> support:", sorted(support(pair)), f"(nonzero values {int((pair != 0).sum())})")

    # additivity: action(0+31) == action(0) + action(31)  (this is what makes the
    # element blocks superposable, i.e. what makes one probe per element exact)
    second = local_action_of(local_direction({31: 13}))
    combined = one + second
    additive = torch.equal(pair, combined)
    print("additivity action(0+31) == action(0)+action(31):", additive)

    # timing: one local batch of 64 vs one global batch of 64 columns
    width = 64
    local_batch = torch.zeros(
        (width, element_count, local_dof), device="cuda:0", dtype=torch.float64
    )
    local_batch[torch.arange(width), :, torch.arange(width) % local_dof] = 1.0
    local_wp = wp.from_torch(local_batch, dtype=DTYPE, requires_grad=False)
    global_batch = torch.zeros((width, dof), device="cuda:0", dtype=torch.float64)
    global_batch[torch.arange(width), torch.arange(width)] = 1.0
    global_wp = wp.from_torch(global_batch, dtype=DTYPE, requires_grad=False)

    for label, call in (
        ("local  (element operator only)", lambda: element_operator._tangent_action_linearized_prechecked(local_wp, linearization)),
        ("global (gather+element+assemble)", lambda: mesh_operator._tangent_action_linearized_prechecked(global_wp, linearization)),
    ):
        call()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        per_call = (time.perf_counter() - start) / 3.0
        print(f"{label}: {1000.0 * per_call:.1f} ms / {width} directions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
