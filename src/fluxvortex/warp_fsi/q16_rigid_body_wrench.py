"""Full-space rigid-body virtual-work wrench of the aero generalized action.

G000 item 2 of FINAL_PROPOSAL q16-v5m-gpu-load-contract-20260831 (§5.1):

    Cn_full_action = (T_RB(q, O)^T Q_aero)[normal] / (q_inf S)

with T_RB in R^{2976x6} the rigid-body virtual displacement operator on the
FULL unconstrained coordinates, applied BEFORE essential constraints:

  * translation column k:  dr_i = e_k for every node, dd_i = 0;
  * rotation column k:     dr_i = e_k x (r_i - O),  dd_i = e_k x d_i.

T_RB^T Q never needs the matrix: with per-node blocks [Q_r | Q_d],

    F      = sum_i Q_r_i
    M_O    = sum_i ( (r_i - O) x Q_r_i + d_i x Q_d_i )

which is exactly the resultant plus the moment couple the director forces
contribute about the fixed origin O (default the world origin).  Q_aero
must contain ONLY aerodynamic action (constant + velocity + Mf1 at the
accepted endpoint); internal/damping/inertia/prescribed structural loads
are forbidden inputs.  All arithmetic is CUDA float64.
"""

from __future__ import annotations

import torch


def aero_generalized_wrench(
    generalized_aero_action: torch.Tensor,
    structural_state: torch.Tensor,
    *,
    reference_rows: torch.Tensor,
    origin: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (force (3,), moment_about_origin (3,)) of T_RB^T Q_aero.

    ``generalized_aero_action``/``structural_state``: flat (2976,) CUDA
    float64 (node-major [r(3), d(3)] per node).  ``reference_rows``: the
    mesh's (node_count, 6) reference layout, used for shape validation.
    """

    node_count = int(reference_rows.shape[0])
    dof_count = node_count * 6
    if tuple(generalized_aero_action.shape) != (dof_count,):
        raise ValueError(
            f"aero action must be full-space ({dof_count},), got "
            f"{tuple(generalized_aero_action.shape)}; the rigid-body wrench "
            "is defined BEFORE essential constraints"
        )
    if tuple(structural_state.shape) != (dof_count,):
        raise ValueError(
            f"structural state must be full-space ({dof_count},), got "
            f"{tuple(structural_state.shape)}"
        )
    if origin is None:
        origin = torch.zeros(3, device=structural_state.device, dtype=torch.float64)
    q = structural_state
    blocks = q.reshape(node_count, 2, 3)
    positions = blocks[:, 0, :]
    directors = blocks[:, 1, :]
    action_blocks = generalized_aero_action.reshape(node_count, 2, 3)
    action_r = action_blocks[:, 0, :]
    action_d = action_blocks[:, 1, :]
    force = torch.sum(action_r, dim=0)
    moment = torch.sum(
        torch.linalg.cross(positions - origin[None, :], action_r, dim=1)
        + torch.linalg.cross(directors, action_d, dim=1),
        dim=0,
    )
    return force, moment
