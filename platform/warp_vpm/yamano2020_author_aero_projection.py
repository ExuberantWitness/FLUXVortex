"""Yamano Q4 aerodynamic-assembly topology projected onto the Q16 field.

The paper code owns one aerodynamic/structural Q4 element per UVLM panel and
assembles ``Mf1`` only into that element's 36-by-36 generalized block.  The
production structure remains Q16-only: this module uses the paper Q4 basis as
an intermediate load-integration and column-support space, then applies the
kinematic map ``q_q4 = T q_q16`` and its virtual-work transpose.

Only immutable polynomial/topology tables are prepared with NumPy.  AIC
solves, normal contractions, dense assembly, and both projections execute in
CUDA float64 with Torch; there is no CPU numerical fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from fluxvortex.q16_ancf_shell import q16_shape


@dataclass(frozen=True, slots=True, eq=False)
class Yamano2020AuthorAddedMassProjection:
    """CUDA operators retained from one author-topology Mf1 assembly."""

    generalized_matrix: torch.Tensor
    neumann_map: torch.Tensor
    gamma_rate_map: torch.Tensor
    pressure_to_generalized: torch.Tensor
    q4_from_q16: torch.Tensor
    q4_generalized_matrix: torch.Tensor
    q4_neumann_map: torch.Tensor
    q4_gamma_rate_map: torch.Tensor
    q4_pressure_to_generalized: torch.Tensor


def _positive_count(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int")
    return value


def _q4_scalar_shape(
    xi: float,
    eta: float,
    chord_extent: float,
    span_extent: float,
) -> np.ndarray:
    """Port of the paper's 12 scalar ``Sc`` functions on ``[0,1]^2``."""

    x = float(xi)
    y = float(eta)
    dl = float(chord_extent)
    dw = float(span_extent)
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("author Q4 local coordinates must lie in [0,1]")
    result = np.asarray(
        [
            -(x - 1.0)
            * (y - 1.0)
            * (2.0 * y**2 - y + 2.0 * x**2 - x - 1.0),
            -dl * x * (x - 1.0) ** 2 * (y - 1.0),
            -dw * y * (y - 1.0) ** 2 * (x - 1.0),
            x * (2.0 * y**2 - y - 3.0 * x + 2.0 * x**2) * (y - 1.0),
            -dl * x**2 * (x - 1.0) * (y - 1.0),
            dw * x * y * (y - 1.0) ** 2,
            -x * y * (1.0 - 3.0 * x - 3.0 * y + 2.0 * y**2 + 2.0 * x**2),
            dl * x**2 * y * (x - 1.0),
            dw * x * y**2 * (y - 1.0),
            y * (x - 1.0) * (2.0 * x**2 - x - 3.0 * y + 2.0 * y**2),
            dl * x * y * (x - 1.0) ** 2,
            -dw * y**2 * (x - 1.0) * (y - 1.0),
        ],
        dtype=np.float64,
    )
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError("author Q4 shape is non-finite")
    return result


def _author_connectivity(chord_count: int, span_count: int) -> np.ndarray:
    result = np.empty((chord_count * span_count, 4), dtype=np.int64)
    for chord in range(chord_count):
        for span in range(span_count):
            element = chord * span_count + span
            lower_left = chord * (span_count + 1) + span
            lower_right = (chord + 1) * (span_count + 1) + span
            result[element] = (
                lower_left,
                lower_right,
                lower_right + 1,
                lower_left + 1,
            )
    return result


def _embed_scalar_local(
    shape: np.ndarray,
    connectivity: np.ndarray,
    scalar_dof_count: int,
) -> np.ndarray:
    element_count = int(connectivity.shape[0])
    result = np.zeros((element_count, scalar_dof_count), dtype=np.float64)
    for element in range(element_count):
        for local_node, global_node in enumerate(connectivity[element]):
            for field in range(3):
                result[element, int(global_node) * 3 + field] += shape[
                    local_node * 3 + field
                ]
    return result


def _author_collocation_scalar_map(
    chord_count: int,
    span_count: int,
    chord_length: float,
    span_length: float,
    connectivity: np.ndarray,
) -> np.ndarray:
    """Reproduce ``generate_panel.m::Sc_mat_col_global`` for a uniform grid."""

    chord_extent = chord_length / chord_count
    span_extent = span_length / span_count
    scalar_dof_count = (chord_count + 1) * (span_count + 1) * 3
    front_upper = _embed_scalar_local(
        _q4_scalar_shape(0.25, 1.0, chord_extent, span_extent),
        connectivity,
        scalar_dof_count,
    )
    front_lower = _embed_scalar_local(
        _q4_scalar_shape(0.25, 0.0, chord_extent, span_extent),
        connectivity,
        scalar_dof_count,
    )
    end_upper = _embed_scalar_local(
        _q4_scalar_shape(1.0, 1.0, chord_extent, span_extent),
        connectivity,
        scalar_dof_count,
    )
    end_lower = _embed_scalar_local(
        _q4_scalar_shape(1.0, 0.0, chord_extent, span_extent),
        connectivity,
        scalar_dof_count,
    )
    back_upper = np.empty_like(front_upper)
    back_lower = np.empty_like(front_lower)
    for chord in range(chord_count):
        for span in range(span_count):
            element = chord * span_count + span
            if chord + 1 < chord_count:
                following = element + span_count
                back_upper[element] = front_upper[following]
                back_lower[element] = front_lower[following]
            else:
                back_upper[element] = (
                    4.0 / 3.0 * (end_upper[element] - front_upper[element])
                    + front_upper[element]
                )
                back_lower[element] = (
                    4.0 / 3.0 * (end_lower[element] - front_lower[element])
                    + front_lower[element]
                )
    result = 0.25 * (front_upper + back_upper + back_lower + front_lower)
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError("author Q4 collocation map is non-finite")
    return np.ascontiguousarray(result)


def _p_interp_uniform_weights(
    local_chord: float,
    chord_panel: int,
    chord_count: int,
) -> tuple[float, float, float]:
    x = float(local_chord)
    if x <= 0.75:
        previous = (3.0 - 4.0 * x) / 4.0
        current = (1.0 + 4.0 * x) / 4.0
        following = 0.0
        if chord_panel == 0:
            previous = 0.0
            current = 1.0
    else:
        previous = 0.0
        if chord_panel + 1 < chord_count:
            current = (7.0 - 4.0 * x) / 4.0
            following = (4.0 * x - 3.0) / 4.0
        else:
            current = 4.0 - 4.0 * x
            following = 0.0
    return previous, current, following


def _author_pressure_scalar_maps(
    chord_count: int,
    span_count: int,
    chord_length: float,
    span_length: float,
) -> np.ndarray:
    """Return each paper element's 12-by-panel pressure integration map."""

    element_count = chord_count * span_count
    result = np.zeros((element_count, 12, element_count), dtype=np.float64)
    chord_extent = chord_length / chord_count
    span_extent = span_length / span_count
    points, weights = np.polynomial.legendre.leggauss(5)
    for chord in range(chord_count):
        for span in range(span_count):
            element = chord * span_count + span
            for chord_point, chord_weight in zip(points, weights, strict=True):
                local_chord = 0.5 * (float(chord_point) + 1.0)
                pressure_weights = _p_interp_uniform_weights(
                    local_chord,
                    chord,
                    chord_count,
                )
                for span_point, span_weight in zip(points, weights, strict=True):
                    local_span = 0.5 * (float(span_point) + 1.0)
                    shape = _q4_scalar_shape(
                        local_chord,
                        local_span,
                        chord_extent,
                        span_extent,
                    )
                    area_weight = (
                        0.25
                        * chord_extent
                        * span_extent
                        * float(chord_weight)
                        * float(span_weight)
                    )
                    for offset, coefficient in zip(
                        (-1, 0, 1), pressure_weights, strict=True
                    ):
                        neighbor_chord = chord + offset
                        if coefficient == 0.0 or not (
                            0 <= neighbor_chord < chord_count
                        ):
                            continue
                        pressure_panel = neighbor_chord * span_count + span
                        result[element, :, pressure_panel] += (
                            area_weight * coefficient * shape
                        )
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError("author Q4 pressure integration is non-finite")
    return np.ascontiguousarray(result)


def _author_local_dofs(connectivity: np.ndarray) -> np.ndarray:
    result = np.empty((connectivity.shape[0], 36), dtype=np.int64)
    for element in range(connectivity.shape[0]):
        cursor = 0
        for global_node in connectivity[element]:
            for field in range(3):
                for component in range(3):
                    result[element, cursor] = (
                        int(global_node) * 9 + field * 3 + component
                    )
                    cursor += 1
    return result


def _q4_from_q16_map(
    mesh: Any,
    *,
    q16_chord_count: int,
    q16_span_count: int,
    author_chord_count: int,
    author_span_count: int,
    chord_length: float,
    span_length: float,
) -> np.ndarray:
    """Map Q16 midsurface values/physical slopes to author nodal ANCF DOFs."""

    q4_node_count = (author_chord_count + 1) * (author_span_count + 1)
    result = np.zeros((q4_node_count * 9, mesh.dof_count), dtype=np.float64)
    connectivity = np.asarray(mesh.connectivity, dtype=np.int64)
    chord_scale = 2.0 * q16_chord_count / chord_length
    span_scale = 2.0 * q16_span_count / span_length
    for author_chord in range(author_chord_count + 1):
        chord_fraction = author_chord / author_chord_count
        scaled_chord = chord_fraction * q16_chord_count
        chord_owner = min(int(math.floor(scaled_chord)), q16_chord_count - 1)
        xi = min(max(2.0 * (scaled_chord - chord_owner) - 1.0, -1.0), 1.0)
        for author_span in range(author_span_count + 1):
            span_fraction = author_span / author_span_count
            scaled_span = span_fraction * q16_span_count
            span_owner = min(int(math.floor(scaled_span)), q16_span_count - 1)
            eta = min(max(2.0 * (scaled_span - span_owner) - 1.0, -1.0), 1.0)
            element = span_owner * q16_chord_count + chord_owner
            shape, dxi, deta = q16_shape(float(xi), float(eta))
            q4_node = author_chord * (author_span_count + 1) + author_span
            # The paper fixes r, dx(r), and dy(r) at the leading edge.  Its
            # constrained coordinates must not be reintroduced by projecting
            # an interior Q16 derivative onto those eliminated Q4 rows.
            if author_chord == 0:
                continue
            for local_node, global_node in enumerate(connectivity[element]):
                for component in range(3):
                    q16_dof = int(global_node) * 6 + component
                    result[q4_node * 9 + component, q16_dof] += shape[local_node]
                    result[q4_node * 9 + 3 + component, q16_dof] += (
                        chord_scale * dxi[local_node]
                    )
                    result[q4_node * 9 + 6 + component, q16_dof] += (
                        span_scale * deta[local_node]
                    )
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError("Q4-from-Q16 kinematic map is non-finite")
    return np.ascontiguousarray(result)


def build_yamano2020_author_added_mass_projection(
    mesh: Any,
    *,
    q16_chord_count: int,
    q16_span_count: int,
    aerodynamic_chord_count: int,
    aerodynamic_span_count: int,
    chord_length: float,
    span_length: float,
    aic: torch.Tensor,
    panel_normals: torch.Tensor,
    density: float,
) -> Yamano2020AuthorAddedMassProjection:
    """Assemble paper-local Mf1 on CUDA and project it to Q16 by ``T.T M T``."""

    q16_chord = _positive_count("q16_chord_count", q16_chord_count)
    q16_span = _positive_count("q16_span_count", q16_span_count)
    chord_count = _positive_count("aerodynamic_chord_count", aerodynamic_chord_count)
    span_count = _positive_count("aerodynamic_span_count", aerodynamic_span_count)
    panel_count = chord_count * span_count
    if mesh.element_count != q16_chord * q16_span:
        raise ValueError("Q16 mesh count differs from declared rectangular topology")
    if (
        aic.device.type != "cuda"
        or panel_normals.device.type != "cuda"
        or aic.dtype is not torch.float64
        or panel_normals.dtype is not torch.float64
    ):
        raise ValueError("author added-mass projection requires CUDA float64 inputs")
    if tuple(aic.shape) != (panel_count, panel_count):
        raise ValueError("author added-mass AIC shape drift")
    if tuple(panel_normals.shape) != (panel_count, 3):
        raise ValueError("author added-mass normal shape drift")
    rho = float(density)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("density must be finite and positive")
    device = aic.device
    connectivity = _author_connectivity(chord_count, span_count)
    scalar_collocation = _author_collocation_scalar_map(
        chord_count,
        span_count,
        chord_length,
        span_length,
        connectivity,
    )
    pressure_scalar = _author_pressure_scalar_maps(
        chord_count,
        span_count,
        chord_length,
        span_length,
    )
    local_dofs_np = _author_local_dofs(connectivity)
    q4_from_q16_np = _q4_from_q16_map(
        mesh,
        q16_chord_count=q16_chord,
        q16_span_count=q16_span,
        author_chord_count=chord_count,
        author_span_count=span_count,
        chord_length=chord_length,
        span_length=span_length,
    )
    q4_node_count = (chord_count + 1) * (span_count + 1)
    q4_scalar_dof_count = q4_node_count * 3
    q4_dof_count = q4_node_count * 9
    scalar_collocation_t = torch.as_tensor(
        scalar_collocation,
        device=device,
        dtype=torch.float64,
    )
    scalar_to_full = torch.as_tensor(
        np.asarray(
            [
                node * 9 + field * 3 + component
                for node in range(q4_node_count)
                for field in range(3)
                for component in range(3)
            ],
            dtype=np.int64,
        ).reshape(q4_scalar_dof_count, 3),
        device=device,
        dtype=torch.int64,
    )
    q4_neumann = torch.zeros(
        (panel_count, q4_dof_count),
        device=device,
        dtype=torch.float64,
    )
    for component in range(3):
        q4_neumann[:, scalar_to_full[:, component]] = (
            scalar_collocation_t * panel_normals[:, component].unsqueeze(1)
        )
    q4_gamma_rate = torch.linalg.solve(aic, q4_neumann)
    pressure_scalar_t = torch.as_tensor(
        pressure_scalar,
        device=device,
        dtype=torch.float64,
    )
    pressure_local = (
        pressure_scalar_t.unsqueeze(2)
        * panel_normals.transpose(0, 1).unsqueeze(0).unsqueeze(0)
    ).reshape(panel_count, 36, panel_count)
    local_dofs = torch.as_tensor(
        local_dofs_np,
        device=device,
        dtype=torch.int64,
    )
    q4_pressure = torch.zeros(
        (q4_dof_count, panel_count),
        device=device,
        dtype=torch.float64,
    )
    q4_matrix = torch.zeros(
        (q4_dof_count, q4_dof_count),
        device=device,
        dtype=torch.float64,
    )
    for element in range(panel_count):
        indices = local_dofs[element]
        q4_pressure[indices] += pressure_local[element]
        # Yamano owns ``A_mat*Gamma = V_normal`` and
        # ``Mf1_mat = A_mat\nvec``.  ``A_mat`` already carries the author's
        # clockwise-ring (negative diagonal) orientation, so the pressure
        # force assembly has no additional Ptera-orientation minus sign.
        local_matrix = rho * (
            pressure_local[element] @ q4_gamma_rate[:, indices]
        )
        q4_matrix[indices.unsqueeze(1), indices.unsqueeze(0)] += local_matrix
    q4_from_q16 = torch.as_tensor(
        q4_from_q16_np,
        device=device,
        dtype=torch.float64,
    )
    neumann = q4_neumann @ q4_from_q16
    gamma_rate = q4_gamma_rate @ q4_from_q16
    pressure = q4_from_q16.transpose(0, 1) @ q4_pressure
    generalized = q4_from_q16.transpose(0, 1) @ q4_matrix @ q4_from_q16
    values = (
        generalized,
        neumann,
        gamma_rate,
        pressure,
        q4_from_q16,
        q4_matrix,
        q4_neumann,
        q4_gamma_rate,
    )
    if any(
        value.device.type != "cuda"
        or value.dtype is not torch.float64
        or not bool(torch.isfinite(value).all().item())
        for value in values
    ):
        raise FloatingPointError("author Q4-to-Q16 added-mass projection is invalid")
    return Yamano2020AuthorAddedMassProjection(
        generalized_matrix=generalized.detach().clone(),
        neumann_map=neumann.detach().clone(),
        gamma_rate_map=gamma_rate.detach().clone(),
        pressure_to_generalized=pressure.detach().clone(),
        q4_from_q16=q4_from_q16.detach().clone(),
        q4_generalized_matrix=q4_matrix.detach().clone(),
        q4_neumann_map=q4_neumann.detach().clone(),
        q4_gamma_rate_map=q4_gamma_rate.detach().clone(),
        q4_pressure_to_generalized=q4_pressure.detach().clone(),
    )


__all__ = [
    "Yamano2020AuthorAddedMassProjection",
    "build_yamano2020_author_added_mass_projection",
]
