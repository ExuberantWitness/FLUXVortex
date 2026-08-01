"""Conservative ledger for a moving-surface vector-circulation inventory.

The ledger checks, but never closes, the cellwise balance

    dG/dt = wall_transfer + internal_transport + external_transport
            - separation_release.

Internal edge fluxes are assembled once with opposite signs on adjacent
cells.  Separation release is a required physical input: this module never
infers it from a residual, pressure, force, LESP, or target load.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SurfaceInventoryError(ValueError):
    """Invalid surface-inventory state or flux ledger."""


def _vectors(name: str, value, count: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise SurfaceInventoryError(
            f"{name} must have shape (n,3), got {array.shape}"
        )
    if count is not None and len(array) != count:
        raise SurfaceInventoryError(
            f"{name} must contain {count} cells, got {len(array)}"
        )
    if not np.all(np.isfinite(array)):
        raise SurfaceInventoryError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class SurfaceInventoryBudgetReport:
    storage_rate: np.ndarray
    wall_transfer_rate: np.ndarray
    internal_transport_net_in_rate: np.ndarray
    external_transport_net_in_rate: np.ndarray
    separation_release_rate: np.ndarray
    residual: np.ndarray
    max_local_residual: float
    global_internal_flux_residual: float
    global_budget_residual: float
    passed: bool


def surface_inventory_budget_report(
    *,
    previous_inventory,
    current_inventory,
    dt: float,
    wall_transfer_rate,
    internal_edges,
    internal_edge_flux_rate,
    external_transport_net_in_rate,
    separation_release_rate,
    tolerance: float = 1.0e-12,
) -> SurfaceInventoryBudgetReport:
    """Assemble a signed finite-volume inventory ledger without a closure."""
    previous = _vectors("previous_inventory", previous_inventory)
    count = len(previous)
    current = _vectors("current_inventory", current_inventory, count)
    wall = _vectors("wall_transfer_rate", wall_transfer_rate, count)
    external = _vectors(
        "external_transport_net_in_rate",
        external_transport_net_in_rate,
        count,
    )
    release = _vectors(
        "separation_release_rate",
        separation_release_rate,
        count,
    )
    if not np.isfinite(dt) or dt <= 0.0:
        raise SurfaceInventoryError("dt must be positive and finite")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise SurfaceInventoryError(
            "tolerance must be finite and non-negative"
        )
    edges = np.asarray(internal_edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise SurfaceInventoryError(
            "internal_edges must have shape (m,2)"
        )
    flux = _vectors(
        "internal_edge_flux_rate",
        internal_edge_flux_rate,
    )
    if len(flux) != len(edges):
        raise SurfaceInventoryError(
            "one vector flux is required per internal edge"
        )
    if (
        np.any(edges < 0)
        or np.any(edges >= count)
        or np.any(edges[:, 0] == edges[:, 1])
    ):
        raise SurfaceInventoryError(
            "internal edges must connect two distinct valid cells"
        )

    internal = np.zeros((count, 3), dtype=float)
    for edge, edge_flux in zip(edges, flux):
        source, destination = map(int, edge)
        internal[source] -= edge_flux
        internal[destination] += edge_flux
    storage = (current-previous)/dt
    residual = storage-wall-internal-external+release
    max_local = float(
        np.max(np.linalg.norm(residual, axis=1), initial=0.0)
    )
    internal_global = float(np.linalg.norm(np.sum(internal, axis=0)))
    global_budget = float(np.linalg.norm(np.sum(residual, axis=0)))
    return SurfaceInventoryBudgetReport(
        storage_rate=storage,
        wall_transfer_rate=wall.copy(),
        internal_transport_net_in_rate=internal,
        external_transport_net_in_rate=external.copy(),
        separation_release_rate=release.copy(),
        residual=residual,
        max_local_residual=max_local,
        global_internal_flux_residual=internal_global,
        global_budget_residual=global_budget,
        passed=max(max_local, internal_global, global_budget) <= tolerance,
    )
