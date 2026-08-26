"""Load-history owner identification (plan §8.7).

Makes the bound_rate vs material choice explicit with physics identity,
not a runtime stability switch."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class LoadHistoryModel:
    """Identifies WHICH pressure-history model is in production use."""
    name: str              # "bound_rate" or "material"
    physics_identity: str  # what it represents physically
    stability_note: str    # why this choice was made
    oracle_status: str     # "validated" | "pending" | "diverged_in_lev_regime"

BOUND_RATE_MODEL = LoadHistoryModel(
    name="bound_rate",
    physics_identity=(
        "dp_add = d(Gamma)/dt: the author's weak-scheme unsteady "
        "circulation time-derivative pressure term"
    ),
    stability_note=(
        "Bounded by construction; the material-derivative (strong-scheme "
        "Mf2_vec1) diverges when every wake row carries large persistent "
        "LEV release circulation"
    ),
    oracle_status="validated_against_author_weak_scheme",
)

MATERIAL_MODEL = LoadHistoryModel(
    name="material",
    physics_identity=(
        "Mf2_vec1 = A^{-1}(wake-motion material derivative): the author's "
        "strong-scheme wake-memory pressure"
    ),
    stability_note=(
        "Diverges (-6.5 qS) in sustained-LEV-release regimes; the Yamano "
        "path (which never sheds particles) uses this as the oracle default"
    ),
    oracle_status="validated_for_zero_shedding_only",
)

LOAD_HISTORY_MODELS: dict[str, LoadHistoryModel] = {
    model.name: model for model in (BOUND_RATE_MODEL, MATERIAL_MODEL)
}
