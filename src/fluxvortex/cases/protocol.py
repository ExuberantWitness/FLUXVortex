"""Case definition protocol (refactor plan §6.5)."""
from __future__ import annotations
from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class CaseDefinition(Protocol):
    case_id: str
    # source_manifest, physical_config, numerical_protocol,
    # expected_observables filled in by later slices
