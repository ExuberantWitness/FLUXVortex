"""Frozen CUDA endpoint loads that are independent of the aerodynamic owner.

The production FSI transaction keeps aerodynamic and prescribed structural
loads as distinct ledgers.  Their numerical sum remains on CUDA float64; the
small host copy in this module is used only to seal provenance evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import warp as wp

from . import config

_DOMAIN = "flux-v5m-q16-prescribed-endpoint-load-v1"


def _load_sha256(
    generalized_force: wp.array,
    *,
    source_id: str,
    endpoint_time_s: float,
) -> str:
    host = np.ascontiguousarray(generalized_force.numpy(), dtype=np.float64)
    header = json.dumps(
        {
            "domain": _DOMAIN,
            "dtype": "float64",
            "endpoint_time_hex": endpoint_time_s.hex(),
            "shape": list(host.shape),
            "source_id": source_id,
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + host.tobytes(order="C")).hexdigest()


def _require_force(value: Any) -> wp.array:
    if not isinstance(value, wp.array):
        raise TypeError("generalized_force must be a Warp array")
    if not value.device.is_cuda:
        raise ValueError("prescribed endpoint load must reside on CUDA")
    if value.dtype != config.DTYPE:
        raise TypeError("prescribed endpoint load must use Warp float64")
    if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] <= 0:
        raise ValueError("prescribed endpoint load must have shape (1, positive_dof)")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class Q16CudaPrescribedEndpointLoad:
    """One immutable non-aerodynamic generalized load at a step endpoint."""

    generalized_force: wp.array
    source_id: str
    endpoint_time_s: float
    load_sha256: str

    @classmethod
    def from_force(
        cls,
        generalized_force: Any,
        *,
        source_id: str,
        endpoint_time_s: float,
    ) -> Q16CudaPrescribedEndpointLoad:
        if cls is not Q16CudaPrescribedEndpointLoad:
            raise TypeError("prescribed endpoint load subclasses are forbidden")
        force = _require_force(generalized_force)
        if type(source_id) is not str or not source_id or not source_id.isascii():
            raise ValueError("source_id must be a non-empty ASCII string")
        if isinstance(endpoint_time_s, bool) or not isinstance(
            endpoint_time_s, (int, float, np.integer, np.floating)
        ):
            raise TypeError("endpoint_time_s must be a real scalar")
        endpoint = float(endpoint_time_s)
        if not math.isfinite(endpoint) or endpoint < 0.0:
            raise ValueError("endpoint_time_s must be finite and non-negative")
        frozen = wp.clone(force)
        return cls(
            generalized_force=frozen,
            source_id=source_id,
            endpoint_time_s=endpoint,
            load_sha256=_load_sha256(
                frozen,
                source_id=source_id,
                endpoint_time_s=endpoint,
            ),
        ).validate()

    def validate(self) -> Q16CudaPrescribedEndpointLoad:
        if type(self) is not Q16CudaPrescribedEndpointLoad:
            raise TypeError("prescribed endpoint load subclasses are forbidden")
        force = _require_force(self.generalized_force)
        if type(self.source_id) is not str or not self.source_id:
            raise ValueError("prescribed endpoint source_id is invalid")
        if type(self.endpoint_time_s) is not float or not math.isfinite(
            self.endpoint_time_s
        ) or self.endpoint_time_s < 0.0:
            raise ValueError("prescribed endpoint time is invalid")
        expected = _load_sha256(
            force,
            source_id=self.source_id,
            endpoint_time_s=self.endpoint_time_s,
        )
        if type(self.load_sha256) is not str or self.load_sha256 != expected:
            raise RuntimeError("prescribed endpoint load content drift")
        return self


__all__ = ["Q16CudaPrescribedEndpointLoad"]
