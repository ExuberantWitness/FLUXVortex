"""Independent device and deterministic-value probe for ParticleField."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import warp as wp

import pfield
from pfield import ParticleField


field = ParticleField(capacity=8)
field.add_particles(
    pos=np.array(
        [[0.0, 0.0, 0.0], [0.8, -0.2, 0.1], [-0.3, 0.6, -0.4]],
        dtype=np.float64,
    ),
    gamma=np.array(
        [[0.2, -0.1, 0.5], [-0.4, 0.3, 0.2], [0.1, 0.6, -0.2]],
        dtype=np.float64,
    ),
    sigma=np.array([0.25, 0.4, 0.3], dtype=np.float64),
)
targets = np.array(
    [[0.2, 0.1, 0.7], [-0.5, -0.1, 0.2], [1.1, 0.4, -0.2]],
    dtype=np.float64,
)
velocity = np.asarray(field.velocity_at(targets), dtype=np.float64)
payload = {
    "schema": "fluxv-v5m-pfield-device-audit-v1",
    "pfield_device_env": os.environ.get("PFIELD_DEVICE"),
    "selected_device": pfield._DEVICE,
    "cuda_device_count": int(wp.get_cuda_device_count()),
    "warp_cuda_available": bool(wp.is_cuda_available()),
    "velocity": velocity.tolist(),
    "velocity_sha256": hashlib.sha256(velocity.tobytes(order="C")).hexdigest(),
    "all_finite": bool(np.isfinite(velocity).all()),
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))

