"""Contract for the DIAGNOSTIC added-mass scale hook (D1 degeneracy probe).

``FLUXV_MF1_SCALE`` uniformly scales ``Mf1 = rho P A^-1 N``.  It exists so the map
``(Mf1 scale, E) -> (Cn, zmax/c)`` can be finite-differenced; it is not a calibration
knob.  Three properties have to hold or the probe is worthless:

1. the default is exactly 1.0 and ``_scale_added_mass`` returns the SAME tensor
   object.  Returning a copy would look harmless but silently defeats the M1b memo
   and the M1 LU cache, which key on ``id()``/``_version`` of this 71 MB matrix --
   the measurement would then be confounded by a caching change;
2. a non-unity scale multiplies the operator literally (no clamping, no
   renormalisation of the diagonal, no AIC rebuild);
3. invalid values fail loudly at import instead of producing a silently wrong
   operator (a NaN scale would poison every substep).

The import-time guard runs in a subprocess: reloading this module inside the test
session would leave other modules bound to the old module object.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

import fluxvortex.warp_fsi.q16_flux_v5m_native_fsi as module

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_default_is_unity_and_returns_the_identical_object():
    matrix = torch.zeros(4, 4, dtype=torch.float64)
    scaled = module._scale_added_mass(matrix)
    assert scaled is matrix, (
        "the unity branch must return the same object, otherwise the M1/M1b caches "
        "key on a new tensor every call"
    )


@pytest.mark.parametrize("scale", [1.05, 0.95, 1.0e-3])
def test_non_unity_scale_multiplies_literally(monkeypatch, scale):
    matrix = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    monkeypatch.setattr(module, "_MF1_SCALE", scale)
    scaled = module._scale_added_mass(matrix)
    assert torch.equal(scaled, matrix * scale)
    assert scaled is not matrix


def test_scaling_is_exactly_linear_in_the_operator(monkeypatch):
    """Two scales must differ by exactly their ratio -- no hidden renormalisation."""

    matrix = torch.randn(6, 6, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    monkeypatch.setattr(module, "_MF1_SCALE", 1.05)
    first = module._scale_added_mass(matrix)
    monkeypatch.setattr(module, "_MF1_SCALE", 0.95)
    second = module._scale_added_mass(matrix)
    assert torch.allclose(first, second * (1.05 / 0.95), rtol=1.0e-15, atol=0.0)


@pytest.mark.parametrize("value", ["-1", "0", "nan", "abc"])
def test_invalid_scale_aborts_at_import(value):
    environment = dict(os.environ)
    environment["FLUXV_MF1_SCALE"] = value
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            os.path.join(REPO_ROOT, "src"),
            os.path.join(REPO_ROOT, "platform"),
            os.path.join(REPO_ROOT, "platform", "warp_vpm"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", "import fluxvortex.warp_fsi.q16_flux_v5m_native_fsi"],
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO_ROOT,
    )
    assert completed.returncode != 0, f"FLUXV_MF1_SCALE={value!r} was accepted"
    assert "FLUXV_MF1_SCALE" in completed.stderr or "could not convert" in completed.stderr


def test_unset_scale_imports_cleanly():
    environment = dict(os.environ)
    environment.pop("FLUXV_MF1_SCALE", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            os.path.join(REPO_ROOT, "src"),
            os.path.join(REPO_ROOT, "platform"),
            os.path.join(REPO_ROOT, "platform", "warp_vpm"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import fluxvortex.warp_fsi.q16_flux_v5m_native_fsi as m; "
            "print('MF1SCALE=', m._MF1_SCALE)",
        ],
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    # Warp prints a device banner on import, so match the token instead of the line.
    assert "MF1SCALE= 1.0" in completed.stdout, completed.stdout[-400:]
