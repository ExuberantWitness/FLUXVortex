"""Load Phase 3b alignment fixture from MATLAB.

Usage:
    from tests.load_matlab_fixture import MatlabFixture
    fx = MatlabFixture("/path/to/fixture_t0.1995.mat")
    fx.summary()
    print(fx.Gamma.shape)       # per-panel ring circulation, (Nx, Ny)
    print(fx.V_surf.shape)      # surface velocity at colloc, (N_element, 3)
    print(fx.Mf2_vec1.shape)    # wake-time-derivative compensation, (N_element,)
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat


# Variables we need from the MATLAB workspace (calc_fluid_force.m scope at dump time)
# Group by purpose so we can fail loudly if anything is missing.
GRADIENT_VARS = ['Gamma', 'Gamma_mat', 'dx_Gamma', 'dy_Gamma', 'old_Gamma']
SURFACE_VELOCITY_VARS = ['V_in', 'V_wake_plate', 'V_gamma', 'V_surf', 'V_surf1',
                         'dt_rc_vec']
PRESSURE_VARS = ['dp_add', 'dp_lift', 'dp_lift1', 'dp_lift2', 'dp_vec']
COUPLING_VARS = ['A_mat', 'B_mat', 'Mf1_mat', 'Mf2_mat', 'Mf2_vec1']
WAKE_DT_VARS = ['Gamma_wake', 'dt_q1234_wake_mat', 'Gamma_wake_dt_q1234',
                'Gamma_wake_dt_q1234_n', 'dt_q1234_mat']
GEOMETRY_VARS = ['Nx', 'Ny', 'N_element', 'tau_x', 'tau_y', 'd_x_mat', 'd_y_mat',
                 'n_vec_i', 'rc_vec']
META_VARS = ['time', 'd_t', 'd_t_wake', 'dt_wake_per_dt', 'i_wake_time',
             'Ma', 'Ua']

ALL_VARS = (GRADIENT_VARS + SURFACE_VELOCITY_VARS + PRESSURE_VARS +
            COUPLING_VARS + WAKE_DT_VARS + GEOMETRY_VARS + META_VARS)


class MatlabFixture:
    """Type-safe wrapper around the .mat fixture file."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Fixture not found: {path}\n"
                "Run dump_fixture_run.m in MATLAB first.")
        self._raw = loadmat(path, squeeze_me=True, struct_as_record=False)
        self._path = path
        self._validate()

    def _validate(self):
        missing = [v for v in ALL_VARS if v not in self._raw]
        if missing:
            print(f"[fixture] WARNING: {len(missing)} vars missing from {self._path}:")
            for v in missing[:10]:
                print(f"  - {v}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._raw:
            raise AttributeError(
                f"Variable '{name}' not in fixture. Available: "
                f"{sorted(k for k in self._raw if not k.startswith('_'))[:20]}...")
        return np.asarray(self._raw[name])

    @property
    def Nx(self) -> int:
        return int(self._raw['Nx'])

    @property
    def Ny(self) -> int:
        return int(self._raw['Ny'])

    @property
    def time(self) -> float:
        return float(self._raw['time'])

    def Gamma_grid(self) -> np.ndarray:
        """Return Γ reshaped to (Nx, Ny) — matches MATLAB Gamma_mat."""
        g = np.asarray(self._raw['Gamma']).ravel()
        return g.reshape(self.Ny, self.Nx).T  # MATLAB column-major then transpose

    def dx_Gamma_grid(self) -> np.ndarray:
        return np.asarray(self._raw['dx_Gamma'])

    def dy_Gamma_grid(self) -> np.ndarray:
        return np.asarray(self._raw['dy_Gamma'])

    def summary(self):
        print(f"\n=== MATLAB fixture: {self._path} ===")
        print(f"  time         = {self.time:.5f}")
        print(f"  grid         = {self.Nx} x {self.Ny}")
        print(f"  N_element    = {int(self._raw.get('N_element', -1))}")
        for group, names in [
            ('GRADIENT', GRADIENT_VARS),
            ('SURFACE V', SURFACE_VELOCITY_VARS),
            ('PRESSURE', PRESSURE_VARS),
            ('COUPLING', COUPLING_VARS),
            ('WAKE_DT', WAKE_DT_VARS),
        ]:
            present = [n for n in names if n in self._raw]
            print(f"  {group:10s} ({len(present)}/{len(names)}): "
                  f"{', '.join(present[:5])}{'...' if len(present)>5 else ''}")

    def assert_close(self, name: str, python_value: np.ndarray,
                     atol: float = 1e-10, rtol: float = 1e-8) -> bool:
        """Compare a Python computed value to MATLAB fixture; print diff stats."""
        if name not in self._raw:
            print(f"[fixture] skip {name}: not in fixture")
            return False
        ml = np.asarray(self._raw[name])
        py = np.asarray(python_value)
        if ml.shape != py.shape:
            print(f"[fixture] {name}: SHAPE MISMATCH ml={ml.shape} py={py.shape}")
            return False
        absdiff = np.abs(ml - py)
        max_abs = float(np.max(absdiff))
        max_rel = float(np.max(absdiff / (np.abs(ml) + 1e-15)))
        ok = max_abs < atol or max_rel < rtol
        tag = "PASS" if ok else "FAIL"
        print(f"[fixture] {tag} {name:20s} max|ml-py|={max_abs:.3e}  "
              f"max_rel={max_rel:.3e}  |ml|_max={float(np.max(np.abs(ml))):.3e}")
        return ok


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV/FSI_by_FEM_and_UVLM/single_sheet/fixture_t0.1995.mat"
    fx = MatlabFixture(path)
    fx.summary()
