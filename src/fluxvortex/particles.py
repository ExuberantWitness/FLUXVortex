"""
VortexParticleField — full Reformulated VPM (rVPM) with RK3 time integration.

Ported from FLOWVPM's ReformulatedVPM{Float64} with:
  - RK3 low-storage scheme (3 substeps)
  - Reformulated VPM stretching (f=0, g=1/5)
  - Gaussian-erf Biot-Savart kernel
  - Pedrizzetti relaxation
  - Stability: Gamma limiter, NaN protection, velocity clamping
"""
import numpy as np
from .kernel import velocity_from_particles, jacobian_from_particles


class VortexParticleField:
    """Structure-of-arrays vortex particle field."""

    def __init__(self, max_particles=50000, nu=0.0, rlxf=0.3):
        self.max_particles = max_particles
        self.nu = nu          # kinematic viscosity (0 = inviscid)
        self.rlxf = rlxf      # Pedrizzetti relaxation factor

        self._pos = np.zeros((max_particles, 3))
        self._gamma = np.zeros((max_particles, 3))
        self._sigma = np.zeros(max_particles)
        self._age = np.zeros(max_particles)
        self.np = 0           # current particle count

        # RK3 storage
        self._rk_X = np.zeros((max_particles, 3))
        self._rk_G = np.zeros((max_particles, 3))
        self._rk_s = np.zeros(max_particles)

    # ── Particle management ───────────────────────────────────────
    def add_particle(self, X, Gamma, sigma):
        """Add a single vortex particle."""
        if self.np >= self.max_particles:
            raise RuntimeError(f"Particle field full ({self.max_particles})")
        i = self.np
        self._pos[i] = X
        self._gamma[i] = Gamma
        self._sigma[i] = sigma
        self._age[i] = 0.0
        self.np += 1

    def add_particles_batch(self, positions, gammas, sigmas):
        """Add multiple particles at once."""
        n = positions.shape[0]
        if self.np + n > self.max_particles:
            raise RuntimeError(f"Particle field would overflow ({self.np}+{n} > {self.max_particles})")
        sl = slice(self.np, self.np + n)
        self._pos[sl] = positions
        self._gamma[sl] = gammas
        self._sigma[sl] = sigmas
        self._age[sl] = 0.0
        self.np += n

    @property
    def positions(self):
        return self._pos[:self.np].copy()

    @property
    def gammas(self):
        return self._gamma[:self.np].copy()

    @property
    def sigmas(self):
        return self._sigma[:self.np].copy()

    # ── Induced velocity ──────────────────────────────────────────
    def induce_velocity_at(self, target_points):
        """Velocity induced by all particles at arbitrary target points."""
        if self.np == 0:
            return np.zeros_like(target_points)
        return velocity_from_particles(
            target_points,
            self._pos[:self.np],
            self._gamma[:self.np],
            self._sigma[:self.np],
        )

    # ── RK3 + rVPM time integration ──────────────────────────────
    def advect_rk3(self, dt, U_inf_func, bound_velocity_func=None):
        """
        Advance particles one timestep using RK3 + Reformulated VPM.

        Parameters
        ----------
        dt : float — timestep
        U_inf_func : callable(X) -> (N,3) — freestream velocity at particle positions
        bound_velocity_func : callable(X) -> (N,3) — velocity induced by bound wing vortices
        """
        if self.np == 0:
            return

        n = self.np
        pos = self._pos[:n].copy()
        gamma = self._gamma[:n].copy()
        sigma = self._sigma[:n].copy()

        # Reset RK storage
        rk_X = np.zeros((n, 3))
        rk_G = np.zeros((n, 3))
        rk_s = np.zeros(n)

        # RK3 coefficients: (a, b)
        rk_coeffs = [(0.0, 1.0 / 3.0), (-5.0 / 9.0, 15.0 / 16.0), (-153.0 / 128.0, 8.0 / 15.0)]

        # Max Gamma^2 to prevent stretching blowup
        max_gamma_sq = 1e4

        for a_coeff, b_coeff in rk_coeffs:
            # Total velocity at current particle positions
            U_part = velocity_from_particles(pos, pos, gamma, sigma)

            if bound_velocity_func is not None:
                U_bound = bound_velocity_func(pos)
            else:
                U_bound = np.zeros_like(pos)

            U_inf = U_inf_func(pos)

            U_total = U_part + U_bound + U_inf

            # Clamp velocity to prevent blowup
            U_speed = np.linalg.norm(U_total, axis=1)
            v_max = 50.0  # max allowed velocity magnitude
            fast = U_speed > v_max
            if np.any(fast):
                U_total[fast] *= (v_max / U_speed[fast])[:, None]

            # Jacobian for stretching
            J = jacobian_from_particles(pos, gamma, pos, gamma, sigma)

            # Stretching: S = J^T · Gamma
            S = np.einsum('tij,tj->ti', J, gamma)

            # rVPM reformulation: Z = g * (S · Gamma) / |Gamma|^2
            Gamma_sq = np.sum(gamma ** 2, axis=1)
            SdotG = np.sum(S * gamma, axis=1)

            mask = Gamma_sq > 1e-20
            Z = np.zeros(n)
            Z[mask] = 0.2 * SdotG[mask] / Gamma_sq[mask]

            # Clamp Z rate to prevent excessive stretching
            Z_max = 10.0 / max(dt, 1e-10)
            Z = np.clip(Z, -Z_max, Z_max)

            # RHS
            dGamma = S - 3.0 * Z[:, None] * gamma
            dsigma = -sigma * Z

            # Viscous core spreading: d(sigma^2)/dt = 2*nu
            if self.nu > 0:
                dsigma += self.nu / np.maximum(sigma, 1e-10)

            # RK update: storage = a*storage + dt*RHS; variable += b*storage
            rk_X = a_coeff * rk_X + dt * U_total
            rk_G = a_coeff * rk_G + dt * dGamma
            rk_s = a_coeff * rk_s + dt * dsigma

            pos += b_coeff * rk_X
            gamma += b_coeff * rk_G
            sigma += b_coeff * rk_s

            # Clamp Gamma magnitude to prevent runaway
            g_sq = np.sum(gamma ** 2, axis=1)
            too_big = g_sq > max_gamma_sq
            if np.any(too_big):
                scale = np.sqrt(max_gamma_sq / g_sq[too_big])
                gamma[too_big] *= scale[:, None]

            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 5.0)

        # NaN/Inf protection: zero out bad particles
        bad_pos = ~np.isfinite(pos).all(axis=1)
        bad_gam = ~np.isfinite(gamma).all(axis=1)
        bad = bad_pos | bad_gam
        if np.any(bad):
            gamma[bad] = 0.0

        # Write back
        self._pos[:n] = pos
        self._gamma[:n] = gamma
        self._sigma[:n] = sigma
        self._age[:n] += dt

        # Apply Pedrizzetti relaxation
        self._relax_pedrizzetti()

    # ── Pedrizzetti relaxation ────────────────────────────────────
    def _relax_pedrizzetti(self):
        """Realign Gamma with local vorticity direction."""
        if self.rlxf <= 0 or self.np == 0:
            return

        n = self.np
        pos = self._pos[:n]
        gamma = self._gamma[:n]
        sigma = self._sigma[:n]

        # Skip particles with zero Gamma (relaxation is meaningless)
        g_norm = np.linalg.norm(gamma, axis=1)
        active = g_norm > 1e-15
        if not np.any(active):
            return

        # Compute Jacobian for vorticity
        J = jacobian_from_particles(pos, gamma, pos, gamma, sigma)

        # Vorticity: omega_i = epsilon_ijk * J[j,k]
        omega = np.zeros((n, 3))
        omega[:, 0] = J[:, 2, 1] - J[:, 1, 2]
        omega[:, 1] = J[:, 0, 2] - J[:, 2, 0]
        omega[:, 2] = J[:, 1, 0] - J[:, 0, 1]

        omega_norm = np.linalg.norm(omega, axis=1)
        mask = active & (omega_norm > 1e-12)

        if not np.any(mask):
            return

        Gamma_norm = g_norm[mask]
        omega_hat = omega[mask] / omega_norm[mask, None]

        gamma[mask] = ((1.0 - self.rlxf) * gamma[mask]
                       + self.rlxf * Gamma_norm[:, None] * omega_hat)

        # Normalize to prevent magnitude drift
        b2 = 1.0 - 2.0 * (1.0 - self.rlxf) * self.rlxf * (
            1.0 - np.sum(gamma[mask] * omega_hat, axis=1) / np.maximum(Gamma_norm, 1e-12)
        )
        b2 = np.maximum(b2, 0.01)
        gamma[mask] /= np.sqrt(b2)[:, None]

        self._gamma[:n] = gamma
