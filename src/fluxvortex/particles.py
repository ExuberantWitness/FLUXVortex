"""
VortexParticleField — full Reformulated VPM (rVPM) with RK3 time integration.

Ported from FLOWVPM's ReformulatedVPM{Float64} with:
  - RK3 low-storage scheme (3 substeps)
  - Reformulated VPM stretching (f=0, g=1/5)
  - Gaussian-erf Biot-Savart kernel
  - Pedrizzetti relaxation
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
        pos = self._pos[:n]
        gamma = self._gamma[:n]
        sigma = self._sigma[:n]

        # Reset RK storage
        rk_X = self._rk_X[:n]
        rk_G = self._rk_G[:n]
        rk_s = self._rk_s[:n]
        rk_X[:] = 0.0
        rk_G[:] = 0.0
        rk_s[:] = 0.0

        # RK3 coefficients: (a, b)
        rk_coeffs = [(0.0, 1.0 / 3.0), (-5.0 / 9.0, 15.0 / 16.0), (-153.0 / 128.0, 8.0 / 15.0)]

        for a_coeff, b_coeff in rk_coeffs:
            # Total velocity at current particle positions
            U_part = self.induce_velocity_at(pos)  # particle-particle

            if bound_velocity_func is not None:
                U_bound = bound_velocity_func(pos)   # bound vortex → particle
            else:
                U_bound = np.zeros_like(pos)

            U_inf = U_inf_func(pos)                  # freestream

            U_total = U_part + U_bound + U_inf       # (n, 3)

            # Jacobian for stretching
            J = jacobian_from_particles(pos, gamma, pos, gamma, sigma)  # (n, 3, 3)

            # Stretching: S = J^T · Gamma
            # J[target, vel_comp, pos_comp]  →  S_i = sum_j J[j,i] * Gamma[j]
            S = np.einsum('tij,tj->ti', J, gamma)   # (n, 3) — this is J^T * Gamma

            # rVPM reformulation: Z = (g) * (S · Gamma) / |Gamma|²
            # with f=0, g=1/5
            Gamma_sq = np.sum(gamma ** 2, axis=1)    # (n,)
            SdotG = np.sum(S * gamma, axis=1)        # (n,)

            # Avoid division by zero
            mask = Gamma_sq > 1e-20
            Z = np.zeros(n)
            Z[mask] = 0.2 * SdotG[mask] / Gamma_sq[mask]

            # RHS
            dGamma = S - 3.0 * Z[:, None] * gamma    # (n, 3)
            dsigma = -sigma * Z                       # (n,)

            # Viscous core spreading: d(sigma²)/dt = 2*nu → dsigma/dt = nu/sigma
            if self.nu > 0:
                dsigma += self.nu / np.maximum(sigma, 1e-10)

            # RK update: storage = a*storage + dt*RHS; variable += b*storage
            rk_X = a_coeff * rk_X + dt * U_total
            rk_G = a_coeff * rk_G + dt * dGamma
            rk_s = a_coeff * rk_s + dt * dsigma

            pos += b_coeff * rk_X
            gamma += b_coeff * rk_G
            sigma += b_coeff * rk_s

        # Write back
        self._pos[:n] = pos
        self._gamma[:n] = gamma
        self._sigma[:n] = sigma
        self._age[:n] += dt

        # Clamp sigma to prevent collapse
        min_sigma = 1e-6
        self._sigma[:n] = np.maximum(self._sigma[:n], min_sigma)

        # Apply Pedrizzetti relaxation
        self._relax_pedrizzetti()

    def _relax_pedrizzetti(self):
        """Realign Gamma with local vorticity direction."""
        if self.rlxf <= 0 or self.np == 0:
            return

        n = self.np
        pos = self._pos[:n]
        gamma = self._gamma[:n]

        # Compute Jacobian for vorticity
        J = jacobian_from_particles(pos, gamma, pos, gamma, self._sigma[:n])

        # Vorticity: omega_i = epsilon_ijk * J[j,k]
        # omega_x = J[2,1] - J[1,2]  =  J[row=2,col=1] - J[row=1,col=2]
        omega = np.zeros((n, 3))
        omega[:, 0] = J[:, 2, 1] - J[:, 1, 2]
        omega[:, 1] = J[:, 0, 2] - J[:, 2, 0]
        omega[:, 2] = J[:, 1, 0] - J[:, 0, 1]

        omega_norm = np.linalg.norm(omega, axis=1)
        mask = omega_norm > 1e-12

        Gamma_norm = np.linalg.norm(gamma, axis=1)

        # Gamma_new = (1-rlxf)*Gamma + rlxf*|Gamma|*omega_hat
        omega_hat = np.zeros_like(omega)
        omega_hat[mask] = omega[mask] / omega_norm[mask, None]

        gamma[mask] = ((1.0 - self.rlxf) * gamma[mask]
                       + self.rlxf * Gamma_norm[mask, None] * omega_hat[mask])

        # Corrected Pedrizzetti: normalize to prevent magnitude drift
        b2 = 1.0 - 2.0 * (1.0 - self.rlxf) * self.rlxf * (
            1.0 - np.sum(gamma[mask] * omega_hat[mask], axis=1) / np.maximum(Gamma_norm[mask], 1e-12)
        )
        b2 = np.maximum(b2, 0.01)  # prevent division by tiny number
        gamma[mask] /= np.sqrt(b2)[:, None]

        self._gamma[:n] = gamma
