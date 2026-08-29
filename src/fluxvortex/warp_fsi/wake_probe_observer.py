"""Inertial-frame wake probe observer for the native V5M solver states.

HANDOFF_USE_FIG06_09_12_15_DIGITIZED_DATA_20260829 §5.3/§7 P3: the rigid
Figure 12/13/15 comparisons need the velocity signal at physical probe
points downstream of the wing — bound circulation, LEV/TEV particles and
the free wake all contribute, through the SAME owners the solver advances.
This observer only READS committed states; it duplicates no induction
algorithm (it calls the solver's own ring-velocity and particle-field
evaluations) and cannot perturb the transaction.

Sampling contract:
  u(t, probe) sampled once per committed aerodynamic step;
  u'(t) = u(t) - mean_t(u) over the analysis window;
  PSD via Welch (Hann window, 50% overlap);
  St = f_peak * c / U_inf with parabolic peak interpolation.

The paper reports the dominant frequency averaged over at least 10 probe
locations; ``summarize`` returns per-probe peaks and that average.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


def build_downstream_probe_grid(
    *,
    trailing_edge_world: torch.Tensor,
    chord_m: float,
    span_m: float,
    freestream_direction: torch.Tensor,
    device: str = "cuda:0",
) -> torch.Tensor:
    """12 inertial probes: {1c, 2c} downstream x 3 spanwise x 2 vertical.

    Downstream is along the freestream direction from the trailing-edge
    mid-span; spanwise offsets are -b/4, 0, +b/4 on the wing span axis
    (world y for the fixed-pitch rig); vertical offsets are 0 and +0.2c in
    world z.  The paper's rig probes at 1c/2c past the TE around the
    quarter-span plane; the wider grid keeps 10+ usable locations even if
    one station sits inside a sheet.
    """

    te_mid = trailing_edge_world.reshape(-1, 3).mean(dim=0)
    e_x = freestream_direction / torch.linalg.vector_norm(freestream_direction)
    e_y = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
    e_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float64)
    points: list[torch.Tensor] = []
    for downstream_c in (1.0, 2.0):
        for span_offset in (-0.25, 0.0, 0.25):
            for z_offset_c in (0.0, 0.2):
                points.append(
                    te_mid
                    + downstream_c * chord_m * e_x
                    + span_offset * span_m * e_y
                    + z_offset_c * chord_m * e_z
                )
    grid = torch.stack(points)
    if not bool(torch.isfinite(grid).all().item()):
        raise FloatingPointError("probe grid is non-finite")
    return grid


class WakeProbeObserver:
    """Records committed-state velocities at fixed inertial probe points."""

    def __init__(self, probe_points: torch.Tensor, *, device: str = "cuda:0"):
        if probe_points.ndim != 2 or probe_points.shape[1] != 3:
            raise ValueError("probe_points must have shape (n, 3)")
        self.probe_points = probe_points.contiguous().to(
            device=device, dtype=torch.float64
        )
        self.device = device
        self._history: list[np.ndarray] = []

    def sample(self, solver, state) -> None:
        """Read one committed state: freestream + bound + wake + particles.

        ``solver`` is a native V5M solver instance (rigid or Q16) whose
        ``_ring_velocity``/``v_inf`` and the state's ``particle_field``
        define the induction, and whose ``surface.evaluate`` provides the
        bound-ring geometry.
        """

        geometry = solver.surface.evaluate(None, None)
        velocity = (
            solver.v_inf[None, :]
            + solver._ring_velocity(
                self.probe_points, geometry.rings, state.gamma_bound, rough=False
            )
            + solver._ring_velocity(
                self.probe_points, state.wake_rings, state.wake_gamma, rough=False
            )
        )
        if state.particle_field.n:
            velocity = velocity + state.particle_field.velocity_at_cuda(
                self.probe_points
            )
        if not bool(torch.isfinite(velocity).all().item()):
            raise FloatingPointError("wake probe sampled a non-finite velocity")
        self._history.append(velocity.detach().cpu().numpy())

    @property
    def sample_count(self) -> int:
        return len(self._history)

    def velocity_history(self) -> np.ndarray:
        """Shape (samples, probes, 3)."""

        if not self._history:
            raise RuntimeError("no probe samples recorded yet")
        return np.stack(self._history)


@dataclass(frozen=True, slots=True)
class WakeProbeSummary:
    st_per_probe: tuple[float, ...]
    st_mean: float
    st_spread: float
    st_modified: float  # st_mean * sin(alpha_deg) exactly (closure = 0)
    peak_frequency_hz: float
    samples: int
    window_start_index: int
    # Dominant probe's spectrum for shape comparison (Figure 12): the St
    # grid and PSD of the probe whose peak power is largest.
    dominant_psd_st: tuple[float, ...] = ()
    dominant_psd: tuple[float, ...] = ()


def _welch_psd(signal: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Hann-window Welch PSD with 50% overlap (8 segments minimum)."""

    n = signal.shape[0]
    segment = min(n, max(64, n // 4))
    while segment > 64 and n // (segment // 2) - 1 < 8:
        segment = (segment // 2) + (segment % 2)
    step = max(1, segment // 2)
    starts = range(0, n - segment + 1, step)
    window = np.hanning(segment)
    scale = sample_rate_hz / (window.sum() ** 2)
    acc = None
    count = 0
    for start in starts:
        chunk = signal[start : start + segment] * window
        spectrum = np.abs(np.fft.rfft(chunk)) ** 2 * scale
        acc = spectrum if acc is None else acc + spectrum
        count += 1
    frequencies = np.fft.rfftfreq(segment, d=1.0 / sample_rate_hz)
    return frequencies, (acc / count if count else spectrum)


def summarize_probes(
    observer: WakeProbeObserver,
    *,
    chord_m: float,
    freestream_m_s: float,
    delta_time_s: float,
    alpha_deg: float,
    start_index: int = 0,
) -> WakeProbeSummary:
    """Peak Strouhal per probe (streamwise component) + paper-style average."""

    history = observer.velocity_history()[start_index:]
    if history.shape[0] < 128:
        raise ValueError(
            f"too few probe samples for a spectrum ({history.shape[0]} < 128)"
        )
    st_peaks: list[float] = []
    best_frequency = float("nan")
    best_power = -1.0
    best_st_grid: np.ndarray | None = None
    best_psd: np.ndarray | None = None
    for probe in range(history.shape[1]):
        signal = history[:, probe, 0]
        fluctuation = signal - signal.mean()
        if not float(np.std(fluctuation)) > 0.0:
            st_peaks.append(float("nan"))
            continue
        frequencies, psd = _welch_psd(fluctuation, 1.0 / delta_time_s)
        # Detrend the slow wake drift so the DC/lowest bins cannot win the
        # peak search; shedding peaks here live at St >= ~0.2 while windowed
        # startup drift piles into the first bins.
        drift = np.polyval(np.polyfit(np.arange(fluctuation.shape[0]), fluctuation, 2), np.arange(fluctuation.shape[0]))
        detrended = fluctuation - drift
        frequencies_d, psd = _welch_psd(detrended, 1.0 / delta_time_s)
        st_grid = frequencies_d * chord_m / freestream_m_s
        searchable = st_grid >= 0.05
        if not np.any(searchable):
            st_peaks.append(float("nan"))
            continue
        masked = np.where(searchable, psd, -np.inf)
        peak_index = int(np.argmax(masked))
        # Peak prominence: a real shedding peak must dominate the broadband
        # floor by 10x; otherwise the probe reports no resolved peak (NaN).
        floor = float(np.median(psd[searchable]))
        if not psd[peak_index] > 10.0 * max(floor, 1e-300):
            st_peaks.append(float("nan"))
            continue
        # Parabolic interpolation for sub-bin peak accuracy.
        if 0 < peak_index < len(psd) - 1:
            left, mid, right = psd[peak_index - 1 : peak_index + 2]
            denominator = left - 2.0 * mid + right
            shift = 0.5 * (left - right) / denominator if denominator != 0 else 0.0
            shift = float(np.clip(shift, -0.5, 0.5))
        else:
            shift = 0.0
        peak_hz = frequencies_d[peak_index] + shift * (
            frequencies_d[1] - frequencies_d[0]
        )
        st_peaks.append(peak_hz * chord_m / freestream_m_s)
        if psd[peak_index] > best_power:
            best_power = float(psd[peak_index])
            best_frequency = peak_hz
            best_st_grid = st_grid
            best_psd = psd
    finite = [s for s in st_peaks if math.isfinite(s)]
    if not finite:
        # No probe resolved a dominant peak (e.g. attached flow): an honest
        # "no shedding" outcome, reported as NaN rather than a fabricated
        # broadband peak.  Callers keep the case successful with empty St.
        return WakeProbeSummary(
            st_per_probe=tuple(st_peaks),
            st_mean=float("nan"),
            st_spread=float("nan"),
            st_modified=float("nan"),
            peak_frequency_hz=float("nan"),
            samples=int(history.shape[0]),
            window_start_index=int(start_index),
        )
    st_mean = float(np.mean(finite))
    return WakeProbeSummary(
        st_per_probe=tuple(st_peaks),
        st_mean=st_mean,
        st_spread=float(np.std(finite)),
        st_modified=st_mean * math.sin(math.radians(alpha_deg)),
        peak_frequency_hz=best_frequency,
        samples=int(history.shape[0]),
        window_start_index=int(start_index),
        dominant_psd_st=tuple(float(v) for v in best_st_grid)
        if best_st_grid is not None
        else (),
        dominant_psd=tuple(float(v) for v in best_psd) if best_psd is not None else (),
    )


__all__ = [
    "WakeProbeObserver",
    "WakeProbeSummary",
    "build_downstream_probe_grid",
    "summarize_probes",
]
