"""Block-stationarity check for statistics windows."""
from __future__ import annotations
import torch

def block_stationarity(
    series: torch.Tensor, n_blocks: int = 4
) -> dict[str, float]:
    """Split the window into blocks; report per-block means and their spread.

    A window is stationary when block means agree within a small fraction
    of the series std. This replaces eyeballing.
    """
    blocks = series.chunk(n_blocks, dim=0)
    means = torch.stack([b.mean() for b in blocks])
    spread = float((means.max() - means.min()).item())
    std = float(series.std().item())
    return {
        "block_means": [float(m.item()) for m in means],
        "block_spread": spread,
        "series_std": std,
        "spread_over_std": spread / max(std, 1e-30),
        # `<=` (not `<`) so a perfectly constant series (spread=0, std=0)
        # counts as stationary rather than failing on the degenerate 0<0.
        "stationary": spread <= 0.5 * std,  # heuristic threshold
    }
