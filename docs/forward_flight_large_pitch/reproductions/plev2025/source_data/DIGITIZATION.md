# Yang et al. (2025) Fig. 11 rigid-wing digitization

## Source

The values in `yang2025_fig11_rigid_digitized.csv` were digitized from Fig. 11, **Type A (rigid wing)**, in:

H.-H. Yang, S.-G. Lee, E.-H. Lee, and J.-H. Han, “Numerical simulation framework of bird-inspired ornithopter in forward flight,” *Journal of Fluids and Structures* 133 (2025), 104263. [https://doi.org/10.1016/j.jfluidstructs.2024.104263](https://doi.org/10.1016/j.jfluidstructs.2024.104263)

Fig. 11 reports cycle-averaged lift and thrust in gram-force (`gf`) at geometric angles of attack 0, 5, 10, 15, 20, and 25 degrees. The `Test` series uses open-circle markers; the `Proposed` series uses star markers and a solid line.

## Method and uncertainty

The embedded Fig. 11 raster was extracted from the formal paper PDF at its native 1800 × 1046 pixel resolution. The vertical pixel-to-`gf` mapping was calibrated from the plotted horizontal grid/tick positions. Open-circle centers were used for `Test`; star centers and their adjoining solid-line segments were used for `Proposed`. The resulting ordinates were rounded to 0.1 gf.

The estimated digitization uncertainty is **±0.4 gf per ordinate**. This is graphical reading uncertainty only; it does not include experimental uncertainty, repeatability, or model-form uncertainty. The AoA values are the six discrete abscissae printed in the figure rather than digitized estimates.

## Interpretation warning

`Proposed` is **not a PLEV-only ablation**. In the paper, it denotes the complete proposed modified-UVLM framework, including PLEV, adaptive wake shedding (AWS), and the paper's free-wake/vortex-core treatment. Therefore, the `Proposed` columns must not be used as isolated evidence for the effect or accuracy of PLEV alone. Fig. 11 provides no PLEV-only or AWS-only curve.
