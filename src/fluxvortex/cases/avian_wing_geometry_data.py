"""Verified ground-truth digitized planforms, airfoil camber profiles, and RoboFalcon 2.0 Eq. S1 kinematics.

Self-contained portable geometry database extracted directly from peer-reviewed figures,
supplementary tables, and UIUC airfoil datasets (zero runtime dependency on external files):

1. Cloud Owl (Meng et al., MDPI Appl. Sci. 2023):
   - Planform digitized from Fig. 7 (single-wing exposed span 800 mm + 110 mm half-fuselage = 1.82 m;
     exposed 2-wing area 0.428 m^2 vs published 0.426 m^2 in Section 2.2).
   - Reference airfoil: Eppler E378 (UIUC).
2. RoboFalcon 2.0 (Chen et al., Science Advances 2025, adx0465):
   - Unfolded & folded planforms digitized from Fig. 3A (1.20 m x 0.60 m scale bars).
   - Full 5-DOF FSF kinematics (Theta, Psi, alpha, beta, gamma) from Supplementary Eq. S1
     re-projected onto exact C^inf periodic k=1 6-harmonic Fourier series (zero cycle-end jump).
   - Rigid 3-segment carbon skeleton (Humerus 110 mm, Radius 180 mm from RoboFalcon Table I;
     Manus 325 mm from Fig. 3A) reproducing BOTH Fig. 3A unfolded/folded contours (<2 mm) AND
     Fig. 3J semispan contraction (600 mm -> 327 mm vs published 595 -> 325 mm).
3. NPU-Sparrow (Cao et al., Nature Communications 2026):
   - Planform digitized from Fig. 2a (0.45 m x 0.33 m scale bars; gross area 0.0414 m^2 vs
     published 0.04185 m^2 in Fig. 2b).
   - Reference airfoil: GOE225 (UIUC) on inner ribs, smoothly flattening outboard.
4. Dove (Yang et al., IJMAV 2018):
   - Normalized planform digitized from Fig. 8 & Fig. 9 (straight LE, constant inboard chord,
     smooth rounded TE; b = 0.50 m, exact published AR = 7.4 -> S = 0.03378 m^2).
   - Reference airfoil: Eppler E378 (UIUC).
5. High-Speed Dove (Xue et al., MDPI Drones 2024; Liu & Xue, ICAS 2024-0473):
   - Baseline wingspan b = 0.60 m (Table 2), exact ICAS Table 1 six-station S-shaped pigeon
     chord profile [100, 100, 120, 125, 115, 85] mm at [75, 130, 185, 220, 250, 285] mm.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DigitizedPlanformSpec:
    """Half-span ground-truth digitized planform (from centerline y=0 to wingtip y=half_span_m)."""

    planform_id: str
    source_note: str
    half_span_m: float
    y_root_body_m: float
    gross_area_m2: float
    y_m: tuple[float, ...]
    x_le_m: tuple[float, ...]
    x_te_m: tuple[float, ...]


DIGITIZED_PLANFORMS: dict[str, DigitizedPlanformSpec] = {
    "cloud_owl": DigitizedPlanformSpec(
        planform_id="cloud_owl",
        source_note="Meng et al. (2023) Fig. 7 (800 mm exposed membrane + 110 mm half-body = 0.91 m half-span)",
        half_span_m=0.910,
        y_root_body_m=0.110,
        gross_area_m2=0.495181,
        y_m=(0.0, 0.11, 0.15, 0.19, 0.23, 0.27, 0.31, 0.35, 0.39, 0.43, 0.47, 0.51, 0.55, 0.59, 0.63, 0.67, 0.71, 0.75, 0.79, 0.83, 0.87, 0.91),
        x_le_m=(0.0, 0.0, -0.000299, -0.000684, -0.001067, -0.00146, -0.001841, -0.002173, -0.002446, -0.002739, -0.003143, -0.003769, -0.004507, -0.005209, -0.00593, -0.006612, -0.00713, -0.007457, -0.007828, -0.008838, -0.009981, -0.01),
        x_te_m=(0.290435, 0.290435, 0.290435, 0.289987, 0.289615, 0.289246, 0.288795, 0.287542, 0.285778, 0.285017, 0.284646, 0.285171, 0.287225, 0.288639, 0.287391, 0.281507, 0.271189, 0.255463, 0.230672, 0.196165, 0.14825, 0.085),
    ),
    "robofalcon_2": DigitizedPlanformSpec(
        planform_id="robofalcon_2",
        source_note="Chen et al. (2025) Fig. 3A unfolded (569 mm visible wing + 31 mm half-body = 0.60 m half-span)",
        half_span_m=0.600,
        y_root_body_m=0.031,
        gross_area_m2=0.231495,
        y_m=(0.0, 0.031, 0.059453, 0.087905, 0.116358, 0.144811, 0.173264, 0.201717, 0.230169, 0.258622, 0.287075, 0.315528, 0.34398, 0.372433, 0.400886, 0.429339, 0.457791, 0.486244, 0.514697, 0.54315, 0.571603, 0.6),
        x_le_m=(0.0, 0.0, -0.009986, -0.015469, -0.022656, -0.029919, -0.037194, -0.045147, -0.053121, -0.061157, -0.063532, -0.058143, -0.052754, -0.047365, -0.041976, -0.036588, -0.031199, -0.027023, -0.024189, -0.021355, -0.01852, -0.0157),
        x_te_m=(0.245787, 0.245787, 0.245787, 0.236638, 0.227489, 0.21834, 0.209191, 0.200042, 0.190892, 0.181743, 0.172567, 0.163392, 0.154216, 0.141497, 0.128538, 0.115599, 0.10266, 0.089721, 0.076782, 0.0545, 0.0505, 0.0563),
    ),
    "npu_sparrow": DigitizedPlanformSpec(
        planform_id="npu_sparrow",
        source_note="Cao et al. (2026) Fig. 2a (201.8 mm visible membrane + 23.2 mm half-body = 0.225 m half-span)",
        half_span_m=0.225,
        y_root_body_m=0.02318,
        gross_area_m2=0.041664,
        y_m=(0.0, 0.02318, 0.033271, 0.043362, 0.053453, 0.063545, 0.073636, 0.083727, 0.093818, 0.103909, 0.114, 0.124091, 0.134182, 0.144273, 0.154365, 0.164456, 0.174547, 0.184638, 0.194729, 0.20482, 0.214911, 0.225),
        x_le_m=(0.0, 0.0, 0.0, 0.0, 0.0, 0.000161, 0.000707, 0.000707, 0.000707, 0.000707, 0.000707, 0.000747, 0.000876, 0.001004, 0.003081, 0.006679, 0.010277, 0.013874, 0.017472, 0.02107, 0.024668, 0.0283),
        x_te_m=(0.107577, 0.107577, 0.107577, 0.106946, 0.106182, 0.105355, 0.104574, 0.103844, 0.103097, 0.102261, 0.101276, 0.100235, 0.099271, 0.09839, 0.097286, 0.09587, 0.094011, 0.091478, 0.088258, 0.081694, 0.0645, 0.0643),
    ),
    "dove_2018": DigitizedPlanformSpec(
        planform_id="dove_2018",
        source_note="Yang et al. (2018) Fig. 8 & Fig. 9 normalized silhouette (b = 0.50 m, exact AR = 7.4)",
        half_span_m=0.250,
        y_root_body_m=0.0,
        gross_area_m2=0.03378,
        y_m=(0.0, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.1125, 0.125, 0.1375, 0.15, 0.1625, 0.175, 0.1875, 0.2, 0.2125, 0.225, 0.2375, 0.25),
        x_le_m=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        x_te_m=(0.0771, 0.0771, 0.0771, 0.0771, 0.0771, 0.07705, 0.07691, 0.0767, 0.0764, 0.07582, 0.075, 0.0738, 0.07209, 0.06993, 0.06703, 0.06362, 0.05908, 0.05184, 0.04166, 0.0315, 0.0275),
    ),
    "dove_highspeed_2024": DigitizedPlanformSpec(
        planform_id="dove_highspeed_2024",
        source_note="Xue et al. (2024) Table 2 (b = 0.60 m) + Liu & Xue (ICAS 2024) Table 1 S-shaped wing stations",
        half_span_m=0.300,
        y_root_body_m=0.030,
        gross_area_m2=0.062851,
        y_m=(0.0, 0.03, 0.055, 0.075, 0.1, 0.13, 0.158, 0.185, 0.205, 0.22, 0.235, 0.25, 0.268, 0.285, 0.294, 0.3),
        x_le_m=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        x_te_m=(0.1, 0.1, 0.1008, 0.1, 0.0992, 0.1, 0.108, 0.12, 0.1235, 0.125, 0.1215, 0.115, 0.103, 0.085, 0.058, 0.040),
    ),
}


@dataclass(frozen=True)
class AirfoilCamberProfile:
    """Normalized thin-surface camber line z_c/c vs chordwise coordinate xi in [0, 1]."""

    airfoil_id: str
    xi: tuple[float, ...]
    zc_over_c: tuple[float, ...]
    outboard_fade: bool = False  # True for NPU-Sparrow (GOE225 inner ribs -> flat outer ribs)


AIRFOIL_CAMBER_PROFILES: dict[str, AirfoilCamberProfile] = {
    "e378": AirfoilCamberProfile(
        airfoil_id="e378",
        xi=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
        zc_over_c=(0.0, 0.02266, 0.04001, 0.05407, 0.06525, 0.07318, 0.07802, 0.07988, 0.07914, 0.07603, 0.07094, 0.06488, 0.05807, 0.05071, 0.04296, 0.03495, 0.02685, 0.01884, 0.01112, 0.00427, 0.0),
        outboard_fade=False,
    ),
    "goe225": AirfoilCamberProfile(
        airfoil_id="goe225",
        xi=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
        zc_over_c=(0.0, 0.01782, 0.03214, 0.04247, 0.05092, 0.05807, 0.06507, 0.06918, 0.07315, 0.07476, 0.07629, 0.07544, 0.07448, 0.06997, 0.06533, 0.05663, 0.04788, 0.03676, 0.02564, 0.01336, 0.0),
        outboard_fade=True,
    ),
}


@dataclass(frozen=True)
class RoboFalcon2EqS1KinematicsData:
    """Exact periodic k=1 Fourier representation of Science Advances Eq. S1 + 3-bar skeleton."""

    # Periodic 6-harmonic Fourier coefficients (deg): (a0, c1, s1, c2, s2, ..., c6, s6)
    flap_Theta_deg: tuple[float, ...]
    sweep_Psi_deg: tuple[float, ...]
    humerus_alpha_deg: tuple[float, ...]
    radius_beta_deg: tuple[float, ...]
    manus_gamma_deg: tuple[float, ...]
    # Rigid carbon skeleton bone lengths [m] (Table I + Fig. 3A)
    y_root_body_m: float = 0.031
    l_humerus_m: float = 0.110
    l_radius_m: float = 0.180
    l_manus_m: float = 0.325
    # Exact Fig. 3A -> Fig. 3J rigid 2-bar Humerus/Radius angle polynomials in u_fold in [0, 1]:
    #   theta_s(u) [deg] = c2*u^2 + c1*u + c0 (shoulder angle from +x aft axis)
    #   theta_e(u) [deg] = c2*u^2 + c1*u + c0 (interior elbow angle between humerus and radius)
    theta_s_poly_deg: tuple[float, float, float] = (-8.1074, -26.8141, 63.4725)
    theta_e_poly_deg: tuple[float, float, float] = (-13.9185, -49.2775, 123.7624)
    gamma_min_deg: float = 14.334862
    gamma_max_deg: float = 62.902456
    # Verified Fig. 3A chord parameters [m] (root, wrist, tip) in unfolded (u=0) vs folded (u=1)
    c_root_unfolded_m: float = 0.256
    c_root_folded_m: float = 0.232
    c_wrist_unfolded_m: float = 0.238
    c_wrist_folded_m: float = 0.186
    c_tip_m: float = 0.072

    @staticmethod
    def eval_fourier_with_derivs(
        coeffs_deg: tuple[float, ...],
        phase_rad: float,
        omega_rad_s: float,
    ) -> tuple[float, float, float]:
        """Return (val_rad, dval_dt_rad_s, d2val_dt2_rad_s2) for a periodic 6-harmonic series."""

        val_deg = coeffs_deg[0]
        d1_deg = 0.0
        d2_deg = 0.0
        n_harm = (len(coeffs_deg) - 1) // 2
        for n in range(1, n_harm + 1):
            cn = coeffs_deg[2 * n - 1]
            sn = coeffs_deg[2 * n]
            ang = n * phase_rad
            ca = math.cos(ang)
            sa = math.sin(ang)
            nw = n * omega_rad_s
            val_deg += cn * ca + sn * sa
            d1_deg += nw * (-cn * sa + sn * ca)
            d2_deg += -(nw * nw) * (cn * ca + sn * sa)
        d2r = math.pi / 180.0
        return val_deg * d2r, d1_deg * d2r, d2_deg * d2r


ROBOFALCON2_EQS1 = RoboFalcon2EqS1KinematicsData(
    flap_Theta_deg=(-4.317907, 4.492213, -43.816834, -0.158263, -0.471915, -0.197414, -1.964809, -0.018702, -0.178617, -0.01821, -0.130816, -0.018292, -0.105061),
    sweep_Psi_deg=(9.350654, 0.758204, 9.821142, -4.454858, 0.659147, -0.245591, -0.446761, -0.466905, 0.25655, -0.157333, -0.051624, -0.013379, 0.021827),
    humerus_alpha_deg=(3.989123, -9.737767, -0.273556, 3.415078, -0.34636, 0.023847, -0.21358, -0.595905, 0.057657, 0.106071, -0.067259, -0.00127, -0.004318),
    radius_beta_deg=(18.429571, -12.76622, -0.037036, 5.760376, -0.112034, -0.789459, -0.082216, -0.685197, 0.168752, 0.289588, -0.080918, 0.002185, -0.001279),
    manus_gamma_deg=(33.582238, -24.782428, -0.751577, 6.888075, -1.139582, 1.027059, -0.66628, -1.417407, 0.138458, -0.052413, -0.092335, -0.008971, -0.010295),
)


__all__ = [
    "AIRFOIL_CAMBER_PROFILES",
    "AirfoilCamberProfile",
    "DIGITIZED_PLANFORMS",
    "DigitizedPlanformSpec",
    "ROBOFALCON2_EQS1",
    "RoboFalcon2EqS1KinematicsData",
]
