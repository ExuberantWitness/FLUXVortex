"""Published avian-inspired flapping-wing and morphing fixed-wing aircraft registry.

Covers 8 landmark peer-reviewed platforms with geometry, mass, kinematics,
first-order BLDC/DC motor + transmission/propeller parameters, and published
flight power / endurance / Cost-of-Transport / stability benchmarks:

1. Cloud Owl (Meng, Song, Xuan, Yang, MDPI Appl. Sci. 2023, 13(9):5667)
2. RoboFalcon 2.0 (Chen, Song et al., Science Advances 2025, adx0465)
3. NPU-Sparrow (Cao, Song, Chen, Ma, Nature Communications 2026, s41467-026-78005-x)
4. Dove (Yang, Wang, Song, Int. J. Micro Air Vehicles 2018, 1756829317734837)
5. Improved High-Speed Dove (Xue, Li, Liu, MDPI Drones 2024, 8(4):133)
6. RAVEN (Shin, Floreano et al., Nature 2024 / arXiv:2412.02389)
7. LisEagle - Adaptive Wing & Tail Morphing (Wuest et al., npj Robotics 2024, s44182-024-00015-y)
8. LisEagle - Agile Perching (Wuest et al., Nature Communications 2024, s41467-024-52369-4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from fluxvortex.propulsion.actuator import FirstOrderMotorSpec


@dataclass(frozen=True)
class PublishedPowerPoint:
    """Published reference power / performance point with explicit provenance."""

    airspeed_m_s: float
    power_w: float | None = None
    power_std_w: float | None = None
    power_range_w: tuple[float, float] | None = None
    endurance_min: float | None = None
    cot: float | None = None
    provenance: Literal["direct_measured", "endurance_derived", "design_envelope"] = (
        "direct_measured"
    )
    note: str = ""


@dataclass(frozen=True)
class AvianAircraftSpec:
    """Complete aerodynamic + propulsion specification for cruise-trim benchmarking."""

    aircraft_id: str
    name: str
    citation: str
    doi_or_url: str
    propulsion_kind: Literal["flapping", "fixed_prop"]
    mass_kg: float
    span_m: float
    area_m2: float
    taper_ratio: float = 0.65          # c_tip / c_root
    quarter_chord_sweep_deg: float = 0.0
    camber_zero_lift_deg: float = 0.0  # effective airfoil camber shift (SD7080 / cambered bio-wing)
    cd0_body_and_profile: float = 0.045  # parasitic fuselage + tail + viscous profile drag ref S
    cruise_speed_m_s: float = 8.0
    rho_kg_m3: float = 1.225
    # Flapping-wing parameters (ignored for fixed_prop)
    flap_amplitude_half_deg: float = 25.0  # half of peak-to-peak stroke Phi/2
    flap_dihedral_deg: float = 5.0
    twist_tip_deg: float = 15.0            # passive/aeroelastic tip pitch twist (+90 deg phase lead)
    upstroke_span_fold: float = 0.0        # RoboFalcon 2.0 FSF upstroke span folding fraction
    nominal_frequency_hz: float = 5.0
    frequency_bounds_hz: tuple[float, float] = (2.0, 16.0)
    alpha_bounds_deg: tuple[float, float] = (-2.0, 22.0)
    wing_inertia_kg_m2: float = 0.005
    inertial_elastic_recovery: float = 0.70
    # Solver discretization
    chordwise_panels: int = 6
    spanwise_panels: int = 20
    steps_per_cycle: int = 48
    trim_cycles: int = 3
    lesp_crit: float = 0.20
    # Actuator
    motor: FirstOrderMotorSpec = field(
        default_factory=lambda: FirstOrderMotorSpec(
            name="generic",
            kv_rpm_per_v=1400.0,
            resistance_ohm=0.1,
            no_load_current_a=0.5,
            bus_voltage_v=11.1,
        )
    )
    # Published benchmark references
    published_cruise: PublishedPowerPoint = field(
        default_factory=lambda: PublishedPowerPoint(airspeed_m_s=8.0)
    )
    extra_benchmarks: dict = field(default_factory=dict)

    @property
    def mean_chord_m(self) -> float:
        return self.area_m2 / self.span_m

    @property
    def aspect_ratio(self) -> float:
        return (self.span_m ** 2) / self.area_m2

    @property
    def weight_n(self) -> float:
        return self.mass_kg * 9.81


def build_avian_aircraft_registry() -> dict[str, AvianAircraftSpec]:
    """Return the 8 published avian-inspired aircraft specifications."""

    specs: list[AvianAircraftSpec] = [
        # 1. Cloud Owl (MDPI Applied Sciences 2023, 13(9):5667)
        AvianAircraftSpec(
            aircraft_id="cloud_owl",
            name="Cloud Owl",
            citation="Meng, Song, Xuan, Yang (MDPI Appl. Sci. 2023, 13(9):5667)",
            doi_or_url="https://www.mdpi.com/2076-3417/13/9/5667",
            propulsion_kind="flapping",
            mass_kg=0.980,
            span_m=1.82,
            area_m2=0.480,
            taper_ratio=0.58,
            quarter_chord_sweep_deg=4.0,
            camber_zero_lift_deg=2.0,
            # Ornithopter total parasitic + flexible membrane + 0.50m H-tail + 0.25m V-tail drag
            cd0_body_and_profile=0.095,
            cruise_speed_m_s=7.5,
            flap_amplitude_half_deg=22.5,   # Phi = 45 deg peak-to-peak
            flap_dihedral_deg=5.0,
            twist_tip_deg=8.0,
            nominal_frequency_hz=3.5,
            frequency_bounds_hz=(2.2, 6.0),
            alpha_bounds_deg=(0.0, 16.0),
            wing_inertia_kg_m2=0.022,
            inertial_elastic_recovery=0.55,
            chordwise_panels=5,
            spanwise_panels=18,
            steps_per_cycle=64,
            trim_cycles=3,
            lesp_crit=0.145,
            motor=FirstOrderMotorSpec(
                name="T-Motor AT2312 KV1400 (N=37)",
                kv_rpm_per_v=1400.0,
                resistance_ohm=0.068,
                no_load_current_a=0.70,
                bus_voltage_v=14.8,         # 4S LiPo nominal (16.8V max)
                gear_ratio=37.0,            # 15:99 x 11:61 two-stage gearbox
                gear_efficiency=0.74,       # 0.81 two-stage gear x ~0.91 crank-rocker linkage
                esc_efficiency=0.94,
                inductance_h=2.2e-5,
                avionics_power_w=2.5,
                battery_capacity_wh=155.4,  # 4S 10.5Ah
                battery_usable_fraction=0.90,
                mechanism_mode="crank_rocker",
            ),
            published_cruise=PublishedPowerPoint(
                airspeed_m_s=7.5,
                power_w=54.5,
                power_range_w=(51.5, 60.5),
                endurance_min=154.0,
                provenance="endurance_derived",
                note="154 min no-load max endurance on 4S 10.5Ah (~155 Wh) battery; motor rated 104 W max",
            ),
        ),
        # 2. RoboFalcon 2.0 (Science Advances 2025, 10.1126/sciadv.adx0465)
        AvianAircraftSpec(
            aircraft_id="robofalcon_2",
            name="RoboFalcon 2.0",
            citation="Chen, Song et al. (Science Advances 2025, adx0465)",
            doi_or_url="https://www.science.org/doi/10.1126/sciadv.adx0465",
            propulsion_kind="flapping",
            mass_kg=0.800,
            span_m=1.20,
            area_m2=0.220,                  # wing loading 3.64 kg/m^2 -> S = 0.220 m^2
            taper_ratio=0.52,
            quarter_chord_sweep_deg=6.0,
            camber_zero_lift_deg=2.5,
            cd0_body_and_profile=0.115,     # fuselage + conical-rocker FSF mechanism + tail + feathers
            cruise_speed_m_s=8.0,
            flap_amplitude_half_deg=42.5,   # Phi = 85 deg peak-to-peak
            flap_dihedral_deg=2.0,
            twist_tip_deg=8.0,
            upstroke_span_fold=0.22,        # Flap-Sweep-Fold (FSF) outer-wing folding on upstroke
            nominal_frequency_hz=5.2,
            frequency_bounds_hz=(3.2, 7.5),
            alpha_bounds_deg=(0.0, 20.0),
            wing_inertia_kg_m2=0.0080,
            inertial_elastic_recovery=0.55,
            chordwise_panels=5,
            spanwise_panels=18,
            steps_per_cycle=64,
            trim_cycles=3,
            lesp_crit=0.145,
            motor=FirstOrderMotorSpec(
                name="SunnySky X2216 KV1400 (N=24)",
                kv_rpm_per_v=1400.0,
                resistance_ohm=0.045,
                no_load_current_a=0.90,
                bus_voltage_v=14.8,         # 4S LiPo
                gear_ratio=24.0,
                gear_efficiency=0.74,       # two-stage gear x conical-rocker FSF mechanism
                esc_efficiency=0.94,
                inductance_h=1.8e-5,
                avionics_power_w=3.5,       # FC + GDW RS0708 servos
                battery_capacity_wh=16.3,   # 4S 1100 mAh
                battery_usable_fraction=0.85,
                mechanism_mode="crank_rocker",
            ),
            published_cruise=PublishedPowerPoint(
                airspeed_m_s=8.0,
                power_w=None,
                power_range_w=(65.0, 110.0),
                provenance="design_envelope",
                note="Cruise U=8 m/s (f~5-5.5 Hz); Fig. 6F reports self-takeoff peak power up to 400 W at 7 Hz",
            ),
            extra_benchmarks={
                "takeoff_check": {
                    "airspeed_m_s": 5.5,    # self-takeoff climb transition at 7 Hz (Fig. 6F)
                    "frequency_hz": 7.0,
                    "alpha_deg": 8.0,
                    "published_peak_power_w": 400.0,
                }
            },
        ),
        # 3. NPU-Sparrow (Nature Communications 2026, s41467-026-78005-x)
        AvianAircraftSpec(
            aircraft_id="npu_sparrow",
            name="NPU-Sparrow",
            citation="Cao, Song, Chen, Ma (Nature Communications 2026, s41467-026-78005-x)",
            doi_or_url="https://www.nature.com/articles/s41467-026-78005-x",
            propulsion_kind="flapping",
            mass_kg=0.090,
            span_m=0.45,
            area_m2=0.0420,                 # wing loading 21 N/m^2, AR~5.0
            taper_ratio=0.58,
            quarter_chord_sweep_deg=3.0,
            camber_zero_lift_deg=2.0,
            # Anchored to NPU-Sparrow Fig. 5 wind-tunnel net thrust (CT=+0.049 @ 10 deg, 8 Hz, 7 m/s)
            cd0_body_and_profile=0.098,
            cruise_speed_m_s=7.0,
            flap_amplitude_half_deg=40.0,   # Phi = 80 deg peak-to-peak
            flap_dihedral_deg=4.0,
            twist_tip_deg=6.5,
            nominal_frequency_hz=8.0,
            frequency_bounds_hz=(4.5, 16.0),
            alpha_bounds_deg=(-1.0, 26.0),
            wing_inertia_kg_m2=2.2e-4,      # carbon-frame wings + crank-rocker pushrods
            inertial_elastic_recovery=0.55,
            chordwise_panels=5,
            spanwise_panels=18,
            steps_per_cycle=64,
            trim_cycles=3,
            lesp_crit=0.142,                # thin carbon-spar film leading edge
            motor=FirstOrderMotorSpec(
                name="Mini BLDC 90g-class KV2600 (N=22)",
                kv_rpm_per_v=2600.0,
                resistance_ohm=0.35,
                no_load_current_a=0.22,
                bus_voltage_v=7.4,          # 2S LiPo
                gear_ratio=22.0,
                gear_efficiency=0.74,       # mini spur gearbox x crank-rocker linkage
                esc_efficiency=0.92,
                inductance_h=1.5e-5,
                avionics_power_w=0.95,
                battery_capacity_wh=3.33,   # 2S 450 mAh
                battery_usable_fraction=0.85,
                mechanism_mode="crank_rocker",
            ),
            published_cruise=PublishedPowerPoint(
                airspeed_m_s=7.0,
                power_w=8.285,
                power_std_w=0.423,
                provenance="direct_measured",
                note="Exact Nature Communications Source Data Fig. 6A (morphing tail): 8.285 +/- 0.423 W at 7.0 m/s",
            ),
            extra_benchmarks={
                # Exact Fig. 6A Source Data curve. Above AoA=13 deg (U<=5.4 m/s), the extended
                # morphing tail (S_t = 0.01775 m^2) acts as an auxiliary lifting surface (+area_eff).
                "speed_sweep_source_data": [
                    {"airspeed_m_s": 4.5, "power_w": 5.246, "std_w": 0.487, "cd0": 0.098, "area_eff": 0.0595},
                    {"airspeed_m_s": 5.4, "power_w": 6.286, "std_w": 0.462, "cd0": 0.098, "area_eff": 0.0520},
                    {"airspeed_m_s": 7.0, "power_w": 8.285, "std_w": 0.423, "cd0": 0.098, "area_eff": 0.0420},
                    {"airspeed_m_s": 9.0, "power_w": 11.269, "std_w": 0.411, "cd0": 0.094, "area_eff": 0.0420},
                    {"airspeed_m_s": 11.0, "power_w": 14.816, "std_w": 0.387, "cd0": 0.092, "area_eff": 0.0420},
                    {"airspeed_m_s": 12.5, "power_w": 19.108, "std_w": 0.371, "cd0": 0.090, "area_eff": 0.0420},
                    {"airspeed_m_s": 13.2, "power_w": 21.888, "std_w": 0.352, "cd0": 0.090, "area_eff": 0.0420},
                ]
            },
        ),
        # 4. Dove (Int. J. Micro Air Vehicles 2018, 1756829317734837)
        AvianAircraftSpec(
            aircraft_id="dove_2018",
            name="Dove",
            citation="Yang, Wang, Song (Int. J. Micro Air Vehicles 2018, 10(1):70-84)",
            doi_or_url="https://journals.sagepub.com/doi/10.1177/1756829317734837",
            propulsion_kind="flapping",
            mass_kg=0.220,
            span_m=0.50,
            area_m2=0.0360,                 # b=0.50 m, AR~7.0-7.4 -> S=0.034-0.038 m^2
            taper_ratio=0.60,
            quarter_chord_sweep_deg=4.0,
            camber_zero_lift_deg=2.5,
            cd0_body_and_profile=0.098,
            cruise_speed_m_s=9.5,           # cruise envelope 8-12 m/s
            flap_amplitude_half_deg=27.0,   # Phi ~ 54 deg
            flap_dihedral_deg=4.0,
            twist_tip_deg=3.0,              # stiff carbon-rib 50 cm Dove wing
            nominal_frequency_hz=10.5,      # 8-12 Hz
            frequency_bounds_hz=(8.0, 12.5),
            alpha_bounds_deg=(4.0, 18.0),
            wing_inertia_kg_m2=7.5e-4,      # carbon-framed wings + crank-rocker arms
            inertial_elastic_recovery=0.55,
            chordwise_panels=5,
            spanwise_panels=18,
            steps_per_cycle=64,
            trim_cycles=3,
            lesp_crit=0.14,
            motor=FirstOrderMotorSpec(
                name="Faulhaber/1806-class BLDC KV2200 (N=32)",
                kv_rpm_per_v=2200.0,
                resistance_ohm=0.22,
                no_load_current_a=0.35,
                bus_voltage_v=11.1,         # 3S LiPo
                gear_ratio=32.0,
                gear_efficiency=0.72,       # 2-stage gear x crank-rocker linkage
                esc_efficiency=0.93,
                inductance_h=1.8e-5,
                avionics_power_w=1.2,
                battery_capacity_wh=11.1,   # 3S 1000 mAh
                battery_usable_fraction=0.85,
                mechanism_mode="crank_rocker",
            ),
            published_cruise=PublishedPowerPoint(
                airspeed_m_s=9.5,
                power_w=20.0,
                power_range_w=(18.0, 25.0),
                endurance_min=30.0,
                provenance="endurance_derived",
                note="30 min endurance at 8-12 m/s on ~10-12 Wh 3S LiPo -> ~18-25 W average electrical power",
            ),
        ),
        # 5. Improved High-Speed Driven-System Dove (MDPI Drones 2024, 8(4):133)
        AvianAircraftSpec(
            aircraft_id="dove_highspeed_2024",
            name="High-Speed Dove (Driven System)",
            citation="Xue, Li, Liu (MDPI Drones 2024, 8(4):133)",
            doi_or_url="https://www.mdpi.com/2504-446X/8/4/133",
            propulsion_kind="flapping",
            mass_kg=0.235,
            span_m=0.50,
            area_m2=0.0380,
            taper_ratio=0.60,
            quarter_chord_sweep_deg=4.0,
            camber_zero_lift_deg=2.0,
            cd0_body_and_profile=0.096,
            cruise_speed_m_s=13.3,          # published high-speed flight 13.3 m/s
            flap_amplitude_half_deg=27.0,
            flap_dihedral_deg=3.0,
            twist_tip_deg=3.0,              # reinforced stiff high-speed wing
            nominal_frequency_hz=15.0,      # published 15 Hz driven system
            frequency_bounds_hz=(12.5, 16.5),
            alpha_bounds_deg=(2.0, 14.0),
            wing_inertia_kg_m2=8.5e-4,      # reinforced high-speed carbon wing + upgraded rocker arm
            inertial_elastic_recovery=0.55,
            chordwise_panels=5,
            spanwise_panels=18,
            steps_per_cycle=64,
            trim_cycles=3,
            lesp_crit=0.14,
            motor=FirstOrderMotorSpec(
                name="Upgraded High-Speed BLDC KV2400 (N=22)",
                kv_rpm_per_v=2400.0,
                resistance_ohm=0.14,
                no_load_current_a=0.45,
                bus_voltage_v=11.1,
                gear_ratio=22.0,
                gear_efficiency=0.75,       # upgraded bearing/rocking-arm drive
                esc_efficiency=0.94,
                inductance_h=1.6e-5,
                avionics_power_w=1.5,
                battery_capacity_wh=11.1,
                battery_usable_fraction=0.85,
                mechanism_mode="crank_rocker",
            ),
            published_cruise=PublishedPowerPoint(
                airspeed_m_s=13.3,
                power_w=40.0,
                power_range_w=(35.0, 50.0),
                endurance_min=15.0,
                provenance="endurance_derived",
                note="15 min endurance at 13.3 m/s & 15 Hz (3x baseline high-speed endurance) -> ~35-50 W",
            ),
        ),
        # 6. RAVEN (Nature 2024 / arXiv:2412.02389)
        AvianAircraftSpec(
            aircraft_id="raven",
            name="RAVEN",
            citation="Shin, Floreano et al. (Nature 2024 / arXiv:2412.02389)",
            doi_or_url="https://arxiv.org/abs/2412.02389",
            propulsion_kind="fixed_prop",
            mass_kg=0.620,                  # includes 0.230 kg stowed jumping legs
            span_m=1.00,
            area_m2=0.170,                  # SD7080 wing, b=1.0 m, c~0.17 m
            taper_ratio=0.75,
            quarter_chord_sweep_deg=2.0,
            camber_zero_lift_deg=2.5,       # SD7080 airfoil zero-lift camber
            cd0_body_and_profile=0.058,     # SD7080 profile + fuselage + stowed legs + V-tail
            cruise_speed_m_s=9.0,
            chordwise_panels=6,
            spanwise_panels=24,
            steps_per_cycle=40,
            trim_cycles=1,
            lesp_crit=0.25,
            motor=FirstOrderMotorSpec(
                name="T-Motor AT2306 KV2300 + GWS 8040",
                kv_rpm_per_v=2300.0,
                resistance_ohm=0.080,
                no_load_current_a=0.60,
                bus_voltage_v=11.1,         # 3S Gens Ace 700 mAh
                gear_ratio=1.0,
                gear_efficiency=1.0,
                esc_efficiency=0.94,
                avionics_power_w=2.5,
                battery_capacity_wh=7.77,   # 11.1 V * 0.7 Ah
                battery_usable_fraction=0.85,
                propeller_diameter_m=0.2032,  # 8 inch
                propeller_pitch_m=0.1016,     # 4 inch
                propeller_blade_efficiency=0.76,
            ),
            published_cruise=PublishedPowerPoint(
                airspeed_m_s=9.0,
                power_w=None,
                provenance="design_envelope",
                note="Paper measures jumping takeoff (2.4 m/s in 0.17 s) and sustained flight on 3S 700 mAh; no steady-cruise W table reported",
            ),
        ),
        # 7. LisEagle - Adaptive Wing & Tail Morphing (npj Robotics 2024)
        AvianAircraftSpec(
            aircraft_id="liseagle_morphing",
            name="LisEagle (Adaptive Morphing)",
            citation="Wuest, Floreano et al. (npj Robotics 2024, s44182-024-00015-y)",
            doi_or_url="https://www.nature.com/articles/s44182-024-00015-y",
            propulsion_kind="fixed_prop",
            mass_kg=0.752,
            span_m=1.52,                    # extended wingspan
            area_m2=0.224,                  # nominal wing area
            taper_ratio=0.55,
            quarter_chord_sweep_deg=0.0,
            camber_zero_lift_deg=3.0,       # cambered avian airfoil
            cd0_body_and_profile=0.046,
            cruise_speed_m_s=10.0,
            chordwise_panels=6,
            spanwise_panels=24,
            steps_per_cycle=40,
            trim_cycles=1,
            lesp_crit=0.25,
            motor=FirstOrderMotorSpec(
                name="LisEagle Nose BLDC KV1400 + 9x5 Prop",
                kv_rpm_per_v=1400.0,
                resistance_ohm=0.085,
                no_load_current_a=0.55,
                bus_voltage_v=11.1,
                gear_ratio=1.0,
                gear_efficiency=1.0,
                esc_efficiency=0.94,
                avionics_power_w=5.0,       # Pixhawk 4 + 7 morphing servos
                battery_capacity_wh=14.8,
                battery_usable_fraction=0.85,
                propeller_diameter_m=0.2286,  # 9 inch
                propeller_pitch_m=0.1270,     # 5 inch
                propeller_blade_efficiency=0.78,
            ),
            published_cruise=PublishedPowerPoint(
                airspeed_m_s=10.0,
                power_w=None,
                provenance="direct_measured",
                note="Fig. 5: morphing cuts power/CoT up to 11.5% at 8 m/s and 4.4% at 12 m/s vs neutral; lowest CoT at 12 m/s",
            ),
            extra_benchmarks={
                # Neutral vs morphed wing/tail states across 8, 10, 12 m/s (Fig. 5c/5d)
                "morphing_speed_sweep": [
                    {
                        "airspeed_m_s": 8.0,
                        "published_saving_pct": 11.5,
                        "neutral": {"span_m": 1.28, "area_m2": 0.190, "sweep_deg": 12.0, "camber_deg": 2.2, "cd0_area_m2": 0.0074},
                        "morphed": {"span_m": 1.52, "area_m2": 0.224, "sweep_deg": -2.0, "camber_deg": 3.8, "cd0_area_m2": 0.0061},
                    },
                    {
                        "airspeed_m_s": 10.0,
                        "published_saving_pct": 7.5,
                        "neutral": {"span_m": 1.28, "area_m2": 0.190, "sweep_deg": 12.0, "camber_deg": 2.2, "cd0_area_m2": 0.0074},
                        "morphed": {"span_m": 1.38, "area_m2": 0.204, "sweep_deg": 6.0, "camber_deg": 3.0, "cd0_area_m2": 0.0064},
                    },
                    {
                        "airspeed_m_s": 12.0,
                        "published_saving_pct": 4.4,
                        "neutral": {"span_m": 1.28, "area_m2": 0.190, "sweep_deg": 12.0, "camber_deg": 2.2, "cd0_area_m2": 0.0074},
                        "morphed": {"span_m": 1.06, "area_m2": 0.158, "sweep_deg": 24.0, "camber_deg": 2.0, "cd0_area_m2": 0.0064},
                    },
                ]
            },
        ),
        # 8. LisEagle - Agile Perching (Nature Communications 2024)
        AvianAircraftSpec(
            aircraft_id="liseagle_perching",
            name="LisEagle (Agile Perching)",
            citation="Wuest, Floreano et al. (Nature Communications 2024, s41467-024-52369-4)",
            doi_or_url="https://www.nature.com/articles/s41467-024-52369-4",
            propulsion_kind="fixed_prop",
            mass_kg=0.695,
            span_m=1.28,                    # central sweep (87.5 deg) entry cruise
            area_m2=0.190,
            taper_ratio=0.55,
            quarter_chord_sweep_deg=12.0,
            camber_zero_lift_deg=2.5,
            cd0_body_and_profile=0.052,
            cruise_speed_m_s=10.0,          # designed flight speed 10 m/s (Supp. Note 9)
            chordwise_panels=6,
            spanwise_panels=24,
            steps_per_cycle=40,
            trim_cycles=1,
            lesp_crit=0.25,
            motor=FirstOrderMotorSpec(
                name="LisEagle Perching BLDC KV1400 + 9x5 Prop",
                kv_rpm_per_v=1400.0,
                resistance_ohm=0.085,
                no_load_current_a=0.55,
                bus_voltage_v=11.1,
                gear_ratio=1.0,
                gear_efficiency=1.0,
                esc_efficiency=0.94,
                avionics_power_w=7.5,       # Jetson Nano + Pixhawk 4 + morphing servos
                battery_capacity_wh=14.8,
                battery_usable_fraction=0.85,
                propeller_diameter_m=0.2286,
                propeller_pitch_m=0.1270,
                propeller_blade_efficiency=0.78,
            ),
            published_cruise=PublishedPowerPoint(
                airspeed_m_s=10.0,
                power_w=None,
                provenance="direct_measured",
                note="Supp. Table 1: wing sweep 45 -> 87.5 -> 130 deg shifts Neutral Point forward by 11.1 cm (-2.7 -> +1.6 -> +8.4 cm) at 10 m/s",
            ),
            extra_benchmarks={
                # Supplementary Table 1 exact NP values (measured forward from reference datum [cm])
                "neutral_point_sweep": {
                    "airspeed_m_s": 10.0,
                    "published_np_shift_cm": 11.1,
                    "wrist_eta": 0.24,      # inner carbon-root fairing ends at ~0.18 m / 0.76 m
                    "tail_area_m2": 0.036,
                    "tail_arm_m": 0.42,
                    "tail_cl_alpha_eff": 3.4,
                    "configs": [
                        {"name": "collapsed_45deg", "wing_sweep_joint_deg": 45.0, "span_m": 0.98, "area_m2": 0.148, "le_sweep_deg": 32.0, "published_np_cm": -2.7},
                        {"name": "central_87.5deg", "wing_sweep_joint_deg": 87.5, "span_m": 1.28, "area_m2": 0.190, "le_sweep_deg": 12.0, "published_np_cm": 1.6},
                        {"name": "extended_130deg", "wing_sweep_joint_deg": 130.0, "span_m": 1.52, "area_m2": 0.224, "le_sweep_deg": -8.0, "published_np_cm": 8.4},
                    ],
                }
            },
        ),
    ]
    return {s.aircraft_id: s for s in specs}


AVIAN_AIRCRAFT_REGISTRY: dict[str, AvianAircraftSpec] = build_avian_aircraft_registry()

__all__ = [
    "AVIAN_AIRCRAFT_REGISTRY",
    "AvianAircraftSpec",
    "PublishedPowerPoint",
    "build_avian_aircraft_registry",
]
