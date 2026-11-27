"""Propulsion, first-order actuator dynamics, and cruise-trim analysis."""

from .actuator import (
    ActuatorPowerBreakdown,
    FirstOrderMotorSpec,
    evaluate_flapping_motor_cycle,
    evaluate_propeller_motor_cruise,
)
from .cruise_trim import (
    CruiseTrimResult,
    MorphingFlappingWingKinematics,
    OperatingPointResult,
    compute_neutral_point_sweep,
    evaluate_fixed_wing_operating_point,
    evaluate_flapping_operating_point,
    solve_fixed_wing_cruise_trim,
    solve_flapping_cruise_trim,
)

__all__ = [
    "ActuatorPowerBreakdown",
    "CruiseTrimResult",
    "FirstOrderMotorSpec",
    "MorphingFlappingWingKinematics",
    "OperatingPointResult",
    "compute_neutral_point_sweep",
    "evaluate_fixed_wing_operating_point",
    "evaluate_flapping_motor_cycle",
    "evaluate_flapping_operating_point",
    "evaluate_propeller_motor_cruise",
    "solve_fixed_wing_cruise_trim",
    "solve_flapping_cruise_trim",
]
