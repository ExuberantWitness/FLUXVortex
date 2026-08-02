"""Production claim components.

The numerical solver currently publishes its already-computed, per-node force
channels into ``solver_channels``.  These components make that accounting
explicit and enforce one provider/one ledger entry while the low-level UVLM
time integrator remains unchanged.
"""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from .core import (
    ClaimComponent,
    ClaimRuntimeError,
    ForceContribution,
    NodeOutput,
    StepContext,
)


class _ChannelComponent(ClaimComponent):
    channel_names: tuple[str, ...] = ()

    def step(self, context: StepContext) -> NodeOutput:
        channels: Mapping[str, Any] = context.values["solver_channels"]
        forces = []
        for name in self.channel_names:
            value = channels.get(name)
            if value is None:
                continue
            forces.append(
                ForceContribution(
                    node_id=self.node_id,
                    channel=name,
                    body_force=np.asarray(value, dtype=float),
                    role=self.role,
                )
            )
        return NodeOutput(values={name: channels.get(name) for name in self.provides}, forces=forces)


class UVLMComponent(_ChannelComponent):
    channel_names = ("uvlm", "vortex_impulse", "leading_edge_suction", "uvlm_remainder")


class KirchhoffLBComponent(_ChannelComponent):
    channel_names = ("separation", "profile_drag", "sep_drag_qs")


class DSVortexComponent(_ChannelComponent):
    channel_names = ("ds_vortex", "vortex_normal")


class CTConsistencyComponent(_ChannelComponent):
    channel_names = ("ct_consistency",)


class TwistResponseObserver(_ChannelComponent):
    channel_names = ()


class RigDragComponent(_ChannelComponent):
    channel_names = ("rig_drag",)


class CycleReductionComponent(_ChannelComponent):
    """Graph-level numerical correction from arithmetic to reported cycle mean."""

    channel_names = ("numerical_cycle_reduction",)

    def step(self, context: StepContext) -> NodeOutput:
        channels: Mapping[str, Any] = context.values["solver_channels"]
        if "numerical_cycle_reduction" not in channels:
            raise ClaimRuntimeError(
                f"{self.node_id}: missing canonical numerical_cycle_reduction"
            )
        if self.provides != {"r0_cycle_reduction"}:
            raise ClaimRuntimeError(
                f"{self.node_id}: must provide only r0_cycle_reduction"
            )
        value = np.asarray(
            channels["numerical_cycle_reduction"], dtype=float
        )
        contribution = ForceContribution(
            node_id=self.node_id,
            channel="numerical_cycle_reduction",
            body_force=value,
            role=self.role,
        )
        return NodeOutput(
            values={"r0_cycle_reduction": value},
            forces=(contribution,),
        )


class LegacyClosureComponent(_ChannelComponent):
    channel_names = ("legacy_closure", "viscous", "profile_drag", "legacy_dynamic_stall")


class DistributedDoubletShadowComponent(ClaimComponent):
    """N3.1j4 spatial-state guard with no pressure or force."""

    def step(self, context: StepContext) -> NodeOutput:
        from .distributed_doublet import (
            QuadraticDoubletAssembly,
            QuadraticDoubletSurface,
        )

        inputs = context.values["spatial_shadow_inputs"]
        if not isinstance(inputs, Mapping):
            raise ClaimRuntimeError(
                f"{self.node_id}: spatial_shadow_inputs must be a mapping"
            )
        surface = inputs.get("surface")
        if not isinstance(surface, QuadraticDoubletSurface):
            raise ClaimRuntimeError(
                f"{self.node_id}: input 'surface' must be QuadraticDoubletSurface"
            )
        continuity = surface.continuity_report(
            tolerance=float(inputs.get("continuity_tolerance", 1.0e-12))
        )
        assembly = inputs.get("assembly")
        if assembly is None:
            topology_report = surface.boundary_report(
                tolerance=float(
                    inputs.get("boundary_strength_tolerance", 1.0e-12)
                )
            )
            topology_passed = topology_report.compatible
        else:
            if not isinstance(assembly, QuadraticDoubletAssembly):
                raise ClaimRuntimeError(
                    f"{self.node_id}: input 'assembly' has wrong type"
                )
            if not any(patch.surface is surface for patch in assembly.patches):
                raise ClaimRuntimeError(
                    f"{self.node_id}: surface is absent from input assembly"
                )
            topology_report = assembly.topology_report(
                strength_tolerance=float(
                    inputs.get("boundary_strength_tolerance", 1.0e-12)
                ),
                geometry_tolerance=float(
                    inputs.get("interface_geometry_tolerance", 1.0e-12)
                ),
            )
            topology_passed = topology_report.compatible

        sheet_orders = tuple(
            inputs.get(
                "sheet_average_orders",
                (48, 80, 128, 192, 256),
            )
        )
        sheet_absolute_tolerance = float(
            inputs.get("sheet_average_absolute_tolerance", 2.0e-10)
        )
        sheet_relative_tolerance = float(
            inputs.get("sheet_average_relative_tolerance", 2.0e-9)
        )
        if assembly is None:
            (
                sheet_points,
                sheet_face_owner,
                sheet_barycentric,
            ) = surface.interior_collocation_points()
            sheet_patch_owner = None
            sheet_velocity, sheet_report = (
                surface.induced_velocity_sheet_average_converged(
                    sheet_face_owner,
                    sheet_barycentric,
                    orders=sheet_orders,
                    absolute_tolerance=sheet_absolute_tolerance,
                    relative_tolerance=sheet_relative_tolerance,
                )
            )
        else:
            (
                sheet_points,
                sheet_patch_owner,
                sheet_face_owner,
                sheet_barycentric,
            ) = assembly.interior_collocation_points()
            sheet_velocity, sheet_report = (
                assembly.induced_velocity_sheet_average_converged(
                    sheet_patch_owner,
                    sheet_face_owner,
                    sheet_barycentric,
                    orders=sheet_orders,
                    absolute_tolerance=sheet_absolute_tolerance,
                    relative_tolerance=sheet_relative_tolerance,
                )
            )
        sheet_passed = sheet_report.converged
        from .sheet_velocity_projection import (
            project_assembly_vertex_star_normal_geometry_velocity,
            project_vertex_star_normal_geometry_velocity,
        )

        if assembly is None:
            geometry_velocity = project_vertex_star_normal_geometry_velocity(
                surface,
                sheet_velocity,
                relative_rank_tolerance=inputs.get(
                    "geometry_velocity_relative_rank_tolerance"
                ),
            )
            geometry_velocity_report = geometry_velocity.report
            projection_passed = geometry_velocity.report.full_rank
        else:
            # Weld only explicitly declared interfaces.  This is the
            # assembly counterpart of the vertex-star normal limit; it does
            # not revive the falsified full-vector P1 projection.
            geometry_velocity = (
                project_assembly_vertex_star_normal_geometry_velocity(
                    assembly,
                    sheet_velocity,
                    geometry_tolerance=float(
                        inputs.get(
                            "interface_geometry_tolerance",
                            1.0e-12,
                        )
                    ),
                    relative_rank_tolerance=inputs.get(
                        "geometry_velocity_relative_rank_tolerance"
                    ),
                )
            )
            geometry_velocity_report = geometry_velocity.report
            projection_passed = geometry_velocity.report.full_rank

        previous = inputs.get("previous_surface")
        if previous is None:
            kelvin_passed = True
            kelvin_report = None
        else:
            if not isinstance(previous, QuadraticDoubletSurface):
                raise ClaimRuntimeError(
                    f"{self.node_id}: previous_surface has wrong type"
                )
            kelvin_report = previous.kelvin_report(
                surface,
                tolerance=float(inputs.get("kelvin_tolerance", 1.0e-12)),
            )
            kelvin_passed = kelvin_report.passed

        field_points = inputs.get("field_points")
        field_velocity = None
        field_report = None
        if field_points is not None:
            field_velocity, field_report = surface.induced_velocity_converged(
                field_points,
                orders=tuple(inputs.get("quadrature_orders", (8, 12, 16, 24))),
                absolute_tolerance=float(
                    inputs.get("field_absolute_tolerance", 1.0e-10)
                ),
                relative_tolerance=float(
                    inputs.get("field_relative_tolerance", 1.0e-8)
                ),
            )
            field_passed = field_report.converged
        else:
            field_passed = False

        # This explicit bit can only be supplied by a canonical field runner.
        # It is deliberately independent of any force target.
        canonical_passed = bool(inputs.get("canonical_field_gate_passed", False))
        eligible = (
            continuity.compatible
            and topology_passed
            and sheet_passed
            and projection_passed
            and kelvin_passed
            and field_passed
            and canonical_passed
        )
        state = {
            "surface": surface,
            "field_velocity": field_velocity,
            "sheet_average_points": sheet_points,
            "sheet_average_patch_owner": sheet_patch_owner,
            "sheet_average_face_owner": sheet_face_owner,
            "sheet_average_barycentric": sheet_barycentric,
            "sheet_average_velocity": sheet_velocity,
            "geometry_velocity": geometry_velocity,
            "eligible_for_pressure": eligible,
            "guards": {
                "trace_continuity": continuity,
                "global_boundary_topology": topology_report,
                "on_sheet_field": sheet_report,
                "geometry_velocity_projection": geometry_velocity_report,
                "material_kelvin": kelvin_report,
                "off_sheet_field": field_report,
                "canonical_field_gate_passed": canonical_passed,
            },
        }
        return NodeOutput(
            values={"spatial_vortex_state": state},
            diagnostics=state["guards"],
        )


class UnifiedPressureShadowComponent(ClaimComponent):
    """N3.1j3 pressure ledger, hard-gated by the spatial field."""

    def step(self, context: StepContext) -> NodeOutput:
        from .unified_panel_pressure import unified_panel_pressure

        spatial = context.values["spatial_vortex_state"]
        if not spatial["eligible_for_pressure"]:
            state = {
                "status": "blocked",
                "reason": "spatial field gate has not passed",
                "pressure": None,
            }
            return NodeOutput(
                values={"panel_pressure_state": state},
                diagnostics={"promotion_blocked": True},
            )
        inputs = context.values["spatial_shadow_inputs"]
        pressure = inputs.get("pressure")
        if not isinstance(pressure, Mapping):
            raise ClaimRuntimeError(
                f"{self.node_id}: eligible spatial state requires pressure inputs"
            )
        result = unified_panel_pressure(
            density=float(pressure["density"]),
            local_velocity=pressure["local_velocity"],
            surface_gradient=pressure["surface_gradient"],
            potential_rate_channels=pressure["potential_rate_channels"],
            area=pressure["area"],
            normal=pressure["normal"],
        )
        state = {
            "status": "shadow",
            "reason": "",
            "pressure": result,
        }
        return NodeOutput(
            values={"panel_pressure_state": state},
            diagnostics={"ledger": result.ledger_report()},
        )


class ConservativeTransferShadowComponent(ClaimComponent):
    """N3.1i2a work-conjugate transfer, never a force provider."""

    def step(self, context: StepContext) -> NodeOutput:
        from .work_conjugate_transfer import (
            rigid_resultant_report,
            transfer_generalized,
        )

        pressure_state = context.values["panel_pressure_state"]
        if pressure_state["status"] == "blocked":
            state = {
                "status": "blocked",
                "reason": pressure_state["reason"],
                "generalized_load": None,
            }
            return NodeOutput(
                values={"structural_load_state": state},
                diagnostics={"promotion_blocked": True},
            )
        inputs = context.values["spatial_shadow_inputs"]
        structure = inputs.get("structure")
        if not isinstance(structure, Mapping):
            raise ClaimRuntimeError(
                f"{self.node_id}: pressure state requires structure inputs"
            )
        pressure = pressure_state["pressure"]
        load = transfer_generalized(
            pressure.total_force,
            structure["kinematic_jacobian"],
        )
        work_report = load.virtual_work_report(
            structure["virtual_work_probes"],
            absolute_tolerance=float(
                structure.get("work_absolute_tolerance", 1.0e-12)
            ),
            relative_tolerance=float(
                structure.get("work_relative_tolerance", 1.0e-12)
            ),
        )
        resultant_report = rigid_resultant_report(
            structure["point_position"],
            pressure.total_force,
            load.values,
            origin=structure.get("origin"),
            translation_columns=tuple(
                structure.get("translation_columns", (0, 1, 2))
            ),
            rotation_columns=tuple(
                structure.get("rotation_columns", (3, 4, 5))
            ),
            tolerance=float(
                structure.get("resultant_tolerance", 1.0e-12)
            ),
        )
        eligible = work_report.passed and resultant_report.passed
        state = {
            "status": "shadow" if eligible else "failed",
            "reason": "" if eligible else "transfer guard failed",
            "generalized_load": load,
            "guards": {
                "virtual_work": work_report,
                "rigid_resultant": resultant_report,
            },
        }
        return NodeOutput(
            values={"structural_load_state": state},
            diagnostics=state["guards"],
        )
