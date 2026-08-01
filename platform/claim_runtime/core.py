"""Typed executable DAG used by the FLUXV claim-chain model.

The runtime deliberately separates scientific dependencies (``depends_on``)
from data-flow dependencies (``requires``/``provides``).  Components never
mutate another component's state: all exchange happens through StepContext and
all forces are booked through ForceLedger.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import yaml


class ClaimRuntimeError(RuntimeError):
    """A graph/schema/accounting invariant was violated."""


@dataclass(frozen=True)
class RunConfig:
    closure: str
    values: Mapping[str, Any]
    explicit: frozenset[str] = frozenset()
    sources: Mapping[str, str] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


@dataclass
class ForceContribution:
    node_id: str
    channel: str
    body_force: np.ndarray
    role: str = "physics"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        force = np.asarray(self.body_force, dtype=float)
        if force.shape not in ((3,),) and not (force.ndim == 2 and force.shape[1] == 3):
            raise ClaimRuntimeError(
                f"{self.node_id}:{self.channel}: body_force must have shape (3,) or (n,3), got {force.shape}"
            )
        if not np.all(np.isfinite(force)):
            raise ClaimRuntimeError(f"{self.node_id}:{self.channel}: non-finite force")
        self.body_force = force

    @property
    def total(self) -> np.ndarray:
        return self.body_force if self.body_force.ndim == 1 else self.body_force.sum(axis=0)


class ForceLedger:
    def __init__(self):
        self._items: list[ForceContribution] = []
        self._keys: set[tuple[str, str]] = set()

    def add(self, contribution: ForceContribution) -> None:
        key = (contribution.node_id, contribution.channel)
        if key in self._keys:
            raise ClaimRuntimeError(f"duplicate force contribution {key}")
        self._keys.add(key)
        self._items.append(contribution)

    @property
    def items(self) -> tuple[ForceContribution, ...]:
        return tuple(self._items)

    def total_body(self, roles: Optional[set[str]] = None) -> np.ndarray:
        selected = self._items if roles is None else [x for x in self._items if x.role in roles]
        return sum((x.total for x in selected), np.zeros(3, dtype=float))

    def by_node(self) -> Dict[str, list[dict[str, Any]]]:
        out: Dict[str, list[dict[str, Any]]] = {}
        for item in self._items:
            out.setdefault(item.node_id, []).append(
                {
                    "channel": item.channel,
                    "role": item.role,
                    "body_force": item.total.tolist(),
                    "metadata": dict(item.metadata),
                }
            )
        return out


@dataclass
class StepContext:
    step: int = -1
    time: float = 0.0
    values: MutableMapping[str, Any] = field(default_factory=dict)
    ledger: ForceLedger = field(default_factory=ForceLedger)

    def require(self, names: Iterable[str], node_id: str) -> None:
        missing = [name for name in names if name not in self.values]
        if missing:
            raise ClaimRuntimeError(f"{node_id}: missing runtime inputs {missing}")

    def publish(self, node_id: str, allowed: set[str], values: Mapping[str, Any]) -> None:
        illegal = sorted(set(values) - allowed)
        if illegal:
            raise ClaimRuntimeError(f"{node_id}: attempted to publish undeclared outputs {illegal}")
        for key, value in values.items():
            if key in self.values:
                raise ClaimRuntimeError(f"{node_id}: runtime output {key!r} already has a provider")
            self.values[key] = value


@dataclass
class NodeOutput:
    values: Mapping[str, Any] = field(default_factory=dict)
    forces: Sequence[ForceContribution] = field(default_factory=tuple)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class GraphRunManifest:
    closure: str
    topology: list[str]
    nodes: list[dict[str, Any]]
    parameter_sources: Dict[str, str]
    internal_stages: list[dict[str, Any]] = field(default_factory=list)
    guards: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Preserve the frozen production-manifest shape for closures that do
        # not declare internal stages.  Candidate closures still expose their
        # complete internal implementation identity.
        if not payload["internal_stages"]:
            payload.pop("internal_stages")
        return payload


class ClaimComponent:
    """Base interface for executable claim nodes."""

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = dict(spec)
        self.node_id = str(spec["id"])
        self.requires = tuple(spec.get("requires", ()))
        self.provides = set(spec.get("provides", ()))
        self.role = str(spec.get("runtime_role", "physics"))

    def initialize(self, config: RunConfig, context: StepContext) -> None:
        self.config = config

    def step(self, context: StepContext) -> NodeOutput:
        return NodeOutput()

    def finalize(self, context: StepContext) -> Mapping[str, Any]:
        return {}


@dataclass
class _RuntimeNode:
    spec: dict[str, Any]
    component: ClaimComponent
    source_hash: str


@dataclass(frozen=True)
class _InternalStageBinding:
    """A traced implementation owned and executed inside a runtime node.

    Internal stages are deliberately not ``ClaimComponent`` instances: they
    do not enter the graph topology, publish StepContext values, or book force
    directly.  The owning component remains the sole runtime provider while
    this binding makes the nested claim's implementation identity auditable.
    """

    spec: dict[str, Any]
    owner_node_id: str
    implementation: type
    source_hash: str


class ClaimGraph:
    """Validated executable claim DAG."""

    SCHEMA_KEYS = {
        "id", "title", "claim", "state", "freeze", "evidence", "refs", "memory",
        "guard", "falsified_when", "depends_on", "provides_to", "children",
        "version", "changed", "notes", "implementation", "implementation_version",
        "implementation_hash", "requires", "provides", "enabled_in", "config",
        "exclusive_with", "runtime_role", "runtime_binding", "runtime_owner",
    }
    EXTERNAL_INPUTS = {"solver_channels", "spatial_shadow_inputs"}

    def __init__(
        self,
        nodes: Sequence[_RuntimeNode],
        config: RunConfig,
        internal_stages: Sequence[_InternalStageBinding] = (),
    ):
        self.config = config
        self.nodes = {node.spec["id"]: node for node in nodes}
        self.internal_stages = {
            stage.spec["id"]: stage for stage in internal_stages
        }
        self._validate_internal_stages()
        self._validate_runtime_roles()
        self.topology = self._topological_order()
        self._validate_dataflow()
        self._validate_exclusivity()
        self.context = StepContext()
        for node_id in self.topology:
            self.nodes[node_id].component.initialize(config, self.context)

    def _validate_runtime_roles(self) -> None:
        allowed_nonclaim_roles = {"diagnostic", "necessary_physics", "compatibility"}
        for node_id, node in self.nodes.items():
            state = node.spec.get("state")
            role = node.spec.get("runtime_role", "physics")
            if state in {"falsified", "dead_end"} and role not in allowed_nonclaim_roles:
                raise ClaimRuntimeError(
                    f"{node_id}: {state} node cannot run as candidate physics (role={role})"
                )

    def _validate_internal_stages(self) -> None:
        executable_ids = set(self.nodes)
        allowed_nonclaim_roles = {
            "diagnostic",
            "necessary_physics",
            "compatibility",
        }
        for stage_id, stage in self.internal_stages.items():
            if stage_id in executable_ids:
                raise ClaimRuntimeError(
                    f"{stage_id}: internal stage duplicates an executable node id"
                )
            if stage.owner_node_id not in executable_ids:
                raise ClaimRuntimeError(
                    f"{stage_id}: internal stage owner "
                    f"{stage.owner_node_id!r} is not an enabled executable node"
                )
            state = stage.spec.get("state")
            role = stage.spec.get("runtime_role", "physics")
            if (
                state in {"falsified", "dead_end"}
                and role not in allowed_nonclaim_roles
            ):
                raise ClaimRuntimeError(
                    f"{stage_id}: {state} internal stage cannot bind as "
                    f"candidate physics (role={role})"
                )

    @classmethod
    def from_yaml(cls, nodes_dir: str | os.PathLike[str], config: RunConfig) -> "ClaimGraph":
        specs = []
        internal_stages: list[_InternalStageBinding] = []
        seen_ids: set[str] = set()

        def visit(spec: Mapping[str, Any], source: str) -> None:
            unknown = set(spec) - cls.SCHEMA_KEYS
            if unknown:
                raise ClaimRuntimeError(
                    f"{source}: unknown claim fields {sorted(unknown)}"
                )
            node_id = str(spec.get("id", ""))
            if not node_id:
                raise ClaimRuntimeError(f"{source}: claim node has no id")
            if node_id in seen_ids:
                raise ClaimRuntimeError(f"{source}: duplicate claim id {node_id}")
            seen_ids.add(node_id)

            runtime_binding = str(
                spec.get("runtime_binding", "component")
            )
            if runtime_binding not in {"component", "internal_stage"}:
                raise ClaimRuntimeError(
                    f"{node_id}: unknown runtime_binding {runtime_binding!r}"
                )
            if runtime_binding == "component" and spec.get("runtime_owner"):
                raise ClaimRuntimeError(
                    f"{node_id}: runtime_owner is valid only for an internal_stage"
                )

            if config.closure in spec.get("enabled_in", []):
                implementation = spec.get("implementation")
                if not implementation:
                    raise ClaimRuntimeError(
                        f"{node_id}: enabled node has no implementation"
                    )
                if runtime_binding == "internal_stage":
                    owner_node_id = str(spec.get("runtime_owner", ""))
                    if not owner_node_id:
                        raise ClaimRuntimeError(
                            f"{node_id}: internal_stage has no runtime_owner"
                        )
                    stage_cls = cls._resolve_internal_stage_implementation(
                        implementation
                    )
                    source_hash = cls._module_source_hash(stage_cls)
                    pinned = str(spec.get("implementation_hash", ""))
                    if not pinned:
                        raise ClaimRuntimeError(
                            f"{node_id}: internal_stage has no pinned "
                            "implementation_hash"
                        )
                    if pinned != source_hash:
                        raise ClaimRuntimeError(
                            f"{node_id}: internal-stage implementation drift: "
                            f"expected {pinned}, got {source_hash}"
                        )
                    internal_stages.append(
                        _InternalStageBinding(
                            spec=dict(spec),
                            owner_node_id=owner_node_id,
                            implementation=stage_cls,
                            source_hash=source_hash,
                        )
                    )
                else:
                    component_cls = cls._resolve_implementation(implementation)
                    source_hash = cls._source_hash(component_cls)
                    pinned = str(spec.get("implementation_hash", ""))
                    if spec.get("freeze") and pinned and pinned != source_hash:
                        raise ClaimRuntimeError(
                            f"{node_id}: frozen implementation drift: "
                            f"expected {pinned}, got {source_hash}"
                        )
                    specs.append(
                        _RuntimeNode(
                            dict(spec),
                            component_cls(spec),
                            source_hash,
                        )
                    )
            for child in spec.get("children", ()) or ():
                if not isinstance(child, Mapping):
                    raise ClaimRuntimeError(
                        f"{node_id}: child claim must be a mapping"
                    )
                visit(child, f"{source}/{child.get('id', '<missing>')}")

        for path in sorted(Path(nodes_dir).glob("*.y*ml")):
            with path.open() as stream:
                spec = yaml.safe_load(stream)
            if not isinstance(spec, Mapping):
                raise ClaimRuntimeError(
                    f"{path.name}: top-level claim must be a mapping"
                )
            visit(spec, path.name)
        return cls(specs, config, internal_stages)

    @staticmethod
    def _resolve_implementation(path: str):
        if ":" not in path:
            raise ClaimRuntimeError(f"invalid implementation path {path!r}")
        module_name, class_name = path.split(":", 1)
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as exc:
            raise ClaimRuntimeError(f"unknown implementation {path!r}") from exc
        if not inspect.isclass(cls) or not issubclass(cls, ClaimComponent):
            raise ClaimRuntimeError(f"{path!r} is not a ClaimComponent")
        return cls

    @staticmethod
    def _resolve_internal_stage_implementation(path: str):
        if ":" not in path:
            raise ClaimRuntimeError(f"invalid implementation path {path!r}")
        module_name, class_name = path.split(":", 1)
        try:
            implementation = getattr(
                importlib.import_module(module_name),
                class_name,
            )
        except (ImportError, AttributeError) as exc:
            raise ClaimRuntimeError(
                f"unknown internal-stage implementation {path!r}"
            ) from exc
        if not inspect.isclass(implementation):
            raise ClaimRuntimeError(
                f"{path!r} is not an internal-stage implementation class"
            )
        return implementation

    @staticmethod
    def _module_source_hash(implementation: type) -> str:
        """Hash the complete source module governed by an internal stage.

        Unlike a standalone ClaimComponent, an internal stage can call
        module-level pressure and state-transition helpers.  Hashing the
        complete module therefore captures the executable surface actually
        owned by the stage.
        """

        source_path = inspect.getsourcefile(implementation)
        if source_path is None:
            raise ClaimRuntimeError(
                f"{implementation!r}: internal-stage source is unavailable"
            )
        try:
            source = Path(source_path).read_bytes()
        except OSError as exc:
            raise ClaimRuntimeError(
                f"{implementation!r}: cannot read internal-stage source "
                f"{source_path!r}"
            ) from exc
        return "sha256:" + hashlib.sha256(source).hexdigest()

    @staticmethod
    def _source_hash(component_cls: type) -> str:
        governed = [
            cls for cls in component_cls.mro()
            if cls is not object and issubclass(cls, ClaimComponent)
        ]
        source = "\n".join(inspect.getsource(cls) for cls in governed).encode("utf-8")
        return "sha256:" + hashlib.sha256(source).hexdigest()

    def _topological_order(self) -> list[str]:
        dependencies = {
            node_id: {dep for dep in node.spec.get("depends_on", []) if dep in self.nodes}
            for node_id, node in self.nodes.items()
        }
        order: list[str] = []
        ready = sorted(node_id for node_id, deps in dependencies.items() if not deps)
        while ready:
            current = ready.pop(0)
            order.append(current)
            for node_id in sorted(dependencies):
                if current in dependencies[node_id]:
                    dependencies[node_id].remove(current)
                    if not dependencies[node_id] and node_id not in order and node_id not in ready:
                        ready.append(node_id)
                        ready.sort()
        if len(order) != len(self.nodes):
            cyclic = sorted(set(self.nodes) - set(order))
            raise ClaimRuntimeError(f"claim dependency cycle involving {cyclic}")
        return order

    def _validate_dataflow(self) -> None:
        providers: Dict[str, str] = {}
        for node_id in self.topology:
            for name in self.nodes[node_id].component.provides:
                if name in providers:
                    raise ClaimRuntimeError(
                        f"runtime value {name!r} has duplicate providers {providers[name]} and {node_id}"
                    )
                providers[name] = node_id
        for node_id in self.topology:
            missing = [
                x for x in self.nodes[node_id].component.requires
                if x not in providers and x not in self.EXTERNAL_INPUTS
            ]
            if missing:
                raise ClaimRuntimeError(f"{node_id}: no provider for runtime inputs {missing}")

    def _validate_exclusivity(self) -> None:
        enabled_specs = {
            node_id: node.spec for node_id, node in self.nodes.items()
        }
        enabled_specs.update({
            stage_id: stage.spec
            for stage_id, stage in self.internal_stages.items()
        })
        enabled = set(enabled_specs)
        for node_id, spec in enabled_specs.items():
            conflicts = enabled.intersection(spec.get("exclusive_with", []))
            if conflicts:
                raise ClaimRuntimeError(f"{node_id}: mutually exclusive nodes enabled: {sorted(conflicts)}")

    def step(self, step: int, time: float, inputs: Optional[Mapping[str, Any]] = None) -> StepContext:
        context = StepContext(step=step, time=time, values=dict(inputs or {}))
        for node_id in self.topology:
            component = self.nodes[node_id].component
            context.require(component.requires, node_id)
            output = component.step(context)
            context.publish(node_id, component.provides, output.values)
            for force in output.forces:
                if force.node_id != node_id:
                    raise ClaimRuntimeError(f"{node_id}: booked force under {force.node_id}")
                context.ledger.add(force)
        self.context = context
        return context

    def finalize(self) -> dict[str, Any]:
        return {
            node_id: dict(self.nodes[node_id].component.finalize(self.context))
            for node_id in self.topology
        }

    def manifest(self, guards: Optional[Mapping[str, Any]] = None) -> GraphRunManifest:
        nodes = []
        for node_id in self.topology:
            runtime = self.nodes[node_id]
            nodes.append(
                {
                    "id": node_id,
                    "state": runtime.spec["state"],
                    "freeze": bool(runtime.spec.get("freeze", False)),
                    "runtime_role": runtime.spec.get("runtime_role", "physics"),
                    "implementation": runtime.spec["implementation"],
                    "implementation_version": runtime.spec.get("implementation_version", ""),
                    "implementation_hash": runtime.source_hash,
                }
            )
        internal_stages = []
        for stage_id in sorted(self.internal_stages):
            stage = self.internal_stages[stage_id]
            internal_stages.append(
                {
                    "id": stage_id,
                    "state": stage.spec["state"],
                    "freeze": bool(stage.spec.get("freeze", False)),
                    "runtime_role": stage.spec.get(
                        "runtime_role",
                        "diagnostic",
                    ),
                    "runtime_binding": "internal_stage",
                    "runtime_owner": stage.owner_node_id,
                    "implementation": stage.spec["implementation"],
                    "implementation_version": stage.spec.get(
                        "implementation_version",
                        "",
                    ),
                    "implementation_hash": stage.source_hash,
                    "implementation_hash_scope": "module_file",
                    "requires": list(stage.spec.get("requires", ())),
                    "provides": list(stage.spec.get("provides", ())),
                }
            )
        return GraphRunManifest(
            closure=self.config.closure,
            topology=list(self.topology),
            nodes=nodes,
            parameter_sources=dict(self.config.sources),
            internal_stages=internal_stages,
            guards=dict(guards or {}),
        )
