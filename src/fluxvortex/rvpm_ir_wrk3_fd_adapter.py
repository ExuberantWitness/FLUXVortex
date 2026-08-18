"""Audited centered-FD adapter for a generic frozen-parent velocity field.

The adapter deliberately knows nothing about Ptera or benchmark observations.
It accepts one already-frozen parent velocity callable, evaluates that callable
at a physical center and the six Cartesian offsets, and exposes callbacks that
compose directly with :mod:`fluxvortex.rvpm_ir_wrk3`.

Every successful evaluator response is an exact, immutable attestation.  The
adapter recomputes all array hashes, checks the parent before and after every
call, and commits its immutable ledger only after the complete evaluation has
passed.  A failed evaluation therefore leaves the adapter ledger unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import inspect
import json
import math
import threading
from types import CodeType, FunctionType
from typing import Callable, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rvpm_ir_wrk3 import (
    IRWRK3Field,
    IRWRK3TracerField,
    make_ir_wrk3_field,
    make_ir_wrk3_tracer_field,
)
from .rvpm_transport import ParticleState

FloatArray = NDArray[np.float64]

DEFAULT_MAX_TARGET_COUNT: Final[int] = 1_000_000
MAX_FD_EVALUATION_COUNT: Final[int] = 12_288
MAX_FD_VELOCITY_CALL_COUNT: Final[int] = 86_016
MAX_LIVE_FD_LEDGER_COUNT: Final[int] = 4096
_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_LEDGER_GENESIS: Final[str] = sha256(
    b"fluxv-ir-wrk3-frozen-parent-fd-ledger-v1"
).hexdigest()

_NUMPY_BINDING_NAMES: Final[tuple[str, ...]] = (
    "all",
    "any",
    "array",
    "array_equal",
    "asarray",
    "ascontiguousarray",
    "bool_",
    "dtype",
    "empty",
    "float64",
    "frombuffer",
    "isfinite",
    "ndarray",
)
_FROZEN_NUMPY_BINDINGS: Final[tuple[tuple[str, object], ...]] = tuple(
    (name, getattr(np, name)) for name in _NUMPY_BINDING_NAMES
)
_FROZEN_PUBLIC_BINDINGS: Final[tuple[tuple[str, object], ...]] = (
    ("np", np),
    ("inspect", inspect),
    ("json", json),
    ("math", math),
    ("threading", threading),
    ("partial", partial),
    ("CodeType", CodeType),
    ("FunctionType", FunctionType),
    ("ParticleState", ParticleState),
    ("make_ir_wrk3_field", make_ir_wrk3_field),
    ("make_ir_wrk3_tracer_field", make_ir_wrk3_tracer_field),
    ("sha256", sha256),
)
_FROZEN_MATH_BINDINGS: Final[tuple[tuple[str, object], ...]] = (
    ("isfinite", math.isfinite),
)
_FROZEN_SCALAR_BINDINGS: Final[tuple[tuple[str, object], ...]] = (
    ("DEFAULT_MAX_TARGET_COUNT", DEFAULT_MAX_TARGET_COUNT),
    ("MAX_FD_EVALUATION_COUNT", MAX_FD_EVALUATION_COUNT),
    ("MAX_FD_VELOCITY_CALL_COUNT", MAX_FD_VELOCITY_CALL_COUNT),
    ("MAX_LIVE_FD_LEDGER_COUNT", MAX_LIVE_FD_LEDGER_COUNT),
    ("_HEX_DIGITS", _HEX_DIGITS),
    ("_LEDGER_GENESIS", _LEDGER_GENESIS),
)


def _assert_runtime_bindings(
    numpy_bindings: tuple[tuple[str, object], ...] = _FROZEN_NUMPY_BINDINGS,
    public_bindings: tuple[tuple[str, object], ...] = _FROZEN_PUBLIC_BINDINGS,
    math_bindings: tuple[tuple[str, object], ...] = _FROZEN_MATH_BINDINGS,
    scalar_bindings: tuple[tuple[str, object], ...] = _FROZEN_SCALAR_BINDINGS,
) -> None:
    if globals().get("_FROZEN_NUMPY_BINDINGS") is not numpy_bindings:
        raise RuntimeError("FD adapter NumPy binding registry drift")
    if globals().get("_FROZEN_PUBLIC_BINDINGS") is not public_bindings:
        raise RuntimeError("FD adapter public binding registry drift")
    if globals().get("_FROZEN_MATH_BINDINGS") is not math_bindings:
        raise RuntimeError("FD adapter math binding registry drift")
    if globals().get("_FROZEN_SCALAR_BINDINGS") is not scalar_bindings:
        raise RuntimeError("FD adapter scalar binding registry drift")
    for name, frozen in numpy_bindings:
        if getattr(np, name) is not frozen:
            raise RuntimeError(f"FD adapter NumPy runtime binding drift: {name}")
    for name, frozen in public_bindings:
        if globals().get(name) is not frozen:
            raise RuntimeError(f"FD adapter public runtime binding drift: {name}")
    for name, frozen in math_bindings:
        if getattr(math, name) is not frozen:
            raise RuntimeError(f"FD adapter math runtime binding drift: {name}")
    for name, frozen in scalar_bindings:
        if globals().get(name) is not frozen:
            raise RuntimeError(f"FD adapter scalar runtime binding drift: {name}")


def _validate_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _validate_parent_token(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("parent_token must be a non-empty exact string")
    return value


def _validate_epsilon(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or type(value) not in (float, int):
        raise ValueError("epsilon must be a built-in real scalar")
    epsilon = float(value)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and strictly positive")
    scale = 0.5 / epsilon
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("epsilon has no finite positive centered-FD scale")
    return epsilon


def _validate_cap(
    value: object,
    hard_cap: int = DEFAULT_MAX_TARGET_COUNT,
) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("max_target_count must be an exact positive integer")
    if value > hard_cap:
        raise ValueError("max_target_count exceeds the frozen hard cap")
    return value


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _frozen_float64(
    name: str, value: ArrayLike, shape_tail: tuple[int, ...]
) -> FloatArray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf":
        raise ValueError(f"{name} must use a real numeric dtype")
    array = np.asarray(original, dtype=np.float64)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        expected = (
            "(n,)" if not shape_tail else f"(n, {', '.join(map(str, shape_tail))})"
        )
        raise ValueError(f"{name} must have shape {expected}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return frozen.reshape(contiguous.shape)


def _require_readonly_float64(
    name: str,
    value: object,
    shape: tuple[int, ...],
) -> FloatArray:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be an exact ndarray")
    if value.dtype != np.dtype(np.float64) or value.shape != shape:
        raise ValueError(f"{name} has an invalid dtype or shape")
    if value.flags.writeable or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be readonly and C-contiguous")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_state(state: object, cap: int) -> tuple[ParticleState, int]:
    if type(state) is not ParticleState:
        raise ValueError("state must be an exact ParticleState")
    if type(state.positions) is not np.ndarray or state.positions.ndim != 2:
        raise ValueError("state.positions must be an exact rank-2 ndarray")
    count = state.positions.shape[0]
    if count > cap:
        raise ValueError("physical target cap exceeded")
    positions = _frozen_float64("state.positions", state.positions, (3,))
    gamma = _frozen_float64("state.gamma", state.gamma, (3,))
    sigma = _frozen_float64("state.sigma", state.sigma, ())
    if (
        positions.shape[0] != count
        or gamma.shape[0] != count
        or sigma.shape[0] != count
    ):
        raise ValueError("state arrays must have the same leading dimension")
    if np.any(sigma <= 0.0):
        raise ValueError("state.sigma must be strictly positive")
    validated = ParticleState(positions, gamma, sigma)
    return validated, count


def _prepare_targets(name: str, value: ArrayLike, cap: int) -> FloatArray:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is not None:
        try:
            shape = tuple(raw_shape)
        except TypeError as error:
            raise ValueError(f"{name}.shape must be an iterable shape") from error
        if len(shape) != 2 or shape[1:] != (3,):
            raise ValueError(f"{name} must have shape (n, 3)")
        if type(shape[0]) is not int or shape[0] < 0:
            raise ValueError(f"{name} leading dimension is invalid")
        if shape[0] > cap:
            raise ValueError(f"{name} cap exceeded")
    else:
        try:
            leading_count = len(value)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            leading_count = None
        if leading_count is not None and leading_count > cap:
            raise ValueError(f"{name} cap exceeded")
    original = np.asarray(value)
    if original.ndim != 2 or original.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (n, 3)")
    if original.shape[0] > cap:
        raise ValueError(f"{name} cap exceeded")
    return _frozen_float64(name, original, (3,))


def _attestation_sha256(
    *,
    target_sha256: str,
    velocity_sha256: str,
    parent_token: str,
    parent_state_sha256: str,
) -> str:
    return sha256(
        json.dumps(
            {
                "kind": "fluxv-frozen-parent-velocity-v1",
                "parent_state_sha256": parent_state_sha256,
                "parent_token": parent_token,
                "target_sha256": target_sha256,
                "velocity_sha256": velocity_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenParentVelocity:
    """Exact response attesting one frozen-parent velocity evaluation."""

    velocity: FloatArray
    target_sha256: str
    velocity_sha256: str
    parent_token: str
    parent_state_sha256: str
    attestation_sha256: str


def make_frozen_parent_velocity(
    targets: ArrayLike,
    velocity: ArrayLike,
    *,
    parent_token: str,
    parent_state_sha256: str,
) -> FrozenParentVelocity:
    """Create a readonly response bound to exact targets and parent identity."""

    _assert_runtime_bindings()
    token = _validate_parent_token(parent_token)
    parent_sha = _validate_sha256("parent_state_sha256", parent_state_sha256)
    target_array = _frozen_float64("targets", targets, (3,))
    velocity_array = _frozen_float64("velocity", velocity, (3,))
    if target_array.shape != velocity_array.shape:
        raise ValueError("targets and velocity must have identical shape")
    target_sha = _array_sha256(target_array)
    velocity_sha = _array_sha256(velocity_array)
    return FrozenParentVelocity(
        velocity=velocity_array,
        target_sha256=target_sha,
        velocity_sha256=velocity_sha,
        parent_token=token,
        parent_state_sha256=parent_sha,
        attestation_sha256=_attestation_sha256(
            target_sha256=target_sha,
            velocity_sha256=velocity_sha,
            parent_token=token,
            parent_state_sha256=parent_sha,
        ),
    )


@dataclass(frozen=True, slots=True)
class CallableProvenance:
    role: str
    kind: str
    module: str
    qualname: str
    code_sha256: str
    dependency_count: int
    provenance_sha256: str


@dataclass(frozen=True, slots=True)
class _DependencyBinding:
    label: str
    source_kind: str
    owner: object
    key: object
    expected: object
    content_sha256: str
    include_scalars: bool


@dataclass(frozen=True, slots=True)
class _FunctionGuard:
    label: str
    function: FunctionType
    code: object
    code_sha256: str
    defaults: object
    kwdefaults: object
    global_bindings: tuple[tuple[dict[str, object], str, object], ...]
    closure_bindings: tuple[tuple[object, object], ...]
    data_bindings: tuple[_DependencyBinding, ...]


@dataclass(frozen=True, slots=True)
class _CallableGuard:
    root: object
    root_kind: str
    root_module: str
    root_qualname: str
    root_identity: tuple[object, ...]
    functions: tuple[_FunctionGuard, ...]
    provenance: CallableProvenance


def _canonical_code_constant(value: object) -> object:
    """Encode only immutable CPython code constants without runtime caches."""

    if value is None:
        return ["none"]
    if value is Ellipsis:
        return ["ellipsis"]
    if value is NotImplemented:
        return ["not-implemented"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) is complex:
        return ["complex", value.real.hex(), value.imag.hex()]
    if type(value) is str:
        return ["str", value]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    if type(value) is tuple:
        return ["tuple", [_canonical_code_constant(item) for item in value]]
    if type(value) is frozenset:
        encoded = [_canonical_code_constant(item) for item in value]
        encoded.sort(
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
        return ["frozenset", encoded]
    if type(value) is CodeType:
        return ["code", _canonical_code_payload(value)]
    raise ValueError(
        "callable code contains an unsupported non-canonical constant: "
        f"{type(value).__qualname__}"
    )


def _canonical_code_payload(code: CodeType) -> dict[str, object]:
    return {
        "argcount": code.co_argcount,
        "cellvars": list(code.co_cellvars),
        "code": code.co_code.hex(),
        "consts": [_canonical_code_constant(value) for value in code.co_consts],
        "exceptiontable": getattr(code, "co_exceptiontable", b"").hex(),
        "firstlineno": code.co_firstlineno,
        "flags": code.co_flags,
        "freevars": list(code.co_freevars),
        "kwonlyargcount": code.co_kwonlyargcount,
        "linetable": getattr(code, "co_linetable", b"").hex(),
        "name": code.co_name,
        "names": list(code.co_names),
        "nlocals": code.co_nlocals,
        "posonlyargcount": code.co_posonlyargcount,
        "qualname": getattr(code, "co_qualname", code.co_name),
        "stacksize": code.co_stacksize,
        "varnames": list(code.co_varnames),
    }


def _code_sha256(function: FunctionType) -> str:
    payload = _canonical_code_payload(function.__code__)
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _dependency_payload(
    value: object,
    seen: set[int] | None = None,
    *,
    include_scalars: bool = False,
) -> tuple[bool, bytes]:
    """Return a deterministic payload only for array/buffer-bearing state."""

    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return False, b""
    if type(value) is np.ndarray:
        array = value
        digest = sha256()
        digest.update(b"ndarray\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(b"writeable=1" if array.flags.writeable else b"writeable=0")
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
        return True, digest.digest()
    if type(value) is bytearray:
        return True, b"bytearray\0" + bytes(value)
    if type(value) is memoryview:
        return (
            True,
            b"memoryview\0"
            + str(value.format).encode("ascii", errors="backslashreplace")
            + b"\0"
            + json.dumps(value.shape, separators=(",", ":")).encode("ascii")
            + b"\0"
            + value.tobytes(),
        )
    if include_scalars and type(value) in (
        type(None),
        bool,
        int,
        float,
        complex,
        str,
        bytes,
    ):
        if type(value) is float:
            rendered = value.hex()
        elif type(value) is complex:
            rendered = f"{value.real.hex()}:{value.imag.hex()}"
        elif type(value) is bytes:
            return True, b"bytes\0" + value
        else:
            rendered = repr(value)
        return (
            True,
            type(value).__name__.encode("ascii") + b"\0" + rendered.encode("utf-8"),
        )
    if type(value) not in (tuple, list, dict):
        return False, b""
    visited.add(identity)
    parts: list[bytes] = []
    if type(value) in (tuple, list):
        for index, item in enumerate(value):
            contains, payload = _dependency_payload(
                item, visited, include_scalars=include_scalars
            )
            if contains:
                parts.append(str(index).encode("ascii") + b":" + payload)
        kind = b"tuple" if type(value) is tuple else b"list"
    else:
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            contains, payload = _dependency_payload(
                value[key], visited, include_scalars=include_scalars
            )
            if contains:
                key_payload = (
                    type(key).__name__.encode("utf-8")
                    + b":"
                    + repr(key).encode("utf-8", errors="backslashreplace")
                )
                parts.append(key_payload + b":" + payload)
        kind = b"dict"
    visited.remove(identity)
    if not parts:
        return False, b""
    return True, kind + b"\0" + b"\0".join(parts)


def _dependency_sha256(
    value: object,
    *,
    include_scalars: bool = False,
) -> str | None:
    contains, payload = _dependency_payload(value, include_scalars=include_scalars)
    return sha256(payload).hexdigest() if contains else None


def _dependency_binding(
    label: str,
    source_kind: str,
    owner: object,
    key: object,
    value: object,
    *,
    include_scalars: bool = False,
) -> _DependencyBinding | None:
    content_sha = _dependency_sha256(value, include_scalars=include_scalars)
    if content_sha is None:
        return None
    return _DependencyBinding(
        label=label,
        source_kind=source_kind,
        owner=owner,
        key=key,
        expected=value,
        content_sha256=content_sha,
        include_scalars=include_scalars,
    )


def _resolved_dependency(binding: _DependencyBinding) -> object:
    if binding.source_kind == "global":
        return binding.owner.get(binding.key)  # type: ignore[union-attr]
    if binding.source_kind == "closure":
        try:
            return binding.owner.cell_contents
        except ValueError as error:
            raise RuntimeError(
                f"dependency closure became empty: {binding.label}"
            ) from error
    if binding.source_kind == "attribute":
        return getattr(binding.owner, binding.key, None)
    if binding.source_kind == "mapping":
        return binding.owner.get(binding.key)  # type: ignore[union-attr]
    if binding.source_kind == "direct":
        return binding.owner
    raise RuntimeError(f"unknown dependency binding kind: {binding.source_kind}")


def _assert_dependency_binding(binding: _DependencyBinding, role: str) -> None:
    current = _resolved_dependency(binding)
    scalar_by_value = binding.include_scalars and type(binding.expected) in (
        type(None),
        bool,
        int,
        float,
        complex,
        str,
        bytes,
    )
    if not scalar_by_value and current is not binding.expected:
        raise RuntimeError(f"{role} dependency identity drift: {binding.label}")
    observed_sha = _dependency_sha256(current, include_scalars=binding.include_scalars)
    if observed_sha != binding.content_sha256:
        raise RuntimeError(f"{role} dependency content drift: {binding.label}")


def _root_callable_parts(
    value: object,
) -> tuple[str, str, str, tuple[object, ...], tuple[FunctionType, ...]]:
    if inspect.isfunction(value):
        function = value
        return (
            "function",
            function.__module__,
            function.__qualname__,
            (function,),
            (function,),
        )
    if inspect.ismethod(value) and inspect.isfunction(value.__func__):
        function = value.__func__
        return (
            "bound_method",
            function.__module__,
            function.__qualname__,
            (value, value.__self__, function),
            (function,),
        )
    if isinstance(value, partial):
        kind, module, qualname, identity, functions = _root_callable_parts(value.func)
        return (
            f"partial[{kind}]",
            module,
            qualname,
            (value, value.func, value.args, value.keywords, *identity),
            functions,
        )
    if callable(value):
        call = getattr(type(value), "__call__", None)
        if inspect.isfunction(call):
            return (
                "callable_object",
                type(value).__module__,
                f"{type(value).__qualname__}.__call__",
                (value, type(value), call),
                (call,),
            )
    raise ValueError("callable must have auditable Python function code")


def _root_state_owners(value: object) -> tuple[object, ...]:
    if inspect.ismethod(value):
        return (value.__self__,)
    if isinstance(value, partial):
        return (*_root_state_owners(value.func), *value.args)
    if inspect.isfunction(value):
        return ()
    if callable(value):
        return (value,)
    return ()


def _freeze_callable_guard(role: str, value: object) -> _CallableGuard:
    if type(role) is not str or not role:
        raise ValueError("callable provenance role is invalid")
    kind, module, qualname, identity, roots = _root_callable_parts(value)
    root_owners = _root_state_owners(value)
    pending: list[tuple[str, FunctionType]] = [
        (f"{role}:root:{index}", function) for index, function in enumerate(roots)
    ]
    seen: set[int] = set()
    guards: list[_FunctionGuard] = []
    while pending:
        label, function = pending.pop(0)
        if id(function) in seen:
            continue
        seen.add(id(function))
        global_bindings: list[tuple[dict[str, object], str, object]] = []
        data_bindings: list[_DependencyBinding] = []
        for name in sorted(set(function.__code__.co_names)):
            if name not in function.__globals__:
                continue
            dependency = function.__globals__[name]
            if not callable(dependency):
                binding = _dependency_binding(
                    f"{label}:global:{name}",
                    "global",
                    function.__globals__,
                    name,
                    dependency,
                )
                if binding is not None:
                    data_bindings.append(binding)
                else:
                    for attribute in sorted(set(function.__code__.co_names)):
                        if inspect.ismodule(dependency):
                            namespace = vars(dependency)
                            if attribute not in namespace:
                                continue
                            nested = namespace[attribute]
                        else:
                            try:
                                nested = inspect.getattr_static(dependency, attribute)
                            except AttributeError:
                                continue
                        if inspect.ismemberdescriptor(nested):
                            try:
                                nested = object.__getattribute__(dependency, attribute)
                            except AttributeError:
                                continue
                        binding = _dependency_binding(
                            f"{label}:global:{name}.{attribute}",
                            "attribute",
                            dependency,
                            attribute,
                            nested,
                        )
                        if binding is not None:
                            data_bindings.append(binding)
                continue
            global_bindings.append((function.__globals__, name, dependency))
            if inspect.isfunction(dependency):
                pending.append((f"{label}:global:{name}", dependency))
            elif inspect.ismethod(dependency) and inspect.isfunction(
                dependency.__func__
            ):
                pending.append((f"{label}:global:{name}", dependency.__func__))
        closure_bindings: list[tuple[object, object]] = []
        if function.__closure__ is not None:
            for index, cell in enumerate(function.__closure__):
                try:
                    dependency = cell.cell_contents
                except ValueError:
                    continue
                if not callable(dependency):
                    binding = _dependency_binding(
                        f"{label}:closure:{index}",
                        "closure",
                        cell,
                        index,
                        dependency,
                    )
                    if binding is not None:
                        data_bindings.append(binding)
                    else:
                        for attribute in sorted(set(function.__code__.co_names)):
                            if inspect.ismodule(dependency):
                                namespace = vars(dependency)
                                if attribute not in namespace:
                                    continue
                                nested = namespace[attribute]
                            else:
                                try:
                                    nested = inspect.getattr_static(
                                        dependency, attribute
                                    )
                                except AttributeError:
                                    continue
                            if inspect.ismemberdescriptor(nested):
                                try:
                                    nested = object.__getattribute__(
                                        dependency, attribute
                                    )
                                except AttributeError:
                                    continue
                            binding = _dependency_binding(
                                f"{label}:closure:{index}.{attribute}",
                                "attribute",
                                dependency,
                                attribute,
                                nested,
                            )
                            if binding is not None:
                                data_bindings.append(binding)
                    continue
                closure_bindings.append((cell, dependency))
                if inspect.isfunction(dependency):
                    pending.append((f"{label}:closure:{index}", dependency))
                elif inspect.ismethod(dependency) and inspect.isfunction(
                    dependency.__func__
                ):
                    pending.append((f"{label}:closure:{index}", dependency.__func__))
        if function.__defaults__ is not None:
            for index, dependency in enumerate(function.__defaults__):
                binding = _dependency_binding(
                    f"{label}:default:{index}",
                    "direct",
                    dependency,
                    index,
                    dependency,
                    include_scalars=True,
                )
                if binding is not None:
                    data_bindings.append(binding)
        if function.__kwdefaults__ is not None:
            for name, dependency in sorted(function.__kwdefaults__.items()):
                binding = _dependency_binding(
                    f"{label}:kwdefault:{name}",
                    "mapping",
                    function.__kwdefaults__,
                    name,
                    dependency,
                    include_scalars=True,
                )
                if binding is not None:
                    data_bindings.append(binding)
        for owner_index, owner in enumerate(root_owners):
            for attribute in sorted(set(function.__code__.co_names)):
                try:
                    dependency = inspect.getattr_static(owner, attribute)
                except AttributeError:
                    continue
                if inspect.ismemberdescriptor(dependency):
                    try:
                        dependency = object.__getattribute__(owner, attribute)
                    except AttributeError:
                        continue
                binding = _dependency_binding(
                    f"{label}:self:{owner_index}.{attribute}",
                    "attribute",
                    owner,
                    attribute,
                    dependency,
                )
                if binding is not None:
                    data_bindings.append(binding)
        if isinstance(value, partial):
            for index, dependency in enumerate(value.args):
                binding = _dependency_binding(
                    f"{label}:partial_arg:{index}",
                    "direct",
                    dependency,
                    index,
                    dependency,
                    include_scalars=True,
                )
                if binding is not None:
                    data_bindings.append(binding)
            if value.keywords is not None:
                for name, dependency in sorted(value.keywords.items()):
                    binding = _dependency_binding(
                        f"{label}:partial_keyword:{name}",
                        "mapping",
                        value.keywords,
                        name,
                        dependency,
                        include_scalars=True,
                    )
                    if binding is not None:
                        data_bindings.append(binding)
        guards.append(
            _FunctionGuard(
                label=label,
                function=function,
                code=function.__code__,
                code_sha256=_code_sha256(function),
                defaults=function.__defaults__,
                kwdefaults=function.__kwdefaults__,
                global_bindings=tuple(global_bindings),
                closure_bindings=tuple(closure_bindings),
                data_bindings=tuple(data_bindings),
            )
        )
    guards_tuple = tuple(guards)
    aggregate = sha256()
    for guard in guards_tuple:
        aggregate.update(guard.label.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(guard.function.__module__.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(guard.function.__qualname__.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(guard.code_sha256.encode("ascii"))
        aggregate.update(b"\0")
        for binding in guard.data_bindings:
            aggregate.update(binding.label.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(binding.content_sha256.encode("ascii"))
            aggregate.update(b"\0")
    code_sha = aggregate.hexdigest()
    provenance_sha = sha256(
        json.dumps(
            {
                "code_sha256": code_sha,
                "dependency_count": len(guards_tuple)
                + sum(len(guard.data_bindings) for guard in guards_tuple),
                "kind": kind,
                "module": module,
                "qualname": qualname,
                "role": role,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    provenance = CallableProvenance(
        role=role,
        kind=kind,
        module=module,
        qualname=qualname,
        code_sha256=code_sha,
        dependency_count=len(guards_tuple)
        + sum(len(guard.data_bindings) for guard in guards_tuple),
        provenance_sha256=provenance_sha,
    )
    return _CallableGuard(
        root=value,
        root_kind=kind,
        root_module=module,
        root_qualname=qualname,
        root_identity=identity,
        functions=guards_tuple,
        provenance=provenance,
    )


def _assert_callable_guard(guard: _CallableGuard) -> None:
    kind, module, qualname, identity, _ = _root_callable_parts(guard.root)
    if (
        kind != guard.root_kind
        or module != guard.root_module
        or qualname != guard.root_qualname
        or len(identity) != len(guard.root_identity)
        or any(
            current is not frozen
            for current, frozen in zip(identity, guard.root_identity, strict=True)
        )
    ):
        raise RuntimeError(f"{guard.provenance.role} callable identity drift")
    for function_guard in guard.functions:
        function = function_guard.function
        if (
            function.__code__ is not function_guard.code
            or function.__defaults__ is not function_guard.defaults
            or function.__kwdefaults__ is not function_guard.kwdefaults
        ):
            raise RuntimeError(
                f"{guard.provenance.role} callable code/default drift: "
                f"{function_guard.label}"
            )
        for namespace, name, frozen in function_guard.global_bindings:
            if namespace.get(name) is not frozen:
                raise RuntimeError(
                    f"{guard.provenance.role} callable global drift: "
                    f"{function_guard.label}:{name}"
                )
        for cell, frozen in function_guard.closure_bindings:
            try:
                current = cell.cell_contents
            except ValueError as error:
                raise RuntimeError(
                    f"{guard.provenance.role} callable closure drift: "
                    f"{function_guard.label}"
                ) from error
            if current is not frozen:
                raise RuntimeError(
                    f"{guard.provenance.role} callable closure drift: "
                    f"{function_guard.label}"
                )
        for binding in function_guard.data_bindings:
            _assert_dependency_binding(binding, guard.provenance.role)


@dataclass(frozen=True, slots=True)
class FDVelocityCall:
    call_index: int
    evaluation_index: int
    target_kind: str
    sample_kind: str
    axis: int
    sign: int
    target_count: int
    target_sha256: str
    velocity_sha256: str
    parent_token: str
    parent_state_sha256: str
    callable_provenance_sha256: str
    attestation_sha256: str
    call_sha256: str


@dataclass(frozen=True, slots=True)
class FDEvaluationRecord:
    evaluation_index: int
    target_kind: str
    target_count: int
    source_state_sha256: str
    target_sha256: str
    velocity_sha256: str
    jacobian_sha256: str | None
    parent_token: str
    parent_state_sha256: str
    epsilon: float
    first_call_index: int
    last_call_index: int
    evaluator_call_count: int
    evaluation_sha256: str
    trace_sha256: str


@dataclass(frozen=True, slots=True)
class FDPhysicalResult:
    field: IRWRK3Field
    evaluation: FDEvaluationRecord


@dataclass(frozen=True, slots=True)
class FDTracerResult:
    field: IRWRK3TracerField
    evaluation: FDEvaluationRecord


@dataclass(frozen=True, slots=True)
class FDCallLedger:
    epsilon: float
    parent_token: str
    parent_state_sha256: str
    velocity_evaluator_provenance: CallableProvenance
    parent_hash_getter_provenance: CallableProvenance
    calls: tuple[FDVelocityCall, ...]
    evaluations: tuple[FDEvaluationRecord, ...]
    physical_evaluation_count: int
    tracer_evaluation_count: int
    center_call_count: int
    offset_call_count: int
    evaluator_call_count: int
    trace_sha256: str
    ledger_sha256: str


FrozenParentVelocityEvaluator = Callable[[FloatArray], FrozenParentVelocity]
ParentStateSHA256Getter = Callable[[], str]


def _call_sha256(payload: dict[str, object]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _make_live_fd_ledger_registry(
    max_count: int = MAX_LIVE_FD_LEDGER_COUNT,
    hash_constructor: Callable[[bytes], object] = sha256,
) -> tuple[Callable[..., object], ...]:
    """Create a sealed issuance registry without exposing its mutable mapping."""

    lock = threading.RLock()
    registry: dict[int, tuple[FDCallLedger, str]] = {}
    issuance_counter = 0

    def compute_seal() -> str:
        payload = (
            issuance_counter,
            tuple(
                (key, id(entry[0]), entry[1], entry[0].ledger_sha256)
                for key, entry in sorted(registry.items())
            ),
        )
        digest = hash_constructor(repr(payload).encode("ascii"))
        return digest.hexdigest()  # type: ignore[no-any-return, union-attr]

    seal = compute_seal()

    def verify_seal() -> None:
        if compute_seal() != seal:
            raise RuntimeError("live FD ledger registry integrity drift")

    def register(ledger: FDCallLedger) -> None:
        nonlocal issuance_counter, seal
        with lock:
            verify_seal()
            if len(registry) >= max_count:
                raise RuntimeError("live FD ledger registry cap exceeded")
            key = id(ledger)
            if key in registry:
                raise RuntimeError("live FD ledger identity collision")
            registry[key] = (ledger, ledger.ledger_sha256)
            issuance_counter += 1
            seal = compute_seal()

    def assert_capacity() -> None:
        with lock:
            verify_seal()
            if len(registry) >= max_count:
                raise RuntimeError("live FD ledger registry cap exceeded")

    def attest(ledger: FDCallLedger) -> None:
        with lock:
            verify_seal()
            entry = registry.get(id(ledger))
            if (
                entry is None
                or entry[0] is not ledger
                or entry[1] != ledger.ledger_sha256
            ):
                raise ValueError("ledger is not the exact live issued snapshot")

    def snapshot() -> tuple[object, ...]:
        with lock:
            verify_seal()
            return (
                issuance_counter,
                seal,
                tuple(
                    (key, entry[0], entry[1]) for key, entry in sorted(registry.items())
                ),
            )

    def assert_snapshot_unchanged(expected: tuple[object, ...]) -> None:
        with lock:
            verify_seal()
            observed = snapshot()
            if (
                len(expected) != 3
                or observed[0] != expected[0]
                or observed[1] != expected[1]
            ):
                raise RuntimeError("live FD ledger registry changed during callback")
            observed_entries = observed[2]
            expected_entries = expected[2]
            if (
                type(observed_entries) is not tuple
                or type(expected_entries) is not tuple
                or len(observed_entries) != len(expected_entries)
            ):
                raise RuntimeError("live FD ledger registry changed during callback")
            for observed_entry, expected_entry in zip(
                observed_entries, expected_entries, strict=True
            ):
                if (
                    observed_entry[0] != expected_entry[0]
                    or observed_entry[1] is not expected_entry[1]
                    or observed_entry[2] != expected_entry[2]
                ):
                    raise RuntimeError(
                        "live FD ledger registry changed during callback"
                    )

    return (register, assert_capacity, attest, snapshot, assert_snapshot_unchanged)


(
    _register_fd_call_ledger,
    _assert_fd_call_ledger_capacity,
    _attest_fd_call_ledger,
    _snapshot_fd_call_ledger_registry,
    _assert_fd_call_ledger_registry_unchanged,
) = _make_live_fd_ledger_registry()


@dataclass(frozen=True, slots=True)
class _InternalFunctionBinding:
    name: str
    function: FunctionType
    code: object
    defaults: object
    kwdefaults: tuple[tuple[str, object], ...] | None


_INTERNAL_FUNCTION_NAMES: Final[tuple[str, ...]] = (
    "_assert_runtime_bindings",
    "_validate_sha256",
    "_validate_parent_token",
    "_validate_epsilon",
    "_validate_cap",
    "_array_sha256",
    "_frozen_float64",
    "_require_readonly_float64",
    "_validate_state",
    "_prepare_targets",
    "_attestation_sha256",
    "make_frozen_parent_velocity",
    "_canonical_code_constant",
    "_canonical_code_payload",
    "_code_sha256",
    "_dependency_payload",
    "_dependency_sha256",
    "_dependency_binding",
    "_resolved_dependency",
    "_assert_dependency_binding",
    "_root_callable_parts",
    "_root_state_owners",
    "_freeze_callable_guard",
    "_assert_callable_guard",
    "_call_sha256",
    "_make_live_fd_ledger_registry",
    "_register_fd_call_ledger",
    "_assert_fd_call_ledger_capacity",
    "_attest_fd_call_ledger",
    "_snapshot_fd_call_ledger_registry",
    "_assert_fd_call_ledger_registry_unchanged",
)
_INTERNAL_BINDING_NAMESPACE: Final[dict[str, object]] = globals()


def _freeze_internal_binding(name: str) -> _InternalFunctionBinding:
    function = _INTERNAL_BINDING_NAMESPACE[name]
    if type(function) is not FunctionType:
        raise RuntimeError(f"FD adapter internal binding is not a function: {name}")
    kwdefaults = (
        None
        if function.__kwdefaults__ is None
        else tuple(sorted(function.__kwdefaults__.items()))
    )
    return _InternalFunctionBinding(
        name=name,
        function=function,
        code=function.__code__,
        defaults=function.__defaults__,
        kwdefaults=kwdefaults,
    )


_FROZEN_INTERNAL_BINDINGS: Final[tuple[_InternalFunctionBinding, ...]] = tuple(
    _freeze_internal_binding(name) for name in _INTERNAL_FUNCTION_NAMES
)


def _assert_internal_bindings(
    frozen: tuple[_InternalFunctionBinding, ...] = _FROZEN_INTERNAL_BINDINGS,
    namespace: dict[str, object] = _INTERNAL_BINDING_NAMESPACE,
) -> None:
    if globals().get("_FROZEN_INTERNAL_BINDINGS") is not frozen:
        raise RuntimeError("FD adapter internal verifier registry drift")
    for binding in frozen:
        current = namespace.get(binding.name)
        if current is not binding.function:
            raise RuntimeError(
                f"FD adapter internal function identity drift: {binding.name}"
            )
        if current.__code__ is not binding.code:
            raise RuntimeError(
                f"FD adapter internal function code drift: {binding.name}"
            )
        if current.__defaults__ is not binding.defaults:
            raise RuntimeError(
                f"FD adapter internal function defaults drift: {binding.name}"
            )
        current_kwdefaults = (
            None
            if current.__kwdefaults__ is None
            else tuple(sorted(current.__kwdefaults__.items()))
        )
        if current_kwdefaults != binding.kwdefaults:
            raise RuntimeError(
                f"FD adapter internal function kwdefaults drift: {binding.name}"
            )


def _validate_callable_provenance(
    name: str,
    value: object,
    *,
    role: str,
) -> CallableProvenance:
    if type(value) is not CallableProvenance:
        raise ValueError(f"{name} must be an exact CallableProvenance")
    if value.role != role:
        raise ValueError(f"{name} role mismatch")
    for label, text_value in (
        ("kind", value.kind),
        ("module", value.module),
        ("qualname", value.qualname),
    ):
        if type(text_value) is not str or not text_value:
            raise ValueError(f"{name}.{label} is invalid")
    _validate_sha256(f"{name}.code_sha256", value.code_sha256)
    _validate_sha256(f"{name}.provenance_sha256", value.provenance_sha256)
    if type(value.dependency_count) is not int or value.dependency_count < 1:
        raise ValueError(f"{name}.dependency_count is invalid")
    expected = _call_sha256(
        {
            "code_sha256": value.code_sha256,
            "dependency_count": value.dependency_count,
            "kind": value.kind,
            "module": value.module,
            "qualname": value.qualname,
            "role": value.role,
        }
    )
    if value.provenance_sha256 != expected:
        raise ValueError(f"{name} digest mismatch")
    return value


def _velocity_call_payload(call: FDVelocityCall) -> dict[str, object]:
    return {
        "attestation_sha256": call.attestation_sha256,
        "axis": call.axis,
        "call_index": call.call_index,
        "callable_provenance_sha256": call.callable_provenance_sha256,
        "evaluation_index": call.evaluation_index,
        "kind": "fluxv-fd-velocity-call-v1",
        "parent_state_sha256": call.parent_state_sha256,
        "parent_token": call.parent_token,
        "sample_kind": call.sample_kind,
        "sign": call.sign,
        "target_count": call.target_count,
        "target_kind": call.target_kind,
        "target_sha256": call.target_sha256,
        "velocity_sha256": call.velocity_sha256,
    }


def _evaluation_payload(
    evaluation: FDEvaluationRecord,
    calls: tuple[FDVelocityCall, ...],
) -> dict[str, object]:
    return {
        "call_sha256": [call.call_sha256 for call in calls],
        "epsilon": evaluation.epsilon,
        "evaluation_index": evaluation.evaluation_index,
        "kind": "fluxv-fd-evaluation-v1",
        "jacobian_sha256": evaluation.jacobian_sha256,
        "parent_state_sha256": evaluation.parent_state_sha256,
        "parent_token": evaluation.parent_token,
        "source_state_sha256": evaluation.source_state_sha256,
        "target_count": evaluation.target_count,
        "target_kind": evaluation.target_kind,
        "target_sha256": evaluation.target_sha256,
        "velocity_sha256": evaluation.velocity_sha256,
    }


def _ledger_payload(ledger: FDCallLedger) -> dict[str, object]:
    return {
        "counts": {
            "center": ledger.center_call_count,
            "evaluator": ledger.evaluator_call_count,
            "offset": ledger.offset_call_count,
            "physical": ledger.physical_evaluation_count,
            "tracer": ledger.tracer_evaluation_count,
        },
        "domain": "fluxv-fd-call-ledger-v1",
        "epsilon": float(ledger.epsilon).hex(),
        "evaluation_sha256": [
            evaluation.evaluation_sha256 for evaluation in ledger.evaluations
        ],
        "parent_hash_getter_provenance_sha256": (
            ledger.parent_hash_getter_provenance.provenance_sha256
        ),
        "parent_state_sha256": ledger.parent_state_sha256,
        "parent_token": ledger.parent_token,
        "trace_sha256": ledger.trace_sha256,
        "velocity_call_sha256": [call.call_sha256 for call in ledger.calls],
        "velocity_evaluator_provenance_sha256": (
            ledger.velocity_evaluator_provenance.provenance_sha256
        ),
    }


def _ledger_sha256(ledger: FDCallLedger) -> str:
    return _call_sha256(_ledger_payload(ledger))


def _validate_fd_call_ledger_tree(
    ledger: object,
    *,
    max_target_count: int = DEFAULT_MAX_TARGET_COUNT,
    max_call_count: int = MAX_FD_VELOCITY_CALL_COUNT,
    max_evaluation_count: int = MAX_FD_EVALUATION_COUNT,
) -> FDCallLedger:
    if type(ledger) is not FDCallLedger:
        raise ValueError("ledger must be an exact FDCallLedger")
    epsilon = _validate_epsilon(ledger.epsilon)
    parent_token = _validate_parent_token(ledger.parent_token)
    parent_sha = _validate_sha256(
        "ledger.parent_state_sha256", ledger.parent_state_sha256
    )
    velocity_provenance = _validate_callable_provenance(
        "ledger.velocity_evaluator_provenance",
        ledger.velocity_evaluator_provenance,
        role="velocity_evaluator",
    )
    _validate_callable_provenance(
        "ledger.parent_hash_getter_provenance",
        ledger.parent_hash_getter_provenance,
        role="parent_state_sha256_getter",
    )
    if type(ledger.calls) is not tuple or type(ledger.evaluations) is not tuple:
        raise ValueError("ledger calls/evaluations must be exact tuples")
    if len(ledger.calls) > max_call_count:
        raise ValueError("ledger velocity-call cap exceeded")
    if len(ledger.evaluations) > max_evaluation_count:
        raise ValueError("ledger evaluation cap exceeded")

    for expected_index, call in enumerate(ledger.calls, start=1):
        if type(call) is not FDVelocityCall:
            raise ValueError("ledger call must have the exact frozen type")
        if call.call_index != expected_index:
            raise ValueError("ledger call index sequence mismatch")
        if type(call.evaluation_index) is not int or call.evaluation_index < 1:
            raise ValueError("ledger call evaluation index is invalid")
        if call.target_kind not in ("physical", "tracer"):
            raise ValueError("ledger call target kind is invalid")
        if call.sample_kind == "center":
            if call.axis != -1 or call.sign != 0:
                raise ValueError("ledger center call coordinate is invalid")
        elif call.sample_kind == "offset":
            if call.axis not in (0, 1, 2) or call.sign not in (-1, 1):
                raise ValueError("ledger offset call coordinate is invalid")
        else:
            raise ValueError("ledger sample kind is invalid")
        if (
            type(call.target_count) is not int
            or not 0 <= call.target_count <= max_target_count
        ):
            raise ValueError("ledger call target count is invalid")
        for label, digest in (
            ("target", call.target_sha256),
            ("velocity", call.velocity_sha256),
            ("callable provenance", call.callable_provenance_sha256),
            ("attestation", call.attestation_sha256),
            ("call", call.call_sha256),
        ):
            _validate_sha256(f"ledger call {label}", digest)
        if call.parent_token != parent_token or call.parent_state_sha256 != parent_sha:
            raise ValueError("ledger call parent binding mismatch")
        if call.callable_provenance_sha256 != velocity_provenance.provenance_sha256:
            raise ValueError("ledger call provenance mismatch")
        expected_attestation = _attestation_sha256(
            target_sha256=call.target_sha256,
            velocity_sha256=call.velocity_sha256,
            parent_token=parent_token,
            parent_state_sha256=parent_sha,
        )
        if call.attestation_sha256 != expected_attestation:
            raise ValueError("ledger call attestation mismatch")
        if call.call_sha256 != _call_sha256(_velocity_call_payload(call)):
            raise ValueError("ledger call digest mismatch")

    trace_sha = _LEDGER_GENESIS
    next_call_index = 1
    physical_count = 0
    tracer_count = 0
    for expected_index, evaluation in enumerate(ledger.evaluations, start=1):
        if type(evaluation) is not FDEvaluationRecord:
            raise ValueError("ledger evaluation must have the exact frozen type")
        if evaluation.evaluation_index != expected_index:
            raise ValueError("ledger evaluation index sequence mismatch")
        if evaluation.target_kind == "physical":
            expected_pattern = (
                ("center", -1, 0),
                ("offset", 0, -1),
                ("offset", 0, 1),
                ("offset", 1, -1),
                ("offset", 1, 1),
                ("offset", 2, -1),
                ("offset", 2, 1),
            )
            physical_count += 1
        elif evaluation.target_kind == "tracer":
            expected_pattern = (("center", -1, 0),)
            tracer_count += 1
        else:
            raise ValueError("ledger evaluation target kind is invalid")
        call_count = len(expected_pattern)
        first = next_call_index
        last = first + call_count - 1
        if (
            evaluation.first_call_index != first
            or evaluation.last_call_index != last
            or evaluation.evaluator_call_count != call_count
        ):
            raise ValueError("ledger evaluation call range mismatch")
        evaluation_calls = ledger.calls[first - 1 : last]
        if len(evaluation_calls) != call_count:
            raise ValueError("ledger evaluation references missing calls")
        for call, pattern in zip(evaluation_calls, expected_pattern, strict=True):
            if (
                call.evaluation_index != expected_index
                or call.target_kind != evaluation.target_kind
                or (call.sample_kind, call.axis, call.sign) != pattern
                or call.target_count != evaluation.target_count
            ):
                raise ValueError("ledger evaluation/call binding mismatch")
        center = evaluation_calls[0]
        if (
            evaluation.target_sha256 != center.target_sha256
            or evaluation.velocity_sha256 != center.velocity_sha256
            or evaluation.parent_token != parent_token
            or evaluation.parent_state_sha256 != parent_sha
            or evaluation.epsilon != epsilon
        ):
            raise ValueError("ledger evaluation content binding mismatch")
        _validate_sha256(
            "ledger evaluation source_state_sha256",
            evaluation.source_state_sha256,
        )
        if evaluation.target_kind == "physical":
            _validate_sha256(
                "ledger evaluation jacobian_sha256",
                evaluation.jacobian_sha256,
            )
        elif evaluation.jacobian_sha256 is not None:
            raise ValueError("tracer evaluation must not attest a Jacobian")
        _validate_sha256(
            "ledger evaluation evaluation_sha256",
            evaluation.evaluation_sha256,
        )
        _validate_sha256("ledger evaluation trace_sha256", evaluation.trace_sha256)
        expected_evaluation_sha = _call_sha256(
            _evaluation_payload(evaluation, evaluation_calls)
        )
        if evaluation.evaluation_sha256 != expected_evaluation_sha:
            raise ValueError("ledger evaluation digest mismatch")
        trace_sha = sha256(
            (trace_sha + expected_evaluation_sha).encode("ascii")
        ).hexdigest()
        if evaluation.trace_sha256 != trace_sha:
            raise ValueError("ledger evaluation trace-chain mismatch")
        next_call_index = last + 1
    if next_call_index != len(ledger.calls) + 1:
        raise ValueError("ledger contains unclaimed velocity calls")

    derived_counts = (
        physical_count,
        tracer_count,
        physical_count + tracer_count,
        6 * physical_count,
        len(ledger.calls),
    )
    reported_counts = (
        ledger.physical_evaluation_count,
        ledger.tracer_evaluation_count,
        ledger.center_call_count,
        ledger.offset_call_count,
        ledger.evaluator_call_count,
    )
    if any(type(value) is not int or value < 0 for value in reported_counts):
        raise ValueError("ledger counters must be exact nonnegative integers")
    if reported_counts != derived_counts:
        raise ValueError("ledger counter recomputation mismatch")
    if ledger.trace_sha256 != trace_sha:
        raise ValueError("ledger terminal trace mismatch")
    _validate_sha256("ledger.ledger_sha256", ledger.ledger_sha256)
    if ledger.ledger_sha256 != _ledger_sha256(ledger):
        raise ValueError("ledger digest mismatch")
    return ledger


def validate_fd_call_ledger(ledger: FDCallLedger) -> FDCallLedger:
    """Recompute the exact ledger tree and require a live-issued identity."""

    validated = _validate_fd_call_ledger_tree(ledger)
    _attest_fd_call_ledger(validated)
    return validated


@dataclass(frozen=True, slots=True)
class _AdapterClassBinding:
    name: str
    value: object
    code: object | None
    defaults: object
    kwdefaults: object


@dataclass(frozen=True, slots=True)
class _AdapterSeal:
    adapter_class: type
    epsilon: float
    parent_token: str
    parent_state_sha256: str
    max_target_count: int
    velocity_evaluator: object
    parent_hash_getter: object
    velocity_guard: _CallableGuard
    parent_hash_guard: _CallableGuard
    velocity_guard_content_sha256: str
    parent_guard_content_sha256: str
    class_bindings: tuple[_AdapterClassBinding, ...]
    constructor_bindings: tuple[tuple[str, object], ...]
    seal_sha256: str


_ADAPTER_CLASS_BINDING_NAMES: Final[tuple[str, ...]] = (
    "__init__",
    "epsilon",
    "parent_token",
    "parent_state_sha256",
    "ledger",
    "_preflight",
    "_current_parent_sha256",
    "_assert_parent_unchanged",
    "_invoke_velocity",
    "_stencil_targets",
    "_make_evaluation",
    "_assert_capacity",
    "_commit",
    "evaluate_physical",
    "physical_field",
    "evaluate_tracer",
    "tracer_field",
    "snapshot",
)
_ADAPTER_CONSTRUCTOR_NAMES: Final[tuple[str, ...]] = (
    "FrozenParentVelocity",
    "CallableProvenance",
    "FDVelocityCall",
    "FDEvaluationRecord",
    "FDPhysicalResult",
    "FDTracerResult",
    "FDCallLedger",
    "_DependencyBinding",
    "_FunctionGuard",
    "_CallableGuard",
    "_AdapterClassBinding",
    "_AdapterSeal",
)


def _guard_content_sha256(guard: _CallableGuard) -> str:
    payload: dict[str, object] = {
        "functions": [],
        "provenance": guard.provenance.provenance_sha256,
        "root": id(guard.root),
        "root_identity": [id(value) for value in guard.root_identity],
        "root_kind": guard.root_kind,
        "root_module": guard.root_module,
        "root_qualname": guard.root_qualname,
    }
    functions: list[dict[str, object]] = []
    for function in guard.functions:
        functions.append(
            {
                "closure": [
                    (id(cell), id(value)) for cell, value in function.closure_bindings
                ],
                "code": id(function.code),
                "code_sha256": function.code_sha256,
                "data": [
                    (
                        binding.label,
                        binding.source_kind,
                        id(binding.owner),
                        repr(binding.key),
                        id(binding.expected),
                        binding.content_sha256,
                        binding.include_scalars,
                    )
                    for binding in function.data_bindings
                ],
                "defaults": id(function.defaults),
                "function": id(function.function),
                "global": [
                    (id(namespace), name, id(value))
                    for namespace, name, value in function.global_bindings
                ],
                "kwdefaults": id(function.kwdefaults),
                "label": function.label,
            }
        )
    payload["functions"] = functions
    return _call_sha256(payload)


def _adapter_seal_sha256(
    *,
    epsilon: float,
    parent_token: str,
    parent_state_sha256: str,
    max_target_count: int,
    velocity_provenance_sha256: str,
    parent_provenance_sha256: str,
    velocity_guard_content_sha256: str,
    parent_guard_content_sha256: str,
) -> str:
    return _call_sha256(
        {
            "domain": "fluxv-fd-adapter-config-seal-v1",
            "epsilon": float(epsilon).hex(),
            "max_target_count": max_target_count,
            "parent_guard_content_sha256": parent_guard_content_sha256,
            "parent_provenance_sha256": parent_provenance_sha256,
            "parent_state_sha256": parent_state_sha256,
            "parent_token": parent_token,
            "velocity_guard_content_sha256": velocity_guard_content_sha256,
            "velocity_provenance_sha256": velocity_provenance_sha256,
        }
    )


def _freeze_adapter_class_binding(
    adapter_class: type,
    name: str,
) -> _AdapterClassBinding:
    value = adapter_class.__dict__.get(name)
    function = value.fget if type(value) is property else value
    if type(function) is FunctionType:
        return _AdapterClassBinding(
            name=name,
            value=value,
            code=function.__code__,
            defaults=function.__defaults__,
            kwdefaults=function.__kwdefaults__,
        )
    return _AdapterClassBinding(name, value, None, None, None)


def _make_adapter_seal(adapter: object) -> _AdapterSeal:
    adapter_class = type(adapter)
    class_bindings = tuple(
        _freeze_adapter_class_binding(adapter_class, name)
        for name in _ADAPTER_CLASS_BINDING_NAMES
    )
    constructor_bindings = tuple(
        (name, globals().get(name)) for name in _ADAPTER_CONSTRUCTOR_NAMES
    )
    velocity_guard_sha = _guard_content_sha256(adapter._velocity_guard)
    parent_guard_sha = _guard_content_sha256(adapter._parent_hash_guard)
    seal_sha = _adapter_seal_sha256(
        epsilon=adapter._epsilon,
        parent_token=adapter._parent_token,
        parent_state_sha256=adapter._parent_state_sha256,
        max_target_count=adapter._max_target_count,
        velocity_provenance_sha256=(
            adapter._velocity_guard.provenance.provenance_sha256
        ),
        parent_provenance_sha256=(
            adapter._parent_hash_guard.provenance.provenance_sha256
        ),
        velocity_guard_content_sha256=velocity_guard_sha,
        parent_guard_content_sha256=parent_guard_sha,
    )
    return _AdapterSeal(
        adapter_class=adapter_class,
        epsilon=adapter._epsilon,
        parent_token=adapter._parent_token,
        parent_state_sha256=adapter._parent_state_sha256,
        max_target_count=adapter._max_target_count,
        velocity_evaluator=adapter._velocity_evaluator,
        parent_hash_getter=adapter._parent_hash_getter,
        velocity_guard=adapter._velocity_guard,
        parent_hash_guard=adapter._parent_hash_guard,
        velocity_guard_content_sha256=velocity_guard_sha,
        parent_guard_content_sha256=parent_guard_sha,
        class_bindings=class_bindings,
        constructor_bindings=constructor_bindings,
        seal_sha256=seal_sha,
    )


def _assert_adapter_seal(adapter: object) -> None:
    seal = adapter._seal
    if type(seal) is not _AdapterSeal or type(adapter) is not seal.adapter_class:
        raise RuntimeError("FD adapter config seal type drift")
    scalar_pairs = (
        (adapter._epsilon, seal.epsilon),
        (adapter._parent_token, seal.parent_token),
        (adapter._parent_state_sha256, seal.parent_state_sha256),
        (adapter._max_target_count, seal.max_target_count),
    )
    if any(
        type(current) is not type(frozen) or current != frozen
        for current, frozen in scalar_pairs
    ):
        raise RuntimeError("FD adapter sealed scalar config drift")
    identity_pairs = (
        (adapter._velocity_evaluator, seal.velocity_evaluator),
        (adapter._parent_hash_getter, seal.parent_hash_getter),
        (adapter._velocity_guard, seal.velocity_guard),
        (adapter._parent_hash_guard, seal.parent_hash_guard),
    )
    if any(current is not frozen for current, frozen in identity_pairs):
        raise RuntimeError("FD adapter sealed identity config drift")
    velocity_guard_sha = _guard_content_sha256(adapter._velocity_guard)
    parent_guard_sha = _guard_content_sha256(adapter._parent_hash_guard)
    if (
        velocity_guard_sha != seal.velocity_guard_content_sha256
        or parent_guard_sha != seal.parent_guard_content_sha256
    ):
        raise RuntimeError("FD adapter guard content drift")
    for binding in seal.class_bindings:
        current = seal.adapter_class.__dict__.get(binding.name)
        if current is not binding.value:
            raise RuntimeError(f"FD adapter class binding drift: {binding.name}")
        function = current.fget if type(current) is property else current
        if binding.code is not None and (
            type(function) is not FunctionType
            or function.__code__ is not binding.code
            or function.__defaults__ is not binding.defaults
            or function.__kwdefaults__ is not binding.kwdefaults
        ):
            raise RuntimeError(f"FD adapter class function drift: {binding.name}")
    for name, frozen in seal.constructor_bindings:
        if globals().get(name) is not frozen:
            raise RuntimeError(f"FD adapter constructor binding drift: {name}")
    observed_seal_sha = _adapter_seal_sha256(
        epsilon=adapter._epsilon,
        parent_token=adapter._parent_token,
        parent_state_sha256=adapter._parent_state_sha256,
        max_target_count=adapter._max_target_count,
        velocity_provenance_sha256=(
            adapter._velocity_guard.provenance.provenance_sha256
        ),
        parent_provenance_sha256=(
            adapter._parent_hash_guard.provenance.provenance_sha256
        ),
        velocity_guard_content_sha256=velocity_guard_sha,
        parent_guard_content_sha256=parent_guard_sha,
    )
    if seal.seal_sha256 != observed_seal_sha:
        raise RuntimeError("FD adapter config seal digest mismatch")


_SEAL_HELPER_FUNCTION_NAMES: Final[tuple[str, ...]] = (
    "_guard_content_sha256",
    "_adapter_seal_sha256",
    "_freeze_adapter_class_binding",
    "_make_adapter_seal",
    "_assert_adapter_seal",
)
_FROZEN_SEAL_HELPER_BINDINGS: Final[tuple[_InternalFunctionBinding, ...]] = tuple(
    _InternalFunctionBinding(
        name=name,
        function=globals()[name],
        code=globals()[name].__code__,
        defaults=globals()[name].__defaults__,
        kwdefaults=(
            None
            if globals()[name].__kwdefaults__ is None
            else tuple(sorted(globals()[name].__kwdefaults__.items()))
        ),
    )
    for name in _SEAL_HELPER_FUNCTION_NAMES
)


def _assert_seal_helper_bindings(
    frozen: tuple[_InternalFunctionBinding, ...] = _FROZEN_SEAL_HELPER_BINDINGS,
) -> None:
    if globals().get("_FROZEN_SEAL_HELPER_BINDINGS") is not frozen:
        raise RuntimeError("FD adapter seal-helper registry drift")
    for binding in frozen:
        current = globals().get(binding.name)
        if current is not binding.function or current.__code__ is not binding.code:
            raise RuntimeError(f"FD adapter seal-helper function drift: {binding.name}")
        if current.__defaults__ is not binding.defaults:
            raise RuntimeError(f"FD adapter seal-helper defaults drift: {binding.name}")
        current_kwdefaults = (
            None
            if current.__kwdefaults__ is None
            else tuple(sorted(current.__kwdefaults__.items()))
        )
        if current_kwdefaults != binding.kwdefaults:
            raise RuntimeError(
                f"FD adapter seal-helper kwdefaults drift: {binding.name}"
            )


class FrozenParentCenteredFDAdapter:
    """Stateful, transactional callback adapter for one frozen parent field."""

    __slots__ = (
        "_calls",
        "_epsilon",
        "_evaluations",
        "_lock",
        "_max_target_count",
        "_parent_hash_getter",
        "_parent_hash_guard",
        "_parent_state_sha256",
        "_parent_token",
        "_seal",
        "_trace_sha256",
        "_velocity_evaluator",
        "_velocity_guard",
    )

    def __init__(
        self,
        velocity_evaluator: FrozenParentVelocityEvaluator,
        parent_state_sha256_getter: ParentStateSHA256Getter,
        *,
        epsilon: float,
        parent_token: str,
        max_target_count: int = DEFAULT_MAX_TARGET_COUNT,
    ) -> None:
        verify_runtime = _assert_runtime_bindings
        verify_callable = _assert_callable_guard
        verify_internal = _assert_internal_bindings
        verify_internal()
        verify_runtime()
        token = _validate_parent_token(parent_token)
        epsilon_value = _validate_epsilon(epsilon)
        cap = _validate_cap(max_target_count)
        velocity_guard = _freeze_callable_guard(
            "velocity_evaluator", velocity_evaluator
        )
        parent_hash_guard = _freeze_callable_guard(
            "parent_state_sha256_getter", parent_state_sha256_getter
        )
        verify_callable(velocity_guard)
        verify_callable(parent_hash_guard)
        parent_sha = parent_state_sha256_getter()
        verify_internal()
        verify_runtime()
        verify_callable(velocity_guard)
        verify_callable(parent_hash_guard)
        parent_sha = _validate_sha256("parent_state_sha256_getter result", parent_sha)

        self._velocity_evaluator = velocity_evaluator
        self._parent_hash_getter = parent_state_sha256_getter
        self._velocity_guard = velocity_guard
        self._parent_hash_guard = parent_hash_guard
        self._epsilon = epsilon_value
        self._parent_token = token
        self._parent_state_sha256 = parent_sha
        self._max_target_count = cap
        self._calls: list[FDVelocityCall] = []
        self._evaluations: list[FDEvaluationRecord] = []
        self._trace_sha256 = _LEDGER_GENESIS
        self._lock = threading.RLock()
        self._seal = _make_adapter_seal(self)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def parent_token(self) -> str:
        return self._parent_token

    @property
    def parent_state_sha256(self) -> str:
        return self._parent_state_sha256

    @property
    def ledger(self) -> FDCallLedger:
        return self.snapshot()

    def _preflight(
        self,
        verify_runtime: Callable[[], None] = _assert_runtime_bindings,
        verify_callable: Callable[[_CallableGuard], None] = _assert_callable_guard,
        verify_internal: Callable[[], None] = _assert_internal_bindings,
        verify_seal_helpers: Callable[[], None] = _assert_seal_helper_bindings,
        verify_seal: Callable[[object], None] = _assert_adapter_seal,
    ) -> None:
        verify_internal()
        verify_seal_helpers()
        verify_seal(self)
        verify_runtime()
        verify_callable(self._velocity_guard)
        verify_callable(self._parent_hash_guard)

    def _current_parent_sha256(self) -> str:
        self._preflight()
        current = self._parent_hash_getter()
        self._preflight()
        return _validate_sha256("parent_state_sha256_getter result", current)

    def _assert_parent_unchanged(self) -> None:
        if self._current_parent_sha256() != self._parent_state_sha256:
            raise RuntimeError("frozen parent state hash drift")

    def _invoke_velocity(
        self,
        targets: FloatArray,
        *,
        evaluation_index: int,
        call_index: int,
        target_kind: str,
        sample_kind: str,
        axis: int,
        sign: int,
    ) -> tuple[FloatArray, FDVelocityCall]:
        trusted_preflight = self._preflight
        preflight_function = trusted_preflight.__func__
        verifier_functions = (
            preflight_function,
            *(preflight_function.__defaults__ or ()),
            _snapshot_fd_call_ledger_registry,
            _assert_fd_call_ledger_registry_unchanged,
        )
        verifier_bindings = tuple(
            (
                function,
                function.__code__,
                function.__defaults__,
                function.__kwdefaults__,
            )
            for function in verifier_functions
            if type(function) is FunctionType
        )
        snapshot_registry = _snapshot_fd_call_ledger_registry
        verify_registry_unchanged = _assert_fd_call_ledger_registry_unchanged
        parent_hash_getter = self._parent_hash_getter
        validate_sha = _validate_sha256
        trusted_preflight()
        registry_snapshot = snapshot_registry()

        def verify_after_callback() -> None:
            current_preflight = type(self).__dict__.get("_preflight")
            if (
                current_preflight is not preflight_function
                or preflight_function.__code__ is not verifier_bindings[0][1]
                or preflight_function.__defaults__ is not verifier_bindings[0][2]
                or preflight_function.__kwdefaults__ is not verifier_bindings[0][3]
            ):
                raise RuntimeError(
                    "FD adapter trusted preflight changed during callback"
                )
            for function, code, defaults, kwdefaults in verifier_bindings[1:]:
                if (
                    function.__code__ is not code
                    or function.__defaults__ is not defaults
                    or function.__kwdefaults__ is not kwdefaults
                ):
                    raise RuntimeError(
                        "FD adapter trusted verifier changed during callback"
                    )
            if (
                globals().get("_snapshot_fd_call_ledger_registry")
                is not snapshot_registry
                or globals().get("_assert_fd_call_ledger_registry_unchanged")
                is not verify_registry_unchanged
            ):
                raise RuntimeError(
                    "FD adapter ledger-registry verifier changed during callback"
                )
            trusted_preflight()
            verify_registry_unchanged(registry_snapshot)

        def verify_parent() -> None:
            verify_after_callback()
            current = parent_hash_getter()
            verify_after_callback()
            current = validate_sha("parent_state_sha256_getter result", current)
            if current != self._parent_state_sha256:
                raise RuntimeError("frozen parent state hash drift")

        verify_parent()
        verify_after_callback()
        response = self._velocity_evaluator(targets)
        verify_after_callback()
        verify_parent()
        if type(response) is not FrozenParentVelocity:
            raise ValueError(
                "velocity_evaluator must return an exact FrozenParentVelocity"
            )
        count = targets.shape[0]
        velocity = _require_readonly_float64(
            "frozen parent velocity", response.velocity, (count, 3)
        )
        target_sha = _array_sha256(targets)
        velocity_sha = _array_sha256(velocity)
        if response.target_sha256 != target_sha:
            raise ValueError("frozen parent response target hash mismatch")
        if response.velocity_sha256 != velocity_sha:
            raise ValueError("frozen parent response velocity hash mismatch")
        if response.parent_token != self._parent_token:
            raise ValueError("frozen parent response token mismatch")
        if response.parent_state_sha256 != self._parent_state_sha256:
            raise ValueError("frozen parent response state hash mismatch")
        expected_attestation = _attestation_sha256(
            target_sha256=target_sha,
            velocity_sha256=velocity_sha,
            parent_token=self._parent_token,
            parent_state_sha256=self._parent_state_sha256,
        )
        if response.attestation_sha256 != expected_attestation:
            raise ValueError("frozen parent response attestation hash mismatch")
        call_payload: dict[str, object] = {
            "attestation_sha256": expected_attestation,
            "axis": axis,
            "call_index": call_index,
            "callable_provenance_sha256": (
                self._velocity_guard.provenance.provenance_sha256
            ),
            "evaluation_index": evaluation_index,
            "kind": "fluxv-fd-velocity-call-v1",
            "parent_state_sha256": self._parent_state_sha256,
            "parent_token": self._parent_token,
            "sample_kind": sample_kind,
            "sign": sign,
            "target_count": count,
            "target_kind": target_kind,
            "target_sha256": target_sha,
            "velocity_sha256": velocity_sha,
        }
        call = FDVelocityCall(
            call_index=call_index,
            evaluation_index=evaluation_index,
            target_kind=target_kind,
            sample_kind=sample_kind,
            axis=axis,
            sign=sign,
            target_count=count,
            target_sha256=target_sha,
            velocity_sha256=velocity_sha,
            parent_token=self._parent_token,
            parent_state_sha256=self._parent_state_sha256,
            callable_provenance_sha256=(
                self._velocity_guard.provenance.provenance_sha256
            ),
            attestation_sha256=expected_attestation,
            call_sha256=_call_sha256(call_payload),
        )
        return velocity, call

    def _stencil_targets(self, targets: FloatArray) -> tuple[FloatArray, ...]:
        offsets: list[FloatArray] = []
        for axis in range(3):
            for sign in (-1, 1):
                shifted = np.array(targets, dtype=np.float64, copy=True, order="C")
                shifted[:, axis] += sign * self._epsilon
                if not np.all(np.isfinite(shifted)):
                    raise FloatingPointError("centered-FD offset target is non-finite")
                if np.any(shifted[:, axis] == targets[:, axis]):
                    raise FloatingPointError(
                        "epsilon does not perturb every centered-FD target"
                    )
                offsets.append(_frozen_float64("offset_targets", shifted, (3,)))
        return tuple(offsets)

    def _make_evaluation(
        self,
        *,
        evaluation_index: int,
        target_kind: str,
        target_count: int,
        source_state_sha256: str,
        target_sha256: str,
        velocity_sha256: str,
        jacobian_sha256: str | None,
        calls: tuple[FDVelocityCall, ...],
    ) -> FDEvaluationRecord:
        payload: dict[str, object] = {
            "call_sha256": [call.call_sha256 for call in calls],
            "epsilon": self._epsilon,
            "evaluation_index": evaluation_index,
            "kind": "fluxv-fd-evaluation-v1",
            "jacobian_sha256": jacobian_sha256,
            "parent_state_sha256": self._parent_state_sha256,
            "parent_token": self._parent_token,
            "source_state_sha256": source_state_sha256,
            "target_count": target_count,
            "target_kind": target_kind,
            "target_sha256": target_sha256,
            "velocity_sha256": velocity_sha256,
        }
        evaluation_sha = _call_sha256(payload)
        trace_sha = sha256(
            (self._trace_sha256 + evaluation_sha).encode("ascii")
        ).hexdigest()
        return FDEvaluationRecord(
            evaluation_index=evaluation_index,
            target_kind=target_kind,
            target_count=target_count,
            source_state_sha256=source_state_sha256,
            target_sha256=target_sha256,
            velocity_sha256=velocity_sha256,
            jacobian_sha256=jacobian_sha256,
            parent_token=self._parent_token,
            parent_state_sha256=self._parent_state_sha256,
            epsilon=self._epsilon,
            first_call_index=calls[0].call_index,
            last_call_index=calls[-1].call_index,
            evaluator_call_count=len(calls),
            evaluation_sha256=evaluation_sha,
            trace_sha256=trace_sha,
        )

    def _assert_capacity(
        self,
        additional_calls: int,
        max_evaluation_count: int = MAX_FD_EVALUATION_COUNT,
        max_call_count: int = MAX_FD_VELOCITY_CALL_COUNT,
    ) -> None:
        if len(self._evaluations) >= max_evaluation_count:
            raise RuntimeError("FD evaluation evidence cap exceeded")
        if len(self._calls) + additional_calls > max_call_count:
            raise RuntimeError("FD velocity-call evidence cap exceeded")

    def _commit(
        self,
        calls: tuple[FDVelocityCall, ...],
        evaluation: FDEvaluationRecord,
    ) -> None:
        self._calls.extend(calls)
        self._evaluations.append(evaluation)
        self._trace_sha256 = evaluation.trace_sha256

    def evaluate_physical(self, state: ParticleState) -> FDPhysicalResult:
        """Evaluate one physical center plus six offsets and construct ``U,J``."""

        with self._lock:
            self._preflight()
            self._assert_capacity(7)
            validated, count = _validate_state(state, self._max_target_count)
            targets = _frozen_float64("physical_targets", validated.positions, (3,))
            offsets = self._stencil_targets(targets)
            evaluation_index = len(self._evaluations) + 1
            next_call = len(self._calls) + 1
            local_calls: list[FDVelocityCall] = []
            center, call = self._invoke_velocity(
                targets,
                evaluation_index=evaluation_index,
                call_index=next_call,
                target_kind="physical",
                sample_kind="center",
                axis=-1,
                sign=0,
            )
            local_calls.append(call)
            offset_velocity: list[FloatArray] = []
            for offset_index, offset_targets in enumerate(offsets):
                axis = offset_index // 2
                sign = -1 if offset_index % 2 == 0 else 1
                velocity, offset_call = self._invoke_velocity(
                    offset_targets,
                    evaluation_index=evaluation_index,
                    call_index=next_call + len(local_calls),
                    target_kind="physical",
                    sample_kind="offset",
                    axis=axis,
                    sign=sign,
                )
                local_calls.append(offset_call)
                offset_velocity.append(velocity)
            jacobian = np.empty((count, 3, 3), dtype=np.float64)
            scale = 0.5 / self._epsilon
            for axis in range(3):
                negative = offset_velocity[2 * axis]
                positive = offset_velocity[2 * axis + 1]
                jacobian[:, :, axis] = (positive - negative) * scale
            if not np.all(np.isfinite(jacobian)):
                raise FloatingPointError("centered-FD Jacobian is non-finite")
            jacobian_frozen = _frozen_float64("jacobian", jacobian, (3, 3))
            field = make_ir_wrk3_field(
                validated,
                center,
                jacobian_frozen,
                parent_token=self._parent_token,
            )
            calls = tuple(local_calls)
            evaluation = self._make_evaluation(
                evaluation_index=evaluation_index,
                target_kind="physical",
                target_count=count,
                source_state_sha256=field.source_state_sha256,
                target_sha256=_array_sha256(targets),
                velocity_sha256=_array_sha256(center),
                jacobian_sha256=_array_sha256(jacobian_frozen),
                calls=calls,
            )
            self._commit(calls, evaluation)
            return FDPhysicalResult(field=field, evaluation=evaluation)

    def physical_field(self, state: ParticleState) -> IRWRK3Field:
        """IR-WRK3 physical callback returning only the attested field."""

        return self.evaluate_physical(state).field

    def evaluate_tracer(
        self,
        state: ParticleState,
        tracer_positions: ArrayLike,
        parent_token: str,
    ) -> FDTracerResult:
        """Evaluate tracer centers exactly once, without finite-difference offsets."""

        with self._lock:
            self._preflight()
            self._assert_capacity(1)
            token = _validate_parent_token(parent_token)
            if token != self._parent_token:
                raise ValueError("tracer callback parent token mismatch")
            # The configured cap governs points sent to the parent evaluator.
            # Tracer evaluation hashes, but does not submit, the physical source.
            validated, _ = _validate_state(state, self._max_target_count)
            targets = _prepare_targets(
                "tracer_positions", tracer_positions, self._max_target_count
            )
            evaluation_index = len(self._evaluations) + 1
            call_index = len(self._calls) + 1
            velocity, call = self._invoke_velocity(
                targets,
                evaluation_index=evaluation_index,
                call_index=call_index,
                target_kind="tracer",
                sample_kind="center",
                axis=-1,
                sign=0,
            )
            field = make_ir_wrk3_tracer_field(
                validated,
                targets,
                velocity,
                parent_token=self._parent_token,
            )
            calls = (call,)
            evaluation = self._make_evaluation(
                evaluation_index=evaluation_index,
                target_kind="tracer",
                target_count=targets.shape[0],
                source_state_sha256=field.source_state_sha256,
                target_sha256=_array_sha256(targets),
                velocity_sha256=_array_sha256(velocity),
                jacobian_sha256=None,
                calls=calls,
            )
            self._commit(calls, evaluation)
            return FDTracerResult(field=field, evaluation=evaluation)

    def tracer_field(
        self,
        state: ParticleState,
        tracer_positions: FloatArray,
        parent_token: str,
    ) -> IRWRK3TracerField:
        """IR-WRK3 tracer callback returning only the attested center field."""

        return self.evaluate_tracer(state, tracer_positions, parent_token).field

    def snapshot(self) -> FDCallLedger:
        """Return an immutable, point-in-time ledger snapshot."""

        with self._lock:
            _assert_adapter_seal(self)
            _assert_fd_call_ledger_capacity()
            physical = sum(
                record.target_kind == "physical" for record in self._evaluations
            )
            tracer = len(self._evaluations) - physical
            draft = FDCallLedger(
                epsilon=self._epsilon,
                parent_token=self._parent_token,
                parent_state_sha256=self._parent_state_sha256,
                velocity_evaluator_provenance=self._velocity_guard.provenance,
                parent_hash_getter_provenance=self._parent_hash_guard.provenance,
                calls=tuple(self._calls),
                evaluations=tuple(self._evaluations),
                physical_evaluation_count=physical,
                tracer_evaluation_count=tracer,
                center_call_count=physical + tracer,
                offset_call_count=6 * physical,
                evaluator_call_count=len(self._calls),
                trace_sha256=self._trace_sha256,
                ledger_sha256="",
            )
            ledger = FDCallLedger(
                epsilon=draft.epsilon,
                parent_token=draft.parent_token,
                parent_state_sha256=draft.parent_state_sha256,
                velocity_evaluator_provenance=(draft.velocity_evaluator_provenance),
                parent_hash_getter_provenance=(draft.parent_hash_getter_provenance),
                calls=draft.calls,
                evaluations=draft.evaluations,
                physical_evaluation_count=draft.physical_evaluation_count,
                tracer_evaluation_count=draft.tracer_evaluation_count,
                center_call_count=draft.center_call_count,
                offset_call_count=draft.offset_call_count,
                evaluator_call_count=draft.evaluator_call_count,
                trace_sha256=draft.trace_sha256,
                ledger_sha256=_ledger_sha256(draft),
            )
            _validate_fd_call_ledger_tree(ledger)
            _register_fd_call_ledger(ledger)
            return ledger


__all__ = (
    "CallableProvenance",
    "DEFAULT_MAX_TARGET_COUNT",
    "FDCallLedger",
    "FDEvaluationRecord",
    "FDPhysicalResult",
    "FDTracerResult",
    "FDVelocityCall",
    "FrozenParentCenteredFDAdapter",
    "FrozenParentVelocity",
    "FrozenParentVelocityEvaluator",
    "ParentStateSHA256Getter",
    "make_frozen_parent_velocity",
    "validate_fd_call_ledger",
)
