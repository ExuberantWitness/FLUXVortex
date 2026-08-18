"""Transactional live-boundary net-state owner for FluxV v5h9 M0/M1.

The v5h8 state is a gross, append-only diagnostic description.  This module
never calls its collapse helper.  Bootstrap accepts only a release-one state;
subsequent commits retain the parent's physical prefix, update exactly its
live gamma slice, and append only the newest physical suffix.

``particle_cap`` bounds the committed net state.  Because this API receives an
already-materialized v5h8 candidate, it cannot preflight or prevent allocation
of that upstream gross candidate.  It does check the net count before this
module materializes a new committed particle array.

Owners, states, events, and proposals are same-process capabilities.  Frozen
dataclasses and SHA-256 chains make their contents auditable; private weak
registries make equal-value objects produced with ``dataclasses.replace``
non-authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import dis
import hashlib
import importlib
from numbers import Real
from pathlib import Path
import sys
from threading import RLock
from types import CodeType, FunctionType, ModuleType
from typing import Any, Literal
import weakref
from weakref import WeakKeyDictionary, WeakSet

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ProposalStatus = Literal["compatible", "remesh_required"]

INTERFACE_ID = "fluxv-v5h9-live-boundary-net-owner-v2"
OPERATOR_ORDER = "pre_rhs_pre_lsrk3_pre_pedrizzetti"
V5H8_FILENAME = "fluxv_v5h8_incremental_sheet.py"
_V5H8_INTERFACE_ID = "fluxv-v5h8-incremental-connected-sheet-v1"
_RING_PHYSICAL_OWNER = "ring"
_DIAGNOSTIC_SHADOW_OWNER = "diagnostic_shadow"
_GENESIS_SHA256 = "0" * 64
_PHYSICAL_ROLES = frozenset(
    ("fresh_upstream", "fresh_downstream", "fresh_left_tip", "fresh_right_tip")
)
# Populated in place after every public entry point has been defined.  Its
# identity is part of the self-seal, so entry verification never trusts a
# caller-replaceable module-level seal object.
_OWNER_RUNTIME_ENVIRONMENT: dict[str, object] = {}


def _exact_positive_integer(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _exact_nonnegative_integer(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _nonempty_string(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _exact_finite_float(name: str, value: object) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _sha256_string(name: str, value: object) -> str:
    if type(value) is not str or len(value) != 64:
        raise TypeError(f"{name} must be a lowercase SHA-256 string")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _readonly_float64(name: str, value: ArrayLike, *, ndim: int) -> FloatArray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or original.dtype.kind == "b":
        raise TypeError(f"{name} must use a real numeric dtype")
    array = np.ascontiguousarray(np.asarray(original, dtype=np.float64))
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(
        array.shape
    )
    immutable.setflags(write=False)
    return immutable


def _digest_token(digest: Any, token: bytes) -> None:
    digest.update(len(token).to_bytes(8, "big"))
    digest.update(token)


def _hash_value(digest: Any, value: object) -> None:
    """Hash supported immutable values with explicit types and lengths."""

    if value is None:
        _digest_token(digest, b"none")
    elif type(value) is bool:
        _digest_token(digest, b"bool:1" if value else b"bool:0")
    elif type(value) is int:
        _digest_token(digest, b"int")
        _digest_token(digest, str(value).encode("ascii"))
    elif type(value) is float:
        _digest_token(digest, b"float")
        _digest_token(digest, value.hex().encode("ascii"))
    elif type(value) is str:
        _digest_token(digest, b"str")
        _digest_token(digest, value.encode("utf-8"))
    elif type(value) is tuple:
        _digest_token(digest, b"tuple")
        _digest_token(digest, str(len(value)).encode("ascii"))
        for item in value:
            _hash_value(digest, item)
    elif type(value) is np.ndarray:
        _digest_token(digest, b"ndarray")
        _digest_token(digest, value.dtype.str.encode("ascii"))
        _hash_value(digest, tuple(value.shape))
        _hash_value(digest, tuple(value.strides))
        _digest_token(digest, memoryview(value).cast("B"))
    elif isinstance(value, np.generic):
        _digest_token(digest, b"numpy-scalar")
        _digest_token(digest, value.dtype.str.encode("ascii"))
        _digest_token(digest, value.tobytes())
    elif is_dataclass(value) and not isinstance(value, type):
        _digest_token(digest, b"dataclass")
        _digest_token(digest, type(value).__qualname__.encode("utf-8"))
        for field in fields(value):
            _digest_token(digest, field.name.encode("utf-8"))
            _hash_value(digest, getattr(value, field.name))
    else:
        raise TypeError(f"unsupported digest value type: {type(value).__name__}")


def _exact_tree_equal(left: object, right: object) -> bool:
    """Compare trusted immutable records without coercive object equality."""

    if type(left) is not type(right):
        return False
    if left is None:
        return True
    if type(left) in (bool, int, str):
        return bool(left == right)
    if type(left) is float:
        return left.hex() == right.hex()
    if type(left) is tuple:
        return len(left) == len(right) and all(
            _exact_tree_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if is_dataclass(left) and not isinstance(left, type):
        left_fields = fields(left)
        right_fields = fields(right)
        return tuple(field.name for field in left_fields) == tuple(
            field.name for field in right_fields
        ) and all(
            _exact_tree_equal(
                _SAFE_OBJECT_GETATTRIBUTE(left, field.name),
                _SAFE_OBJECT_GETATTRIBUTE(right, field.name),
            )
            for field in left_fields
        )
    raise TypeError(f"unsupported exact comparison type: {type(left).__name__}")


def _exact_array_equal(left: object, right: object) -> bool:
    """Require identical ndarray metadata and physical bytes."""

    return (
        type(left) is np.ndarray
        and type(right) is np.ndarray
        and left.dtype.str == right.dtype.str
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.strides) == tuple(right.strides)
        and memoryview(left).cast("B") == memoryview(right).cast("B")
    )


def _digest_pairs(domain: str, pairs: tuple[tuple[str, object], ...]) -> str:
    digest = hashlib.sha256()
    _digest_token(digest, domain.encode("utf-8"))
    for label, value in pairs:
        _digest_token(digest, label.encode("utf-8"))
        _hash_value(digest, value)
    return digest.hexdigest()


def _record_digest(domain: str, value: object, excluded: frozenset[str]) -> str:
    return _digest_pairs(
        domain,
        tuple(
            (field.name, getattr(value, field.name))
            for field in fields(value)
            if field.name not in excluded
        ),
    )


_V5H8_PATH = Path(__file__).resolve().with_name(V5H8_FILENAME)
_V5H8_SOURCE_BYTES = _V5H8_PATH.read_bytes()
_V5H8_SOURCE_SHA256 = hashlib.sha256(_V5H8_SOURCE_BYTES).hexdigest()


def _top_level_code_manifest(source: bytes, filename: str) -> dict[str, CodeType]:
    module_code = compile(source, filename, "exec")
    result: dict[str, CodeType] = {}
    for constant in module_code.co_consts:
        if isinstance(constant, CodeType):
            result[constant.co_name] = constant
    return result


def _recursive_code_manifest(
    source: bytes, filename: str
) -> dict[tuple[str, int, int], CodeType]:
    result: dict[tuple[str, int, int], CodeType] = {}
    occurrences: dict[tuple[str, int], int] = {}

    def visit(code: CodeType) -> None:
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                base = (constant.co_qualname, constant.co_firstlineno)
                occurrence = occurrences.get(base, 0)
                occurrences[base] = occurrence + 1
                key = (*base, occurrence)
                result[key] = constant
                visit(constant)

    visit(compile(source, filename, "exec"))
    return result


def _top_level_defined_names(source: bytes, filename: str) -> frozenset[str]:
    """Return names the authoritative source itself installs in its module."""

    module_code = compile(source, filename, "exec")
    names: set[str] = set()
    for instruction in dis.get_instructions(module_code):
        if instruction.opname in ("DELETE_NAME", "DELETE_GLOBAL"):
            raise RuntimeError("authoritative source deletes a module binding")
        if instruction.opname in ("STORE_NAME", "STORE_GLOBAL"):
            if type(instruction.argval) is not str:
                raise RuntimeError("authoritative module binding name is invalid")
            names.add(instruction.argval)
    return frozenset(names)


@dataclass(frozen=True, slots=True)
class _GlobalResolution:
    function_name: str
    function: FunctionType
    global_name: str
    source: Literal["module", "builtins", "missing"]
    value: object
    builtins_mapping: dict[str, object]


@dataclass(frozen=True, slots=True)
class _FunctionRuntimeSeal:
    function_name: str
    function: FunctionType
    code: CodeType
    globals_mapping: dict[str, object]
    builtins_mapping: dict[str, object]
    global_resolutions: tuple[_GlobalResolution, ...]
    defaults: tuple[object, ...] | None
    kwdefaults: dict[str, object] | None
    kwdefault_items: tuple[tuple[str, object], ...]
    closure: tuple[object, ...] | None
    closure_values: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _ClassNamespaceSeal:
    class_name: str
    class_type: type
    namespace_names: frozenset[str]
    namespace_items: tuple[tuple[str, object], ...]
    member_runtime_seals: tuple[_FunctionRuntimeSeal, ...]


@dataclass(frozen=True, slots=True)
class _ModuleAttributeSeal:
    function_name: str
    global_name: str
    path: tuple[str, ...]
    root_module: ModuleType
    value: object


_MISSING_GLOBAL = object()
_SAFE_ANY = any
_CLASSMETHOD_TYPE = classmethod
_SAFE_FROZENSET = frozenset
_SAFE_ID = id
_SAFE_OBJECT_GETATTRIBUTE = object.__getattribute__
_PROPERTY_TYPE = property
_SAFE_SORTED = sorted
_STATICMETHOD_TYPE = staticmethod
_SAFE_TUPLE = tuple
_SAFE_TYPE = type
_SAFE_ZIP = zip


def _make_function_runtime_seal(
    function_name: str, function: FunctionType
) -> _FunctionRuntimeSeal:
    globals_mapping = function.__globals__
    builtins_mapping = function.__builtins__
    if _SAFE_TYPE(globals_mapping) is not dict:
        raise RuntimeError(f"runtime callable globals are invalid: {function_name}")
    if _SAFE_TYPE(builtins_mapping) is not dict:
        raise RuntimeError(f"runtime callable builtins are invalid: {function_name}")
    resolutions: list[_GlobalResolution] = []
    for global_name in _code_global_names(function.__code__):
        if global_name in globals_mapping:
            source: Literal["module", "builtins", "missing"] = "module"
            value = globals_mapping[global_name]
        elif global_name in builtins_mapping:
            source = "builtins"
            value = builtins_mapping[global_name]
        else:
            source = "missing"
            value = _MISSING_GLOBAL
        resolutions.append(
            _GlobalResolution(
                function_name=function_name,
                function=function,
                global_name=global_name,
                source=source,
                value=value,
                builtins_mapping=builtins_mapping,
            )
        )
    kwdefaults = function.__kwdefaults__
    if kwdefaults is not None and _SAFE_TYPE(kwdefaults) is not dict:
        raise RuntimeError(f"runtime closure kwdefaults are invalid: {function_name}")
    closure = function.__closure__
    return _FunctionRuntimeSeal(
        function_name=function_name,
        function=function,
        code=function.__code__,
        globals_mapping=globals_mapping,
        builtins_mapping=builtins_mapping,
        global_resolutions=_SAFE_TUPLE(resolutions),
        defaults=function.__defaults__,
        kwdefaults=kwdefaults,
        kwdefault_items=(
            () if kwdefaults is None else _SAFE_TUPLE(_SAFE_SORTED(kwdefaults.items()))
        ),
        closure=closure,
        closure_values=(
            ()
            if closure is None
            else _SAFE_TUPLE(cell.cell_contents for cell in closure)
        ),
    )


def _verify_function_runtime_seal(seal: _FunctionRuntimeSeal) -> None:
    function = seal.function
    if (
        function.__code__ is not seal.code
        or function.__globals__ is not seal.globals_mapping
        or function.__builtins__ is not seal.builtins_mapping
        or function.__defaults__ is not seal.defaults
        or function.__kwdefaults__ is not seal.kwdefaults
        or function.__closure__ is not seal.closure
    ):
        raise RuntimeError(f"runtime callable metadata changed: {seal.function_name}")
    for resolution in seal.global_resolutions:
        if resolution.global_name in seal.globals_mapping:
            source = "module"
            value = seal.globals_mapping[resolution.global_name]
        elif resolution.global_name in seal.builtins_mapping:
            source = "builtins"
            value = seal.builtins_mapping[resolution.global_name]
        else:
            source = "missing"
            value = _MISSING_GLOBAL
        if source != resolution.source or value is not resolution.value:
            raise RuntimeError(
                "runtime callable global changed: "
                f"{seal.function_name}:{resolution.global_name}"
            )
    kwdefaults = seal.kwdefaults
    if kwdefaults is not None:
        if _SAFE_TUPLE(_SAFE_SORTED(kwdefaults)) != _SAFE_TUPLE(
            name for name, _ in seal.kwdefault_items
        ):
            raise RuntimeError(
                f"runtime callable kwdefaults changed: {seal.function_name}"
            )
        if _SAFE_ANY(
            kwdefaults[name] is not expected for name, expected in seal.kwdefault_items
        ):
            raise RuntimeError(
                f"runtime callable kwdefaults changed: {seal.function_name}"
            )
    closure = seal.closure
    if closure is not None and _SAFE_ANY(
        cell.cell_contents is not expected
        for cell, expected in _SAFE_ZIP(closure, seal.closure_values, strict=True)
    ):
        raise RuntimeError(f"runtime callable closure changed: {seal.function_name}")


def _code_global_names(code: CodeType) -> tuple[str, ...]:
    names: set[str] = set()
    for instruction in dis.get_instructions(code):
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            if type(instruction.argval) is not str:
                raise RuntimeError("callable global bytecode name is invalid")
            names.add(instruction.argval)
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names.update(_code_global_names(constant))
    return tuple(sorted(names))


def _freeze_global_closure(
    module: ModuleType,
    root_names: tuple[str, ...],
    module_defined_names: frozenset[str],
) -> tuple[
    tuple[tuple[str, FunctionType], ...],
    tuple[_GlobalResolution, ...],
    tuple[_FunctionRuntimeSeal, ...],
]:
    """Freeze the actual global/builtin lookup closure of root functions."""

    namespace = module.__dict__
    queue: list[tuple[str, FunctionType]] = []
    for name in root_names:
        function = namespace.get(name)
        if _SAFE_TYPE(function) is not FunctionType:
            raise RuntimeError(f"runtime closure root is not a function: {name}")
        queue.append((name, function))
    seen: set[int] = set()
    functions: list[tuple[str, FunctionType]] = []
    resolutions: list[_GlobalResolution] = []
    runtime_seals: list[_FunctionRuntimeSeal] = []
    while queue:
        function_name, function = queue.pop()
        if id(function) in seen:
            continue
        seen.add(id(function))
        if function.__globals__ is not module.__dict__:
            raise RuntimeError(
                f"runtime closure function globals are invalid: {function_name}"
            )
        builtins_mapping = function.__builtins__
        if _SAFE_TYPE(builtins_mapping) is not dict:
            raise RuntimeError(
                f"runtime closure builtins mapping is invalid: {function_name}"
            )
        runtime_seals.append(_make_function_runtime_seal(function_name, function))
        functions.append((function_name, function))
        for global_name in _code_global_names(function.__code__):
            if global_name in module_defined_names:
                if global_name not in module.__dict__:
                    raise RuntimeError(
                        "authoritative module global is missing: "
                        f"{function_name}:{global_name}"
                    )
                source: Literal["module", "builtins", "missing"] = "module"
                value = module.__dict__[global_name]
            elif global_name in module.__dict__:
                # A name absent from the compiled module's STORE_NAME set is
                # a builtin fallback (or unresolved name).  Refuse to bless a
                # shadow that was injected before this owner was imported.
                raise RuntimeError(
                    "runtime builtin fallback is shadowed: "
                    f"{function_name}:{global_name}"
                )
            elif global_name in builtins_mapping:
                source = "builtins"
                value = builtins_mapping[global_name]
            else:
                source = "missing"
                value = _MISSING_GLOBAL
            resolutions.append(
                _GlobalResolution(
                    function_name=function_name,
                    function=function,
                    global_name=global_name,
                    source=source,
                    value=value,
                    builtins_mapping=builtins_mapping,
                )
            )
            if (
                source == "module"
                and _SAFE_TYPE(value) is FunctionType
                and value.__globals__ is module.__dict__
            ):
                queue.append((global_name, value))
    return (
        tuple(sorted(functions, key=lambda item: item[0])),
        tuple(
            sorted(
                resolutions,
                key=lambda item: (item.function_name, item.global_name),
            )
        ),
        tuple(sorted(runtime_seals, key=lambda item: item.function_name)),
    )


def _verify_global_closure(
    module: ModuleType,
    functions: tuple[tuple[str, FunctionType], ...],
    resolutions: tuple[_GlobalResolution, ...],
    runtime_seals: tuple[_FunctionRuntimeSeal, ...],
    code_manifest: dict[str, CodeType],
) -> None:
    namespace = module.__dict__
    for name, expected_function in functions:
        function = namespace.get(name)
        if (
            function is not expected_function
            or _SAFE_TYPE(function) is not FunctionType
            or function.__globals__ is not module.__dict__
            or function.__code__ != code_manifest[name]
        ):
            raise RuntimeError(f"runtime closure callable changed: {name}")
    for seal in runtime_seals:
        _verify_function_runtime_seal(seal)
    for resolution in resolutions:
        function = resolution.function
        if function.__builtins__ is not resolution.builtins_mapping:
            raise RuntimeError(
                f"runtime builtins mapping changed: {resolution.function_name}"
            )
        if resolution.global_name in module.__dict__:
            source = "module"
            value = module.__dict__[resolution.global_name]
        elif resolution.global_name in resolution.builtins_mapping:
            source = "builtins"
            value = resolution.builtins_mapping[resolution.global_name]
        else:
            source = "missing"
            value = _MISSING_GLOBAL
        if source != resolution.source or value is not resolution.value:
            raise RuntimeError(
                "runtime global resolution changed: "
                f"{resolution.function_name}:{resolution.global_name}"
            )


def _class_member_runtime_seals(
    class_name: str, namespace: object
) -> tuple[_FunctionRuntimeSeal, ...]:
    functions: list[tuple[str, FunctionType]] = []
    for member_name, member in namespace.items():
        if _SAFE_TYPE(member) is FunctionType:
            functions.append((f"{class_name}.{member_name}", member))
        elif _SAFE_TYPE(member) is _PROPERTY_TYPE:
            for accessor_name in ("fget", "fset", "fdel"):
                accessor = _SAFE_OBJECT_GETATTRIBUTE(member, accessor_name)
                if accessor is not None:
                    functions.append(
                        (
                            f"{class_name}.{member_name}.{accessor_name}",
                            accessor,
                        )
                    )
        elif _SAFE_TYPE(member) in (_STATICMETHOD_TYPE, _CLASSMETHOD_TYPE):
            functions.append(
                (
                    f"{class_name}.{member_name}.__func__",
                    _SAFE_OBJECT_GETATTRIBUTE(member, "__func__"),
                )
            )
    queue = list(_SAFE_SORTED(functions, key=lambda item: item[0]))
    seen: set[int] = set()
    seals: list[_FunctionRuntimeSeal] = []
    while queue:
        function_name, function = queue.pop()
        if _SAFE_ID(function) in seen:
            continue
        seen.add(_SAFE_ID(function))
        seal = _make_function_runtime_seal(function_name, function)
        seals.append(seal)
        for resolution in seal.global_resolutions:
            if _SAFE_TYPE(resolution.value) is FunctionType:
                queue.append(
                    (
                        f"{function_name}->{resolution.global_name}",
                        resolution.value,
                    )
                )
        for closure_index, value in enumerate(seal.closure_values):
            if _SAFE_TYPE(value) is FunctionType:
                queue.append(
                    (
                        f"{function_name}->closure[{closure_index}]",
                        value,
                    )
                )
    return _SAFE_TUPLE(_SAFE_SORTED(seals, key=lambda seal: seal.function_name))


def _freeze_class_namespaces(
    module: ModuleType,
    class_names: tuple[str, ...],
    module_defined_names: frozenset[str],
    source_code_manifest: dict[tuple[str, int, int], CodeType],
) -> tuple[_ClassNamespaceSeal, ...]:
    module_namespace = module.__dict__
    result: list[_ClassNamespaceSeal] = []
    for class_name in class_names:
        class_type = module_namespace.get(class_name)
        if _SAFE_TYPE(class_type) is not _SAFE_TYPE:
            raise RuntimeError(f"runtime closure class is invalid: {class_name}")
        namespace = _SAFE_TYPE.__getattribute__(class_type, "__dict__")
        member_runtime_seals = _class_member_runtime_seals(class_name, namespace)
        for runtime_seal in member_runtime_seals:
            if runtime_seal.globals_mapping is module_namespace:
                for resolution in runtime_seal.global_resolutions:
                    source_defined = resolution.global_name in module_defined_names
                    if source_defined != (resolution.source == "module"):
                        raise RuntimeError(
                            "runtime class callable global source is invalid: "
                            f"{runtime_seal.function_name}:"
                            f"{resolution.global_name}"
                        )
            logical_name = runtime_seal.function_name
            if "->" in logical_name:
                continue
            for suffix in (".fget", ".fset", ".fdel", ".__func__"):
                if logical_name.endswith(suffix):
                    logical_name = logical_name[: -len(suffix)]
                    break
            expected_codes = _SAFE_TUPLE(
                source_code
                for (qualname, _, _), source_code in source_code_manifest.items()
                if qualname == logical_name
            )
            if expected_codes and not _SAFE_ANY(
                runtime_seal.code == expected for expected in expected_codes
            ):
                raise RuntimeError(
                    "runtime class callable disagrees with source: "
                    f"{runtime_seal.function_name}"
                )
        result.append(
            _ClassNamespaceSeal(
                class_name=class_name,
                class_type=class_type,
                namespace_names=_SAFE_FROZENSET(namespace),
                namespace_items=_SAFE_TUPLE(
                    _SAFE_SORTED(namespace.items(), key=lambda item: item[0])
                ),
                member_runtime_seals=member_runtime_seals,
            )
        )
    return _SAFE_TUPLE(result)


def _verify_class_namespaces(
    module: ModuleType, seals: tuple[_ClassNamespaceSeal, ...]
) -> None:
    module_namespace = module.__dict__
    for seal in seals:
        class_type = module_namespace.get(seal.class_name)
        if class_type is not seal.class_type:
            raise RuntimeError(f"runtime closure class changed: {seal.class_name}")
        namespace = _SAFE_TYPE.__getattribute__(class_type, "__dict__")
        if _SAFE_FROZENSET(namespace) != seal.namespace_names:
            raise RuntimeError(
                f"runtime closure class namespace changed: {seal.class_name}"
            )
        if _SAFE_ANY(
            namespace.get(name) is not expected
            for name, expected in seal.namespace_items
        ):
            raise RuntimeError(
                f"runtime closure class binding changed: {seal.class_name}"
            )
        for runtime_seal in seal.member_runtime_seals:
            _verify_function_runtime_seal(runtime_seal)


def _code_module_attribute_paths(
    code: CodeType, module_global_names: frozenset[str]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    paths: set[tuple[str, tuple[str, ...]]] = set()
    instructions = _SAFE_TUPLE(dis.get_instructions(code))
    for index, instruction in enumerate(instructions):
        if (
            instruction.opname not in ("LOAD_GLOBAL", "LOAD_NAME")
            or instruction.argval not in module_global_names
        ):
            continue
        global_name = instruction.argval
        path: list[str] = []
        for following in instructions[index + 1 :]:
            if following.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                break
            if _SAFE_TYPE(following.argval) is not str:
                raise RuntimeError("runtime module attribute name is invalid")
            path.append(following.argval)
            paths.add((global_name, _SAFE_TUPLE(path)))
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            paths.update(_code_module_attribute_paths(constant, module_global_names))
    return _SAFE_TUPLE(_SAFE_SORTED(paths, key=lambda item: (item[0], item[1])))


def _freeze_module_attributes(
    functions: tuple[tuple[str, FunctionType], ...],
    resolutions: tuple[_GlobalResolution, ...],
) -> tuple[_ModuleAttributeSeal, ...]:
    modules_by_function: dict[FunctionType, dict[str, ModuleType]] = {}
    for resolution in resolutions:
        if _SAFE_TYPE(resolution.value) is ModuleType:
            modules_by_function.setdefault(resolution.function, {})[
                resolution.global_name
            ] = resolution.value
    seals: list[_ModuleAttributeSeal] = []
    for function_name, function in functions:
        module_globals = modules_by_function.get(function, {})
        paths = _code_module_attribute_paths(
            function.__code__, _SAFE_FROZENSET(module_globals)
        )
        for global_name, path in paths:
            root_module = module_globals[global_name]
            current: object = root_module
            static_module_path = True
            for attribute_name in path:
                if _SAFE_TYPE(current) is not ModuleType:
                    static_module_path = False
                    break
                if attribute_name == "__dict__":
                    current = _SAFE_OBJECT_GETATTRIBUTE(current, "__dict__")
                    continue
                namespace = current.__dict__
                if attribute_name not in namespace:
                    raise RuntimeError(
                        "runtime module attribute is unavailable: "
                        f"{function_name}:{global_name}.{'.'.join(path)}"
                    )
                current = namespace[attribute_name]
            if not static_module_path:
                continue
            seals.append(
                _ModuleAttributeSeal(
                    function_name=function_name,
                    global_name=global_name,
                    path=path,
                    root_module=root_module,
                    value=current,
                )
            )
    return _SAFE_TUPLE(
        _SAFE_SORTED(
            seals,
            key=lambda seal: (seal.function_name, seal.global_name, seal.path),
        )
    )


def _verify_module_attributes(seals: tuple[_ModuleAttributeSeal, ...]) -> None:
    for seal in seals:
        current: object = seal.root_module
        for attribute_name in seal.path:
            if _SAFE_TYPE(current) is not ModuleType:
                raise RuntimeError(
                    "runtime module attribute path changed: "
                    f"{seal.function_name}:{seal.global_name}.{'.'.join(seal.path)}"
                )
            if attribute_name == "__dict__":
                current = _SAFE_OBJECT_GETATTRIBUTE(current, "__dict__")
                continue
            namespace = current.__dict__
            if attribute_name not in namespace:
                raise RuntimeError(
                    "runtime module attribute disappeared: "
                    f"{seal.function_name}:{seal.global_name}.{'.'.join(seal.path)}"
                )
            current = namespace[attribute_name]
        if current is not seal.value:
            raise RuntimeError(
                "runtime module attribute changed: "
                f"{seal.function_name}:{seal.global_name}.{'.'.join(seal.path)}"
            )


_V5H8_CODE_MANIFEST = _top_level_code_manifest(_V5H8_SOURCE_BYTES, str(_V5H8_PATH))
_V5H8_RECURSIVE_CODE_MANIFEST = _recursive_code_manifest(
    _V5H8_SOURCE_BYTES, str(_V5H8_PATH)
)
_V5H8_MODULE_DEFINED_NAMES = _top_level_defined_names(
    _V5H8_SOURCE_BYTES, str(_V5H8_PATH)
)

# Pin the authoritative edge-bridge module independently of aliases exposed by
# a dynamically loaded v5h8 module.  Metadata such as ``__module__`` and
# ``__qualname__`` is spoofable; exact binding identity is not.
_EDGE_MODULE_NAME = "fluxvortex.rvpm_edge_bridge"
_EDGE_MODULE = importlib.import_module(_EDGE_MODULE_NAME)
if sys.modules.get(_EDGE_MODULE_NAME) is not _EDGE_MODULE:
    raise RuntimeError("authoritative edge-bridge module identity is unavailable")
_EDGE_NAMESPACE = _EDGE_MODULE.__dict__
if type(_EDGE_NAMESPACE.get("__file__")) is not str:
    raise RuntimeError("authoritative edge-bridge source path is unavailable")
_EDGE_PATH = Path(_EDGE_NAMESPACE["__file__"]).resolve()
_EDGE_SOURCE_BYTES = _EDGE_PATH.read_bytes()
_EDGE_SOURCE_SHA256 = hashlib.sha256(_EDGE_SOURCE_BYTES).hexdigest()
_EDGE_CODE_MANIFEST = _top_level_code_manifest(_EDGE_SOURCE_BYTES, str(_EDGE_PATH))
_EDGE_RECURSIVE_CODE_MANIFEST = _recursive_code_manifest(
    _EDGE_SOURCE_BYTES, str(_EDGE_PATH)
)
_EDGE_MODULE_DEFINED_NAMES = _top_level_defined_names(
    _EDGE_SOURCE_BYTES, str(_EDGE_PATH)
)
_EDGE_ROOT_FUNCTIONS = (
    "canonical_edge_key",
    "assemble_ring_edge_graph",
    "deposit_edge_graph_prescribed_sigma_and_spacing",
)
(
    _EDGE_REQUIRED_FUNCTION_BINDINGS,
    _EDGE_GLOBAL_RESOLUTIONS,
    _EDGE_FUNCTION_RUNTIME_SEALS,
) = _freeze_global_closure(
    _EDGE_MODULE,
    _EDGE_ROOT_FUNCTIONS,
    _EDGE_MODULE_DEFINED_NAMES,
)
_EDGE_REQUIRED_FUNCTIONS = tuple(name for name, _ in _EDGE_REQUIRED_FUNCTION_BINDINGS)
_EDGE_REQUIRED_CLASSES = (
    "BridgeNode",
    "DirectedRing",
    "EdgeIncidence",
    "EdgeLedger",
    "EdgeGraph",
    "ParticleLineage",
    "BridgeDiagnostics",
    "ShadowBridgeResult",
)
_EDGE_CLASS_NAMESPACE_SEALS = _freeze_class_namespaces(
    _EDGE_MODULE,
    _EDGE_REQUIRED_CLASSES,
    _EDGE_MODULE_DEFINED_NAMES,
    _EDGE_RECURSIVE_CODE_MANIFEST,
)
_EDGE_CLASS_FUNCTION_RUNTIME_SEALS = _SAFE_TUPLE(
    runtime_seal
    for class_seal in _EDGE_CLASS_NAMESPACE_SEALS
    for runtime_seal in class_seal.member_runtime_seals
)
_EDGE_CLASS_FUNCTION_BINDINGS = _SAFE_TUPLE(
    (runtime_seal.function_name, runtime_seal.function)
    for runtime_seal in _EDGE_CLASS_FUNCTION_RUNTIME_SEALS
)
_EDGE_CLASS_GLOBAL_RESOLUTIONS = _SAFE_TUPLE(
    resolution
    for runtime_seal in _EDGE_CLASS_FUNCTION_RUNTIME_SEALS
    for resolution in runtime_seal.global_resolutions
)
_EDGE_MODULE_ATTRIBUTE_SEALS = _freeze_module_attributes(
    _EDGE_REQUIRED_FUNCTION_BINDINGS + _EDGE_CLASS_FUNCTION_BINDINGS,
    _EDGE_GLOBAL_RESOLUTIONS + _EDGE_CLASS_GLOBAL_RESOLUTIONS,
)
_EDGE_FROZEN_NAMES = (
    *_EDGE_REQUIRED_FUNCTIONS,
    *_EDGE_REQUIRED_CLASSES,
    "FROZEN_OVERLAP_LAMBDA",
    "RING_PHYSICAL_OWNER",
    "DIAGNOSTIC_SHADOW_OWNER",
    "np",
    "fsum",
    "ceil",
)
_EDGE_FROZEN_BINDINGS = tuple(
    (name, _EDGE_NAMESPACE[name]) for name in _EDGE_FROZEN_NAMES
)
_EDGE_INCIDENCE_TYPE = _EDGE_NAMESPACE["EdgeIncidence"]
_EDGE_CANONICAL_EDGE_KEY = _EDGE_NAMESPACE["canonical_edge_key"]
_EDGE_PARTICLE_SCHEMA = "rvpm-edge-shadow-prescribed-sigma-spacing-v1"
_V5H8_REQUIRED_FUNCTIONS = (
    "_finite_real",
    "_positive_real",
    "_positive_integer",
    "_point",
    "_readonly_array",
    "_validate_readonly_particle_arrays",
    "make_panel",
    "_validate_panel",
    "_panel_node_ids",
    "_panel_edge_keys",
    "_deposit_panel",
    "_role_for_edge",
    "_snapshot",
    "_validate_state",
)
_V5H8_IMPORTED_BINDINGS = (
    "BridgeNode",
    "DirectedRing",
    "ParticleLineage",
    "assemble_ring_edge_graph",
    "canonical_edge_key",
    "deposit_edge_graph_prescribed_sigma_and_spacing",
)


@dataclass(frozen=True, slots=True)
class _OracleBinding:
    module: ModuleType
    module_namespace: dict[str, object]
    source_sha256: str
    state_type: type
    panel_type: type
    lineage_type: type
    particle_lineage_type: type
    edge_incidence_type: type
    validate_state: Any
    validate_panel: Any
    panel_edge_keys: Any
    deposit_panel: Any
    role_for_edge: Any
    frozen_bindings: tuple[tuple[str, object], ...]
    required_function_bindings: tuple[tuple[str, FunctionType], ...]
    global_resolutions: tuple[_GlobalResolution, ...]
    function_runtime_seals: tuple[_FunctionRuntimeSeal, ...]
    module_attribute_seals: tuple[_ModuleAttributeSeal, ...]
    class_namespace_seals: tuple[_ClassNamespaceSeal, ...]


_ORACLE_BINDINGS: dict[ModuleType, _OracleBinding] = {}


def _verify_owner_runtime_environment() -> None:
    environment = _OWNER_RUNTIME_ENVIRONMENT
    module = environment.get("module")
    namespace = environment.get("namespace")
    if _SAFE_TYPE(module) is not ModuleType or module.__dict__ is not namespace:
        raise RuntimeError("owner runtime module identity changed")
    module_name = namespace.get("__name__")
    if _SAFE_TYPE(module_name) is not str or sys.modules.get(module_name) is not module:
        raise RuntimeError("owner runtime module registration changed")
    file_value = namespace.get("__file__")
    if _SAFE_TYPE(file_value) is not str:
        raise RuntimeError("owner runtime source path is unavailable")
    _verify_global_closure(
        module,
        environment["required_function_bindings"],
        environment["global_resolutions"],
        environment["function_runtime_seals"],
        environment["code_manifest"],
    )
    _verify_module_attributes(environment["module_attribute_seals"])
    _verify_class_namespaces(module, environment["class_namespace_seals"])
    source_path = environment["source_path"]
    if Path(file_value).resolve() != source_path:
        raise RuntimeError("owner runtime source path changed")
    if (
        hashlib.sha256(source_path.read_bytes()).hexdigest()
        != environment["source_sha256"]
    ):
        raise RuntimeError("owner runtime source changed")


def _verify_edge_module() -> None:
    if sys.modules.get(_EDGE_MODULE_NAME) is not _EDGE_MODULE:
        raise RuntimeError("authoritative edge-bridge module identity changed")
    if _EDGE_MODULE.__dict__ is not _EDGE_NAMESPACE:
        raise RuntimeError("authoritative edge-bridge namespace identity changed")
    file_value = _EDGE_NAMESPACE.get("__file__")
    if _SAFE_TYPE(file_value) is not str:
        raise RuntimeError("authoritative edge-bridge source path is unavailable")
    # Verify all builtin/module resolutions before invoking helpers such as
    # Path that could themselves consult a process-wide changed builtin.
    _verify_global_closure(
        _EDGE_MODULE,
        _EDGE_REQUIRED_FUNCTION_BINDINGS,
        _EDGE_GLOBAL_RESOLUTIONS,
        _EDGE_FUNCTION_RUNTIME_SEALS,
        _EDGE_CODE_MANIFEST,
    )
    _verify_module_attributes(_EDGE_MODULE_ATTRIBUTE_SEALS)
    _verify_class_namespaces(_EDGE_MODULE, _EDGE_CLASS_NAMESPACE_SEALS)
    if Path(file_value).resolve() != _EDGE_PATH:
        raise RuntimeError("authoritative edge-bridge source path changed")
    if hashlib.sha256(_EDGE_PATH.read_bytes()).hexdigest() != _EDGE_SOURCE_SHA256:
        raise RuntimeError("authoritative edge-bridge source changed")
    for name, expected in _EDGE_FROZEN_BINDINGS:
        if _EDGE_NAMESPACE.get(name) is not expected:
            raise RuntimeError(f"authoritative edge-bridge binding changed: {name}")
    for name in _EDGE_REQUIRED_FUNCTIONS:
        function = _EDGE_NAMESPACE.get(name)
        if (
            _SAFE_TYPE(function) is not FunctionType
            or function.__globals__ is not _EDGE_MODULE.__dict__
            or function.__code__ != _EDGE_CODE_MANIFEST[name]
        ):
            raise RuntimeError(
                f"authoritative edge-bridge callable provenance is invalid: {name}"
            )
    for name in _EDGE_REQUIRED_CLASSES:
        class_type = _EDGE_NAMESPACE.get(name)
        if (
            _SAFE_TYPE(class_type) is not _SAFE_TYPE
            or class_type.__module__ != _EDGE_MODULE_NAME
            or class_type.__qualname__ != name
        ):
            raise RuntimeError(
                f"authoritative edge-bridge class provenance is invalid: {name}"
            )


def _verify_oracle_binding(binding: _OracleBinding) -> None:
    _verify_edge_module()
    module = binding.module
    namespace = module.__dict__
    if namespace is not binding.module_namespace:
        raise RuntimeError("v5h8 oracle namespace identity changed")
    module_name = namespace.get("__name__")
    if _SAFE_TYPE(module_name) is not str or sys.modules.get(module_name) is not module:
        raise RuntimeError("v5h8 oracle module identity changed")
    if _SAFE_TYPE(namespace.get("__file__")) is not str:
        raise RuntimeError("v5h8 oracle source path is unavailable")
    _verify_global_closure(
        module,
        binding.required_function_bindings,
        binding.global_resolutions,
        binding.function_runtime_seals,
        _V5H8_CODE_MANIFEST,
    )
    _verify_module_attributes(binding.module_attribute_seals)
    _verify_class_namespaces(module, binding.class_namespace_seals)
    if Path(namespace["__file__"]).resolve() != _V5H8_PATH:
        raise RuntimeError("v5h8 oracle source path changed")
    source = _V5H8_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != binding.source_sha256:
        raise RuntimeError("v5h8 oracle source changed")
    for name, expected in binding.frozen_bindings:
        if namespace.get(name) is not expected:
            raise RuntimeError(f"v5h8 oracle binding changed: {name}")
    for name in _V5H8_REQUIRED_FUNCTIONS:
        function = namespace.get(name)
        if (
            type(function) is not FunctionType
            or function.__globals__ is not module.__dict__
        ):
            raise RuntimeError(f"v5h8 oracle callable provenance is invalid: {name}")
        # Code-object structural equality is stable across an isolated import
        # of the same source.  Marshalled bytes are not: marshal preserves
        # reference/interning details that may differ despite equal code.
        if function.__code__ != _V5H8_CODE_MANIFEST[name]:
            raise RuntimeError(f"v5h8 oracle callable code changed: {name}")


def _oracle_binding(value: object, *, validate_state: bool = True) -> _OracleBinding:
    _verify_edge_module()
    value_type = type(value)
    module = sys.modules.get(value_type.__module__)
    if type(module) is not ModuleType:
        raise TypeError("gross state does not come from a live v5h8 oracle module")
    namespace = module.__dict__
    if type(namespace.get("__file__")) is not str:
        raise TypeError("gross state does not come from a live v5h8 oracle module")
    if Path(namespace["__file__"]).resolve() != _V5H8_PATH:
        raise TypeError("gross state does not come from the frozen v5h8 source")
    if hashlib.sha256(_V5H8_PATH.read_bytes()).hexdigest() != _V5H8_SOURCE_SHA256:
        raise RuntimeError("frozen v5h8 source changed after v5h9 import")
    if namespace.get("INTERFACE_ID") != _V5H8_INTERFACE_ID:
        raise RuntimeError("v5h8 interface binding is invalid")
    if namespace.get("RING_PHYSICAL_OWNER") != _RING_PHYSICAL_OWNER:
        raise RuntimeError("v5h8 physical-owner binding is invalid")
    if namespace.get("DIAGNOSTIC_SHADOW_OWNER") != _DIAGNOSTIC_SHADOW_OWNER:
        raise RuntimeError("v5h8 diagnostic-owner binding is invalid")
    state_type = namespace.get("IncrementalSheetState")
    panel_type = namespace.get("SheetPanel")
    lineage_type = namespace.get("IncrementalLineage")
    if not all(type(item) is type for item in (state_type, panel_type, lineage_type)):
        raise RuntimeError("v5h8 oracle schema bindings are invalid")
    if value_type is not state_type:
        raise TypeError("gross state must be an exact v5h8 IncrementalSheetState")
    for name in _V5H8_IMPORTED_BINDINGS:
        if namespace.get(name) is not _EDGE_NAMESPACE.get(name):
            raise RuntimeError(f"v5h8 transitive binding identity is invalid: {name}")
    frozen_names = (
        tuple(_V5H8_REQUIRED_FUNCTIONS)
        + tuple(_V5H8_IMPORTED_BINDINGS)
        + (
            "IncrementalSheetState",
            "SheetPanel",
            "IncrementalLineage",
            "np",
        )
    )
    binding = _ORACLE_BINDINGS.get(module)
    if binding is None:
        (
            required_function_bindings,
            global_resolutions,
            function_runtime_seals,
        ) = _freeze_global_closure(
            module,
            _V5H8_REQUIRED_FUNCTIONS,
            _V5H8_MODULE_DEFINED_NAMES,
        )
        class_names = set(("IncrementalSheetState", "SheetPanel", "IncrementalLineage"))
        module_name = namespace["__name__"]
        for resolution in global_resolutions:
            resolved_value = resolution.value
            if (
                _SAFE_TYPE(resolved_value) is _SAFE_TYPE
                and _SAFE_TYPE.__getattribute__(resolved_value, "__module__")
                == module_name
            ):
                class_names.add(resolution.global_name)
        class_namespace_seals = _freeze_class_namespaces(
            module,
            _SAFE_TUPLE(_SAFE_SORTED(class_names)),
            _V5H8_MODULE_DEFINED_NAMES,
            _V5H8_RECURSIVE_CODE_MANIFEST,
        )
        class_function_runtime_seals = _SAFE_TUPLE(
            runtime_seal
            for class_seal in class_namespace_seals
            for runtime_seal in class_seal.member_runtime_seals
        )
        class_function_bindings = _SAFE_TUPLE(
            (runtime_seal.function_name, runtime_seal.function)
            for runtime_seal in class_function_runtime_seals
        )
        class_global_resolutions = _SAFE_TUPLE(
            resolution
            for runtime_seal in class_function_runtime_seals
            for resolution in runtime_seal.global_resolutions
        )
        module_attribute_seals = _freeze_module_attributes(
            required_function_bindings + class_function_bindings,
            global_resolutions + class_global_resolutions,
        )
        binding = _OracleBinding(
            module=module,
            module_namespace=namespace,
            source_sha256=_V5H8_SOURCE_SHA256,
            state_type=state_type,
            panel_type=panel_type,
            lineage_type=lineage_type,
            particle_lineage_type=namespace["ParticleLineage"],
            edge_incidence_type=_EDGE_INCIDENCE_TYPE,
            validate_state=namespace["_validate_state"],
            validate_panel=namespace["_validate_panel"],
            panel_edge_keys=namespace["_panel_edge_keys"],
            deposit_panel=namespace["_deposit_panel"],
            role_for_edge=namespace["_role_for_edge"],
            frozen_bindings=tuple((name, namespace[name]) for name in frozen_names),
            required_function_bindings=required_function_bindings,
            global_resolutions=global_resolutions,
            function_runtime_seals=function_runtime_seals,
            module_attribute_seals=module_attribute_seals,
            class_namespace_seals=class_namespace_seals,
        )
        _ORACLE_BINDINGS[module] = binding
    _verify_oracle_binding(binding)
    if validate_state:
        binding.validate_state(value)
    return binding


def _validate_gross_container(value: object, binding: _OracleBinding) -> None:
    """Validate types/memory safety before classifying mechanical mismatch.

    The frozen v5h8 validator intentionally mixes container errors with exact
    support errors.  The former raise; the latter are valid remesh requests.
    """

    if type(value) is not binding.state_type:
        raise TypeError("candidate must be the bound exact v5h8 state type")
    arrays = (value.positions, value.gamma, value.sigma)
    for array in arrays:
        if type(array) is not np.ndarray or array.dtype != np.dtype(np.float64):
            raise TypeError("candidate arrays must be exact float64 ndarrays")
        if not array.flags.c_contiguous or array.flags.writeable:
            raise ValueError("candidate arrays must be immutable and C-contiguous")
        if not np.all(np.isfinite(array)):
            raise ValueError("candidate arrays must be finite")
    count = value.positions.shape[0]
    if (
        value.positions.shape != (count, 3)
        or value.gamma.shape != (count, 3)
        or value.sigma.shape != (count,)
        or np.any(value.sigma <= 0.0)
    ):
        raise ValueError("candidate particle-array shapes are invalid")
    for name in (
        "particle_ids",
        "lineage",
        "panels",
        "downstream_particle_indices",
        "clone_pairs",
    ):
        if type(getattr(value, name)) is not tuple:
            raise TypeError(f"candidate {name} must be an exact tuple")
    if len(value.particle_ids) != count or len(value.lineage) != count:
        raise ValueError("candidate arrays, IDs, and lineage disagree")
    if any(type(item) is not tuple for item in value.particle_ids):
        raise TypeError("candidate particle IDs must be exact tuples")
    if any(type(record) is not binding.lineage_type for record in value.lineage):
        raise TypeError("candidate lineage types are invalid")
    if not value.panels or any(
        type(panel) is not binding.panel_type for panel in value.panels
    ):
        raise TypeError("candidate panel types are invalid")
    if not value.downstream_particle_indices or any(
        type(index) is not int or not 0 <= index < count
        for index in value.downstream_particle_indices
    ):
        raise ValueError("candidate downstream indices are invalid")
    for pair in value.clone_pairs:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(index) is not int for index in pair)
        ):
            raise TypeError("candidate clone pairs must be exact integer pairs")
        old, clone = pair
        if not 0 <= old < clone < count:
            raise ValueError("candidate clone-pair indices are out of range")


def _gross_digest_components(
    *,
    interface_id: str,
    positions: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    particle_ids: tuple[tuple[Any, ...], ...],
    lineage: tuple[object, ...],
    panels: tuple[object, ...],
    downstream_particle_indices: tuple[int, ...],
    clone_pairs: tuple[tuple[int, int], ...],
    smoothing_radius_m: float,
    target_spacing_m: float,
) -> str:
    return _digest_pairs(
        "fluxv-v5h9-gross-state-v2",
        (
            ("interface_id", interface_id),
            ("positions", positions),
            ("gamma", gamma),
            ("sigma", sigma),
            ("particle_ids", particle_ids),
            ("lineage", lineage),
            ("panels", panels),
            ("downstream_particle_indices", downstream_particle_indices),
            ("clone_pairs", clone_pairs),
            ("smoothing_radius_m", smoothing_radius_m),
            ("target_spacing_m", target_spacing_m),
        ),
    )


def _gross_digest(value: object) -> str:
    return _gross_digest_components(
        interface_id=value.interface_id,
        positions=value.positions,
        gamma=value.gamma,
        sigma=value.sigma,
        particle_ids=value.particle_ids,
        lineage=value.lineage,
        panels=value.panels,
        downstream_particle_indices=value.downstream_particle_indices,
        clone_pairs=value.clone_pairs,
        smoothing_radius_m=value.smoothing_radius_m,
        target_spacing_m=value.target_spacing_m,
    )


def _gross_prefix_digest(value: object, prefix_count: int) -> str:
    parent_release = len(value.panels) - 1
    downstream = tuple(
        index
        for index in range(prefix_count)
        if value.lineage[index].role == "fresh_downstream"
        and value.lineage[index].release_index == parent_release
    )
    clone_pairs = tuple(pair for pair in value.clone_pairs if pair[1] < prefix_count)
    return _gross_digest_components(
        interface_id=value.interface_id,
        positions=value.positions[:prefix_count],
        gamma=value.gamma[:prefix_count],
        sigma=value.sigma[:prefix_count],
        particle_ids=tuple(value.particle_ids[:prefix_count]),
        lineage=tuple(value.lineage[:prefix_count]),
        panels=tuple(value.panels[:-1]),
        downstream_particle_indices=downstream,
        clone_pairs=clone_pairs,
        smoothing_radius_m=value.smoothing_radius_m,
        target_spacing_m=value.target_spacing_m,
    )


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class LiveBoundaryState:
    """Immutable canonical net particle state."""

    interface_id: str
    positions: FloatArray
    gamma: FloatArray
    sigma: FloatArray
    particle_ids: tuple[tuple[Any, ...], ...]
    lineage: tuple[object, ...]
    panels: tuple[object, ...]
    live_boundary_indices: tuple[int, ...]
    release_index: int
    circulation_m2_s: float
    state_epoch: int
    state_sha256: str
    clone_count: int = 0
    counter_particle_count: int = 0
    fresh_upstream_particle_count: int = 0


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class LiveBoundaryEvent:
    """Append-only before/add/after audit record for one zero-time release."""

    proposal_id: str
    release_index: int
    changed_indices: tuple[int, ...]
    gamma_previous_m2_s: float
    gamma_current_m2_s: float
    gamma_scale: float
    parent_state_sha256: str
    candidate_state_sha256: str
    before_gamma_sha256: str
    added_gamma_sha256: str
    after_gamma_sha256: str
    committed_state_sha256: str
    operator_order: str
    clone_count: int
    counter_particle_count: int
    fresh_upstream_particle_count: int
    remesh_count: int
    target_read_count: int
    ptera_call_count: int
    load_call_count: int
    feedback_write_count: int
    parent_write_count: int
    source_time_s: float
    parent_owner_sha256: str
    previous_event_sha256: str
    event_sha256: str


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class LiveBoundaryOwner:
    """Versioned unique owner; commits return a new live capability."""

    owner_id: str
    particle_cap: int
    epoch: int
    wing_id: str
    source_time_s: float
    state: LiveBoundaryState
    events: tuple[LiveBoundaryEvent, ...]
    root_source_time_s: float
    release_dt_s: float
    gross_particle_count: int
    gross_state_sha256: str
    oracle_source_sha256: str
    owner_sha256: str


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class UpdateProposal:
    """Immutable proposal; only a live compatible proposal may commit."""

    status: ProposalStatus
    proposal_id: str
    parent_epoch: int
    changed_indices: tuple[int, ...]
    planned_particle_count: int
    appended_particle_count: int
    first_mismatch: str | None
    gamma_previous_m2_s: float
    gamma_current_m2_s: float
    gamma_scale: float
    wing_id: str
    source_time_s: float
    parent_state_sha256: str
    candidate_state_sha256: str
    proposal_sha256: str
    candidate_state: object
    clone_count: int = 0
    counter_particle_count: int = 0
    fresh_upstream_particle_count: int = 0
    parent_owner_sha256: str = ""
    oracle_source_sha256: str = ""
    candidate_identity: int = 0


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Non-authoritative convenience view of one committed owner generation.

    Callers that cross a trust boundary must validate ``owner`` with
    :func:`validate_live_boundary_owner` and require
    ``owner.state is state`` and ``owner.events[-1] is event``.  CommitResult
    itself is deliberately not accepted as an authoritative API input.
    """

    owner: LiveBoundaryOwner
    state: LiveBoundaryState
    event: LiveBoundaryEvent


@dataclass(frozen=True, slots=True)
class _CommitPlan:
    parent_owner: weakref.ReferenceType[LiveBoundaryOwner]
    binding: _OracleBinding
    candidate_state: object
    candidate_sha256: str
    suffix_indices: tuple[int, ...]
    suffix_positions: FloatArray
    suffix_gamma: FloatArray
    suffix_sigma: FloatArray
    suffix_particle_ids: tuple[tuple[Any, ...], ...]
    suffix_lineage: tuple[object, ...]
    latest_panel: object
    new_boundary_indices: tuple[int, ...]
    issued_proposal: UpdateProposal
    issued_proposal_sha256: str


def _state_digest(value: LiveBoundaryState) -> str:
    return _record_digest(
        "fluxv-v5h9-live-state-v2", value, frozenset(("state_sha256",))
    )


def _event_digest(value: LiveBoundaryEvent) -> str:
    return _record_digest(
        "fluxv-v5h9-live-event-v2", value, frozenset(("event_sha256",))
    )


def _owner_digest(value: LiveBoundaryOwner) -> str:
    return _record_digest(
        "fluxv-v5h9-live-owner-v2", value, frozenset(("owner_sha256",))
    )


def _proposal_digest(value: UpdateProposal) -> str:
    pairs: list[tuple[str, object]] = []
    for field in fields(value):
        if field.name == "proposal_sha256":
            continue
        if field.name == "candidate_state":
            pairs.append(
                ("candidate_state_semantic_sha256", value.candidate_state_sha256)
            )
            continue
        pairs.append((field.name, getattr(value, field.name)))
    return _digest_pairs("fluxv-v5h9-live-proposal-v2", tuple(pairs))


def _proposal_matches_issued(current: UpdateProposal, issued: UpdateProposal) -> bool:
    """Compare every proposal field with exact types and candidate identity."""

    current_fields = fields(current)
    issued_fields = fields(issued)
    if tuple(field.name for field in current_fields) != tuple(
        field.name for field in issued_fields
    ):
        return False
    for field in current_fields:
        actual = getattr(current, field.name)
        expected = getattr(issued, field.name)
        if type(actual) is not type(expected):
            return False
        if field.name == "candidate_state":
            if actual is not expected:
                return False
        elif actual != expected:
            return False
    return True


def _issued_proposal(proposal: UpdateProposal, plan: _CommitPlan) -> UpdateProposal:
    """Authenticate a live proposal against its private issuance snapshot."""

    issued = plan.issued_proposal
    issued_sha = plan.issued_proposal_sha256
    if type(issued) is not UpdateProposal:
        raise RuntimeError("proposal issuance snapshot type is invalid")
    if issued.proposal_sha256 != issued_sha:
        raise RuntimeError("proposal issuance digest changed")
    if _proposal_digest(issued) != issued_sha:
        raise RuntimeError("proposal issuance snapshot is invalid")
    if not _proposal_matches_issued(proposal, issued):
        raise RuntimeError("proposal fields differ from their issued snapshot")
    # Exact field types are established before any equality comparison on a
    # caller-mutated digest field, so a foreign object cannot run its own
    # comparator while pretending to be an issuer digest.
    if proposal.proposal_sha256 != issued_sha:
        raise RuntimeError("proposal differs from its issued digest")
    # Recompute only after identity, original digest, and the complete field
    # snapshot have matched.  A caller-visible digest helper is not an issuer.
    if _proposal_digest(proposal) != issued_sha:
        raise RuntimeError("proposal digest is invalid")
    return issued


def _array_sha256(value: np.ndarray) -> str:
    return _digest_pairs("fluxv-v5h9-array-v2", (("array", value),))


def _validate_immutable_scalar_tree(name: str, value: object) -> None:
    if type(value) in (str, int, bool, type(None)):
        return
    if type(value) is float:
        _exact_finite_float(name, value)
        return
    if type(value) is tuple:
        for item in value:
            _validate_immutable_scalar_tree(name, item)
        return
    raise TypeError(f"{name} must contain only exact immutable scalar values")


def _validate_stable_id(name: str, value: object) -> None:
    if type(value) is int:
        return
    if type(value) is str and value:
        return
    raise TypeError(f"{name} must be an exact integer or nonempty string")


def _validate_physical_lineage_record(
    record: object,
    particle_id: tuple[Any, ...],
    sigma_m: object,
    panel_circulation_m2_s: object,
    binding: _OracleBinding,
) -> None:
    """Validate a v5h8 birth record without invoking caller equality."""

    if type(record) is not binding.lineage_type:
        raise TypeError("state lineage must use the bound exact v5h8 lineage type")
    if not _exact_tree_equal(record.particle_id, particle_id):
        raise ValueError("state lineage particle ID disagrees with particle state")
    if type(record.release_index) is not int or record.release_index < 1:
        raise ValueError("state lineage release index is invalid")
    if type(record.role) is not str or record.role not in _PHYSICAL_ROLES:
        raise ValueError("state lineage role is not physical")
    if record.role == "fresh_upstream" and record.release_index != 1:
        raise ValueError("only release one may contain fresh-upstream rows")
    if record.parent_particle_id is not None:
        raise ValueError("physical state lineage must not name a clone parent")
    if type(record.physical_owner) is not str or (
        record.physical_owner != _RING_PHYSICAL_OWNER
    ):
        raise ValueError("ring must remain the physical owner")
    if type(record.owner_state) is not str or (
        record.owner_state != _DIAGNOSTIC_SHADOW_OWNER
    ):
        raise ValueError("particle state must remain diagnostic_shadow")
    if (
        type(record.source_edge) is not tuple
        or len(record.source_edge) != 2
        or any(
            type(item) not in (str, int) or type(item) is bool
            for item in record.source_edge
        )
    ):
        raise TypeError("state lineage source_edge is invalid")

    source = record.source_lineage
    if type(source) is not binding.particle_lineage_type:
        raise TypeError("physical lineage must retain exact bridge provenance")
    if not _exact_tree_equal(source.particle_id, record.particle_id):
        raise ValueError("bridge particle ID disagrees with incremental lineage")
    if not _exact_tree_equal(source.source_edge, record.source_edge):
        raise ValueError("bridge source edge disagrees with incremental lineage")
    if (
        type(source.subdivision_index) is not int
        or type(source.subdivision_count) is not int
        or not 0 <= source.subdivision_index < source.subdivision_count
        or type(source.ring_incidences) is not tuple
        or len(source.ring_incidences) != 1
        or type(source.step) is not int
        or source.step != record.release_index
        or type(source.physical_owner) is not str
        or source.physical_owner != _RING_PHYSICAL_OWNER
        or type(source.owner_state) is not str
        or source.owner_state != _DIAGNOSTIC_SHADOW_OWNER
    ):
        raise ValueError("bridge birth provenance is inconsistent")

    incidence = source.ring_incidences[0]
    if type(incidence) is not binding.edge_incidence_type:
        raise TypeError("ring incidence must use the authoritative exact type")
    _validate_stable_id("ring incidence ring_id", incidence.ring_id)
    _validate_stable_id("ring incidence source_start_id", incidence.source_start_id)
    _validate_stable_id("ring incidence source_end_id", incidence.source_end_id)
    if type(incidence.traversal_index) is not int or not (
        0 <= incidence.traversal_index < 4
    ):
        raise ValueError("ring incidence traversal index is invalid")
    if type(incidence.canonical_sign) is not int or incidence.canonical_sign not in (
        -1,
        1,
    ):
        raise ValueError("ring incidence canonical sign is invalid")
    ring_circulation = _exact_finite_float(
        "ring incidence ring_circulation", incidence.ring_circulation
    )
    signed_circulation = _exact_finite_float(
        "ring incidence signed_circulation", incidence.signed_circulation
    )
    panel_circulation = _exact_finite_float(
        "lineage panel circulation", panel_circulation_m2_s
    )
    if not _exact_tree_equal(ring_circulation, panel_circulation):
        raise ValueError("ring incidence circulation disagrees with its panel")

    prefix = f"v5h8:release:{record.release_index}"
    upstream_left = f"{prefix}:upstream:left"
    upstream_right = f"{prefix}:upstream:right"
    downstream_left = f"{prefix}:downstream:left"
    downstream_right = f"{prefix}:downstream:right"
    expected_traversals = (
        (upstream_left, upstream_right, "fresh_upstream"),
        (upstream_right, downstream_right, "fresh_right_tip"),
        (downstream_right, downstream_left, "fresh_downstream"),
        (downstream_left, upstream_left, "fresh_left_tip"),
    )
    expected_start, expected_end, expected_role = expected_traversals[
        incidence.traversal_index
    ]
    if not _exact_tree_equal(incidence.ring_id, f"v5h8:ring:{record.release_index}"):
        raise ValueError("ring incidence ring ID is invalid")
    if not _exact_tree_equal(incidence.source_start_id, expected_start) or not (
        _exact_tree_equal(incidence.source_end_id, expected_end)
    ):
        raise ValueError("ring incidence traversal endpoints are invalid")
    expected_edge = _EDGE_CANONICAL_EDGE_KEY(expected_start, expected_end)
    if not _exact_tree_equal(record.source_edge, expected_edge):
        raise ValueError("lineage source edge disagrees with ring traversal")
    if record.role != expected_role:
        raise ValueError("lineage role disagrees with ring traversal")
    expected_sign = 1 if (expected_start, expected_end) == expected_edge else -1
    if incidence.canonical_sign != expected_sign:
        raise ValueError("ring incidence sign disagrees with canonical traversal")
    if not _exact_tree_equal(signed_circulation, expected_sign * ring_circulation):
        raise ValueError("ring incidence signed circulation is inconsistent")

    incidence_signature = (
        (
            incidence.ring_id,
            incidence.traversal_index,
            incidence.source_start_id,
            incidence.source_end_id,
            incidence.canonical_sign,
            incidence.ring_circulation,
            incidence.signed_circulation,
        ),
    )
    sigma = (
        _exact_finite_float("particle ID smoothing radius", particle_id[4])
        if (type(particle_id) is tuple and len(particle_id) == 8)
        else None
    )
    if sigma is None or not _exact_tree_equal(sigma, float(sigma_m)):
        raise ValueError("particle ID smoothing radius is invalid")
    expected_particle_id = (
        _EDGE_PARTICLE_SCHEMA,
        expected_edge,
        source.subdivision_index,
        source.subdivision_count,
        sigma,
        incidence_signature,
        record.release_index,
        _DIAGNOSTIC_SHADOW_OWNER,
    )
    if not _exact_tree_equal(particle_id, expected_particle_id):
        raise ValueError("particle ID does not encode exact birth provenance")


def _validate_live_state_contents(
    value: object, binding: _OracleBinding, *, verify_oracle: bool = True
) -> LiveBoundaryState:
    if type(value) is not LiveBoundaryState:
        raise TypeError("state must be an exact LiveBoundaryState")
    if type(value.interface_id) is not str or value.interface_id != INTERFACE_ID:
        raise ValueError("state interface_id is invalid")
    for name, array, ndim in (
        ("positions", value.positions, 2),
        ("gamma", value.gamma, 2),
        ("sigma", value.sigma, 1),
    ):
        if type(array) is not np.ndarray or array.dtype != np.dtype(np.float64):
            raise TypeError(f"state {name} must be an exact float64 ndarray")
        if array.ndim != ndim or not array.flags.c_contiguous or array.flags.writeable:
            raise ValueError(f"state {name} must be immutable and C-contiguous")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"state {name} must be finite")
    count = value.positions.shape[0]
    if value.positions.shape != (count, 3) or value.gamma.shape != (count, 3):
        raise ValueError("state positions/gamma must have shape (n, 3)")
    if value.sigma.shape != (count,) or np.any(value.sigma <= 0.0):
        raise ValueError("state sigma must be positive with shape (n,)")
    if (
        type(value.particle_ids) is not tuple
        or type(value.lineage) is not tuple
        or type(value.panels) is not tuple
    ):
        raise TypeError("state IDs, lineage, and panels must be exact tuples")
    if len(value.particle_ids) != count or len(value.lineage) != count:
        raise ValueError("state arrays, IDs, and lineage disagree")
    for particle_id in value.particle_ids:
        if type(particle_id) is not tuple:
            raise TypeError("state particle IDs must be exact tuples")
        _validate_immutable_scalar_tree("particle_id", particle_id)
    try:
        unique_count = len(set(value.particle_ids))
    except TypeError as error:
        raise TypeError("state particle IDs must be hashable") from error
    if unique_count != count:
        raise ValueError("state particle IDs must be unique")
    if type(value.release_index) is not int or value.release_index < 1:
        raise ValueError("state release_index is invalid")
    if type(value.state_epoch) is not int or value.state_epoch < 1:
        raise ValueError("state epoch is invalid")
    if (
        value.release_index != value.state_epoch
        or len(value.panels) != value.release_index
    ):
        raise ValueError("state release, epoch, and panel count disagree")
    circulation = _exact_finite_float("state circulation", value.circulation_m2_s)
    for release_index, panel in enumerate(value.panels, start=1):
        if type(panel) is not binding.panel_type:
            raise TypeError("state panels must use the bound exact v5h8 panel type")
        if type(panel.release_index) is not int or panel.release_index != release_index:
            raise ValueError("state panel release index is invalid")
        if verify_oracle:
            binding.validate_panel(panel, expected_release_index=release_index)
    if value.panels[-1].circulation_m2_s != circulation:
        raise ValueError("state circulation disagrees with its latest panel")

    expected_boundary: list[int] = []
    for index, record in enumerate(value.lineage):
        if type(record) is not binding.lineage_type:
            raise TypeError("state lineage must use the bound exact v5h8 lineage type")
        if type(record.release_index) is not int or not (
            1 <= record.release_index <= value.release_index
        ):
            raise ValueError("state lineage release index is invalid")
        _validate_physical_lineage_record(
            record,
            value.particle_ids[index],
            value.sigma[index],
            value.panels[record.release_index - 1].circulation_m2_s,
            binding,
        )
        if verify_oracle:
            panel_keys = binding.panel_edge_keys(value.panels[record.release_index - 1])
            if record.role != binding.role_for_edge(record.source_edge, panel_keys):
                raise ValueError("state lineage role disagrees with its panel edge")
        if (
            record.role == "fresh_downstream"
            and record.release_index == value.release_index
        ):
            expected_boundary.append(index)

    if (
        type(value.live_boundary_indices) is not tuple
        or not value.live_boundary_indices
    ):
        raise ValueError("state live boundary must be a nonempty exact tuple")
    if any(type(index) is not int for index in value.live_boundary_indices):
        raise TypeError("state live boundary indices must be exact integers")
    if value.live_boundary_indices != tuple(expected_boundary):
        raise ValueError(
            "state live boundary does not identify the latest downstream rows"
        )
    for name in (
        "clone_count",
        "counter_particle_count",
        "fresh_upstream_particle_count",
    ):
        if _exact_nonnegative_integer(f"state {name}", getattr(value, name)) != 0:
            raise ValueError("committed physical state must contain zero gross rows")
    expected_sha = _state_digest(value)
    if _sha256_string("state_sha256", value.state_sha256) != expected_sha:
        raise ValueError("state SHA-256 is invalid")
    return value


def _validate_event_contents(
    event: object,
    *,
    owner: LiveBoundaryOwner,
    expected_release: int,
    previous_event_sha256: str,
    previous_committed_state_sha256: str | None,
) -> LiveBoundaryEvent:
    if type(event) is not LiveBoundaryEvent:
        raise TypeError("owner events must be exact LiveBoundaryEvent objects")
    _nonempty_string("event proposal_id", event.proposal_id)
    if type(event.release_index) is not int or event.release_index != expected_release:
        raise ValueError("event release index is invalid")
    if type(event.changed_indices) is not tuple or any(
        type(index) is not int for index in event.changed_indices
    ):
        raise TypeError("event changed_indices must be exact integers in a tuple")
    expected_changed = tuple(
        index
        for index, record in enumerate(owner.state.lineage)
        if record.role == "fresh_downstream"
        and record.release_index == expected_release - 1
    )
    if event.changed_indices != expected_changed:
        raise ValueError("event changed indices are invalid")
    previous = _exact_finite_float(
        "event gamma_previous_m2_s", event.gamma_previous_m2_s
    )
    current = _exact_finite_float("event gamma_current_m2_s", event.gamma_current_m2_s)
    scale = _exact_finite_float("event gamma_scale", event.gamma_scale)
    if (
        previous != owner.state.panels[expected_release - 2].circulation_m2_s
        or current != owner.state.panels[expected_release - 1].circulation_m2_s
        or scale != 1.0 - current / previous
    ):
        raise ValueError("event circulation facts are invalid")
    expected_time = (
        owner.root_source_time_s + (expected_release - 1) * owner.release_dt_s
    )
    if _exact_finite_float("event source_time_s", event.source_time_s) != expected_time:
        raise ValueError("event source time violates the frozen cadence")
    for name in (
        "parent_state_sha256",
        "candidate_state_sha256",
        "before_gamma_sha256",
        "added_gamma_sha256",
        "after_gamma_sha256",
        "committed_state_sha256",
        "parent_owner_sha256",
        "previous_event_sha256",
        "event_sha256",
    ):
        _sha256_string(f"event {name}", getattr(event, name))
    if event.previous_event_sha256 != previous_event_sha256:
        raise ValueError("event digest chain is broken")
    if (
        previous_committed_state_sha256 is not None
        and event.parent_state_sha256 != previous_committed_state_sha256
    ):
        raise ValueError("event state chain is broken")
    if type(event.operator_order) is not str or event.operator_order != OPERATOR_ORDER:
        raise ValueError("event operator order is invalid")
    for name in (
        "clone_count",
        "counter_particle_count",
        "fresh_upstream_particle_count",
        "remesh_count",
        "target_read_count",
        "ptera_call_count",
        "load_call_count",
        "feedback_write_count",
        "parent_write_count",
    ):
        if _exact_nonnegative_integer(f"event {name}", getattr(event, name)) != 0:
            raise ValueError("committed event counters must be exactly zero")
    if _event_digest(event) != event.event_sha256:
        raise ValueError("event SHA-256 is invalid")
    return event


def _validate_owner_contents(
    value: object, binding: _OracleBinding, *, verify_oracle: bool = True
) -> LiveBoundaryOwner:
    if type(value) is not LiveBoundaryOwner:
        raise TypeError("owner must be an exact LiveBoundaryOwner")
    if verify_oracle:
        _verify_oracle_binding(binding)
    _nonempty_string("owner_id", value.owner_id)
    cap = _exact_positive_integer("particle_cap", value.particle_cap)
    if type(value.epoch) is not int or value.epoch < 1:
        raise ValueError("owner epoch is invalid")
    _nonempty_string("wing_id", value.wing_id)
    source_time = _exact_finite_float("owner source_time_s", value.source_time_s)
    root_time = _exact_finite_float(
        "owner root_source_time_s", value.root_source_time_s
    )
    release_dt = _exact_finite_float("owner release_dt_s", value.release_dt_s)
    if root_time <= 0.0 or release_dt <= 0.0:
        raise ValueError("owner source cadence must be positive")
    if source_time != root_time + (value.epoch - 1) * release_dt:
        raise ValueError("owner source time violates the frozen cadence")
    state = _validate_live_state_contents(
        value.state, binding, verify_oracle=verify_oracle
    )
    if value.epoch != state.state_epoch:
        raise ValueError("owner and state epochs disagree")
    if cap < state.positions.shape[0]:
        raise ValueError("owner particle cap is smaller than its state")
    _exact_positive_integer("gross_particle_count", value.gross_particle_count)
    _sha256_string("gross_state_sha256", value.gross_state_sha256)
    if value.oracle_source_sha256 != binding.source_sha256:
        raise ValueError("owner oracle source digest is invalid")
    _sha256_string("oracle_source_sha256", value.oracle_source_sha256)
    if type(value.events) is not tuple or len(value.events) != value.epoch - 1:
        raise ValueError("owner event history disagrees with its epoch")
    previous_event_sha = _GENESIS_SHA256
    previous_state_sha: str | None = None
    for release_index, event in enumerate(value.events, start=2):
        _validate_event_contents(
            event,
            owner=value,
            expected_release=release_index,
            previous_event_sha256=previous_event_sha,
            previous_committed_state_sha256=previous_state_sha,
        )
        previous_event_sha = event.event_sha256
        previous_state_sha = event.committed_state_sha256
    if value.events and value.events[-1].committed_state_sha256 != state.state_sha256:
        raise ValueError("owner latest event does not bind its state")
    if _sha256_string("owner_sha256", value.owner_sha256) != _owner_digest(value):
        raise ValueError("owner SHA-256 is invalid")
    return value


_LOCK = RLock()
_LIVE_STATES: WeakKeyDictionary[LiveBoundaryState, str] = WeakKeyDictionary()
_LIVE_EVENTS: WeakKeyDictionary[LiveBoundaryEvent, str] = WeakKeyDictionary()
_LIVE_OWNERS: WeakKeyDictionary[LiveBoundaryOwner, str] = WeakKeyDictionary()
_OWNER_ORACLES: WeakKeyDictionary[
    LiveBoundaryOwner, _OracleBinding
] = WeakKeyDictionary()
# There is no owner-id release API.  Keep the latest generation strongly so a
# process-global owner-id claim cannot disappear under GC and restart a second
# genesis chain with the same audit identity.
_CURRENT_OWNERS: dict[str, LiveBoundaryOwner] = {}
_LIVE_PROPOSALS: WeakKeyDictionary[UpdateProposal, _CommitPlan] = WeakKeyDictionary()
_CONSUMED_PROPOSALS: WeakSet[UpdateProposal] = WeakSet()
_CONSUMED_GENERATIONS: set[tuple[str, int, str, str]] = set()


def _generation_key(owner: LiveBoundaryOwner) -> tuple[str, int, str, str]:
    return (owner.owner_id, owner.epoch, owner.owner_sha256, owner.state.state_sha256)


def _current_owner(owner_id: str) -> LiveBoundaryOwner | None:
    return _CURRENT_OWNERS.get(owner_id)


def _validate_live_owner_capability(
    value: object, *, verify_oracle: bool
) -> LiveBoundaryOwner:
    if type(value) is not LiveBoundaryOwner:
        raise TypeError("owner must be an exact LiveBoundaryOwner")
    with _LOCK:
        binding = _OWNER_ORACLES.get(value)
        if binding is None:
            raise RuntimeError("owner is not a live issued capability")
        owner = _validate_owner_contents(value, binding, verify_oracle=verify_oracle)
        if _LIVE_OWNERS.get(owner) != owner.owner_sha256:
            raise RuntimeError("owner live attestation is invalid")
        if _LIVE_STATES.get(owner.state) != owner.state.state_sha256:
            raise RuntimeError("owner state is not live-attested")
        for event in owner.events:
            if _LIVE_EVENTS.get(event) != event.event_sha256:
                raise RuntimeError("owner event is not live-attested")
        return owner


def validate_live_boundary_owner(value: object) -> LiveBoundaryOwner:
    _verify_owner_runtime_environment()
    return _validate_live_owner_capability(value, verify_oracle=True)


def _make_state(
    *,
    positions: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    particle_ids: tuple[tuple[Any, ...], ...],
    lineage: tuple[object, ...],
    panels: tuple[object, ...],
    live_boundary_indices: tuple[int, ...],
    circulation_m2_s: float,
    state_epoch: int,
    binding: _OracleBinding,
    verify_oracle: bool = True,
) -> LiveBoundaryState:
    state = LiveBoundaryState(
        interface_id=INTERFACE_ID,
        positions=_readonly_float64("positions", positions, ndim=2),
        gamma=_readonly_float64("gamma", gamma, ndim=2),
        sigma=_readonly_float64("sigma", sigma, ndim=1),
        particle_ids=tuple(particle_ids),
        lineage=tuple(lineage),
        panels=tuple(panels),
        live_boundary_indices=tuple(live_boundary_indices),
        release_index=len(panels),
        circulation_m2_s=float(circulation_m2_s),
        state_epoch=state_epoch,
        state_sha256="",
        clone_count=0,
        counter_particle_count=0,
        fresh_upstream_particle_count=0,
    )
    state = replace(state, state_sha256=_state_digest(state))
    return _validate_live_state_contents(state, binding, verify_oracle=verify_oracle)


def _rebuild_release_one(
    gross_state: object, binding: _OracleBinding
) -> tuple[object, tuple[object, ...], tuple[int, ...]]:
    """Re-deposit and authenticate every release-one derived state field."""

    if type(gross_state.interface_id) is not str:
        raise TypeError("release-one interface_id must be an exact string")
    if gross_state.interface_id != _V5H8_INTERFACE_ID:
        raise ValueError("release-one interface_id is invalid")
    if type(gross_state.panels) is not tuple or len(gross_state.panels) != 1:
        raise ValueError("bootstrap requires exactly one release panel")
    panel = gross_state.panels[0]
    if type(panel) is not binding.panel_type:
        raise TypeError("release-one panel must use the bound exact panel type")
    for name in (
        "upstream_left",
        "upstream_right",
        "downstream_left",
        "downstream_right",
    ):
        point = _SAFE_OBJECT_GETATTRIBUTE(panel, name)
        if type(point) is not tuple or len(point) != 3:
            raise TypeError(f"release-one panel {name} must be an exact 3-tuple")
        for component in point:
            _exact_finite_float(f"release-one panel {name}", component)
    circulation = _exact_finite_float(
        "release-one panel circulation_m2_s", panel.circulation_m2_s
    )
    if circulation == 0.0:
        raise ValueError("release-one panel circulation_m2_s must be nonzero")
    if type(panel.release_index) is not int or panel.release_index != 1:
        raise ValueError("release-one panel release_index must be exact integer one")
    smoothing = _exact_finite_float(
        "release-one smoothing_radius_m", gross_state.smoothing_radius_m
    )
    spacing = _exact_finite_float(
        "release-one target_spacing_m", gross_state.target_spacing_m
    )
    if smoothing <= 0.0 or spacing <= 0.0:
        raise ValueError("release-one smoothing and spacing must be positive")
    if type(gross_state.clone_pairs) is not tuple or gross_state.clone_pairs:
        raise ValueError("release-one clone_pairs must be the empty exact tuple")

    binding.validate_panel(panel, expected_release_index=1)
    deposited, keys = binding.deposit_panel(panel, smoothing, spacing)
    expected_lineage = tuple(
        binding.lineage_type(
            particle_id=particle_id,
            release_index=1,
            role=binding.role_for_edge(source.source_edge, keys),
            source_edge=source.source_edge,
            parent_particle_id=None,
            source_lineage=source,
        )
        for particle_id, source in zip(
            deposited.particle_ids, deposited.lineage, strict=True
        )
    )
    expected_downstream = tuple(
        index
        for index, source in enumerate(deposited.lineage)
        if _exact_tree_equal(source.source_edge, keys["downstream"])
    )
    if not expected_downstream:
        raise FloatingPointError("release-one deposition has no downstream boundary")

    comparisons = (
        ("positions", _exact_array_equal(gross_state.positions, deposited.positions)),
        ("gamma", _exact_array_equal(gross_state.gamma, deposited.gamma)),
        ("sigma", _exact_array_equal(gross_state.sigma, deposited.sigma)),
        (
            "particle_ids",
            _exact_tree_equal(gross_state.particle_ids, deposited.particle_ids),
        ),
        ("lineage", _exact_tree_equal(gross_state.lineage, expected_lineage)),
        (
            "downstream_particle_indices",
            _exact_tree_equal(
                gross_state.downstream_particle_indices, expected_downstream
            ),
        ),
    )
    for name, matches in comparisons:
        if not matches:
            raise ValueError(
                "release-one gross state disagrees with authoritative "
                f"deposition: {name}"
            )
    return deposited, expected_lineage, expected_downstream


def bootstrap_live_boundary_owner(
    gross_state: object,
    *,
    particle_cap: int = 1_000,
    owner_id: str,
    wing_id: str,
    source_time_s: float,
) -> LiveBoundaryOwner:
    """Create epoch one directly from a validated release-one gross state."""

    _verify_owner_runtime_environment()
    cap = _exact_positive_integer("particle_cap", particle_cap)
    identifier = _nonempty_string("owner_id", owner_id)
    wing = _nonempty_string("wing_id", wing_id)
    source_time = _finite_real("source_time_s", source_time_s)
    if source_time <= 0.0:
        raise ValueError("source_time_s must be positive")
    binding = _oracle_binding(gross_state)
    deposited, expected_lineage, expected_downstream = _rebuild_release_one(
        gross_state, binding
    )
    count = deposited.positions.shape[0]
    if count > cap:
        raise RuntimeError(
            "v5h9 committed-net particle cap exceeded before materialization"
        )
    state = _make_state(
        positions=deposited.positions,
        gamma=deposited.gamma,
        sigma=deposited.sigma,
        particle_ids=tuple(deposited.particle_ids),
        lineage=expected_lineage,
        panels=(replace(gross_state.panels[0]),),
        live_boundary_indices=expected_downstream,
        circulation_m2_s=float(gross_state.panels[-1].circulation_m2_s),
        state_epoch=1,
        binding=binding,
    )
    owner = LiveBoundaryOwner(
        owner_id=identifier,
        particle_cap=cap,
        epoch=1,
        wing_id=wing,
        source_time_s=source_time,
        state=state,
        events=(),
        root_source_time_s=source_time,
        release_dt_s=source_time,
        gross_particle_count=count,
        gross_state_sha256=_gross_digest(gross_state),
        oracle_source_sha256=binding.source_sha256,
        owner_sha256="",
    )
    owner = replace(owner, owner_sha256=_owner_digest(owner))
    _validate_owner_contents(owner, binding)
    with _LOCK:
        if _current_owner(identifier) is not None:
            raise RuntimeError("owner_id already has a live current generation")
        try:
            _LIVE_STATES[state] = state.state_sha256
            _LIVE_OWNERS[owner] = owner.owner_sha256
            _OWNER_ORACLES[owner] = binding
            _CURRENT_OWNERS[identifier] = owner
        except BaseException:
            _LIVE_STATES.pop(state, None)
            _LIVE_OWNERS.pop(owner, None)
            _OWNER_ORACLES.pop(owner, None)
            if _current_owner(identifier) is owner:
                _CURRENT_OWNERS.pop(identifier, None)
            raise
    return validate_live_boundary_owner(owner)


def _finalize_proposal(proposal: UpdateProposal) -> UpdateProposal:
    return replace(proposal, proposal_sha256=_proposal_digest(proposal))


def _remesh_proposal(
    owner: LiveBoundaryOwner,
    candidate_state: object,
    *,
    proposal_id: str,
    mismatch: str,
    planned_count: int,
    gamma_previous: float,
    gamma_current: float,
    gamma_scale: float,
    wing_id: str,
    source_time: float,
    candidate_digest: str,
    binding: _OracleBinding,
) -> UpdateProposal:
    return _finalize_proposal(
        UpdateProposal(
            status="remesh_required",
            proposal_id=proposal_id,
            parent_epoch=owner.epoch,
            changed_indices=(),
            planned_particle_count=planned_count,
            appended_particle_count=max(
                0, planned_count - owner.state.positions.shape[0]
            ),
            first_mismatch=mismatch,
            gamma_previous_m2_s=gamma_previous,
            gamma_current_m2_s=gamma_current,
            gamma_scale=gamma_scale,
            wing_id=wing_id,
            source_time_s=source_time,
            parent_state_sha256=owner.state.state_sha256,
            candidate_state_sha256=candidate_digest,
            proposal_sha256="",
            # A non-committable diagnostic must not retain the already-large
            # gross object.  Its semantic digest and same-process identity are
            # still bound into the proposal for audit.
            candidate_state=None,
            clone_count=0,
            counter_particle_count=0,
            fresh_upstream_particle_count=0,
            parent_owner_sha256=owner.owner_sha256,
            oracle_source_sha256=binding.source_sha256,
            candidate_identity=id(candidate_state),
        )
    )


def propose_live_boundary_update(
    owner: object,
    parent_state: object,
    candidate_state: object,
    gamma_previous: object,
    gamma_current: object,
    *,
    proposal_id: str,
    wing_id: str | None = None,
    source_time_s: float | None = None,
) -> UpdateProposal:
    """Validate one exact-compatible gross-to-net transition without commit.

    The gross candidate already exists on entry.  The cap gate therefore
    protects only committed-net materialization performed by this module.
    """

    _verify_owner_runtime_environment()
    parent_owner = validate_live_boundary_owner(owner)
    if parent_state is not parent_owner.state:
        raise ValueError("parent state identity is stale or not owned")
    identifier = _nonempty_string("proposal_id", proposal_id)
    wing = (
        parent_owner.wing_id
        if wing_id is None
        else _nonempty_string("wing_id", wing_id)
    )
    source_time = (
        parent_owner.root_source_time_s + parent_owner.epoch * parent_owner.release_dt_s
        if source_time_s is None
        else _finite_real("source_time_s", source_time_s)
    )
    previous = _finite_real("gamma_previous", gamma_previous)
    current = _finite_real("gamma_current", gamma_current)
    binding = _oracle_binding(candidate_state, validate_state=False)
    parent_binding = _OWNER_ORACLES.get(parent_owner)
    if binding is not parent_binding:
        raise RuntimeError("candidate does not use the owner's frozen v5h8 oracle")
    _validate_gross_container(candidate_state, binding)
    oracle_support_invalid = False
    try:
        binding.validate_state(candidate_state)
    except ValueError:
        # Exact support/topology failures are remesh classifications, not
        # malformed-container failures.  The independent checks below retain
        # a more specific first mismatch whenever possible.
        oracle_support_invalid = True
    candidate_digest = _gross_digest(candidate_state)

    mismatch: str | None = None
    parent_count = parent_owner.state.positions.shape[0]
    prefix_count = parent_owner.gross_particle_count
    current_release = parent_owner.state.release_index + 1
    if wing != parent_owner.wing_id:
        mismatch = "wing_id"
    elif source_time != (
        parent_owner.root_source_time_s
        + (current_release - 1) * parent_owner.release_dt_s
    ):
        mismatch = "source_time_s"
    elif previous == 0.0 or previous != parent_owner.state.circulation_m2_s:
        mismatch = "gamma_previous"
    elif len(candidate_state.panels) != current_release:
        mismatch = "release_index"
    elif tuple(candidate_state.panels[:-1]) != parent_owner.state.panels:
        mismatch = "panel_prefix"
    elif candidate_state.panels[-2].circulation_m2_s != previous:
        mismatch = "gamma_previous"
    elif candidate_state.panels[-1].circulation_m2_s != current:
        mismatch = "gamma_current"
    elif current == 0.0 or np.signbit(current) != np.signbit(previous):
        mismatch = "circulation_lifecycle"
    elif abs(current) <= abs(previous):
        mismatch = "circulation_lifecycle"
    elif candidate_state.positions.shape[0] <= prefix_count:
        mismatch = "particle_count"
    elif (
        _gross_prefix_digest(candidate_state, prefix_count)
        != parent_owner.gross_state_sha256
    ):
        mismatch = "gross_prefix"

    removed = {clone for _, clone in candidate_state.clone_pairs}
    physical_prefix = tuple(
        index for index in range(prefix_count) if index not in removed
    )
    if mismatch is None and len(physical_prefix) != parent_count:
        mismatch = "particle_count"
    if mismatch is None:
        prefix_index = np.asarray(physical_prefix, dtype=np.int64)
        if not np.array_equal(
            candidate_state.positions[prefix_index], parent_owner.state.positions
        ):
            mismatch = "positions"
        elif not np.array_equal(
            candidate_state.sigma[prefix_index], parent_owner.state.sigma
        ):
            mismatch = "sigma"
        elif tuple(
            candidate_state.particle_ids[index] for index in physical_prefix
        ) != (parent_owner.state.particle_ids):
            mismatch = "particle_id"
        elif tuple(candidate_state.lineage[index] for index in physical_prefix) != (
            parent_owner.state.lineage
        ):
            mismatch = "lineage"

    latest_pairs = tuple(
        pair for pair in candidate_state.clone_pairs if pair[1] >= prefix_count
    )
    mapping = {gross: net for net, gross in enumerate(physical_prefix)}
    if mismatch is None and len(latest_pairs) != len(
        parent_owner.state.live_boundary_indices
    ):
        mismatch = "clone_count"
    if mismatch is None:
        try:
            mapped_old = tuple(mapping[old] for old, _ in latest_pairs)
        except KeyError:
            mismatch = "clone_order"
        else:
            if mapped_old != parent_owner.state.live_boundary_indices:
                mismatch = "clone_order"
    if mismatch is None:
        expected_clone_indices = tuple(
            range(prefix_count, prefix_count + len(latest_pairs))
        )
        if tuple(clone for _, clone in latest_pairs) != expected_clone_indices:
            mismatch = "clone_order"
    clone_scale = -current / previous if previous != 0.0 else float("nan")
    if mismatch is None:
        for old_gross, clone_gross in latest_pairs:
            parent_net = mapping[old_gross]
            if not np.array_equal(
                candidate_state.positions[old_gross],
                parent_owner.state.positions[parent_net],
            ) or not np.array_equal(
                candidate_state.positions[clone_gross],
                parent_owner.state.positions[parent_net],
            ):
                mismatch = "positions"
                break
            if (
                candidate_state.sigma[old_gross] != parent_owner.state.sigma[parent_net]
                or candidate_state.sigma[clone_gross]
                != parent_owner.state.sigma[parent_net]
            ):
                mismatch = "sigma"
                break
            if not np.array_equal(
                candidate_state.gamma[old_gross], parent_owner.state.gamma[parent_net]
            ):
                mismatch = "gamma_prefix"
                break
            if not np.array_equal(
                candidate_state.gamma[clone_gross],
                clone_scale * parent_owner.state.gamma[parent_net],
            ):
                mismatch = "gamma"
                break

    newest_rows = tuple(range(prefix_count, candidate_state.positions.shape[0]))
    clone_indices = {clone for _, clone in latest_pairs}
    suffix_indices = tuple(index for index in newest_rows if index not in clone_indices)
    if mismatch is None:
        for index in suffix_indices:
            record = candidate_state.lineage[index]
            if record.release_index != current_release or record.role not in (
                "fresh_downstream",
                "fresh_left_tip",
                "fresh_right_tip",
            ):
                mismatch = "new_suffix"
                break
    if mismatch is None and any(
        candidate_state.lineage[index].role != "inherited_upstream_counter"
        or candidate_state.lineage[index].release_index != current_release
        for index in clone_indices
    ):
        mismatch = "clone_lineage"
    planned_count = parent_count + len(suffix_indices)
    if mismatch is None and planned_count > parent_owner.particle_cap:
        mismatch = "particle_cap"

    if mismatch is None and oracle_support_invalid:
        mismatch = "candidate_state"

    trusted_suffix_positions: FloatArray | None = None
    trusted_suffix_gamma: FloatArray | None = None
    trusted_suffix_sigma: FloatArray | None = None
    trusted_suffix_particle_ids: tuple[tuple[Any, ...], ...] = ()
    trusted_suffix_lineage: tuple[object, ...] = ()

    # Rebuild only the newest panel deposition after the committed-net cap gate.
    # This detects forged fresh suffix rows; it is not a gross preallocation gate.
    if mismatch is None:
        deposited, keys = binding.deposit_panel(
            candidate_state.panels[-1],
            candidate_state.smoothing_radius_m,
            candidate_state.target_spacing_m,
        )
        kept = tuple(
            index
            for index, record in enumerate(deposited.lineage)
            if record.source_edge != keys["upstream"]
        )
        if len(kept) != len(suffix_indices):
            mismatch = "new_suffix_count"
        else:
            suffix_array = np.asarray(suffix_indices, dtype=np.int64)
            kept_array = np.asarray(kept, dtype=np.int64)
            trusted_suffix_positions = _readonly_float64(
                "trusted suffix positions",
                deposited.positions[kept_array],
                ndim=2,
            )
            trusted_suffix_gamma = _readonly_float64(
                "trusted suffix gamma",
                deposited.gamma[kept_array],
                ndim=2,
            )
            trusted_suffix_sigma = _readonly_float64(
                "trusted suffix sigma",
                deposited.sigma[kept_array],
                ndim=1,
            )
            trusted_suffix_particle_ids = tuple(
                deposited.particle_ids[index] for index in kept
            )
            trusted_suffix_lineage = tuple(
                binding.lineage_type(
                    particle_id=deposited.particle_ids[index],
                    release_index=current_release,
                    role=binding.role_for_edge(
                        deposited.lineage[index].source_edge, keys
                    ),
                    source_edge=deposited.lineage[index].source_edge,
                    parent_particle_id=None,
                    source_lineage=deposited.lineage[index],
                )
                for index in kept
            )
            if not _exact_array_equal(
                candidate_state.positions[suffix_array], trusted_suffix_positions
            ):
                mismatch = "positions"
            elif not _exact_array_equal(
                candidate_state.gamma[suffix_array], trusted_suffix_gamma
            ):
                mismatch = "gamma"
            elif not _exact_array_equal(
                candidate_state.sigma[suffix_array], trusted_suffix_sigma
            ):
                mismatch = "sigma"
            elif not _exact_tree_equal(
                tuple(candidate_state.particle_ids[index] for index in suffix_indices),
                trusted_suffix_particle_ids,
            ):
                mismatch = "particle_id"
            elif not _exact_tree_equal(
                tuple(candidate_state.lineage[index] for index in suffix_indices),
                trusted_suffix_lineage,
            ):
                mismatch = "lineage"

    new_boundary: tuple[int, ...] = ()
    if mismatch is None:
        suffix_mapping = {
            gross: parent_count + order for order, gross in enumerate(suffix_indices)
        }
        try:
            candidate_boundary = tuple(
                suffix_mapping[index]
                for index in candidate_state.downstream_particle_indices
            )
        except KeyError:
            mismatch = "downstream_boundary"
        if mismatch is None:
            new_boundary = tuple(
                parent_count + order
                for order, record in enumerate(trusted_suffix_lineage)
                if record.role == "fresh_downstream"
            )
            if (
                not new_boundary
                or len(set(new_boundary)) != len(new_boundary)
                or candidate_boundary != new_boundary
            ):
                mismatch = "downstream_boundary"

    gamma_scale = 1.0 - current / previous if previous != 0.0 else float("nan")
    if mismatch is not None:
        return _remesh_proposal(
            parent_owner,
            candidate_state,
            proposal_id=identifier,
            mismatch=mismatch,
            planned_count=planned_count,
            gamma_previous=previous,
            gamma_current=current,
            gamma_scale=gamma_scale,
            wing_id=wing,
            source_time=source_time,
            candidate_digest=candidate_digest,
            binding=binding,
        )

    if (
        trusted_suffix_positions is None
        or trusted_suffix_gamma is None
        or trusted_suffix_sigma is None
        or not trusted_suffix_particle_ids
        or not trusted_suffix_lineage
    ):
        raise RuntimeError("compatible proposal lacks trusted suffix materialization")

    proposal = _finalize_proposal(
        UpdateProposal(
            status="compatible",
            proposal_id=identifier,
            parent_epoch=parent_owner.epoch,
            changed_indices=parent_owner.state.live_boundary_indices,
            planned_particle_count=planned_count,
            appended_particle_count=len(suffix_indices),
            first_mismatch=None,
            gamma_previous_m2_s=previous,
            gamma_current_m2_s=current,
            gamma_scale=gamma_scale,
            wing_id=wing,
            source_time_s=source_time,
            parent_state_sha256=parent_owner.state.state_sha256,
            candidate_state_sha256=candidate_digest,
            proposal_sha256="",
            candidate_state=candidate_state,
            clone_count=0,
            counter_particle_count=0,
            fresh_upstream_particle_count=0,
            parent_owner_sha256=parent_owner.owner_sha256,
            oracle_source_sha256=binding.source_sha256,
            candidate_identity=id(candidate_state),
        )
    )
    issued_proposal = replace(proposal)
    plan = _CommitPlan(
        parent_owner=weakref.ref(parent_owner),
        binding=binding,
        candidate_state=candidate_state,
        candidate_sha256=candidate_digest,
        suffix_indices=suffix_indices,
        suffix_positions=trusted_suffix_positions,
        suffix_gamma=trusted_suffix_gamma,
        suffix_sigma=trusted_suffix_sigma,
        suffix_particle_ids=trusted_suffix_particle_ids,
        suffix_lineage=trusted_suffix_lineage,
        latest_panel=replace(candidate_state.panels[-1]),
        new_boundary_indices=new_boundary,
        issued_proposal=issued_proposal,
        issued_proposal_sha256=proposal.proposal_sha256,
    )
    with _LOCK:
        if _current_owner(parent_owner.owner_id) is not parent_owner:
            raise RuntimeError(
                "owner generation was consumed during proposal validation"
            )
        if _generation_key(parent_owner) in _CONSUMED_GENERATIONS:
            raise RuntimeError("owner generation is already consumed")
        _LIVE_PROPOSALS[proposal] = plan
    return proposal


def _reattest_candidate_suffix(plan: _CommitPlan) -> None:
    """Match caller-owned candidate rows to the private issuance material."""

    candidate = plan.candidate_state
    try:
        suffix = np.asarray(plan.suffix_indices, dtype=np.int64)
        positions_match = _exact_array_equal(
            candidate.positions[suffix], plan.suffix_positions
        )
        gamma_match = _exact_array_equal(candidate.gamma[suffix], plan.suffix_gamma)
        sigma_match = _exact_array_equal(candidate.sigma[suffix], plan.suffix_sigma)
        ids_match = _exact_tree_equal(
            tuple(candidate.particle_ids[index] for index in plan.suffix_indices),
            plan.suffix_particle_ids,
        )
        lineage_match = _exact_tree_equal(
            tuple(candidate.lineage[index] for index in plan.suffix_indices),
            plan.suffix_lineage,
        )
        panel_match = _exact_tree_equal(candidate.panels[-1], plan.latest_panel)
    except (IndexError, TypeError, ValueError) as error:
        raise RuntimeError("proposal candidate suffix changed before commit") from error
    if not all(
        (
            positions_match,
            gamma_match,
            sigma_match,
            ids_match,
            lineage_match,
            panel_match,
        )
    ):
        raise RuntimeError("proposal candidate suffix changed before commit")


def _build_committed_state(
    owner: LiveBoundaryOwner, proposal: UpdateProposal, plan: _CommitPlan
) -> LiveBoundaryState:
    parent_gamma = np.array(owner.state.gamma, dtype=np.float64, order="C", copy=True)
    live = np.asarray(owner.state.live_boundary_indices, dtype=np.int64)
    parent_gamma[live] = proposal.gamma_scale * owner.state.gamma[live]
    positions = np.vstack((owner.state.positions, plan.suffix_positions))
    gamma = np.vstack((parent_gamma, plan.suffix_gamma))
    sigma = np.concatenate((owner.state.sigma, plan.suffix_sigma))
    particle_ids = owner.state.particle_ids + plan.suffix_particle_ids
    lineage = owner.state.lineage + plan.suffix_lineage
    return _make_state(
        positions=positions,
        gamma=gamma,
        sigma=sigma,
        particle_ids=particle_ids,
        lineage=lineage,
        panels=owner.state.panels + (plan.latest_panel,),
        live_boundary_indices=plan.new_boundary_indices,
        circulation_m2_s=proposal.gamma_current_m2_s,
        state_epoch=owner.epoch + 1,
        binding=plan.binding,
        verify_oracle=False,
    )


def commit_live_boundary_update(owner: object, proposal: object) -> CommitResult:
    """Atomically consume one owner generation and publish one net release."""

    _verify_owner_runtime_environment()
    if type(owner) is not LiveBoundaryOwner:
        raise TypeError("owner must be an exact LiveBoundaryOwner")
    if type(proposal) is not UpdateProposal:
        raise TypeError("proposal must be an exact UpdateProposal")
    with _LOCK:
        # Proposal creation froze and validated all v5h8-dependent facts.
        # Commit deliberately neither resolves nor calls mutable oracle
        # bindings (including validator and collapse helpers).
        # Re-attest callable identity/source without invoking a changed
        # binding.  A drifted validator fails here before it can execute; the
        # still-live proposal remains retryable after the binding is restored.
        parent_owner = _validate_live_owner_capability(owner, verify_oracle=True)
        if _current_owner(parent_owner.owner_id) is not parent_owner:
            raise RuntimeError("owner generation is stale or already consumed")
        generation = _generation_key(parent_owner)
        if generation in _CONSUMED_GENERATIONS:
            raise RuntimeError("owner generation was already committed")
        if proposal in _CONSUMED_PROPOSALS:
            raise RuntimeError("proposal replay after commit")
        plan = _LIVE_PROPOSALS.get(proposal)
        if plan is None:
            raise RuntimeError("proposal is stale or not live")
        issued = _issued_proposal(proposal, plan)
        if issued.status != "compatible":
            raise RuntimeError("issued proposal is not compatible")
        if plan.parent_owner() is not parent_owner:
            raise RuntimeError("proposal owner is stale")
        if (
            issued.parent_epoch != parent_owner.epoch
            or issued.parent_owner_sha256 != parent_owner.owner_sha256
            or issued.parent_state_sha256 != parent_owner.state.state_sha256
            or issued.candidate_state is not plan.candidate_state
            or issued.candidate_identity != id(plan.candidate_state)
            or issued.candidate_state_sha256 != plan.candidate_sha256
            or issued.oracle_source_sha256 != plan.binding.source_sha256
        ):
            raise RuntimeError("proposal binding is stale")
        if _gross_digest(plan.candidate_state) != plan.candidate_sha256:
            raise RuntimeError("proposal candidate changed before commit")
        _reattest_candidate_suffix(plan)

        # All fallible mechanics and validation precede registry publication.
        new_state = _build_committed_state(parent_owner, issued, plan)
        if new_state.positions.shape[0] != issued.planned_particle_count:
            raise RuntimeError("committed count disagrees with proposal")
        before_gamma = parent_owner.state.gamma[
            np.asarray(parent_owner.state.live_boundary_indices, dtype=np.int64)
        ]
        after_gamma = new_state.gamma[
            np.asarray(issued.changed_indices, dtype=np.int64)
        ]
        added_gamma = after_gamma - before_gamma
        previous_event_sha = (
            parent_owner.events[-1].event_sha256
            if parent_owner.events
            else _GENESIS_SHA256
        )
        event = LiveBoundaryEvent(
            proposal_id=issued.proposal_id,
            release_index=new_state.release_index,
            changed_indices=issued.changed_indices,
            gamma_previous_m2_s=issued.gamma_previous_m2_s,
            gamma_current_m2_s=issued.gamma_current_m2_s,
            gamma_scale=issued.gamma_scale,
            parent_state_sha256=parent_owner.state.state_sha256,
            candidate_state_sha256=issued.candidate_state_sha256,
            before_gamma_sha256=_array_sha256(before_gamma),
            added_gamma_sha256=_array_sha256(added_gamma),
            after_gamma_sha256=_array_sha256(after_gamma),
            committed_state_sha256=new_state.state_sha256,
            operator_order=OPERATOR_ORDER,
            clone_count=0,
            counter_particle_count=0,
            fresh_upstream_particle_count=0,
            remesh_count=0,
            target_read_count=0,
            ptera_call_count=0,
            load_call_count=0,
            feedback_write_count=0,
            parent_write_count=0,
            source_time_s=issued.source_time_s,
            parent_owner_sha256=parent_owner.owner_sha256,
            previous_event_sha256=previous_event_sha,
            event_sha256="",
        )
        event = replace(event, event_sha256=_event_digest(event))
        next_owner = LiveBoundaryOwner(
            owner_id=parent_owner.owner_id,
            particle_cap=parent_owner.particle_cap,
            epoch=parent_owner.epoch + 1,
            wing_id=parent_owner.wing_id,
            source_time_s=issued.source_time_s,
            state=new_state,
            events=parent_owner.events + (event,),
            root_source_time_s=parent_owner.root_source_time_s,
            release_dt_s=parent_owner.release_dt_s,
            gross_particle_count=plan.candidate_state.positions.shape[0],
            gross_state_sha256=plan.candidate_sha256,
            oracle_source_sha256=plan.binding.source_sha256,
            owner_sha256="",
        )
        next_owner = replace(next_owner, owner_sha256=_owner_digest(next_owner))
        _validate_owner_contents(next_owner, plan.binding, verify_oracle=False)

        # Detect caller-visible mutation that raced with construction.  All
        # mechanics above used only the private issued snapshot, and failure
        # here still precedes every registry publication.
        _issued_proposal(proposal, plan)

        siblings = tuple(
            (candidate_proposal, candidate_plan)
            for candidate_proposal, candidate_plan in tuple(_LIVE_PROPOSALS.items())
            if candidate_plan.parent_owner() is parent_owner
        )
        old_current = _CURRENT_OWNERS.get(parent_owner.owner_id)
        try:
            _LIVE_STATES[new_state] = new_state.state_sha256
            _LIVE_EVENTS[event] = event.event_sha256
            _LIVE_OWNERS[next_owner] = next_owner.owner_sha256
            _OWNER_ORACLES[next_owner] = plan.binding
            _CONSUMED_GENERATIONS.add(generation)
            for sibling, _ in siblings:
                _CONSUMED_PROPOSALS.add(sibling)
                _LIVE_PROPOSALS.pop(sibling, None)
            _CURRENT_OWNERS[parent_owner.owner_id] = next_owner
        except BaseException:
            _LIVE_STATES.pop(new_state, None)
            _LIVE_EVENTS.pop(event, None)
            _LIVE_OWNERS.pop(next_owner, None)
            _OWNER_ORACLES.pop(next_owner, None)
            _CONSUMED_GENERATIONS.discard(generation)
            for sibling, sibling_plan in siblings:
                _CONSUMED_PROPOSALS.discard(sibling)
                _LIVE_PROPOSALS[sibling] = sibling_plan
            if old_current is None:
                _CURRENT_OWNERS.pop(parent_owner.owner_id, None)
            else:
                _CURRENT_OWNERS[parent_owner.owner_id] = old_current
            raise
    return CommitResult(owner=next_owner, state=new_state, event=event)


# Seal the complete owner implementation only after all entry points and their
# transitive helpers exist.  Populate the placeholder in place: public entry
# functions resolve this exact dict object, and verification freezes that
# resolution without freezing the registries' legitimately changing contents.
_OWNER_MODULE = sys.modules[__name__]
_OWNER_NAMESPACE = _OWNER_MODULE.__dict__
if type(_OWNER_NAMESPACE.get("__file__")) is not str:
    raise RuntimeError("owner runtime source path is unavailable")
_OWNER_SOURCE_PATH = Path(_OWNER_NAMESPACE["__file__"]).resolve()
_OWNER_SOURCE_BYTES = _OWNER_SOURCE_PATH.read_bytes()
_OWNER_SOURCE_SHA256 = hashlib.sha256(_OWNER_SOURCE_BYTES).hexdigest()
_OWNER_CODE_MANIFEST = _top_level_code_manifest(
    _OWNER_SOURCE_BYTES, str(_OWNER_SOURCE_PATH)
)
_OWNER_RECURSIVE_CODE_MANIFEST = _recursive_code_manifest(
    _OWNER_SOURCE_BYTES, str(_OWNER_SOURCE_PATH)
)
_OWNER_MODULE_DEFINED_NAMES = _top_level_defined_names(
    _OWNER_SOURCE_BYTES, str(_OWNER_SOURCE_PATH)
)
_OWNER_ROOT_FUNCTIONS = (
    "bootstrap_live_boundary_owner",
    "propose_live_boundary_update",
    "commit_live_boundary_update",
    "validate_live_boundary_owner",
)
(
    _OWNER_REQUIRED_FUNCTION_BINDINGS,
    _OWNER_GLOBAL_RESOLUTIONS,
    _OWNER_FUNCTION_RUNTIME_SEALS,
) = _freeze_global_closure(
    _OWNER_MODULE,
    _OWNER_ROOT_FUNCTIONS,
    _OWNER_MODULE_DEFINED_NAMES,
)
_OWNER_REQUIRED_CLASSES_SET: set[str] = set()
for _OWNER_RESOLUTION in _OWNER_GLOBAL_RESOLUTIONS:
    _OWNER_VALUE = _OWNER_RESOLUTION.value
    if (
        _SAFE_TYPE(_OWNER_VALUE) is _SAFE_TYPE
        and _SAFE_TYPE.__getattribute__(_OWNER_VALUE, "__module__") == __name__
    ):
        _OWNER_REQUIRED_CLASSES_SET.add(_OWNER_RESOLUTION.global_name)
_OWNER_CLASS_NAMESPACE_SEALS = _freeze_class_namespaces(
    _OWNER_MODULE,
    _SAFE_TUPLE(_SAFE_SORTED(_OWNER_REQUIRED_CLASSES_SET)),
    _OWNER_MODULE_DEFINED_NAMES,
    _OWNER_RECURSIVE_CODE_MANIFEST,
)
_OWNER_CLASS_FUNCTION_RUNTIME_SEALS = _SAFE_TUPLE(
    runtime_seal
    for class_seal in _OWNER_CLASS_NAMESPACE_SEALS
    for runtime_seal in class_seal.member_runtime_seals
)
_OWNER_CLASS_FUNCTION_BINDINGS = _SAFE_TUPLE(
    (runtime_seal.function_name, runtime_seal.function)
    for runtime_seal in _OWNER_CLASS_FUNCTION_RUNTIME_SEALS
)
_OWNER_CLASS_GLOBAL_RESOLUTIONS = _SAFE_TUPLE(
    resolution
    for runtime_seal in _OWNER_CLASS_FUNCTION_RUNTIME_SEALS
    for resolution in runtime_seal.global_resolutions
)
_OWNER_MODULE_ATTRIBUTE_SEALS = _freeze_module_attributes(
    _OWNER_REQUIRED_FUNCTION_BINDINGS + _OWNER_CLASS_FUNCTION_BINDINGS,
    _OWNER_GLOBAL_RESOLUTIONS + _OWNER_CLASS_GLOBAL_RESOLUTIONS,
)
_OWNER_RUNTIME_ENVIRONMENT.update(
    {
        "module": _OWNER_MODULE,
        "namespace": _OWNER_NAMESPACE,
        "source_path": _OWNER_SOURCE_PATH,
        "source_sha256": _OWNER_SOURCE_SHA256,
        "code_manifest": _OWNER_CODE_MANIFEST,
        "required_function_bindings": _OWNER_REQUIRED_FUNCTION_BINDINGS,
        "global_resolutions": _OWNER_GLOBAL_RESOLUTIONS,
        "function_runtime_seals": _OWNER_FUNCTION_RUNTIME_SEALS,
        "module_attribute_seals": _OWNER_MODULE_ATTRIBUTE_SEALS,
        "class_namespace_seals": _OWNER_CLASS_NAMESPACE_SEALS,
    }
)


__all__ = [
    "CommitResult",
    "INTERFACE_ID",
    "LiveBoundaryEvent",
    "LiveBoundaryOwner",
    "LiveBoundaryState",
    "OPERATOR_ORDER",
    "UpdateProposal",
    "bootstrap_live_boundary_owner",
    "commit_live_boundary_update",
    "propose_live_boundary_update",
    "validate_live_boundary_owner",
]
