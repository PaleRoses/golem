"""Content-addressed circuit-solve lineage: keys, on-disk store, parallel realize."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from golem.kernel.anatomy.realize.carriers import _index_structural_cost_field
from golem.kernel.anatomy.realize.constructor import (
    DEFAULT_CIRCUIT_CONSTRUCTOR,
    realize_circuit,
)

if TYPE_CHECKING:
    from golem.kernel.anatomy.graph import RejectedVasculature
    from golem.kernel.anatomy.realize.carriers import (
        _CircuitGeometry,
        _StructuralCostField,
    )
    from golem.kernel.anatomy.realize.constructor import CircuitConstructor
    from golem.kernel.anatomy.vocabulary import (
        AcceptedAnatomy,
        CirculationCircuit,
        SealedVascularConfig,
    )


_KEY_SCHEMA = "golem.circuit-solve/1"
_RESULT_KEY_SCHEMA = "golem.vasculature-result/1"
_COLD_ENV = "GOLEM_CIRCUIT_CACHE_COLD"
_STORE_DIRNAME = ".golem-cache"
_STORE_SUBDIR = "circuit"
_RESULT_STORE_SUBDIR = "vasculature"
_PARALLEL_TERMINAL_PAIR_FLOOR = 256
# Salt over every module whose source shapes a circuit solve or its result
# types; any kernel edit within these roots invalidates persisted verdicts.
_SALT_ROOTS = ("kernel/anatomy", "kernel/engine")


@dataclass(frozen=True)
class _CircuitSolveInputs:
    circuit: CirculationCircuit
    terminal_pair_count: int
    accepted: AcceptedAnatomy
    part_by_id: dict[str, dict]
    provenance: dict[str, str]
    landmarks: dict[str, object]
    bone_use_count: dict[str, int]
    shared_pump_interfaces: tuple[tuple[float, float, float], ...]
    config: SealedVascularConfig
    structural_field: _StructuralCostField | None
    constructor: CircuitConstructor


def _leaf(tag: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(tag + b"\x00" + payload).digest()


def _node(tag: bytes, children: tuple[bytes, ...]) -> bytes:
    return hashlib.sha256(tag + b"\x00" + b"".join(children)).digest()


def _digest(value: object) -> bytes:
    match value:
        case bool():
            return _leaf(b"bool", b"\x01" if value else b"\x00")
        case Enum():
            return _node(
                b"enum",
                (_digest(type(value).__qualname__), _digest(value.value)),
            )
        case int():
            return _leaf(b"int", str(value).encode())
        case float():
            return _leaf(b"float", repr(value).encode())
        case str():
            return _leaf(b"str", value.encode())
        case bytes():
            return _leaf(b"bytes", value)
        case None:
            return _leaf(b"none", b"")
        case tuple():
            return _node(b"tuple", tuple(_digest(item) for item in value))
        case list():
            return _node(b"list", tuple(_digest(item) for item in value))
        case dict():
            return _node(
                b"map",
                tuple(
                    _node(b"entry", (key_digest, _digest(mapped)))
                    for key_digest, mapped in sorted(
                        (
                            (_digest(key), mapped)
                            for key, mapped in value.items()
                        ),
                        key=lambda pair: pair[0],
                    )
                ),
            )
        case _ if is_dataclass(value) and not isinstance(value, type):
            return _node(
                b"dataclass:" + type(value).__qualname__.encode(),
                tuple(
                    _node(
                        b"field",
                        (
                            _digest(field.name),
                            _digest(getattr(value, field.name)),
                        ),
                    )
                    for field in fields(value)
                ),
            )
        case _:
            raise TypeError(
                f"non-canonical circuit-key input of type {type(value)!r}"
            )


@cache
def _kernel_source_salt() -> bytes:
    package_root = Path(__file__).resolve().parents[3]
    sources = sorted(
        path
        for root in _SALT_ROOTS
        for path in (package_root / root).rglob("*.py")
    )
    return _node(
        b"kernel-salt",
        tuple(
            _node(
                b"source",
                (
                    _digest(path.relative_to(package_root).as_posix()),
                    _leaf(b"content", path.read_bytes()),
                ),
            )
            for path in sources
        ),
    )


def _pump_bone(accepted: AcceptedAnatomy) -> str:
    return accepted.overall.circulation.pump.region.host_bone_id


def _circuit_interface_bones(circuit: CirculationCircuit) -> frozenset[str]:
    return frozenset(
        address.removeprefix("skeleton/").removesuffix("/joint")
        for address in (
            *(
                segment.interface_address
                for segment in circuit.distributing_arteries
            ),
            circuit.resistance_arteriole.interface_address,
        )
    )


def _referenced_parts(
    inputs: _CircuitSolveInputs,
    territory_bones: frozenset[str],
) -> tuple[tuple[str, str, dict], ...]:
    return tuple(
        sorted(
            (
                (part_id, address, inputs.part_by_id[part_id])
                for part_id, address in inputs.provenance.items()
                if part_id in inputs.part_by_id
                and isinstance(address, str)
                and any(
                    address.startswith(f"skeleton/{bone}/")
                    for bone in territory_bones
                )
            ),
            key=lambda entry: entry[0],
        )
    )


def _referenced_landmarks(
    inputs: _CircuitSolveInputs,
    landmark_bones: frozenset[str],
) -> tuple[tuple[str, object], ...]:
    return tuple(
        sorted(
            (
                (name, value)
                for name, value in inputs.landmarks.items()
                if any(name.startswith(f"{bone}/") for bone in landmark_bones)
            ),
            key=lambda entry: entry[0],
        )
    )


def _circuit_key(inputs: _CircuitSolveInputs) -> str:
    circuit = inputs.circuit
    territory_bones = frozenset(circuit.bones)
    landmark_bones = territory_bones | _circuit_interface_bones(circuit)
    return _digest(
        (
            _KEY_SCHEMA,
            circuit,
            inputs.terminal_pair_count,
            _pump_bone(inputs.accepted),
            _referenced_parts(inputs, territory_bones),
            _referenced_landmarks(inputs, landmark_bones),
            tuple(
                (bone, inputs.bone_use_count.get(bone, 0))
                for bone in sorted(circuit.bones)
            ),
            inputs.shared_pump_interfaces,
            inputs.config,
            inputs.structural_field,
            _kernel_source_salt().hex(),
        )
    ).hex()


def _solve_one_circuit(
    inputs: _CircuitSolveInputs,
) -> _CircuitGeometry | RejectedVasculature:
    structural_cost_index = (
        _index_structural_cost_field(inputs.structural_field)
        if inputs.structural_field is not None
        else None
    )
    return realize_circuit(
        inputs.constructor,
        inputs.circuit,
        inputs.terminal_pair_count,
        inputs.accepted,
        inputs.part_by_id,
        inputs.provenance,
        inputs.landmarks,
        inputs.bone_use_count,
        inputs.shared_pump_interfaces,
        inputs.config,
        structural_cost_index,
    )


def _solve_parallel(
    inputs: tuple[_CircuitSolveInputs, ...],
) -> tuple[_CircuitGeometry | RejectedVasculature, ...]:
    worker_count = min(len(inputs), os.cpu_count() or 1)
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context
        ) as pool:
            return tuple(pool.map(_solve_one_circuit, inputs))
    except (OSError, BrokenProcessPool):
        # Spawn pools are unavailable in some sandboxed/forkless hosts;
        # the sequential result is the semantic equal of the parallel one.
        return tuple(map(_solve_one_circuit, inputs))


def _solve_missing(
    inputs: tuple[_CircuitSolveInputs, ...],
) -> tuple[_CircuitGeometry | RejectedVasculature, ...]:
    estimated_work = sum(
        item.terminal_pair_count for item in inputs
    )
    return (
        tuple(map(_solve_one_circuit, inputs))
        if len(inputs) <= 1
        or estimated_work <= _PARALLEL_TERMINAL_PAIR_FLOOR
        else _solve_parallel(inputs)
    )


def _vasculature_result_key(
    accepted: AcceptedAnatomy,
    canonical_body: str,
    config: SealedVascularConfig,
    structural_field: _StructuralCostField | None,
) -> str:
    return _digest(
        (
            _RESULT_KEY_SCHEMA,
            accepted,
            canonical_body,
            config,
            structural_field,
            _kernel_source_salt().hex(),
        )
    ).hex()


@cache
def _result_store_root() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / _STORE_DIRNAME
        / _RESULT_STORE_SUBDIR
    )


def _result_path(key: str) -> Path:
    return _result_store_root() / f"{key}.pkl"


def _load_vasculature_result(key: str) -> object | None:
    path = _result_path(key)
    return pickle.loads(path.read_bytes()) if path.exists() else None


def _store_vasculature_result(key: str, result: object) -> None:
    root = _result_store_root()
    root.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    temporary = root / f"{key}.{os.getpid()}.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, _result_path(key))


@cache
def _store_root() -> Path:
    return Path(__file__).resolve().parents[4] / _STORE_DIRNAME / _STORE_SUBDIR


def _circuit_path(key: str) -> Path:
    return _store_root() / f"{key}.pkl"


def _store_bypassed() -> bool:
    return os.environ.get(_COLD_ENV, "").strip().lower() in ("1", "true", "yes")


def _load_circuit(
    key: str,
) -> _CircuitGeometry | RejectedVasculature | None:
    path = _circuit_path(key)
    return pickle.loads(path.read_bytes()) if path.exists() else None


def _store_circuit(
    key: str, result: _CircuitGeometry | RejectedVasculature
) -> None:
    root = _store_root()
    root.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    temporary = root / f"{key}.{os.getpid()}.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, _circuit_path(key))


def _persist_circuits(
    keys: tuple[str, ...],
    results: tuple[_CircuitGeometry | RejectedVasculature, ...],
) -> None:
    for key, result in zip(keys, results, strict=True):
        _store_circuit(key, result)


def _resolve_with_store(
    inputs: tuple[_CircuitSolveInputs, ...],
) -> tuple[_CircuitGeometry | RejectedVasculature, ...]:
    keys = tuple(map(_circuit_key, inputs))
    loaded = tuple(map(_load_circuit, keys))
    miss_indices = tuple(
        index for index, result in enumerate(loaded) if result is None
    )
    solved = _solve_missing(tuple(inputs[index] for index in miss_indices))
    _persist_circuits(tuple(keys[index] for index in miss_indices), solved)
    solved_by_index = dict(zip(miss_indices, solved, strict=True))
    return tuple(
        result if result is not None else solved_by_index[index]
        for index, result in enumerate(loaded)
    )


def _realize_circuits(
    circuits: tuple[CirculationCircuit, ...],
    allocation_by_region: dict[str, int],
    accepted: AcceptedAnatomy,
    part_by_id: dict[str, dict],
    provenance: dict[str, str],
    landmarks: dict[str, object],
    bone_use_count: dict[str, int],
    shared_pump_interfaces: tuple[tuple[float, float, float], ...],
    config: SealedVascularConfig,
    structural_field: _StructuralCostField | None,
    *,
    constructor: CircuitConstructor = DEFAULT_CIRCUIT_CONSTRUCTOR,
) -> tuple[_CircuitGeometry | RejectedVasculature, ...]:
    inputs = tuple(
        _CircuitSolveInputs(
            circuit=circuit,
            terminal_pair_count=allocation_by_region[
                circuit.capillary_bed.region.region_id
            ],
            accepted=accepted,
            part_by_id=part_by_id,
            provenance=provenance,
            landmarks=landmarks,
            bone_use_count=bone_use_count,
            shared_pump_interfaces=shared_pump_interfaces,
            config=config,
            structural_field=structural_field,
            constructor=constructor,
        )
        for circuit in circuits
    )
    return (
        _solve_missing(inputs)
        if _store_bypassed() or constructor is not DEFAULT_CIRCUIT_CONSTRUCTOR
        else _resolve_with_store(inputs)
    )
