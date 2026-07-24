"""Assembly ingestion over one immutable source snapshot."""

from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import assert_never

from golem.assembly.carriers import ElementRole, _LoadedElement
from golem.assembly.fit import _element_fit_input_obstructions
from golem.assembly.obstructions import (
    AssemblyInputObstruction,
    DuplicateElementIdObstruction,
    ElementBodyObstruction,
    ElementSourceObstruction,
    EquipmentBodyDialectObstruction,
    MalformedAssemblyObstruction,
    UnknownElementRoleObstruction,
)
from golem.assembly.service import (BoneAddress, ElementAddress, MountAddress, PortAddress, RegionAddress, SemanticAddress)
from golem.kernel.anatomy import AcceptedAnatomy
from golem.kernel.body import CompiledBody, Compiler, RejectedBody
from golem.paths import PILOTS


class _ElementSourceKind(StrEnum):
    INLINE_GRAPH = "inline_graph"
    GRAPH_FILE = "graph_file"
    BODY_FILE = "body_file"


@dataclass(frozen=True)
class _SourcePayload:
    label: str
    content: bytes


@dataclass(frozen=True)
class _ElementSourceSnapshot:
    element_id: str
    kind: _ElementSourceKind
    primary: _SourcePayload | None
    contract: _SourcePayload | None = None
    mounts: tuple[_SourcePayload | None, ...] = ()


@dataclass(frozen=True)
class _AssemblySourceSnapshot:
    elements: tuple[_ElementSourceSnapshot, ...] = field(
        compare=False,
        hash=False,
    )
    digest: str


def _assembly_input_obstructions(
    spec: object,
) -> tuple[AssemblyInputObstruction, ...]:
    if not isinstance(spec, dict):
        return (MalformedAssemblyObstruction("/", "expected an object"),)
    elements = spec.get("elements")
    if not isinstance(elements, list) or not elements:
        return (
            MalformedAssemblyObstruction(
                "/elements", "expected a non-empty array"
            ),
        )
    malformed_entries = tuple(
        MalformedAssemblyObstruction(
            f"/elements/{index}", "expected an object"
        )
        for index, entry in enumerate(elements)
        if not isinstance(entry, dict)
    )
    if malformed_entries:
        return malformed_entries
    typed_entries = tuple(entry for entry in elements if isinstance(entry, dict))
    identifiers = tuple(entry.get("id") for entry in typed_entries)
    identifier_obstructions = tuple(
        MalformedAssemblyObstruction(
            f"/elements/{index}/id", "expected a non-empty string"
        )
        for index, identifier in enumerate(identifiers)
        if not isinstance(identifier, str) or not identifier
    )
    source_obstructions = tuple(
        MalformedAssemblyObstruction(
            f"/elements/{index}",
            "expected exactly one of graph or body",
        )
        for index, entry in enumerate(typed_entries)
        if int("graph" in entry) + int("body" in entry) != 1
    )
    duplicate_obstructions = tuple(
        DuplicateElementIdObstruction(str(identifier))
        for index, identifier in enumerate(identifiers)
        if isinstance(identifier, str) and identifier in identifiers[:index]
    )
    role_obstructions = tuple(
        obstruction
        for entry in typed_entries
        for obstruction in _element_role_obstructions(entry)
    )
    return (
        *identifier_obstructions,
        *source_obstructions,
        *duplicate_obstructions,
        *role_obstructions,
        *_element_fit_input_obstructions(typed_entries),
    )


def _element_role_obstructions(
    entry: dict,
) -> tuple[AssemblyInputObstruction, ...]:
    identifier = entry.get("id")
    element_id = identifier if isinstance(identifier, str) else "<unknown>"
    role = entry.get("role", ElementRole.EQUIPMENT.value)
    decoded_role = next(
        (candidate for candidate in ElementRole if candidate.value == role),
        None,
    )
    return (
        (UnknownElementRoleObstruction(element_id, role),)
        if decoded_role is None
        else (
            (EquipmentBodyDialectObstruction(element_id),)
            if decoded_role is ElementRole.EQUIPMENT and "body" in entry
            else ()
        )
    )


def _read_payload(path: Path, label: str | None = None) -> _SourcePayload:
    return _SourcePayload(label or str(path), path.read_bytes())


def _body_contract_snapshot(
    source_dir: Path,
    body_spec: dict,
) -> _SourcePayload | None:
    contract = body_spec.get("contract")
    return (
        _read_payload(source_dir / contract, f"body-contract:{contract}")
        if isinstance(contract, str)
        else None
    )


def _body_mount_snapshot(
    source_dir: Path,
    mount: object,
) -> _SourcePayload | None:
    reference = mount.get("spec") if isinstance(mount, dict) else None
    candidates = (
        (source_dir / reference, PILOTS / reference)
        if isinstance(reference, str)
        else ()
    )
    selected = next((candidate for candidate in candidates if candidate.is_file()), None)
    return (
        _read_payload(selected, f"body-mount:{reference}")
        if selected is not None
        else None
    )


def _capture_element_source(
    entry: dict,
    base_dir: Path,
) -> _ElementSourceSnapshot:
    element_id = str(entry["id"])
    if "body" in entry:
        body_source = entry["body"]
        body_path = base_dir / body_source if isinstance(body_source, str) else None
        primary = (
            _read_payload(body_path, f"body:{body_source}")
            if body_path is not None
            else _SourcePayload(
                f"body:inline:{element_id}",
                json.dumps(
                    body_source,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8"),
            )
        )
        body_spec = json.loads(primary.content)
        source_dir = body_path.parent if body_path is not None else base_dir
        return _ElementSourceSnapshot(
            element_id=element_id,
            kind=_ElementSourceKind.BODY_FILE,
            primary=primary,
            contract=_body_contract_snapshot(source_dir, body_spec),
            mounts=tuple(
                _body_mount_snapshot(source_dir, mount)
                for mount in body_spec.get("mounts", ())
            ),
        )
    graph = entry["graph"]
    return (
        _ElementSourceSnapshot(
            element_id,
            _ElementSourceKind.GRAPH_FILE,
            _read_payload(base_dir / graph, f"graph:{graph}"),
        )
        if isinstance(graph, str)
        else _ElementSourceSnapshot(
            element_id,
            _ElementSourceKind.INLINE_GRAPH,
            None,
        )
    )


def _snapshot_payloads(
    snapshot: _ElementSourceSnapshot,
) -> tuple[_SourcePayload, ...]:
    return tuple(
        payload
        for payload in (
            snapshot.primary,
            snapshot.contract,
            *snapshot.mounts,
        )
        if payload is not None
    )


def _snapshot_digest(
    spec: dict,
    elements: tuple[_ElementSourceSnapshot, ...],
) -> str:
    canonical_spec = json.dumps(
        spec,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload = b"\0".join(
        (
            canonical_spec,
            *tuple(
                source.label.encode("utf-8") + b"\0" + source.content
                for element in elements
                for source in _snapshot_payloads(element)
            ),
        )
    )
    return hashlib.sha256(payload).hexdigest()


def _capture_assembly_snapshot(
    spec: dict,
    base_dir: Path,
) -> _AssemblySourceSnapshot | ElementSourceObstruction:
    try:
        elements = tuple(
            _capture_element_source(entry, base_dir) for entry in spec["elements"]
        )
        return _AssemblySourceSnapshot(elements, _snapshot_digest(spec, elements))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as failure:
        return ElementSourceObstruction("<assembly>", str(failure))


def _assembly_source_digest(
    spec: dict,
    base_dir: Path,
) -> str | ElementSourceObstruction:
    snapshot = _capture_assembly_snapshot(spec, base_dir)
    return snapshot.digest if isinstance(snapshot, _AssemblySourceSnapshot) else snapshot


def _snapshot_body_spec(
    snapshot: _ElementSourceSnapshot,
) -> dict:
    body_spec = json.loads(snapshot.primary.content) if snapshot.primary is not None else {}
    contract = (
        json.loads(snapshot.contract.content)
        if snapshot.contract is not None
        else body_spec.get("contract")
    )
    mounts = tuple(
        {
            **mount,
            "spec": (
                f"mount-{index}.json"
                if source is not None
                else f"missing-mount-{index}.json"
            ),
        }
        if isinstance(mount, dict)
        else mount
        for index, (mount, source) in enumerate(
            zip(body_spec.get("mounts", ()), snapshot.mounts, strict=True)
        )
    )
    return {
        **body_spec,
        **({"contract": contract} if contract is not None else {}),
        **({"mounts": mounts} if "mounts" in body_spec else {}),
    }


def _compile_body_snapshot(
    snapshot: _ElementSourceSnapshot,
) -> CompiledBody | ElementBodyObstruction:
    body_spec = _snapshot_body_spec(snapshot)
    with TemporaryDirectory(prefix="golem-assembly-snapshot-") as directory:
        spec_dir = Path(directory)
        tuple(
            (spec_dir / f"mount-{index}.json").write_bytes(source.content)
            for index, source in enumerate(snapshot.mounts)
            if source is not None
        )
        result = Compiler(body_spec, spec_dir=spec_dir).compile()
        match result:
            case CompiledBody() as compiled:
                graph_extensions = {
                    key: body_spec[key]
                    for key in ("_intents", "appendages", "conduits")
                    if key in body_spec
                }
                return replace(
                    compiled,
                    graph={**compiled.graph, **graph_extensions},
                )
            case RejectedBody(obstructions):
                return ElementBodyObstruction(
                    snapshot.element_id,
                    obstructions,
                )
            case _ as unreachable:
                assert_never(unreachable)


def _load_element_snapshot(
    entry: dict,
    snapshot: _ElementSourceSnapshot,
) -> _LoadedElement | ElementBodyObstruction | ElementSourceObstruction:
    try:
        loaded: dict | CompiledBody | ElementBodyObstruction = (
            entry["graph"]
            if snapshot.kind is _ElementSourceKind.INLINE_GRAPH
            else json.loads(snapshot.primary.content)
            if snapshot.kind is _ElementSourceKind.GRAPH_FILE
            and snapshot.primary is not None
            else _compile_body_snapshot(snapshot)
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as failure:
        return ElementSourceObstruction(snapshot.element_id, str(failure))
    match loaded:
        case ElementBodyObstruction():
            return loaded
        case dict():
            source_graph, anatomy, eye_globes = loaded, None, ()
        case CompiledBody(graph, _, anatomy, eye_globes):
            source_graph = graph
        case _ as unreachable:
            assert_never(unreachable)
    return _LoadedElement(
        entry=entry,
        source_graph=source_graph,
        anatomy=anatomy,
        eye_globes=eye_globes,
    )


def load_element_graph(
    entry: dict, base_dir: Path
) -> dict | ElementBodyObstruction | ElementSourceObstruction:
    loaded = _load_element(entry, base_dir)
    return loaded.source_graph if isinstance(loaded, _LoadedElement) else loaded


def _load_element(
    entry: dict,
    base_dir: Path,
) -> _LoadedElement | ElementBodyObstruction | ElementSourceObstruction:
    try:
        snapshot = _capture_element_source(entry, base_dir)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as failure:
        return ElementSourceObstruction(str(entry.get("id", "<unknown>")), str(failure))
    return _load_element_snapshot(entry, snapshot)


def _known_semantic_addresses(
    loaded: tuple[_LoadedElement, ...],
) -> frozenset[SemanticAddress]:
    return frozenset(
        address
        for element in loaded
        for address in _element_semantic_addresses(element)
    )


def _element_semantic_addresses(
    element: _LoadedElement,
) -> tuple[SemanticAddress, ...]:
    element_id = element.entry["id"]
    landmarks = element.source_graph.get("intent", {}).get("landmarks", {})
    bone_ids = tuple(
        dict.fromkeys(
            landmark.partition("/")[0]
            for landmark in landmarks
            if isinstance(landmark, str) and "/" in landmark
        )
    )
    regions = (
        element.anatomy.overall.regions
        if isinstance(element.anatomy, AcceptedAnatomy)
        else ()
    )
    pump_ports: tuple[SemanticAddress, ...] = (
        (
            PortAddress(element_id, "pump_outlet"),
            PortAddress(element_id, "pump_inlet"),
        )
        if isinstance(element.anatomy, AcceptedAnatomy)
        else ()
    )
    return (
        ElementAddress(element_id),
        MountAddress(element_id),
        *tuple(BoneAddress(element_id, bone_id) for bone_id in bone_ids),
        *tuple(
            RegionAddress(element_id, region.region_id) for region in regions
        ),
        *pump_ports,
    )
