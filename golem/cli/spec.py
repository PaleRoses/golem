from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from golem.json_value import is_json_value
from golem.kernel.body import DIALECT as BODY_DIALECT

BODY_DOCUMENT = BODY_DIALECT
GRAPH_DOCUMENT = "graph/parts"
ASSEMBLY_DOCUMENT = "assembly/elements"


@dataclass(frozen=True)
class LoadedSpec:
    path: Path
    payload: dict[str, object]

    @property
    def base_dir(self) -> Path:
        return self.path.resolve().parent


@dataclass(frozen=True)
class SpecSourceObstruction:
    path: Path
    kind: str
    reason: str


@dataclass(frozen=True)
class UnrecognizedDocumentObstruction:
    path: Path
    accepted_dialects: tuple[str, ...]


type SpecLoadResult = LoadedSpec | SpecSourceObstruction


def load_spec(path: Path) -> SpecLoadResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as failure:
        return SpecSourceObstruction(path, type(failure).__name__, str(failure))
    return (
        SpecSourceObstruction(
            path,
            "MalformedSpec",
            "top-level JSON value must be an object",
        )
        if not isinstance(payload, dict)
        else SpecSourceObstruction(
            path,
            "NonFiniteJsonNumber",
            "JSON numbers must be finite",
        )
        if not is_json_value(payload)
        else LoadedSpec(path, payload)
    )


def is_body_document(payload: dict[str, object]) -> bool:
    return payload.get("dialect") == BODY_DIALECT


def is_raw_graph(payload: dict[str, object]) -> bool:
    return "parts" in payload


def is_assembly_document(payload: dict[str, object]) -> bool:
    return isinstance(payload.get("elements"), list)


_DOCUMENT_PREDICATES: dict[str, Callable[[dict[str, object]], bool]] = {
    BODY_DOCUMENT: is_body_document,
    GRAPH_DOCUMENT: is_raw_graph,
    ASSEMBLY_DOCUMENT: is_assembly_document,
}


def classify_document(
    source: LoadedSpec,
    accepted: tuple[str, ...],
) -> UnrecognizedDocumentObstruction | None:
    return (
        None
        if any(_DOCUMENT_PREDICATES[name](source.payload) for name in accepted)
        else UnrecognizedDocumentObstruction(source.path, accepted)
    )


def element_identifier(source: LoadedSpec) -> str:
    name = source.payload.get("name")
    return name if isinstance(name, str) and name else source.path.stem


def element_entry(source: LoadedSpec) -> dict[str, object]:
    payload = source.payload
    source_key = "body" if is_body_document(payload) else "graph"
    projected_options = {
        key: payload[key]
        for key in (
            "appearance_material",
            "appearance_palette",
            "plate_policy",
            "surface_color",
            "surface_detail",
        )
        if key in payload
    }
    return {
        "id": element_identifier(source),
        "role": "creature",
        source_key: payload,
        **projected_options,
    }


def assembly_document(source: LoadedSpec) -> dict[str, object]:
    payload = source.payload
    return (
        payload
        if isinstance(payload.get("elements"), list)
        else {
            "name": element_identifier(source),
            "elements": [element_entry(source)],
            **({"pitch": payload["pitch"]} if "pitch" in payload else {}),
        }
    )


def render_source_obstruction(obstruction: SpecSourceObstruction) -> str:
    return (
        f"REJECTED [{obstruction.kind}] {obstruction.path}: "
        f"{obstruction.reason}\n"
    )


def render_unrecognized_document(
    obstruction: UnrecognizedDocumentObstruction,
) -> str:
    return (
        f"REJECTED [UnrecognizedDocument] {obstruction.path}: "
        f"document matches no accepted dialect "
        f"(accepted: {', '.join(obstruction.accepted_dialects)})\n"
    )
