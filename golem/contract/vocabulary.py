from __future__ import annotations

import json
from dataclasses import dataclass

from golem.assembly import AssemblyStratum, ElementRole
from golem.contract.model import ContractSection
from golem.kernel.body.relations import (
    RELATION_ALIASES,
    ConventionId,
    RelationKind,
    RelationSolvePolicy,
)
from golem.materials import MATERIAL_CATALOGUE
from golem.paths import SPECS


@dataclass(frozen=True)
class VocabularyEntry:
    kind_name: str
    frame_doc: str


def element_role_values() -> tuple[str, ...]:
    return tuple(role.value for role in ElementRole)


def appearance_material_values() -> tuple[str, ...]:
    return tuple(
        material.material_id.value
        for material in MATERIAL_CATALOGUE.appearance_materials
    )


def assembly_stratum_values() -> tuple[str, ...]:
    return tuple(stratum.value for stratum in AssemblyStratum)


def _decode_entry(value: object) -> VocabularyEntry:
    if not isinstance(value, dict):
        raise TypeError("vocabulary entry must be an object")
    kind_name = value.get("kind_name")
    frame_doc = value.get("frame_doc")
    if not isinstance(kind_name, str) or not isinstance(frame_doc, str):
        raise TypeError("vocabulary entry requires string kind_name and frame_doc")
    return VocabularyEntry(kind_name, frame_doc)


def vocabulary_entries() -> tuple[VocabularyEntry, ...]:
    payload = json.loads((SPECS / "vocabulary.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("vocab"), list):
        raise TypeError("specs/vocabulary.json requires a vocab array")
    return tuple(map(_decode_entry, payload["vocab"]))


def render() -> str:
    geometry = "\n\n".join(
        f"{entry.kind_name}\n{entry.frame_doc}"
        for entry in vocabulary_entries()
    )
    closed_sets = "\n".join(
        (
            "TYPED CLOSED SETS",
            "ElementRole: " + ", ".join(element_role_values()),
            "AppearanceMaterialId: " + ", ".join(appearance_material_values()),
            "AssemblyStratum: " + ", ".join(assembly_stratum_values()),
            "RelationKind: "
            + ", ".join(kind.value for kind in RelationKind),
            "RelationSolvePolicy: "
            + ", ".join(policy.value for policy in RelationSolvePolicy),
            "ConventionId: "
            + ", ".join(convention.value for convention in ConventionId),
            "RelationKind aliases (surface sugar, canonicalized on decode): "
            + ", ".join(
                f"{alias}->{kind.value}"
                for alias, kind in RELATION_ALIASES.items()
            ),
        )
    )
    return "\n\n".join(("LIVE GEOMETRY VOCABULARY", geometry, closed_sets))


SECTION = ContractSection("vocabulary", render)
