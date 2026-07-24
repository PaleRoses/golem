"""Closed relation vocabulary, declaration decode, and relational obstruction ADTs."""

from __future__ import annotations

import difflib
import functools
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from golem.addressing.core import parse_scope
from golem.addressing.scope import (
    Bone,
    Landmark,
    Part,
    RejectedScope,
    Scope,
    ScopeKind,
)


class RelationKind(StrEnum):
    ATTACH_AT_NAMED_SITE = "attach_at_named_site"
    ABOVE = "above"
    BELOW = "below"
    MIRROR_OF = "mirror_of"
    PERPENDICULAR_TO = "perpendicular_to"
    COINCIDENT = "coincident"
    ALIGNED = "aligned"
    BETWEEN = "between"


class RelationSolvePolicy(StrEnum):
    SUBJECT = "subject"
    REFERENCE = "reference"
    NEGOTIATE = "negotiate"


class ConventionId(StrEnum):
    ALIGNED_PARALLEL = "aligned_parallel"
    ALIGNED_ANTIPARALLEL = "aligned_antiparallel"
    PERPENDICULAR_POSITIVE_QUARTER = "perpendicular_positive_quarter"
    PERPENDICULAR_NEGATIVE_QUARTER = "perpendicular_negative_quarter"
    MIRROR_PROPER = "mirror_proper"
    BETWEEN_FORWARD = "between_forward"
    BETWEEN_REVERSED = "between_reversed"


RELATION_ALIASES: Mapping[str, RelationKind] = MappingProxyType(
    {
        "attach_at": RelationKind.ATTACH_AT_NAMED_SITE,
        "attached_at": RelationKind.ATTACH_AT_NAMED_SITE,
        "attach_to_site": RelationKind.ATTACH_AT_NAMED_SITE,
        "over": RelationKind.ABOVE,
        "under": RelationKind.BELOW,
        "mirrored_from": RelationKind.MIRROR_OF,
        "reflection_of": RelationKind.MIRROR_OF,
        "perpendicular": RelationKind.PERPENDICULAR_TO,
        "orthogonal": RelationKind.PERPENDICULAR_TO,
        "orthogonal_to": RelationKind.PERPENDICULAR_TO,
        "coincident_with": RelationKind.COINCIDENT,
        "coincides_with": RelationKind.COINCIDENT,
        "align": RelationKind.ALIGNED,
        "aligned_with": RelationKind.ALIGNED,
        "parallel_to": RelationKind.ALIGNED,
        "spanning": RelationKind.BETWEEN,
        "bridges": RelationKind.BETWEEN,
        "squeezed_between": RelationKind.BETWEEN,
    }
)

RELATION_CONVENTIONS: Mapping[RelationKind, tuple[ConventionId, ...]] = (
    MappingProxyType(
        {
            RelationKind.ALIGNED: (
                ConventionId.ALIGNED_PARALLEL,
                ConventionId.ALIGNED_ANTIPARALLEL,
            ),
            RelationKind.PERPENDICULAR_TO: (
                ConventionId.PERPENDICULAR_POSITIVE_QUARTER,
                ConventionId.PERPENDICULAR_NEGATIVE_QUARTER,
            ),
            RelationKind.MIRROR_OF: (ConventionId.MIRROR_PROPER,),
            RelationKind.BETWEEN: (
                ConventionId.BETWEEN_FORWARD,
                ConventionId.BETWEEN_REVERSED,
            ),
        }
    )
)

SUPPORTED_RELATION_SCOPES: Mapping[RelationKind, tuple[ScopeKind, ...]] = (
    MappingProxyType(
        {
            RelationKind.ATTACH_AT_NAMED_SITE: (ScopeKind.LANDMARK,),
            RelationKind.ABOVE: (ScopeKind.BONE, ScopeKind.PART),
            RelationKind.BELOW: (ScopeKind.BONE, ScopeKind.PART),
            RelationKind.MIRROR_OF: (ScopeKind.BONE, ScopeKind.PART),
            RelationKind.PERPENDICULAR_TO: (
                ScopeKind.BONE,
                ScopeKind.PART,
                ScopeKind.LANDMARK,
            ),
            RelationKind.COINCIDENT: (ScopeKind.PART, ScopeKind.LANDMARK),
            RelationKind.ALIGNED: (
                ScopeKind.BONE,
                ScopeKind.PART,
                ScopeKind.LANDMARK,
            ),
            RelationKind.BETWEEN: (ScopeKind.LANDMARK,),
        }
    )
)

_SUBJECT_SCOPE_OVERRIDES: Mapping[RelationKind, tuple[ScopeKind, ...]] = (
    MappingProxyType({RelationKind.BETWEEN: (ScopeKind.BONE,)})
)

_DISTANCE_KINDS = frozenset((RelationKind.ABOVE, RelationKind.BELOW))


@dataclass(frozen=True)
class MalformedRelationObstruction:
    address: str
    reason: str


@dataclass(frozen=True)
class DuplicateRelationIdObstruction:
    address: str
    relation_id: str


@dataclass(frozen=True)
class UnknownRelationKindObstruction:
    address: str
    authored_kind: str
    closest_valid_candidates: tuple[str, ...]


@dataclass(frozen=True)
class UnresolvedRelationSelectorObstruction:
    address: str
    selector: str
    closest_valid_candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnsupportedRelationSelectorObstruction:
    address: str
    selector: str
    supported_scope_kinds: tuple[ScopeKind, ...]


@dataclass(frozen=True)
class UnreferenceableDerivedGeometryObstruction:
    address: str
    selector: str
    reason: str


@dataclass(frozen=True)
class FixedPlacementConflictObstruction:
    relation_id: str
    subject_address: str
    reference_address: str
    required: float
    observed: float


@dataclass(frozen=True)
class GoalRelationAuthorityObstruction:
    relation_id: str
    goal_id: str
    bone_addresses: tuple[str, ...]


@dataclass(frozen=True)
class UnderconstrainedRelationObstruction:
    owner_address: str
    variable_count: int
    rank: int
    free_dimension_count: int
    parameter_addresses: tuple[str, ...]


@dataclass(frozen=True)
class RelationalSolveExhaustedObstruction:
    owner_address: str
    stage: str
    depth: int
    relation_id: str
    required: float
    observed: float
    maximum_normalized_residual: float
    attempted_evaluations: int


@dataclass(frozen=True)
class RelationalSolverBudgetObstruction:
    owner_address: str
    required_evaluations: int
    observed_evaluations: int
    best_residual: float


@dataclass(frozen=True)
class BranchCycleObstruction:
    relation_id: str
    active_sets: tuple[tuple[str, ...], ...]
    required: float
    observed: float


@dataclass(frozen=True)
class BranchSearchBudgetObstruction:
    relation_id: str
    required_active_sets: int
    observed_active_sets: int
    best_residual: float


type RelationDecodeObstruction = (
    MalformedRelationObstruction
    | DuplicateRelationIdObstruction
    | UnknownRelationKindObstruction
    | UnresolvedRelationSelectorObstruction
    | UnsupportedRelationSelectorObstruction
    | UnreferenceableDerivedGeometryObstruction
)

type RelationSolveObstruction = (
    FixedPlacementConflictObstruction
    | GoalRelationAuthorityObstruction
    | UnderconstrainedRelationObstruction
    | RelationalSolveExhaustedObstruction
    | RelationalSolverBudgetObstruction
    | BranchCycleObstruction
    | BranchSearchBudgetObstruction
)

type BodyRelationObstruction = (
    RelationDecodeObstruction | RelationSolveObstruction
)


@dataclass(frozen=True)
class FixedLocalAttachment:
    t: float = 1.0
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class NamedSiteAttachment:
    site: Landmark
    selector: str


type Attachment = FixedLocalAttachment | NamedSiteAttachment


@dataclass(frozen=True)
class RelationDeclaration:
    relation_id: str
    kind: RelationKind
    authored_kind: str
    subject: Scope
    subject_selector: str
    reference: Scope
    reference_selector: str
    solve: RelationSolvePolicy
    distance: float
    address: str
    reference_b: Scope | None = None
    reference_b_selector: str | None = None


@dataclass(frozen=True)
class AcceptedRelationDecode:
    declarations: tuple[RelationDeclaration, ...]


@dataclass(frozen=True)
class RejectedRelationDecode:
    obstructions: tuple[RelationDecodeObstruction, ...]


type RelationDecodeResult = AcceptedRelationDecode | RejectedRelationDecode


def _normalize_kind(authored: str) -> str:
    return authored.strip().lower().replace("-", "_")


def closest_relation_kinds(authored: str) -> tuple[str, ...]:
    matches = difflib.get_close_matches(
        _normalize_kind(authored),
        [*RelationKind, *RELATION_ALIASES],
        n=3,
    )
    canonical = tuple(
        RELATION_ALIASES.get(match, match) for match in matches
    )
    return tuple(dict.fromkeys(str(kind) for kind in canonical))


def resolve_relation_kind(
    authored: str, address: str
) -> tuple[RelationKind, str] | UnknownRelationKindObstruction:
    normalized = _normalize_kind(authored)
    try:
        return RelationKind(normalized), authored
    except ValueError:
        alias = RELATION_ALIASES.get(normalized)
        return (
            (alias, authored)
            if alias is not None
            else UnknownRelationKindObstruction(
                address, authored, closest_relation_kinds(authored)
            )
        )


def _scope_kind(scope: Scope) -> ScopeKind:
    match scope:
        case Bone():
            return ScopeKind.BONE
        case Part():
            return ScopeKind.PART
        case Landmark():
            return ScopeKind.LANDMARK
        case _:
            return ScopeKind(type(scope).__name__.lower())


def _decode_selector(
    selector: object,
    kind: RelationKind,
    address: str,
    supported: tuple[ScopeKind, ...] | None = None,
) -> Scope | RelationDecodeObstruction:
    if not isinstance(selector, str) or not selector:
        return MalformedRelationObstruction(
            address, f"selector must be a non-empty string, got {selector!r}"
        )
    scope = parse_scope(selector)
    if isinstance(scope, RejectedScope):
        return UnresolvedRelationSelectorObstruction(address, selector)
    scopes = SUPPORTED_RELATION_SCOPES[kind] if supported is None else supported
    return (
        scope
        if _scope_kind(scope) in scopes
        else UnsupportedRelationSelectorObstruction(
            address, selector, scopes
        )
    )


def _decode_distance(
    value: object, kind: RelationKind, address: str
) -> float | MalformedRelationObstruction:
    if value is None:
        return 0.0
    if kind not in _DISTANCE_KINDS:
        return MalformedRelationObstruction(
            address,
            f"distance is only lawful on {sorted(k.value for k in _DISTANCE_KINDS)},"
            f" not {kind.value}",
        )
    try:
        distance = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return MalformedRelationObstruction(
            address, f"distance must be a finite non-negative number, got {value!r}"
        )
    return (
        distance
        if math.isfinite(distance) and distance >= 0.0
        else MalformedRelationObstruction(
            address, f"distance must be finite and non-negative, got {distance!r}"
        )
    )


def _decode_declaration(
    item: object, index: int
) -> RelationDeclaration | tuple[RelationDecodeObstruction, ...]:
    address = f"pose/relations/{index}"
    if not isinstance(item, Mapping):
        return (
            MalformedRelationObstruction(
                address, f"relation must be an object, got {type(item).__name__}"
            ),
        )
    relation_id = item.get("id")
    if not isinstance(relation_id, str) or not relation_id:
        return (
            MalformedRelationObstruction(
                address, "relation requires a non-empty string id"
            ),
        )
    address = f"pose/relations/{relation_id}"
    authored_kind = item.get("kind")
    if not isinstance(authored_kind, str) or not authored_kind:
        return (
            MalformedRelationObstruction(
                address, "relation requires a non-empty string kind"
            ),
        )
    resolved = resolve_relation_kind(authored_kind, address)
    if isinstance(resolved, UnknownRelationKindObstruction):
        return (resolved,)
    kind, authored = resolved
    if kind is RelationKind.ATTACH_AT_NAMED_SITE:
        return (
            MalformedRelationObstruction(
                address,
                "attachment is owned by the bone attach field;"
                ' author {"attach": {"at": "landmark:..."}} on the bone'
                " instead of a pose relation",
            ),
        )
    solve_raw = item.get("solve")
    try:
        solve = RelationSolvePolicy(solve_raw)  # type: ignore[arg-type]
    except ValueError:
        return (
            MalformedRelationObstruction(
                address,
                f"solve must be one of {[p.value for p in RelationSolvePolicy]},"
                f" got {solve_raw!r}",
            ),
        )
    if kind is RelationKind.BETWEEN and solve is RelationSolvePolicy.REFERENCE:
        return (
            MalformedRelationObstruction(
                address,
                "between admits solve policies ['negotiate', 'subject']:"
                " its two references are peers with no single yielding side",
            ),
        )
    if kind is RelationKind.BETWEEN and "reference" in item:
        return (
            MalformedRelationObstruction(
                address,
                'between requires exactly two references: author'
                ' "references": [first, second], not "reference"',
            ),
        )
    if kind is not RelationKind.BETWEEN and "references" in item:
        return (
            MalformedRelationObstruction(
                address,
                f'{kind.value} admits exactly one reference;'
                ' author "reference"',
            ),
        )
    reference_key = (
        "references" if kind is RelationKind.BETWEEN else "reference"
    )
    unknown_keys = tuple(
        sorted(
            set(item)
            - {"id", "kind", "subject", reference_key, "solve", "distance"}
        )
    )
    if unknown_keys:
        return (
            MalformedRelationObstruction(
                address, f"unknown relation keys {list(unknown_keys)}"
            ),
        )
    subject = _decode_selector(
        item.get("subject"),
        kind,
        f"{address}/subject",
        _SUBJECT_SCOPE_OVERRIDES.get(kind),
    )
    if kind is RelationKind.BETWEEN:
        references_raw = item.get("references")
        if (
            not isinstance(references_raw, Sequence)
            or isinstance(references_raw, str)
            or len(references_raw) != 2
        ):
            return (
                MalformedRelationObstruction(
                    address,
                    "between requires exactly two references,"
                    f" got {references_raw!r}",
                ),
            )
        reference = _decode_selector(
            references_raw[0], kind, f"{address}/references/0"
        )
        reference_b = _decode_selector(
            references_raw[1], kind, f"{address}/references/1"
        )
        reference_selectors = (str(references_raw[0]), str(references_raw[1]))
    else:
        reference = _decode_selector(
            item.get("reference"), kind, f"{address}/reference"
        )
        reference_b = None
        reference_selectors = (str(item.get("reference")), None)
    distance = _decode_distance(item.get("distance"), kind, address)
    obstruction_types = (
        MalformedRelationObstruction,
        UnresolvedRelationSelectorObstruction,
        UnsupportedRelationSelectorObstruction,
        UnreferenceableDerivedGeometryObstruction,
    )
    obstructions = tuple(
        component
        for component in (subject, reference, reference_b, distance)
        if isinstance(component, obstruction_types)
    )
    if obstructions:
        return obstructions
    assert not isinstance(subject, obstruction_types)
    assert not isinstance(reference, obstruction_types)
    assert not isinstance(reference_b, obstruction_types)
    assert isinstance(distance, float)
    return RelationDeclaration(
        relation_id=relation_id,
        kind=kind,
        authored_kind=authored,
        subject=subject,
        subject_selector=str(item["subject"]),
        reference=reference,
        reference_selector=reference_selectors[0],
        solve=solve,
        distance=distance,
        address=address,
        reference_b=reference_b,
        reference_b_selector=reference_selectors[1],
    )


def decode_relations(relations: object) -> RelationDecodeResult:
    if relations is None:
        return AcceptedRelationDecode(())
    if not isinstance(relations, Sequence) or isinstance(relations, str):
        return RejectedRelationDecode(
            (
                MalformedRelationObstruction(
                    "pose/relations",
                    f"relations must be a list, got {type(relations).__name__}",
                ),
            )
        )

    authored_ids = tuple(
        item["id"]
        for item in relations
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    )
    duplicates = tuple(
        DuplicateRelationIdObstruction(
            f"pose/relations/{relation_id}", relation_id
        )
        for relation_id, count in Counter(authored_ids).items()
        if count > 1
    )

    def fold(
        state: tuple[
            tuple[RelationDeclaration, ...],
            tuple[RelationDecodeObstruction, ...],
        ],
        indexed: tuple[int, object],
    ) -> tuple[
        tuple[RelationDeclaration, ...],
        tuple[RelationDecodeObstruction, ...],
    ]:
        declarations, obstructions = state
        index, item = indexed
        decoded = _decode_declaration(item, index)
        return (
            (declarations + (decoded,), obstructions)
            if isinstance(decoded, RelationDeclaration)
            else (declarations, obstructions + decoded)
        )

    declarations, obstructions = functools.reduce(
        fold, enumerate(relations), ((), ())
    )
    return (
        RejectedRelationDecode(duplicates + obstructions)
        if duplicates or obstructions
        else AcceptedRelationDecode(declarations)
    )


def decode_attachment(
    attach: object, address: str
) -> Attachment | RelationDecodeObstruction:
    if attach is None:
        return FixedLocalAttachment()
    if not isinstance(attach, Mapping):
        return MalformedRelationObstruction(
            address, f"attach must be an object, got {type(attach).__name__}"
        )
    if "at" in attach:
        overlap = tuple(sorted({"t", "offset"} & set(attach)))
        if overlap:
            return MalformedRelationObstruction(
                address,
                f"attach mixes 'at' with {list(overlap)}: a record admits"
                " exactly one placement owner",
            )
        selector = attach["at"]
        site = _decode_selector(
            selector, RelationKind.ATTACH_AT_NAMED_SITE, address
        )
        return (
            NamedSiteAttachment(site=site, selector=str(selector))
            if isinstance(site, Landmark)
            else site
        )
    unknown_keys = tuple(sorted(set(attach) - {"t", "offset"}))
    if unknown_keys:
        return MalformedRelationObstruction(
            address, f"unknown attach keys {list(unknown_keys)}"
        )
    try:
        t = float(attach.get("t", 1.0))
        offset_raw = attach.get("offset", (0.0, 0.0, 0.0))
        offset = tuple(float(component) for component in offset_raw)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return MalformedRelationObstruction(
            address, f"attach t/offset must be numeric, got {dict(attach)!r}"
        )
    if len(offset) != 3 or not all(map(math.isfinite, (t, *offset))):
        return MalformedRelationObstruction(
            address, "attach offset must be three finite numbers"
        )
    return FixedLocalAttachment(t=t, offset=(offset[0], offset[1], offset[2]))


__all__ = [
    "AcceptedRelationDecode",
    "Attachment",
    "BodyRelationObstruction",
    "BranchCycleObstruction",
    "BranchSearchBudgetObstruction",
    "ConventionId",
    "DuplicateRelationIdObstruction",
    "FixedLocalAttachment",
    "FixedPlacementConflictObstruction",
    "GoalRelationAuthorityObstruction",
    "MalformedRelationObstruction",
    "NamedSiteAttachment",
    "RELATION_ALIASES",
    "RELATION_CONVENTIONS",
    "RejectedRelationDecode",
    "RelationDeclaration",
    "RelationDecodeObstruction",
    "RelationDecodeResult",
    "RelationKind",
    "RelationSolveObstruction",
    "RelationSolvePolicy",
    "RelationalSolveExhaustedObstruction",
    "RelationalSolverBudgetObstruction",
    "SUPPORTED_RELATION_SCOPES",
    "UnderconstrainedRelationObstruction",
    "UnknownRelationKindObstruction",
    "UnreferenceableDerivedGeometryObstruction",
    "UnresolvedRelationSelectorObstruction",
    "UnsupportedRelationSelectorObstruction",
    "closest_relation_kinds",
    "decode_attachment",
    "decode_relations",
]
