from __future__ import annotations

import difflib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import assert_never

import numpy as np

from golem.addressing.core import parse_scope
from golem.addressing.scope import (
    Bone,
    Landmark,
    Part,
    RejectedScope,
    Scope,
    ScopeKind,
)
from golem.addressing.scope_grammar import scope_kind
from golem.kernel.body.canonical import _r, _rvec
from golem.kernel.body.decode import _finite_number
from golem.kernel.body.gencyl import _loft_sections
from golem.kernel.body.geometry import (
    BodyGeometryView,
    build_tree,
    flesh_owner_map,
    landmark_owner,
    port_owner_map,
)
from golem.kernel.body.types import (
    AbsoluteMuscleBulk,
    CyclicMuscleMirrorObstruction,
    DegenerateMuscleSpanObstruction,
    DuplicateMuscleIdObstruction,
    InvalidMuscleProfileObstruction,
    LoftSection,
    LoftSectionObstruction,
    MalformedMuscleBulkObstruction,
    MalformedMuscleDeclarationObstruction,
    MalformedMuscleSectionsObstruction,
    MuscleBulk,
    MuscleBulkBasis,
    MuscleBulkDimensions,
    MuscleBulkReference,
    MuscleBulkRule,
    MuscleDeclarationRule,
    MuscleEndpoint,
    MuscleMirrorAnchorMismatchObstruction,
    MuscleObstruction,
    MusclePartCollisionObstruction,
    MuscleProfileRule,
    RelativeMuscleBulk,
    UnresolvableMuscleAnchorObstruction,
    UnresolvableMuscleBulkBasisObstruction,
    UnresolvedMuscleMirrorObstruction,
    UnsupportedMuscleAnchorObstruction,
    Vector3,
)
from golem.kernel.engine.types import (
    Accepted,
    BlobPart,
    BoxPart,
    CompositionOperator,
    GencylPart,
    GeometryObstruction,
    GeometryRule,
    MuscleFormationEvidence,
    MuscleFormationKind,
    MuscleFormationObstruction,
    Rejected,
    decode_part,
    profile_anchor_depths,
)
from golem.kernel.engine.algebra import solve_muscle_formation


MINIMUM_MUSCLE_SPAN = 1.0e-4
MUSCLE_MIRROR_TOLERANCE = 1.0e-6
_MIRROR = np.asarray((-1.0, 1.0, 1.0), dtype=np.float64)


@dataclass(frozen=True)
class _MuscleDeclaration:
    muscle_id: str
    origin_selector: str
    insertion_selector: str
    blend: float | None
    operator: CompositionOperator | None
    formation: MuscleFormationKind | None
    address: str


@dataclass(frozen=True)
class BaseMuscleDeclaration(_MuscleDeclaration):
    sections: tuple[LoftSection, ...]


@dataclass(frozen=True)
class NaturalMuscleDeclaration(_MuscleDeclaration):
    bulk: MuscleBulk
    definition: float


@dataclass(frozen=True)
class MirroredMuscleDeclaration(_MuscleDeclaration):
    mirror_of: str


type MuscleDeclaration = (
    BaseMuscleDeclaration
    | NaturalMuscleDeclaration
    | MirroredMuscleDeclaration
)


@dataclass(frozen=True)
class ResolvedMuscleAnchor:
    selector: str
    point: Vector3
    up: Vector3


@dataclass(frozen=True)
class ResolvedMuscle:
    declaration: MuscleDeclaration
    origin: ResolvedMuscleAnchor
    insertion: ResolvedMuscleAnchor


@dataclass(frozen=True)
class RealizedMuscle:
    part: dict
    sections: tuple[LoftSection, ...]
    formation_evidence: MuscleFormationEvidence | None = None


@dataclass(frozen=True)
class AcceptedMyology:
    parts: tuple[dict, ...]
    provenance: tuple[tuple[str, str], ...]
    receipt: tuple[dict, ...]


@dataclass(frozen=True)
class RejectedMyology:
    obstructions: tuple[MuscleObstruction, ...]


type MyologyResult = AcceptedMyology | RejectedMyology


def _field_obstruction(
    address: str,
    field: str,
    rule: MuscleDeclarationRule,
    authored_value: object,
) -> MalformedMuscleDeclarationObstruction:
    return MalformedMuscleDeclarationObstruction(
        address=f"{address}/{field}",
        field=field,
        rule=rule,
        authored_value=authored_value,
    )


def _decode_operator(
    value: object, address: str
) -> CompositionOperator | MalformedMuscleDeclarationObstruction | None:
    return (
        None
        if value is None
        else next(
            (
                operator
                for operator in CompositionOperator
                if value == operator.value
            ),
            _field_obstruction(
                address,
                "operator",
                MuscleDeclarationRule.OPERATOR,
                value,
            ),
        )
    )


def _decode_formation(
    value: object, address: str
) -> MuscleFormationKind | MalformedMuscleDeclarationObstruction | None:
    return (
        None
        if value is None
        else next(
            (
                formation
                for formation in MuscleFormationKind
                if value == formation.value
            ),
            _field_obstruction(
                address,
                "formation",
                MuscleDeclarationRule.FORMATION,
                value,
            ),
        )
    )


def _loft_section_obstruction(
    address: str,
    muscle_id: str,
    obstruction: LoftSectionObstruction,
) -> MalformedMuscleSectionsObstruction:
    return MalformedMuscleSectionsObstruction(
        address=(
            f"{address}/sections[{obstruction.section_index}]"
            if obstruction.section_index is not None
            else f"{address}/sections"
        ),
        muscle_id=muscle_id,
        section_index=obstruction.section_index,
        rule=obstruction.rule,
        field=obstruction.field,
        observed_value=obstruction.observed_value,
        missing_fields=obstruction.missing_fields,
        extra_fields=obstruction.extra_fields,
    )


def _profile_obstruction(
    address: str,
    muscle_id: str,
    sections: tuple[LoftSection, ...],
) -> InvalidMuscleProfileObstruction | None:
    first = sections[0] if sections else None
    last = sections[-1] if sections else None
    interior = sections[1:-1]
    maximum_interior_width = (
        max(section.width for section in interior) if interior else None
    )
    maximum_interior_depth = (
        max(section.depth for section in interior) if interior else None
    )
    shared = {
        "address": f"{address}/sections",
        "muscle_id": muscle_id,
        "station_count": len(sections),
        "first_station": first.station if first is not None else None,
        "last_station": last.station if last is not None else None,
        "endpoint_widths": (
            (first.width, last.width)
            if first is not None and last is not None
            else None
        ),
        "endpoint_depths": (
            (first.depth, last.depth)
            if first is not None and last is not None
            else None
        ),
        "maximum_interior_width": maximum_interior_width,
        "maximum_interior_depth": maximum_interior_depth,
    }
    if len(sections) < 3:
        return InvalidMuscleProfileObstruction(
            rule=MuscleProfileRule.MINIMUM_ARITY,
            **shared,
        )
    if first is None or last is None or (first.station, last.station) != (0.0, 1.0):
        return InvalidMuscleProfileObstruction(
            rule=MuscleProfileRule.UNIT_SPAN,
            **shared,
        )
    return (
        InvalidMuscleProfileObstruction(
            rule=MuscleProfileRule.FUSIFORM_TAPER,
            **shared,
        )
        if maximum_interior_width is None
        or maximum_interior_depth is None
        or maximum_interior_width <= max(first.width, last.width)
        or maximum_interior_depth <= max(first.depth, last.depth)
        else None
    )


def _bulk_obstruction(
    address: str,
    muscle_id: str,
    rule: MuscleBulkRule,
    authored_value: object,
    missing_fields: tuple[str, ...] = (),
    extra_fields: tuple[str, ...] = (),
) -> MalformedMuscleBulkObstruction:
    return MalformedMuscleBulkObstruction(
        address=address,
        muscle_id=muscle_id,
        rule=rule,
        authored_value=authored_value,
        missing_fields=missing_fields,
        extra_fields=extra_fields,
    )


def _decode_bulk_dimensions(
    payload: object,
    address: str,
    muscle_id: str,
    fields: frozenset[str],
) -> MuscleBulkDimensions | tuple[MalformedMuscleBulkObstruction, ...]:
    if not isinstance(payload, Mapping):
        return (
            _bulk_obstruction(
                address,
                muscle_id,
                MuscleBulkRule.DIMENSIONS_OBJECT,
                payload,
            ),
        )
    authored_fields = frozenset(map(str, payload))
    if authored_fields != fields:
        return (
            _bulk_obstruction(
                address,
                muscle_id,
                MuscleBulkRule.EXACT_FIELDS,
                tuple(sorted(authored_fields)),
                tuple(sorted(fields - authored_fields)),
                tuple(sorted(authored_fields - fields)),
            ),
        )
    dimension_obstructions = tuple(
        _bulk_obstruction(
            f"{address}/{field}",
            muscle_id,
            MuscleBulkRule.FINITE_POSITIVE_DIMENSION,
            payload[field],
        )
        for field in ("width", "depth")
        if not _finite_number(payload[field]) or float(payload[field]) <= 0.0
    )
    return (
        dimension_obstructions
        if dimension_obstructions
        else MuscleBulkDimensions(
            width=float(payload["width"]),
            depth=float(payload["depth"]),
        )
    )


def _decode_bulk(
    payload: object,
    address: str,
    muscle_id: str,
) -> MuscleBulk | tuple[MalformedMuscleBulkObstruction, ...]:
    if not isinstance(payload, Mapping):
        return (
            _bulk_obstruction(
                address,
                muscle_id,
                MuscleBulkRule.BULK_OBJECT,
                payload,
            ),
        )
    fields = frozenset(map(str, payload))
    if fields not in (frozenset(("relative",)), frozenset(("absolute",))):
        return (
            _bulk_obstruction(
                address,
                muscle_id,
                MuscleBulkRule.RELATIVE_XOR_ABSOLUTE,
                tuple(sorted(fields)),
                () if fields else ("relative|absolute",),
                tuple(sorted(fields - frozenset(("relative", "absolute")))),
            ),
        )
    if "absolute" in payload:
        dimensions = _decode_bulk_dimensions(
            payload["absolute"],
            f"{address}/absolute",
            muscle_id,
            frozenset(("width", "depth")),
        )
        return (
            dimensions
            if isinstance(dimensions, tuple)
            else AbsoluteMuscleBulk(dimensions)
        )
    relative_payload = payload["relative"]
    dimensions = _decode_bulk_dimensions(
        relative_payload,
        f"{address}/relative",
        muscle_id,
        frozenset(("to", "width", "depth")),
    )
    if isinstance(dimensions, tuple):
        return dimensions
    reference_value = relative_payload["to"]
    reference = next(
        (
            candidate
            for candidate in MuscleBulkReference
            if reference_value == candidate.value
        ),
        None,
    )
    return (
        RelativeMuscleBulk(reference, dimensions)
        if reference is not None
        else (
            _bulk_obstruction(
                f"{address}/relative/to",
                muscle_id,
                MuscleBulkRule.REFERENCE,
                reference_value,
            ),
        )
    )


def _decode_declaration(
    payload: object,
    index: int,
) -> MuscleDeclaration | tuple[MuscleObstruction, ...]:
    address = f"muscles/{index}"
    if not isinstance(payload, Mapping):
        return (
            _field_obstruction(
                address,
                "declaration",
                MuscleDeclarationRule.DECLARATION_OBJECT,
                payload,
            ),
        )
    muscle_id = payload.get("id")
    origin = payload.get("origin")
    insertion = payload.get("insertion")
    mirror_of = payload.get("mirror_of")
    has_sections = "sections" in payload
    has_bulk = "bulk" in payload
    has_mirror = "mirror_of" in payload
    form_count = len(tuple(filter(None, (has_sections, has_bulk, has_mirror))))
    blend = payload.get("blend")
    definition = payload.get("definition")
    operator_result = _decode_operator(payload.get("operator"), address)
    formation_result = _decode_formation(payload.get("formation"), address)
    common_fields = frozenset(
        (
            "kind",
            "id",
            "origin",
            "insertion",
            "blend",
            "operator",
            "formation",
        )
    )
    form_fields = (
        frozenset(("sections",))
        if has_sections and form_count == 1
        else frozenset(("bulk", "definition"))
        if has_bulk and form_count == 1
        else frozenset(("mirror_of",))
        if has_mirror and form_count == 1
        else frozenset(("sections", "bulk", "definition", "mirror_of"))
    )
    extra_fields = tuple(sorted(frozenset(payload) - common_fields - form_fields))
    structural_obstructions = (
        (
            _field_obstruction(
                address,
                "kind",
                MuscleDeclarationRule.KIND,
                payload.get("kind"),
            ),
        )
        if payload.get("kind") != "muscle"
        else ()
    ) + (
        (
            _field_obstruction(
                address,
                "id",
                MuscleDeclarationRule.IDENTIFIER,
                muscle_id,
            ),
        )
        if not isinstance(muscle_id, str) or not muscle_id
        else ()
    ) + tuple(
        _field_obstruction(
            address,
            endpoint.value,
            MuscleDeclarationRule.ANCHOR_SELECTOR,
            selector,
        )
        for endpoint, selector in (
            (MuscleEndpoint.ORIGIN, origin),
            (MuscleEndpoint.INSERTION, insertion),
        )
        if not isinstance(selector, str) or not selector
    ) + (
        (
            _field_obstruction(
                address,
                "sections|bulk|mirror_of",
                MuscleDeclarationRule.PROFILE_FORM,
                {
                    "sections": payload.get("sections"),
                    "bulk": payload.get("bulk"),
                    "mirror_of": mirror_of,
                },
            ),
        )
        if form_count != 1
        else ()
    ) + (
        (
            _field_obstruction(
                address,
                "mirror_of",
                MuscleDeclarationRule.MIRROR_IDENTIFIER,
                mirror_of,
            ),
        )
        if has_mirror and (not isinstance(mirror_of, str) or not mirror_of)
        else ()
    ) + (
        (
            _field_obstruction(
                address,
                "definition",
                MuscleDeclarationRule.DEFINITION,
                definition,
            ),
        )
        if has_bulk
        and (
            not _finite_number(definition)
            or not 0.0 <= float(definition) <= 1.0
        )
        else ()
    ) + (
        tuple(
            _field_obstruction(
                address,
                field,
                MuscleDeclarationRule.EXACT_FIELDS,
                payload[field],
            )
            for field in extra_fields
        )
    ) + (
        (
            _field_obstruction(
                address,
                "blend",
                MuscleDeclarationRule.FINITE_BLEND,
                blend,
            ),
        )
        if blend is not None
        and (
            not isinstance(blend, (int, float))
            or isinstance(blend, bool)
            or not math.isfinite(float(blend))
            or float(blend) < 0.0
        )
        else ()
    ) + (
        (operator_result,)
        if isinstance(operator_result, MalformedMuscleDeclarationObstruction)
        else ()
    ) + (
        (formation_result,)
        if isinstance(
            formation_result,
            MalformedMuscleDeclarationObstruction,
        )
        else ()
    )
    if structural_obstructions:
        return structural_obstructions
    checked_id = str(muscle_id)
    common = {
        "muscle_id": checked_id,
        "origin_selector": str(origin),
        "insertion_selector": str(insertion),
        "blend": float(blend) if blend is not None else None,
        "operator": (
            operator_result
            if isinstance(operator_result, CompositionOperator)
            else None
        ),
        "formation": (
            formation_result
            if isinstance(formation_result, MuscleFormationKind)
            else None
        ),
        "address": address,
    }
    section_result = (
        _loft_sections({"sections": payload.get("sections")})
        if has_sections
        else None
    )
    if isinstance(section_result, LoftSectionObstruction):
        return (_loft_section_obstruction(address, checked_id, section_result),)
    profile_obstruction = (
        _profile_obstruction(address, checked_id, section_result)
        if isinstance(section_result, tuple)
        else None
    )
    bulk_result = (
        _decode_bulk(payload.get("bulk"), f"{address}/bulk", checked_id)
        if has_bulk
        else None
    )
    if isinstance(bulk_result, tuple):
        return bulk_result
    return (
        (profile_obstruction,)
        if profile_obstruction is not None
        else (
            BaseMuscleDeclaration(
                **common,
                sections=section_result,
            )
            if isinstance(section_result, tuple)
            else NaturalMuscleDeclaration(
                **common,
                bulk=bulk_result,
                definition=float(definition),
            )
            if isinstance(
                bulk_result,
                (AbsoluteMuscleBulk, RelativeMuscleBulk),
            )
            else MirroredMuscleDeclaration(
                **common,
                mirror_of=str(mirror_of),
            )
        )
    )


def _declaration_obstructions(
    declarations: tuple[MuscleDeclaration, ...],
    existing_part_ids: frozenset[str],
) -> tuple[MuscleObstruction, ...]:
    identifiers = tuple(declaration.muscle_id for declaration in declarations)
    duplicates = tuple(
        DuplicateMuscleIdObstruction(
            muscle_id,
            declarations[identifiers.index(muscle_id)].address,
            declarations[index].address,
        )
        for index, muscle_id in enumerate(identifiers)
        if muscle_id in identifiers[:index]
    )
    collisions = tuple(
        MusclePartCollisionObstruction(
            declaration.muscle_id,
            declaration.muscle_id,
        )
        for declaration in declarations
        if declaration.muscle_id in existing_part_ids
    )
    unresolved_mirrors = tuple(
        UnresolvedMuscleMirrorObstruction(
            declaration.muscle_id,
            str(declaration.mirror_of),
            tuple(
                difflib.get_close_matches(
                    str(declaration.mirror_of),
                    identifiers,
                    n=3,
                )
            ),
        )
        for declaration in declarations
        if isinstance(declaration, MirroredMuscleDeclaration)
        and declaration.mirror_of not in identifiers
    )
    return (*duplicates, *collisions, *unresolved_mirrors)


def _cycle_from(
    muscle_id: str,
    declarations: Mapping[str, MuscleDeclaration],
    path: tuple[str, ...] = (),
) -> tuple[str, ...] | None:
    if muscle_id in path:
        return (*path[path.index(muscle_id) :], muscle_id)
    declaration = declarations[muscle_id]
    match declaration:
        case BaseMuscleDeclaration() | NaturalMuscleDeclaration():
            return None
        case MirroredMuscleDeclaration(mirror_of=mirror_of):
            return _cycle_from(
                mirror_of,
                declarations,
                (*path, muscle_id),
            )
        case _ as unreachable:
            assert_never(unreachable)


def _mirror_cycle_obstruction(
    declarations: tuple[MuscleDeclaration, ...],
) -> CyclicMuscleMirrorObstruction | None:
    by_id = {declaration.muscle_id: declaration for declaration in declarations}
    cycle = next(
        (
            result
            for declaration in declarations
            for result in (_cycle_from(declaration.muscle_id, by_id),)
            if result is not None
        ),
        None,
    )
    return CyclicMuscleMirrorObstruction(cycle) if cycle is not None else None


def _available_anchor_selectors(view: BodyGeometryView) -> tuple[str, ...]:
    mirrored_bones = tuple(
        f"bone:{bone_id}_m"
        for bone_id in view.tree.order
        if bool(view.table[bone_id].get("mirrored"))
    )
    mirrored_parts = tuple(
        f"part:{part_id}_m"
        for part_id, (bone_id, _flesh) in view.flesh_map.items()
        if bool(view.table[bone_id].get("mirrored"))
    )
    mirrored_landmarks = tuple(
        f"landmark:{alias}"
        for name in view._points
        for alias in (_mirrored_landmark_alias(name, view),)
        if alias is not None
    )
    return (
        *tuple(f"bone:{bone_id}" for bone_id in view.tree.order),
        *tuple(f"part:{part_id}" for part_id in view.flesh_map),
        *tuple(f"landmark:{name}" for name in view._points),
        *mirrored_bones,
        *mirrored_parts,
        *mirrored_landmarks,
    )


def _mirrored_landmark_alias(
    name: str,
    view: BodyGeometryView,
) -> str | None:
    owner = landmark_owner(name, view.tree, view.ports)
    if owner is None or not bool(view.table[owner].get("mirrored")):
        return None
    prefix = f"{owner}/"
    return (
        f"{owner}_m/{name.removeprefix(prefix)}"
        if name.startswith(prefix)
        else f"{name}_m"
    )


def _direct_scope_exists(scope: Scope, view: BodyGeometryView) -> bool:
    match scope:
        case Bone(bone_id):
            return bone_id in view.tree.records
        case Part(part_id):
            return part_id in view.flesh_map
        case Landmark(name):
            return name in view._points
        case _:
            return False


def _mirrored_base_scope(
    scope: Scope,
    view: BodyGeometryView,
) -> Bone | Part | Landmark | None:
    match scope:
        case Bone(bone_id) if bone_id.endswith("_m"):
            base_id = bone_id.removesuffix("_m")
            return (
                Bone(base_id)
                if base_id in view.tree.records
                and bool(view.table[base_id].get("mirrored"))
                else None
            )
        case Part(part_id) if part_id.endswith("_m"):
            base_id = part_id.removesuffix("_m")
            owner = view.flesh_map.get(base_id)
            return (
                Part(base_id)
                if owner is not None
                and bool(view.table[owner[0]].get("mirrored"))
                else None
            )
        case Landmark(name):
            base_name = next(
                (
                    candidate
                    for candidate in view._points
                    if _mirrored_landmark_alias(candidate, view) == name
                ),
                None,
            )
            return Landmark(base_name) if base_name is not None else None
        case _:
            return None


def _scope_exists(scope: Scope, view: BodyGeometryView) -> bool:
    return (
        _direct_scope_exists(scope, view)
        or _mirrored_base_scope(scope, view) is not None
    )


def _resolve_anchor(
    muscle_id: str,
    endpoint: MuscleEndpoint,
    selector: str,
    address: str,
    view: BodyGeometryView,
) -> ResolvedMuscleAnchor | MuscleObstruction:
    scope = parse_scope(selector)
    candidates = _available_anchor_selectors(view)
    if isinstance(scope, RejectedScope):
        return UnresolvableMuscleAnchorObstruction(
            address=f"{address}/{endpoint.value}",
            muscle_id=muscle_id,
            endpoint=endpoint,
            selector=selector,
            closest_valid_candidates=tuple(
                difflib.get_close_matches(selector, candidates, n=3)
            ),
        )
    if not isinstance(scope, (Bone, Part, Landmark)):
        return UnsupportedMuscleAnchorObstruction(
            address=f"{address}/{endpoint.value}",
            muscle_id=muscle_id,
            endpoint=endpoint,
            selector=selector,
            scope_kind=scope_kind(scope),
            supported_scope_kinds=(
                ScopeKind.BONE,
                ScopeKind.PART,
                ScopeKind.LANDMARK,
            ),
        )
    if not _scope_exists(scope, view):
        return UnresolvableMuscleAnchorObstruction(
            address=f"{address}/{endpoint.value}",
            muscle_id=muscle_id,
            endpoint=endpoint,
            selector=selector,
            closest_valid_candidates=tuple(
                difflib.get_close_matches(selector, candidates, n=3)
            ),
        )
    mirrored_base = (
        None if _direct_scope_exists(scope, view) else _mirrored_base_scope(scope, view)
    )
    resolved_scope = mirrored_base if mirrored_base is not None else scope
    reflection = _MIRROR if mirrored_base is not None else 1.0
    point = view.point(resolved_scope) * reflection
    up = view.frame(resolved_scope)[:, 1] * reflection
    return ResolvedMuscleAnchor(
        selector=selector,
        point=tuple(map(float, point)),
        up=tuple(map(float, up)),
    )


def _part_girth(part: object) -> float | None:
    decoded = decode_part(part)
    match decoded:
        case Accepted(value=GencylPart(radii=radii, profile=profile)):
            depths = (
                profile_anchor_depths(profile, radii)
                if profile is not None
                else radii
            )
            return 2.0 * max((*radii, *depths))
        case Accepted(value=BlobPart(size=size)):
            return 2.0 * max(size)
        case Accepted(value=BoxPart(size=size, round_radius=round_radius)):
            return 2.0 * (max(size) + round_radius)
        case _:
            return None


def _relative_basis(
    declaration: NaturalMuscleDeclaration,
    origin: ResolvedMuscleAnchor,
    insertion: ResolvedMuscleAnchor,
    span: float,
    view: BodyGeometryView,
    parts: Mapping[str, object],
) -> tuple[MuscleBulkBasis, float] | UnresolvableMuscleBulkBasisObstruction:
    bulk = declaration.bulk
    if not isinstance(bulk, RelativeMuscleBulk):
        assert_never(bulk)
    if bulk.reference is MuscleBulkReference.SPAN:
        return MuscleBulkBasis.ANCHOR_SPAN, span
    anchor = (
        origin
        if bulk.reference is MuscleBulkReference.ORIGIN
        else insertion
    )
    parsed_scope = parse_scope(anchor.selector)
    scope = (
        _mirrored_base_scope(parsed_scope, view)
        if isinstance(parsed_scope, (Bone, Part, Landmark))
        else None
    ) or parsed_scope
    if isinstance(scope, Part):
        girth = _part_girth(parts.get(scope.part_id))
        return (
            (MuscleBulkBasis.PART_GIRTH, girth)
            if girth is not None and girth > 0.0
            else UnresolvableMuscleBulkBasisObstruction(
                address=f"{declaration.address}/bulk/relative/to",
                muscle_id=declaration.muscle_id,
                reference=bulk.reference,
                selector=anchor.selector,
                basis=MuscleBulkBasis.PART_GIRTH,
                observed_value=girth,
            )
        )
    bone_id = (
        scope.bone_id
        if isinstance(scope, Bone)
        else landmark_owner(scope.name, view.tree, view.ports)
        if isinstance(scope, Landmark)
        else None
    )
    length = (
        float(view.table[bone_id]["length"])
        if bone_id is not None and bone_id in view.table
        else None
    )
    return (
        (MuscleBulkBasis.BONE_LENGTH, length)
        if length is not None and length > 0.0
        else UnresolvableMuscleBulkBasisObstruction(
            address=f"{declaration.address}/bulk/relative/to",
            muscle_id=declaration.muscle_id,
            reference=bulk.reference,
            selector=anchor.selector,
            basis=MuscleBulkBasis.BONE_LENGTH,
            observed_value=length,
        )
    )


def _derived_sections(
    dimensions: MuscleBulkDimensions,
    definition: float,
    formation: MuscleFormationKind | None,
) -> tuple[LoftSection, ...]:
    tendon_ratio = 0.28 - 0.16 * definition
    shoulder_ratio = 0.82 - 0.12 * definition
    shoulder_station = 0.22 + 0.06 * definition
    section_exponent = (
        2.0
        if formation is MuscleFormationKind.SKELETON_INTEGRAL
        else None
    )
    endpoint_exponent = (
        section_exponent
        if section_exponent is not None
        else 2.2
    )
    shoulder_exponent = (
        section_exponent
        if section_exponent is not None
        else 2.6 + 1.0 * definition
    )
    belly_exponent = (
        section_exponent
        if section_exponent is not None
        else 3.0 + 1.5 * definition
    )
    return (
        LoftSection(
            station=0.0,
            width=dimensions.width * tendon_ratio,
            depth=dimensions.depth * tendon_ratio,
            exponent=endpoint_exponent,
            roll=0.0,
        ),
        LoftSection(
            station=shoulder_station,
            width=dimensions.width * shoulder_ratio,
            depth=dimensions.depth * shoulder_ratio,
            exponent=shoulder_exponent,
            roll=0.0,
        ),
        LoftSection(
            station=0.5,
            width=dimensions.width,
            depth=dimensions.depth,
            exponent=belly_exponent,
            roll=0.0,
        ),
        LoftSection(
            station=1.0 - shoulder_station,
            width=dimensions.width * shoulder_ratio,
            depth=dimensions.depth * shoulder_ratio,
            exponent=shoulder_exponent,
            roll=0.0,
        ),
        LoftSection(
            station=1.0,
            width=dimensions.width * tendon_ratio,
            depth=dimensions.depth * tendon_ratio,
            exponent=endpoint_exponent,
            roll=0.0,
        ),
    )


def _materialize_declaration(
    declaration: MuscleDeclaration,
    origin: ResolvedMuscleAnchor,
    insertion: ResolvedMuscleAnchor,
    span: float,
    view: BodyGeometryView,
    parts: Mapping[str, object],
) -> (
    BaseMuscleDeclaration
    | MirroredMuscleDeclaration
    | UnresolvableMuscleBulkBasisObstruction
):
    if not isinstance(declaration, NaturalMuscleDeclaration):
        return declaration
    bulk = declaration.bulk
    dimensions = (
        bulk.dimensions
        if isinstance(bulk, AbsoluteMuscleBulk)
        else None
    )
    if isinstance(bulk, RelativeMuscleBulk):
        basis = _relative_basis(
            declaration,
            origin,
            insertion,
            span,
            view,
            parts,
        )
        if isinstance(basis, UnresolvableMuscleBulkBasisObstruction):
            return basis
        _basis_kind, scale = basis
        dimensions = MuscleBulkDimensions(
            width=bulk.fractions.width * scale,
            depth=bulk.fractions.depth * scale,
        )
    if dimensions is None:
        assert_never(bulk)
    return BaseMuscleDeclaration(
        muscle_id=declaration.muscle_id,
        origin_selector=declaration.origin_selector,
        insertion_selector=declaration.insertion_selector,
        blend=declaration.blend,
        operator=declaration.operator,
        formation=declaration.formation,
        address=declaration.address,
        sections=_derived_sections(
            dimensions,
            declaration.definition,
            declaration.formation,
        ),
    )


def _resolve_muscle(
    declaration: MuscleDeclaration,
    view: BodyGeometryView,
    parts: Mapping[str, object],
) -> ResolvedMuscle | tuple[MuscleObstruction, ...]:
    origin = _resolve_anchor(
        declaration.muscle_id,
        MuscleEndpoint.ORIGIN,
        declaration.origin_selector,
        declaration.address,
        view,
    )
    insertion = _resolve_anchor(
        declaration.muscle_id,
        MuscleEndpoint.INSERTION,
        declaration.insertion_selector,
        declaration.address,
        view,
    )
    anchor_obstructions = tuple(
        result
        for result in (origin, insertion)
        if not isinstance(result, ResolvedMuscleAnchor)
    )
    if isinstance(origin, ResolvedMuscleAnchor) and isinstance(
        insertion, ResolvedMuscleAnchor
    ):
        span = math.dist(origin.point, insertion.point)
        if span < MINIMUM_MUSCLE_SPAN:
            return (
                DegenerateMuscleSpanObstruction(
                    address=declaration.address,
                    muscle_id=declaration.muscle_id,
                    origin_selector=declaration.origin_selector,
                    insertion_selector=declaration.insertion_selector,
                    observed_span=span,
                    minimum_span=MINIMUM_MUSCLE_SPAN,
                ),
            )
        materialized = _materialize_declaration(
            declaration,
            origin,
            insertion,
            span,
            view,
            parts,
        )
        return (
            (materialized,)
            if isinstance(materialized, UnresolvableMuscleBulkBasisObstruction)
            else ResolvedMuscle(materialized, origin, insertion)
        )
    return anchor_obstructions


def _mirror_anchor_obstructions(
    resolved: tuple[ResolvedMuscle, ...],
) -> tuple[MuscleMirrorAnchorMismatchObstruction, ...]:
    by_id = {muscle.declaration.muscle_id: muscle for muscle in resolved}
    return tuple(
        MuscleMirrorAnchorMismatchObstruction(
            address=muscle.declaration.address,
            muscle_id=muscle.declaration.muscle_id,
            mirror_of=source.declaration.muscle_id,
            endpoint=endpoint,
            source_selector=source_anchor.selector,
            target_selector=target_anchor.selector,
            observed_distance=math.dist(
                tuple(
                    map(
                        float,
                        np.asarray(source_anchor.point, dtype=np.float64)
                        * _MIRROR,
                    )
                ),
                target_anchor.point,
            ),
            maximum_distance=MUSCLE_MIRROR_TOLERANCE,
        )
        for muscle in resolved
        if isinstance(muscle.declaration, MirroredMuscleDeclaration)
        for source in (by_id[muscle.declaration.mirror_of],)
        for endpoint, source_anchor, target_anchor in (
            (MuscleEndpoint.ORIGIN, source.origin, muscle.origin),
            (MuscleEndpoint.INSERTION, source.insertion, muscle.insertion),
        )
        if math.dist(
            tuple(
                map(
                    float,
                    np.asarray(source_anchor.point, dtype=np.float64) * _MIRROR,
                )
            ),
            target_anchor.point,
        )
        > MUSCLE_MIRROR_TOLERANCE
    )


def _union_fields(
    part: dict,
    declaration: MuscleDeclaration,
) -> dict:
    return {
        **part,
        **(
            {"blend": _r(declaration.blend)}
            if declaration.blend is not None
            else {}
        ),
        **(
            {"operator": declaration.operator.value}
            if declaration.operator is not None
            else {}
        ),
        **(
            {"formation": declaration.formation.value}
            if declaration.formation is not None
            else {}
        ),
    }


def _spine_point(
    point: np.ndarray,
    formation: MuscleFormationKind | None,
) -> list[float]:
    return (
        list(map(float, np.asarray(point, dtype=np.float64)))
        if formation is MuscleFormationKind.SKELETON_INTEGRAL
        else _rvec(point)
    )


def _base_muscle(resolved: ResolvedMuscle) -> RealizedMuscle:
    declaration = resolved.declaration
    if not isinstance(declaration, BaseMuscleDeclaration):
        assert_never(declaration)
    sections = declaration.sections
    origin = np.asarray(resolved.origin.point, dtype=np.float64)
    insertion = np.asarray(resolved.insertion.point, dtype=np.float64)
    displacement = insertion - origin
    part = {
        "id": declaration.muscle_id,
        "type": "gencyl",
        "spine": [
            _spine_point(
                origin + section.station * displacement,
                declaration.formation,
            )
            for section in sections
        ],
        "radii": [_r(section.width) for section in sections],
        "profile": {
            "n": [_r(section.exponent) for section in sections],
            "depth": [_r(section.depth) for section in sections],
            "roll": [_r(section.roll) for section in sections],
            "up": _rvec(resolved.origin.up),
        },
    }
    return RealizedMuscle(_union_fields(part, declaration), sections)


def _reflected_muscle(
    declaration: MirroredMuscleDeclaration,
    source: RealizedMuscle,
) -> RealizedMuscle:
    reflected_sections = tuple(
        LoftSection(
            station=section.station,
            width=section.width,
            depth=section.depth,
            exponent=section.exponent,
            roll=-section.roll,
        )
        for section in source.sections
    )
    source_profile = source.part["profile"]
    formation = (
        MuscleFormationKind.SKELETON_INTEGRAL
        if source.part.get("formation")
        == MuscleFormationKind.SKELETON_INTEGRAL.value
        else None
    )
    part = {
        **source.part,
        "id": declaration.muscle_id,
        "spine": [
            _spine_point(
                np.asarray(point, dtype=np.float64) * _MIRROR,
                formation,
            )
            for point in source.part["spine"]
        ],
        "profile": {
            **source_profile,
            "roll": [_r(section.roll) for section in reflected_sections],
            "up": _rvec(
                np.asarray(source_profile["up"], dtype=np.float64) * _MIRROR
            ),
        },
    }
    inherited = {
        key: value
        for key, value in part.items()
        if key not in ("blend", "operator")
    }
    union_fields = {
        **inherited,
        **(
            {"blend": part["blend"]}
            if declaration.blend is None and "blend" in part
            else {}
        ),
        **(
            {"operator": part["operator"]}
            if declaration.operator is None and "operator" in part
            else {}
        ),
    }
    return RealizedMuscle(
        _union_fields(union_fields, declaration),
        reflected_sections,
    )


def _realize_muscle(
    muscle_id: str,
    resolved: Mapping[str, ResolvedMuscle],
) -> RealizedMuscle:
    muscle = resolved[muscle_id]
    match muscle.declaration:
        case BaseMuscleDeclaration():
            return _base_muscle(muscle)
        case MirroredMuscleDeclaration(mirror_of=mirror_of) as declaration:
            return _reflected_muscle(
                declaration,
                _realize_muscle(mirror_of, resolved),
            )
        case _ as unreachable:
            assert_never(unreachable)


def _solve_realized_muscle_formation(
    resolved: ResolvedMuscle,
    realized: RealizedMuscle,
) -> RealizedMuscle | tuple[MuscleObstruction, ...]:
    if "formation" not in realized.part:
        return realized
    mirrored = isinstance(resolved.declaration, MirroredMuscleDeclaration)
    match decode_part(realized.part):
        case Accepted(value=GencylPart() as part):
            match solve_muscle_formation(part, mirrored=mirrored):
                case MuscleFormationObstruction() as obstruction:
                    return (obstruction,)
                case MuscleFormationEvidence() as evidence:
                    return replace(realized, formation_evidence=evidence)
        case Rejected(obstructions=obstructions):
            return obstructions
        case Accepted(value=part):
            return (
                GeometryObstruction(
                    part_id=str(realized.part["id"]),
                    kind=str(realized.part["type"]),
                    rule=GeometryRule.MALFORMED_PART,
                    detail=(
                        "myology formation decoded to non-gencyl carrier "
                        f"{type(part).__name__}"
                    ),
                    fatal=True,
                ),
            )


def _section_dict(section: LoftSection) -> dict[str, float]:
    return {
        "station": section.station,
        "width": section.width,
        "depth": section.depth,
        "exponent": section.exponent,
        "roll": section.roll,
    }


def _receipt_row(
    resolved: ResolvedMuscle,
    realized: RealizedMuscle,
) -> dict:
    declaration = resolved.declaration
    return {
        "id": declaration.muscle_id,
        "kind": "muscle",
        "address": declaration.address,
        "origin": declaration.origin_selector,
        "insertion": declaration.insertion_selector,
        "mirror_of": (
            declaration.mirror_of
            if isinstance(declaration, MirroredMuscleDeclaration)
            else None
        ),
        "span": _r(math.dist(resolved.origin.point, resolved.insertion.point)),
        "sections": tuple(map(_section_dict, realized.sections)),
        "operator": realized.part.get("operator", CompositionOperator.BLEND.value),
        **(
            {"formation": asdict(realized.formation_evidence)}
            if realized.formation_evidence is not None
            else {}
        ),
    }


def compile_myology(
    payload: object,
    spec: Mapping,
    bones: Mapping[str, Mapping],
    landmarks: Mapping[str, object],
    existing_parts: tuple[Mapping[str, object], ...],
) -> MyologyResult:
    if not isinstance(payload, (list, tuple)):
        return RejectedMyology(
            (
                MalformedMuscleDeclarationObstruction(
                    address="muscles",
                    field="muscles",
                    rule=MuscleDeclarationRule.MUSCLES_ARRAY,
                    authored_value=payload,
                ),
            )
        )
    decoded = tuple(
        _decode_declaration(declaration, index)
        for index, declaration in enumerate(payload)
    )
    decode_obstructions = tuple(
        obstruction
        for result in decoded
        if isinstance(result, tuple)
        for obstruction in result
    )
    if decode_obstructions:
        return RejectedMyology(decode_obstructions)
    declarations = tuple(
        result
        for result in decoded
        if isinstance(
            result,
            (
                BaseMuscleDeclaration,
                NaturalMuscleDeclaration,
                MirroredMuscleDeclaration,
            ),
        )
    )
    declaration_obstructions = _declaration_obstructions(
        declarations,
        frozenset(str(part["id"]) for part in existing_parts),
    )
    if declaration_obstructions:
        return RejectedMyology(declaration_obstructions)
    cycle_obstruction = _mirror_cycle_obstruction(declarations)
    if cycle_obstruction is not None:
        return RejectedMyology((cycle_obstruction,))
    tree = build_tree(spec)
    flesh_map = flesh_owner_map(tree)
    view = BodyGeometryView(
        table=bones,
        tree=tree,
        flesh_map=flesh_map,
        ports=port_owner_map(tree),
        _points={
            name: np.asarray(point, dtype=np.float64)
            for name, point in landmarks.items()
        },
    )
    parts = {str(part["id"]): part for part in existing_parts}
    resolved_results = tuple(
        _resolve_muscle(declaration, view, parts)
        for declaration in declarations
    )
    resolution_obstructions = tuple(
        obstruction
        for result in resolved_results
        if isinstance(result, tuple)
        for obstruction in result
    )
    if resolution_obstructions:
        return RejectedMyology(resolution_obstructions)
    resolved = tuple(
        result
        for result in resolved_results
        if isinstance(result, ResolvedMuscle)
    )
    mirror_obstructions = _mirror_anchor_obstructions(resolved)
    if mirror_obstructions:
        return RejectedMyology(mirror_obstructions)
    resolved_by_id = {
        muscle.declaration.muscle_id: muscle for muscle in resolved
    }
    realized = tuple(
        _realize_muscle(declaration.muscle_id, resolved_by_id)
        for declaration in declarations
    )
    formation_results = tuple(
        _solve_realized_muscle_formation(resolved_muscle, realized_muscle)
        for resolved_muscle, realized_muscle in zip(
            resolved,
            realized,
            strict=True,
        )
    )
    formation_obstructions = tuple(
        obstruction
        for result in formation_results
        if isinstance(result, tuple)
        for obstruction in result
    )
    if formation_obstructions:
        return RejectedMyology(formation_obstructions)
    formed = tuple(
        result
        for result in formation_results
        if isinstance(result, RealizedMuscle)
    )
    return AcceptedMyology(
        parts=tuple(muscle.part for muscle in formed),
        provenance=tuple(
            (declaration.muscle_id, declaration.address)
            for declaration in declarations
        ),
        receipt=tuple(
            _receipt_row(muscle, formed_muscle)
            for muscle, formed_muscle in zip(resolved, formed, strict=True)
        ),
    )


__all__ = [
    "AcceptedMyology",
    "MINIMUM_MUSCLE_SPAN",
    "MUSCLE_MIRROR_TOLERANCE",
    "MyologyResult",
    "RejectedMyology",
    "compile_myology",
]
