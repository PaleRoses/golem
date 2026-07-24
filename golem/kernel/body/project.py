"""Total pure projection of body compile obstructions to text and JSON."""

from __future__ import annotations

from dataclasses import asdict
from typing import assert_never

from golem.addressing.session import Address, FieldIndex, FieldName
from golem.addressing.session_grammar import (
    encode_address_segment,
    parse_address,
    render_address,
)
from golem.kernel.body.relations import (
    BranchCycleObstruction,
    BranchSearchBudgetObstruction,
    DuplicateRelationIdObstruction,
    FixedPlacementConflictObstruction,
    GoalRelationAuthorityObstruction,
    MalformedRelationObstruction,
    RelationalSolveExhaustedObstruction,
    RelationalSolverBudgetObstruction,
    UnderconstrainedRelationObstruction,
    UnknownRelationKindObstruction,
    UnreferenceableDerivedGeometryObstruction,
    UnresolvedRelationSelectorObstruction,
    UnsupportedRelationSelectorObstruction,
)
from golem.kernel.body.types import (
    BODY_SPEC_TOP_LEVEL_KEYS,
    BodyObstruction,
    CarveRule,
    WebRule,
    CyclicMuscleMirrorObstruction,
    DegenerateMuscleSpanObstruction,
    DuplicateCarveIdObstruction,
    CarvePartCollisionObstruction,
    DuplicateMuscleIdObstruction,
    DuplicateWebIdObstruction,
    WebPartCollisionObstruction,
    UnknownWebAnchorObstruction,
    DuplicateEyeIdObstruction,
    EyeRule,
    FleshCompositionRule,
    EyePartCollisionObstruction,
    InvalidMuscleProfileObstruction,
    LoftSectionRule,
    MalformedBodySpecSectionObstruction,
    MalformedCarveObstruction,
    MalformedWebObstruction,
    MalformedFleshCompositionObstruction,
    MalformedEyeObstruction,
    MalformedMuscleBulkObstruction,
    MalformedMuscleDeclarationObstruction,
    MalformedMuscleSectionsObstruction,
    MisplacedBodySpecFieldObstruction,
    MissingBodySpecSectionObstruction,
    MuscleBulkRule,
    MuscleDeclarationRule,
    MuscleMirrorAnchorMismatchObstruction,
    MuscleProfileRule,
    MusclePartCollisionObstruction,
    NonGlossyEyeMaterialObstruction,
    RejectedBody,
    SkinFormationAnatomyObstruction,
    UnresolvableMuscleAnchorObstruction,
    UnresolvableMuscleBulkBasisObstruction,
    UnresolvedMuscleMirrorObstruction,
    UnsupportedMuscleAnchorObstruction,
    UnknownBodySpecKeyObstruction,
    UnknownEyeHostObstruction,
    UnknownEyeMaterialObstruction,
)
from golem.kernel.engine.types import (
    CanalEnvelopeObstruction,
    GeometryObstruction,
    IntegralRadiusUnsatisfied,
    MuscleFormationObstruction,
    UnsupportedMuscleFormation,
)
from golem.materials import AppearanceMaterialId


def _scalars(value: object) -> object:
    if isinstance(value, tuple):
        return [_scalars(item) for item in value]
    return str(value) if hasattr(value, "value") and not isinstance(value, str) else value


def _projection_address(projected: dict[str, object]) -> str:
    return next(
        (
            str(projected[field])
            for field in (
                "address",
                "owner_address",
                "subject_address",
                "interface_address",
                "element_id",
                "record_id",
                "node_id",
                "edge_id",
                "region_id",
                "part_id",
            )
            if projected.get(field) is not None
        ),
        "body",
    )


def _render_authored_session_address(
    section: str,
    fields: tuple[str, ...],
) -> str:
    candidate = render_address(
        Address(
            ("meta",),
            (
                FieldName(section),
                *tuple(
                    FieldIndex(int(field))
                    if field.isdecimal()
                    else FieldName(field)
                    for field in fields
                ),
            ),
        )
    )
    return (
        candidate
        if isinstance(parse_address(candidate), Address)
        else _render_authored_session_address(section, fields[:-1])
    )


def _authored_session_address(address: str) -> str:
    components = tuple(filter(None, address.split("/")))
    match components:
        case (section, *fields) if section in ("eyes", "muscles"):
            return _render_authored_session_address(section, tuple(fields))
        case _:
            return address


def _malformed_sections_evidence(
    obstruction: MalformedMuscleSectionsObstruction,
) -> tuple[str, object, object]:
    match obstruction.rule:
        case LoftSectionRule.SECTIONS_ARRAY:
            return "muscle sections are an array", "array", obstruction.observed_value
        case LoftSectionRule.MINIMUM_ARITY:
            return (
                "muscle loft has at least two sections",
                2,
                obstruction.observed_value,
            )
        case LoftSectionRule.SECTION_OBJECT:
            return "muscle loft section is an object", "object", obstruction.observed_value
        case LoftSectionRule.EXACT_FIELDS:
            return (
                "muscle loft section has exactly the declared fields",
                {"missing_fields": (), "extra_fields": ()},
                {
                    "value": obstruction.observed_value,
                    "missing_fields": obstruction.missing_fields,
                    "extra_fields": obstruction.extra_fields,
                },
            )
        case LoftSectionRule.FINITE_FIELD:
            return "muscle loft field is finite", "finite", obstruction.observed_value
        case LoftSectionRule.POSITIVE_WIDTH:
            return "muscle loft width is strictly positive", 0.0, obstruction.observed_value
        case LoftSectionRule.POSITIVE_DEPTH:
            return "muscle loft depth is strictly positive", 0.0, obstruction.observed_value
        case LoftSectionRule.EXPONENT_RANGE:
            return (
                "muscle loft exponent lies within its interval",
                (2.0, 12.0),
                obstruction.observed_value,
            )
        case LoftSectionRule.STRICT_STATION_ORDER:
            return (
                "muscle loft stations are strictly increasing",
                "strictly increasing",
                obstruction.observed_value,
            )
        case _ as unreachable:
            assert_never(unreachable)


def _flesh_composition_predicate(rule: FleshCompositionRule) -> str:
    match rule:
        case FleshCompositionRule.FINITE_NON_NEGATIVE_BLEND:
            return "flesh blend is finite and non-negative"
        case FleshCompositionRule.OPERATOR:
            return "flesh operator belongs to the closed composition vocabulary"
        case _ as unreachable:
            assert_never(unreachable)


def _eye_rule_predicate(rule: EyeRule) -> str:
    match rule:
        case EyeRule.EYES_ARRAY:
            return "eyes declaration is an array"
        case EyeRule.OBJECT:
            return "eye section is an object"
        case EyeRule.KNOWN_FIELD:
            return "eye field belongs to the closed vocabulary"
        case EyeRule.IDENTIFIER:
            return "eye identifier has the declared syntax"
        case EyeRule.FINITE_NUMBER:
            return "eye scalar is finite"
        case EyeRule.FINITE_VECTOR3:
            return "eye vector has three finite components"
        case EyeRule.BOOLEAN:
            return "eye flag is boolean"
        case EyeRule.CANONICAL_MIRROR_SIDE:
            return "mirrored eye is authored on the canonical sagittal side"
        case EyeRule.FINITE_POSITIVE_NUMBER:
            return "eye scalar is finite and strictly positive"
        case EyeRule.FINITE_POSITIVE_VECTOR3:
            return "eye vector has three finite strictly positive components"
        case EyeRule.FINITE_NON_NEGATIVE_NUMBER:
            return "eye scalar is finite and non-negative"
        case EyeRule.SOCKET_RADIUS_EXCEEDS_EYE_RADIUS:
            return "eye socket radius strictly exceeds eye radius"
        case EyeRule.SOCKET_DEPTH_BELOW_RADIUS:
            return "eye socket depth is strictly below socket radius"
        case _ as unreachable:
            assert_never(unreachable)


def _carve_rule_predicate(rule: CarveRule) -> str:
    match rule:
        case CarveRule.CARVES_ARRAY:
            return "carves declaration is an array"
        case CarveRule.OBJECT:
            return "carve section is an object"
        case CarveRule.KNOWN_FIELD:
            return "carve field belongs to the closed vocabulary"
        case CarveRule.IDENTIFIER:
            return "carve identifier has the declared syntax"
        case CarveRule.KIND:
            return "carve kind belongs to the closed vocabulary"
        case CarveRule.FINITE_NUMBER:
            return "carve scalar is finite"
        case CarveRule.FINITE_VECTOR3:
            return "carve vector has three finite components"
        case CarveRule.FINITE_POSITIVE_VECTOR3:
            return "carve vector has three finite strictly positive components"
        case CarveRule.FINITE_NON_NEGATIVE_NUMBER:
            return "carve scalar is finite and non-negative"
        case CarveRule.FINITE_STRICT_PAIR:
            return "carve span is a strictly increasing finite pair"
        case CarveRule.FINITE_POSITIVE_ARRAY:
            return "carve radii are finite and strictly positive"
        case CarveRule.FINITE_STRICT_ARRAY:
            return "carve stations are strictly increasing"
        case CarveRule.MATCHING_ARITY:
            return "carve stations match radii arity"
        case CarveRule.STATIONS_MATCH_SPAN:
            return "carve station endpoints match the declared span"
        case _ as unreachable:
            assert_never(unreachable)


def _web_rule_predicate(rule: WebRule) -> str:
    match rule:
        case WebRule.WEBS_ARRAY:
            return "webs declaration is an array"
        case WebRule.OBJECT:
            return "web section is an object"
        case WebRule.KNOWN_FIELD:
            return "web field belongs to the closed vocabulary"
        case WebRule.IDENTIFIER:
            return "web identifier has the declared syntax"
        case WebRule.ANCHORS_ARRAY:
            return "web anchors are an array"
        case WebRule.ANCHOR_ARITY:
            return "web anchors at least two anchor curves"
        case WebRule.ANCHOR_OBJECT:
            return "web anchor is an object"
        case WebRule.BOOLEAN:
            return "web flag is boolean"
        case WebRule.FINITE_NON_NEGATIVE_NUMBER:
            return "web scalar is finite and non-negative"
        case WebRule.OPERATOR:
            return "web operator belongs to the closed composition vocabulary"
        case WebRule.ROLE:
            return "web role belongs to the closed flesh-role vocabulary"
        case WebRule.FINITE_STRICT_PAIR:
            return "web anchor span is a strictly increasing finite pair"
        case WebRule.FINITE_POSITIVE_ARRAY:
            return "web anchor radii are finite and strictly positive"
        case WebRule.FINITE_STRICT_ARRAY:
            return "web anchor stations are strictly increasing"
        case WebRule.MATCHING_ARITY:
            return "web anchor stations match radii arity"
        case WebRule.STATIONS_MATCH_SPAN:
            return "web anchor station endpoints match the declared span"
        case WebRule.STATION_ARITY:
            return "web anchors emit identical station counts"
        case _ as unreachable:
            assert_never(unreachable)


def _semantic_projection(
    obstruction: BodyObstruction,
    projected: dict[str, object],
) -> dict[str, object]:
    match obstruction:
        case SkinFormationAnatomyObstruction(obstructions):
            return {
                "address": "anatomy/integument_layers",
                "predicate": (
                    "opted-in skin formation has accepted anatomy descent"
                ),
                "required": "AcceptedAnatomy",
                "observed": tuple(
                    type(item).__name__ for item in obstructions
                ),
            }
        case UnknownBodySpecKeyObstruction(address, authored_key, _):
            return {
                "address": address,
                "predicate": (
                    "top-level body key belongs to the declared body dialect"
                ),
                "required": tuple(sorted(BODY_SPEC_TOP_LEVEL_KEYS)),
                "observed": authored_key,
            }
        case MisplacedBodySpecFieldObstruction(
            address,
            field,
            authored_parent,
            legal_parent,
            legal_address,
        ):
            return {
                "address": address,
                "predicate": "body field is authored under its legal parent",
                "required": {
                    "parent": legal_parent,
                    "address": legal_address,
                },
                "observed": {
                    "parent": authored_parent,
                    "field": field,
                },
            }
        case MissingBodySpecSectionObstruction(address, section, legal_parent):
            return {
                "address": address,
                "predicate": "required body section is present",
                "required": {"parent": legal_parent, "section": section},
                "observed": "missing",
            }
        case MalformedBodySpecSectionObstruction(
            address,
            _,
            expected_shape,
            observed_type,
        ):
            return {
                "address": address,
                "predicate": "body section has the declared shape",
                "required": expected_shape,
                "observed": observed_type,
            }
        case MalformedFleshCompositionObstruction(
            address, rule, authored_value, required
        ):
            return {
                "address": address,
                "predicate": _flesh_composition_predicate(rule),
                "required": required,
                "observed": authored_value,
            }
        case MalformedCarveObstruction(address, rule, authored, required):
            return {
                "address": address,
                "predicate": _carve_rule_predicate(rule),
                "required": required,
                "observed": authored,
            }
        case MalformedWebObstruction(address, rule, authored, required):
            return {
                "address": address,
                "predicate": _web_rule_predicate(rule),
                "required": required,
                "observed": authored,
            }
        case UnknownWebAnchorObstruction(address, bone, candidates):
            return {
                "address": address,
                "predicate": "web anchor names a declared bone",
                "required": {"closest_valid_candidates": candidates},
                "observed": bone,
            }
        case MalformedEyeObstruction(address, rule, authored, required):
            return {
                "address": _authored_session_address(address),
                "predicate": _eye_rule_predicate(rule),
                "required": required,
                "observed": authored,
            }
        case UnknownEyeMaterialObstruction(address, appearance_material):
            return {
                "address": _authored_session_address(address),
                "predicate": (
                    "eye appearance material belongs to the closed vocabulary"
                ),
                "required": tuple(
                    material.value for material in AppearanceMaterialId
                ),
                "observed": appearance_material,
            }
        case NonGlossyEyeMaterialObstruction(
            address,
            _,
            roughness,
            maximum_roughness,
        ):
            return {
                "address": _authored_session_address(address),
                "predicate": (
                    "eye material roughness does not exceed the glossy limit"
                ),
                "required": maximum_roughness,
                "observed": roughness,
            }
        case MalformedMuscleDeclarationObstruction(
            address,
            _,
            rule,
            authored_value,
        ):
            match rule:
                case MuscleDeclarationRule.DEFINITION:
                    predicate = (
                        "muscle definition is finite and lies within the "
                        "closed unit interval"
                    )
                    required = (0.0, 1.0)
                case MuscleDeclarationRule.FINITE_BLEND:
                    predicate = "muscle blend is finite and non-negative"
                    required = {"finite": True, "minimum": 0.0}
                case _:
                    predicate = "muscle declaration satisfies its field rule"
                    required = rule.value
            return {
                "address": _authored_session_address(address),
                "predicate": predicate,
                "required": required,
                "observed": authored_value,
            }
        case MalformedMuscleBulkObstruction() as failure:
            if failure.rule is MuscleBulkRule.FINITE_POSITIVE_DIMENSION:
                return {
                    "address": _authored_session_address(failure.address),
                    "predicate": (
                        "muscle bulk dimension is finite and strictly positive"
                    ),
                    "required": {
                        "finite": True,
                        "exclusive_minimum": 0.0,
                    },
                    "observed": failure.authored_value,
                }
            return {
                "address": _authored_session_address(failure.address),
                "predicate": "muscle bulk satisfies its declaration rule",
                "required": {
                    "rule": failure.rule.value,
                    "missing_fields": (),
                    "extra_fields": (),
                },
                "observed": {
                    "value": failure.authored_value,
                    "missing_fields": failure.missing_fields,
                    "extra_fields": failure.extra_fields,
                },
            }
        case UnresolvableMuscleBulkBasisObstruction() as failure:
            return {
                "address": _authored_session_address(failure.address),
                "predicate": "muscle bulk basis is strictly positive",
                "required": 0.0,
                "observed": failure.observed_value,
            }
        case MalformedMuscleSectionsObstruction() as failure:
            predicate, required, observed = _malformed_sections_evidence(failure)
            return {
                "address": _authored_session_address(failure.address),
                "predicate": predicate,
                "required": required,
                "observed": observed,
            }
        case InvalidMuscleProfileObstruction() as failure:
            match failure.rule:
                case MuscleProfileRule.MINIMUM_ARITY:
                    predicate = "muscle profile has at least three stations"
                    required = 3
                    observed = failure.station_count
                case MuscleProfileRule.UNIT_SPAN:
                    predicate = "muscle profile spans the closed unit interval"
                    required = (0.0, 1.0)
                    observed = (failure.first_station, failure.last_station)
                case MuscleProfileRule.FUSIFORM_TAPER:
                    predicate = (
                        "muscle profile endpoints taper below interior maxima"
                    )
                    required = {
                        "maximum_endpoint_width": failure.maximum_interior_width,
                        "maximum_endpoint_depth": failure.maximum_interior_depth,
                    }
                    observed = {
                        "maximum_endpoint_width": (
                            max(failure.endpoint_widths)
                            if failure.endpoint_widths is not None
                            else None
                        ),
                        "maximum_endpoint_depth": (
                            max(failure.endpoint_depths)
                            if failure.endpoint_depths is not None
                            else None
                        ),
                    }
                case _ as unreachable:
                    assert_never(unreachable)
            return {
                "address": _authored_session_address(failure.address),
                "predicate": predicate,
                "required": required,
                "observed": observed,
            }
        case MuscleFormationObstruction(part_id, mirrored, failure):
            match failure:
                case IntegralRadiusUnsatisfied(
                    section_index,
                    prescribed_radius,
                    solved_radius,
                    residual,
                    tolerance,
                ):
                    predicate = (
                        "skeleton-integral formation preserves the prescribed "
                        "section radius"
                    )
                    required = {
                        "maximum_residual": tolerance,
                        "prescribed_radius": prescribed_radius,
                    }
                    observed = {
                        "section_index": section_index,
                        "solved_radius": solved_radius,
                        "residual": residual,
                    }
                case CanalEnvelopeObstruction(
                    interval_index,
                    lower,
                    upper,
                    radius_derivative,
                    support_scale,
                    envelope_ratio,
                    required_upper_bound,
                ):
                    predicate = (
                        "skeleton-integral radius derivative remains inside "
                        "the sealed canal envelope"
                    )
                    required = {
                        "exclusive_upper_bound": required_upper_bound,
                        "support_scale": support_scale,
                    }
                    observed = {
                        "interval_index": interval_index,
                        "interval": (lower, upper),
                        "radius_derivative": radius_derivative,
                        "envelope_ratio": envelope_ratio,
                    }
                case UnsupportedMuscleFormation(reason, legal_family):
                    predicate = (
                        "muscle formation belongs to the closed analytic "
                        "skeleton-integral family"
                    )
                    required = legal_family
                    observed = reason
            return {
                "address": f"muscle/{encode_address_segment(part_id)}/formation",
                "predicate": predicate,
                "required": required,
                "observed": observed,
                "mirrored": mirrored,
            }
        case GeometryObstruction(part_id, kind, rule, detail, fatal):
            return {
                "address": f"muscle/{encode_address_segment(part_id)}",
                "predicate": "derived muscle geometry satisfies the engine contract",
                "required": {
                    "kind": kind,
                    "rule": rule.value,
                    "fatal": False,
                },
                "observed": {
                    "detail": detail,
                    "fatal": fatal,
                },
            }
        case DegenerateMuscleSpanObstruction(
            address,
            muscle_id,
            _,
            _,
            observed_span,
            minimum_span,
        ):
            return {
                "address": _authored_session_address(address),
                "predicate": "muscle span meets the minimum length",
                "required": minimum_span,
                "observed": observed_span,
            }
        case MuscleMirrorAnchorMismatchObstruction(
            address,
            muscle_id,
            _,
            _,
            _,
            _,
            observed_distance,
            maximum_distance,
        ):
            return {
                "address": _authored_session_address(address),
                "predicate": (
                    "mirrored muscle anchor lies within reflection tolerance"
                ),
                "required": maximum_distance,
                "observed": observed_distance,
            }
        case FixedPlacementConflictObstruction(
            _,
            subject_address,
            _,
            required,
            observed,
        ):
            return {
                "address": subject_address,
                "contract_domain": "relation",
                "predicate": "fixed placement satisfies the relation constraint",
                "required": required,
                "observed": observed,
            }
        case RelationalSolveExhaustedObstruction(
            owner_address,
            _,
            _,
            _,
            required,
            observed,
            _,
            _,
        ):
            return {
                "address": owner_address,
                "contract_domain": "relation",
                "predicate": "relation solve satisfies the controlling constraint",
                "required": required,
                "observed": observed,
            }
        case UnderconstrainedRelationObstruction() as failure:
            return {
                "address": failure.owner_address,
                "contract_domain": "relation",
                "predicate": "relation solve is fully constrained",
                "required": {
                    "rank": failure.variable_count,
                    "free_dimension_count": 0,
                },
                "observed": {
                    "rank": failure.rank,
                    "free_dimension_count": failure.free_dimension_count,
                },
            }
        case RelationalSolverBudgetObstruction(
            owner_address,
            required,
            observed,
            best_residual,
        ):
            return {
                "address": owner_address,
                "contract_domain": "relation",
                "predicate": (
                    "relation solve completes within its evaluation budget"
                ),
                "required": {
                    "completed": True,
                    "maximum_evaluations": required,
                },
                "observed": {
                    "completed": False,
                    "evaluated": observed,
                    "best_residual": best_residual,
                },
            }
        case BranchCycleObstruction(relation_id, _, required, observed):
            return {
                "address": (
                    "pose/relations/"
                    f"{encode_address_segment(relation_id)}"
                ),
                "contract_domain": "relation",
                "predicate": (
                    "relation constraint is satisfied before an active-set cycle"
                ),
                "required": required,
                "observed": observed,
            }
        case BranchSearchBudgetObstruction(
            relation_id,
            required,
            observed,
            best_residual,
        ):
            return {
                "address": (
                    "pose/relations/"
                    f"{encode_address_segment(relation_id)}"
                ),
                "contract_domain": "relation",
                "predicate": (
                    "relation branch search completes within its active-set budget"
                ),
                "required": {
                    "completed": True,
                    "maximum_active_sets": required,
                },
                "observed": {
                    "completed": False,
                    "evaluated_active_sets": observed,
                    "best_residual": best_residual,
                },
            }
        case (
            MalformedRelationObstruction()
            | DuplicateRelationIdObstruction()
            | UnknownRelationKindObstruction()
            | UnresolvedRelationSelectorObstruction()
            | UnsupportedRelationSelectorObstruction()
            | UnreferenceableDerivedGeometryObstruction()
            | GoalRelationAuthorityObstruction()
        ):
            return {
                "address": _projection_address(projected),
                "contract_domain": "relation",
                "predicate": type(obstruction).__name__,
                "required": projected.get("required"),
                "observed": projected.get("observed", projected.get("reason")),
            }
        case _:
            return {
                "address": _projection_address(projected),
                "predicate": type(obstruction).__name__,
                "required": projected.get("required"),
                "observed": projected.get("observed", projected.get("reason")),
            }


def project_obstruction(obstruction: BodyObstruction) -> dict:
    projected = {
        "obstruction": type(obstruction).__name__,
        **{key: _scalars(value) for key, value in asdict(obstruction).items()},
    }
    return {**projected, **_semantic_projection(obstruction, projected)}


def obstruction_text(obstruction: BodyObstruction) -> str:
    match obstruction:
        case SkinFormationAnatomyObstruction(obstructions):
            return (
                "anatomy/integument_layers: skin formation anatomy rejected "
                f"({', '.join(type(item).__name__ for item in obstructions)})"
            )
        case UnknownBodySpecKeyObstruction(address, authored_key, candidates):
            return (
                f"{address}: unknown top-level body key {authored_key!r}"
                f" (closest: {', '.join(candidates) or 'none'})"
            )
        case MisplacedBodySpecFieldObstruction(
            address, field, authored_parent, legal_parent, legal_address
        ):
            return (
                f"{address}: {field} is not valid inside a {authored_parent};"
                f" move to {legal_parent}/{legal_address.removeprefix('pose/')}"
            )
        case MissingBodySpecSectionObstruction(address, section, legal_parent):
            return (
                f"{address}: missing required section {section!r}"
                f" in {legal_parent}"
            )
        case MalformedBodySpecSectionObstruction(
            address, section, expected_shape, observed_type
        ):
            return (
                f"{address}: malformed section {section!r};"
                f" expected {expected_shape}, observed {observed_type}"
            )
        case MalformedFleshCompositionObstruction(
            address, rule, authored_value, required
        ):
            return (
                f"{address}: malformed flesh composition ({rule.value};"
                f" required {required!r}, observed {authored_value!r})"
            )
        case MalformedEyeObstruction(address, rule, authored, required):
            return (
                f"{address}: malformed eye ({rule.value};"
                f" required {required!r}, observed {authored!r})"
            )
        case DuplicateEyeIdObstruction(address, eye_id):
            return f"{address}: duplicate eye id {eye_id!r}"
        case UnknownEyeHostObstruction(address, host_bone_id, candidates):
            return (
                f"{address}: unknown eye host {host_bone_id!r}"
                f" (closest: {', '.join(candidates) or 'none'})"
            )
        case UnknownEyeMaterialObstruction(address, appearance_material):
            return (
                f"{address}: unknown eye appearance material "
                f"{appearance_material!r}"
            )
        case NonGlossyEyeMaterialObstruction(
            address,
            appearance_material,
            roughness,
            maximum_roughness,
        ):
            return (
                f"{address}: eye material {appearance_material!r} has roughness "
                f"{roughness:g}; required <= {maximum_roughness:g}"
            )
        case EyePartCollisionObstruction(address, part_id):
            return f"{address}: derived eye part id {part_id!r} already exists"
        case MalformedMuscleDeclarationObstruction(address, field, rule, value):
            return (
                f"{address}: malformed muscle field {field!r}"
                f" ({rule.value}; observed {value!r})"
            )
        case MalformedMuscleBulkObstruction(
            address,
            muscle_id,
            rule,
            value,
            missing,
            extra,
        ):
            return (
                f"{address}: malformed bulk for muscle {muscle_id!r}"
                f" ({rule.value}; observed {value!r};"
                f" missing {missing!r}; extra {extra!r})"
            )
        case UnresolvableMuscleBulkBasisObstruction(
            address,
            muscle_id,
            reference,
            selector,
            basis,
            observed,
        ):
            return (
                f"{address}: cannot derive {basis.value} for muscle"
                f" {muscle_id!r} from {reference.value} host {selector!r}"
                f" (observed {observed!r})"
            )
        case DuplicateMuscleIdObstruction(muscle_id, first, duplicate):
            return (
                f"{duplicate}: duplicate muscle id {muscle_id!r}"
                f" first declared at {first}"
            )
        case MusclePartCollisionObstruction(muscle_id, existing_part_id):
            return (
                f"muscle {muscle_id!r}: emitted id collides with"
                f" existing part {existing_part_id!r}"
            )
        case MalformedMuscleSectionsObstruction(
            address,
            muscle_id,
            section_index,
            rule,
            field,
            observed,
            missing,
            extra,
        ):
            return (
                f"{address}: malformed sections for muscle {muscle_id!r}"
                f" ({rule.value}; field {field!r}; observed {observed!r};"
                f" missing {missing!r}; extra {extra!r}; index {section_index!r})"
            )
        case InvalidMuscleProfileObstruction(
            address,
            muscle_id,
            rule,
            count,
            first,
            last,
            endpoint_widths,
            endpoint_depths,
            belly_width,
            belly_depth,
        ):
            return (
                f"{address}: invalid profile for muscle {muscle_id!r}"
                f" ({rule.value}; stations {count}, span {first!r}..{last!r},"
                f" endpoint widths {endpoint_widths!r},"
                f" endpoint depths {endpoint_depths!r},"
                f" interior maxima {(belly_width, belly_depth)!r})"
            )
        case UnresolvableMuscleAnchorObstruction(
            address, muscle_id, endpoint, selector, candidates
        ):
            return (
                f"{address}: unresolved {endpoint.value} anchor {selector!r}"
                f" for muscle {muscle_id!r}"
                f" (closest: {', '.join(candidates) or 'none'})"
            )
        case UnsupportedMuscleAnchorObstruction(
            address, muscle_id, endpoint, selector, scope, supported
        ):
            return (
                f"{address}: unsupported {endpoint.value} anchor {selector!r}"
                f" for muscle {muscle_id!r}; scope {scope!r},"
                f" supported {', '.join(supported)}"
            )
        case DegenerateMuscleSpanObstruction(
            _, muscle_id, origin, insertion, observed, minimum
        ):
            return (
                f"muscle {muscle_id!r}: degenerate span {observed:g}"
                f" from {origin!r} to {insertion!r}; required >= {minimum:g}"
            )
        case UnresolvedMuscleMirrorObstruction(muscle_id, mirror_of, candidates):
            return (
                f"muscle {muscle_id!r}: unresolved mirror source {mirror_of!r}"
                f" (closest: {', '.join(candidates) or 'none'})"
            )
        case CyclicMuscleMirrorObstruction(muscle_ids):
            return f"cyclic muscle mirror chain: {' -> '.join(muscle_ids)}"
        case MuscleMirrorAnchorMismatchObstruction(
            _,
            muscle_id,
            mirror_of,
            endpoint,
            source_selector,
            target_selector,
            observed,
            maximum,
        ):
            return (
                f"muscle {muscle_id!r}: {endpoint.value} anchor {target_selector!r}"
                f" is {observed:g} from reflected {source_selector!r}"
                f" of {mirror_of!r}; required <= {maximum:g}"
            )
        case MuscleFormationObstruction(part_id, mirrored, failure):
            side = "mirrored " if mirrored else ""
            match failure:
                case IntegralRadiusUnsatisfied(
                    section_index,
                    prescribed_radius,
                    solved_radius,
                    residual,
                    tolerance,
                ):
                    detail = (
                        f"section {section_index} prescribed {prescribed_radius:g}, "
                        f"solved {solved_radius:g}, residual {residual:g}, "
                        f"tolerance {tolerance:g}"
                    )
                case CanalEnvelopeObstruction(
                    interval_index,
                    lower,
                    upper,
                    radius_derivative,
                    support_scale,
                    envelope_ratio,
                    required_upper_bound,
                ):
                    detail = (
                        f"interval {interval_index} [{lower:g}, {upper:g}], "
                        f"radius derivative {radius_derivative:g}, support "
                        f"{support_scale:g}, envelope ratio {envelope_ratio:g}, "
                        f"required < {required_upper_bound:g}"
                    )
                case UnsupportedMuscleFormation(reason, legal_family):
                    detail = (
                        f"{reason}; legal family "
                        f"{', '.join(legal_family)}"
                    )
            return (
                f"muscle {part_id!r}: {side}skeleton-integral formation "
                f"rejected ({detail})"
            )
        case GeometryObstruction(part_id, kind, rule, detail, fatal):
            severity = "fatal" if fatal else "advisory"
            return (
                f"muscle {part_id!r}: {severity} {kind} geometry obstruction "
                f"{rule.value} ({detail})"
            )
        case MalformedCarveObstruction(address, rule, authored, required):
            return (
                f"{address}: malformed carve ({rule.value};"
                f" required {required!r}, observed {authored!r})"
            )
        case DuplicateCarveIdObstruction(address, carve_id):
            return f"{address}: duplicate carve id {carve_id!r}"
        case CarvePartCollisionObstruction(address, part_id):
            return (
                f"{address}: carve id {part_id!r} collides with"
                f" an existing part"
            )
        case MalformedWebObstruction(address, rule, authored, required):
            return (
                f"{address}: malformed web ({rule.value};"
                f" required {required!r}, observed {authored!r})"
            )
        case DuplicateWebIdObstruction(address, web_id):
            return f"{address}: duplicate web id {web_id!r}"
        case WebPartCollisionObstruction(address, part_id):
            return (
                f"{address}: web id {part_id!r} collides with"
                f" an existing part"
            )
        case UnknownWebAnchorObstruction(address, bone, candidates):
            return (
                f"{address}: unknown web anchor bone {bone!r}"
                f" (closest: {', '.join(candidates) or 'none'})"
            )
        case MalformedRelationObstruction(address, reason):
            return f"{address}: malformed relation -- {reason}"
        case DuplicateRelationIdObstruction(address, relation_id):
            return f"{address}: duplicate relation id {relation_id!r}"
        case UnknownRelationKindObstruction(address, authored_kind, candidates):
            return (
                f"{address}: unknown relation kind {authored_kind!r}"
                f" (closest: {', '.join(candidates) or 'none'})"
            )
        case UnresolvedRelationSelectorObstruction(address, selector, candidates):
            return (
                f"{address}: unresolved selector {selector!r}"
                f" (closest: {', '.join(candidates) or 'none'})"
            )
        case UnsupportedRelationSelectorObstruction(address, selector, supported):
            return (
                f"{address}: selector {selector!r} unsupported;"
                f" allowed scopes {', '.join(str(kind) for kind in supported)}"
            )
        case UnreferenceableDerivedGeometryObstruction(address, selector, reason):
            return f"{address}: {selector!r} references derived geometry -- {reason}"
        case FixedPlacementConflictObstruction(
            relation_id, subject, reference, required, observed
        ):
            return (
                f"{relation_id}: fixed placement conflict between {subject!r}"
                f" and {reference!r} (required {required}, observed {observed})"
            )
        case GoalRelationAuthorityObstruction(relation_id, goal_id, bones):
            return (
                f"{relation_id}: relation authority overlaps goal {goal_id!r}"
                f" on bones {', '.join(bones)}"
            )
        case UnderconstrainedRelationObstruction(
            owner, variable_count, rank, free, addresses
        ):
            return (
                f"{owner}: under-constrained ({rank}/{variable_count} rank,"
                f" {free} free) on {', '.join(addresses)}"
            )
        case RelationalSolveExhaustedObstruction(
            owner, stage, depth, relation_id, required, observed, residual, evaluations
        ):
            return (
                f"{owner}: solve exhausted at {stage} depth {depth} on"
                f" {relation_id!r} (required {required}, observed {observed},"
                f" residual {residual}, {evaluations} evaluations)"
            )
        case RelationalSolverBudgetObstruction(owner, required, observed, best):
            return (
                f"{owner}: solver budget exhausted"
                f" ({observed}/{required} evaluations, best residual {best})"
            )
        case BranchCycleObstruction(relation_id, active_sets, required, observed):
            return (
                f"{relation_id}: active-set cycle over {len(active_sets)} states"
                f" (required {required}, observed {observed})"
            )
        case BranchSearchBudgetObstruction(relation_id, required, observed, best):
            return (
                f"{relation_id}: active-set budget exhausted"
                f" ({observed}/{required} states, best residual {best})"
            )


def project_rejected_body(rejected: RejectedBody) -> dict:
    return {
        "rejected": "body",
        "obstructions": [
            project_obstruction(obstruction) for obstruction in rejected.obstructions
        ],
    }


def rejected_body_text(rejected: RejectedBody) -> str:
    return "\n".join(
        obstruction_text(obstruction) for obstruction in rejected.obstructions
    )


__all__ = [
    "obstruction_text",
    "project_obstruction",
    "project_rejected_body",
    "rejected_body_text",
]
