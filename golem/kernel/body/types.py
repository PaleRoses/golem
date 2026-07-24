"""Body-compiler carriers: dialect constants, gencyl-station result, compiled product."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from golem.addressing.scope import ScopeKind
from golem.materials import AppearanceMaterialId
from golem.kernel.body.relation_solve import LocalSolveReceipt
from golem.kernel.body.relations import BodyRelationObstruction, RelationDeclaration
from golem.kernel.engine.types import GeometryObstruction, MuscleFormationObstruction

if TYPE_CHECKING:
    from golem.kernel.anatomy import AnatomyObstruction, AnatomyResult

DIALECT = "body/0.3"
BODY_SPEC_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "_intents",
        "anatomy",
        "appendages",
        "appearance_material",
        "appearance_palette",
        "blend",
        "conduits",
        "contacts",
        "contract",
        "dialect",
        "emit_target",
        "eyes",
        "ground_tol",
        "ground_y",
        "mounts",
        "muscles",
        "name",
        "pitch",
        "plate_policy",
        "pose",
        "props",
        "schematic",
        "skeleton",
        "surface_color",
        "surface_detail",
        "webs",
    }
)
_DOF = ("fixed", "hinge", "ball")

# Dialect ontology roles (wave-3, R3): opt-in declarations that reclassify a
# bone or flesh record out of the default perfused-organ/carrier ontology.
# Absence of `role` keeps the default law; the roles are vocabulary, never a
# silent compilation default.
BONE_ROLE_HANDLE = "handle"  # transform handle: frames compose, no perfusion endpoint
BONE_ROLES = (BONE_ROLE_HANDLE,)
FLESH_ROLE_NON_CARRIER = "non_carrier"  # membrane: authored dims, no carrier law
FLESH_ROLES = (FLESH_ROLE_NON_CARRIER,)
_ROUND = 6  # emitted-coordinate rounding (cross-run byte stability)
_PARALLEL_COS = math.cos(math.radians(1.0))  # up-hint fallback threshold


@dataclass(frozen=True)
class UnknownBodySpecKeyObstruction:
    address: str
    authored_key: str
    closest_valid_candidates: tuple[str, ...]


@dataclass(frozen=True)
class MisplacedBodySpecFieldObstruction:
    address: str
    field: str
    authored_parent: str
    legal_parent: str
    legal_address: str


@dataclass(frozen=True)
class MissingBodySpecSectionObstruction:
    address: str
    section: str
    legal_parent: str


@dataclass(frozen=True)
class MalformedBodySpecSectionObstruction:
    address: str
    section: str
    expected_shape: str
    observed_type: str


class FleshCompositionRule(StrEnum):
    FINITE_NON_NEGATIVE_BLEND = "finite_non_negative_blend"
    OPERATOR = "operator"


@dataclass(frozen=True)
class MalformedFleshCompositionObstruction:
    address: str
    rule: FleshCompositionRule
    authored_value: object
    required: object


type BodySpecDecodeObstruction = (
    UnknownBodySpecKeyObstruction
    | MisplacedBodySpecFieldObstruction
    | MissingBodySpecSectionObstruction
    | MalformedBodySpecSectionObstruction
    | MalformedFleshCompositionObstruction
)


@dataclass(frozen=True)
class _GencylStationObstruction:
    reason: str


type _GencylStationResult = tuple[float, ...] | _GencylStationObstruction


@dataclass(frozen=True)
class LoftSection:
    station: float
    width: float
    depth: float
    exponent: float
    roll: float


class LoftSectionRule(StrEnum):
    SECTIONS_ARRAY = "sections_array"
    MINIMUM_ARITY = "minimum_arity"
    SECTION_OBJECT = "section_object"
    EXACT_FIELDS = "exact_fields"
    FINITE_FIELD = "finite_field"
    POSITIVE_WIDTH = "positive_width"
    POSITIVE_DEPTH = "positive_depth"
    EXPONENT_RANGE = "exponent_range"
    STRICT_STATION_ORDER = "strict_station_order"


@dataclass(frozen=True)
class LoftSectionObstruction:
    section_index: int | None
    rule: LoftSectionRule
    field: str | None = None
    observed_value: object = None
    missing_fields: tuple[str, ...] = ()
    extra_fields: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        match self.rule:
            case LoftSectionRule.SECTIONS_ARRAY:
                return "sections must be an array"
            case LoftSectionRule.MINIMUM_ARITY:
                return "loft flesh requires at least two sections"
            case LoftSectionRule.SECTION_OBJECT:
                return "section must be an object"
            case LoftSectionRule.EXACT_FIELDS:
                required = ("depth", "exponent", "roll", "station", "width")
                return (
                    f"section fields must be exactly {required!r}; "
                    f"missing={self.missing_fields!r}, extra={self.extra_fields!r}"
                )
            case LoftSectionRule.FINITE_FIELD:
                return f"{self.field} must be a finite number"
            case LoftSectionRule.POSITIVE_WIDTH:
                return "width must be positive"
            case LoftSectionRule.POSITIVE_DEPTH:
                return "depth must be positive"
            case LoftSectionRule.EXPONENT_RANGE:
                return "exponent must be in [2, 12]"
            case LoftSectionRule.STRICT_STATION_ORDER:
                return "stations must be strictly increasing"


class MuscleEndpoint(StrEnum):
    ORIGIN = "origin"
    INSERTION = "insertion"


class MuscleBulkReference(StrEnum):
    ORIGIN = "origin"
    INSERTION = "insertion"
    SPAN = "span"


class MuscleBulkBasis(StrEnum):
    BONE_LENGTH = "bone_length"
    PART_GIRTH = "part_girth"
    ANCHOR_SPAN = "anchor_span"


@dataclass(frozen=True)
class MuscleBulkDimensions:
    width: float
    depth: float


@dataclass(frozen=True)
class AbsoluteMuscleBulk:
    dimensions: MuscleBulkDimensions


@dataclass(frozen=True)
class RelativeMuscleBulk:
    reference: MuscleBulkReference
    fractions: MuscleBulkDimensions


type MuscleBulk = AbsoluteMuscleBulk | RelativeMuscleBulk


class MuscleDeclarationRule(StrEnum):
    MUSCLES_ARRAY = "muscles_array"
    DECLARATION_OBJECT = "declaration_object"
    KIND = "kind"
    IDENTIFIER = "identifier"
    ANCHOR_SELECTOR = "anchor_selector"
    PROFILE_XOR_MIRROR = "profile_xor_mirror"
    PROFILE_FORM = "profile_form"
    MIRROR_IDENTIFIER = "mirror_identifier"
    DEFINITION = "definition"
    EXACT_FIELDS = "exact_fields"
    FINITE_BLEND = "finite_blend"
    OPERATOR = "operator"
    FORMATION = "formation"


class MuscleBulkRule(StrEnum):
    BULK_OBJECT = "bulk_object"
    RELATIVE_XOR_ABSOLUTE = "relative_xor_absolute"
    DIMENSIONS_OBJECT = "dimensions_object"
    EXACT_FIELDS = "exact_fields"
    FINITE_POSITIVE_DIMENSION = "finite_positive_dimension"
    REFERENCE = "reference"


class MuscleProfileRule(StrEnum):
    MINIMUM_ARITY = "minimum_arity"
    UNIT_SPAN = "unit_span"
    FUSIFORM_TAPER = "fusiform_taper"


@dataclass(frozen=True)
class MalformedMuscleDeclarationObstruction:
    address: str
    field: str
    rule: MuscleDeclarationRule
    authored_value: object


@dataclass(frozen=True)
class MalformedMuscleBulkObstruction:
    address: str
    muscle_id: str
    rule: MuscleBulkRule
    authored_value: object
    missing_fields: tuple[str, ...] = ()
    extra_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnresolvableMuscleBulkBasisObstruction:
    address: str
    muscle_id: str
    reference: MuscleBulkReference
    selector: str | None
    basis: MuscleBulkBasis
    observed_value: float | None


@dataclass(frozen=True)
class DuplicateMuscleIdObstruction:
    muscle_id: str
    first_address: str
    duplicate_address: str


@dataclass(frozen=True)
class MusclePartCollisionObstruction:
    muscle_id: str
    existing_part_id: str


@dataclass(frozen=True)
class MalformedMuscleSectionsObstruction:
    address: str
    muscle_id: str
    section_index: int | None
    rule: LoftSectionRule
    field: str | None
    observed_value: object
    missing_fields: tuple[str, ...]
    extra_fields: tuple[str, ...]


@dataclass(frozen=True)
class InvalidMuscleProfileObstruction:
    address: str
    muscle_id: str
    rule: MuscleProfileRule
    station_count: int
    first_station: float | None
    last_station: float | None
    endpoint_widths: tuple[float, float] | None
    endpoint_depths: tuple[float, float] | None
    maximum_interior_width: float | None
    maximum_interior_depth: float | None


@dataclass(frozen=True)
class UnresolvableMuscleAnchorObstruction:
    address: str
    muscle_id: str
    endpoint: MuscleEndpoint
    selector: str
    closest_valid_candidates: tuple[str, ...]


@dataclass(frozen=True)
class UnsupportedMuscleAnchorObstruction:
    address: str
    muscle_id: str
    endpoint: MuscleEndpoint
    selector: str
    scope_kind: ScopeKind
    supported_scope_kinds: tuple[ScopeKind, ...]


@dataclass(frozen=True)
class DegenerateMuscleSpanObstruction:
    address: str
    muscle_id: str
    origin_selector: str
    insertion_selector: str
    observed_span: float
    minimum_span: float


@dataclass(frozen=True)
class UnresolvedMuscleMirrorObstruction:
    muscle_id: str
    mirror_of: str
    closest_valid_candidates: tuple[str, ...]


@dataclass(frozen=True)
class CyclicMuscleMirrorObstruction:
    muscle_ids: tuple[str, ...]


@dataclass(frozen=True)
class MuscleMirrorAnchorMismatchObstruction:
    address: str
    muscle_id: str
    mirror_of: str
    endpoint: MuscleEndpoint
    source_selector: str
    target_selector: str
    observed_distance: float
    maximum_distance: float


type MuscleObstruction = (
    GeometryObstruction
    | MalformedMuscleDeclarationObstruction
    | MalformedMuscleBulkObstruction
    | UnresolvableMuscleBulkBasisObstruction
    | DuplicateMuscleIdObstruction
    | MusclePartCollisionObstruction
    | MalformedMuscleSectionsObstruction
    | InvalidMuscleProfileObstruction
    | UnresolvableMuscleAnchorObstruction
    | UnsupportedMuscleAnchorObstruction
    | DegenerateMuscleSpanObstruction
    | UnresolvedMuscleMirrorObstruction
    | CyclicMuscleMirrorObstruction
    | MuscleMirrorAnchorMismatchObstruction
    | MuscleFormationObstruction
)


type LoftSectionResult = tuple[LoftSection, ...] | LoftSectionObstruction


@dataclass(frozen=True)
class CompiledBody:
    """Named compiler product; typed anatomy is never serialized then decoded."""

    graph: dict
    receipt: dict
    anatomy: "AnatomyResult | None"
    eye_globes: tuple["EyeGlobe", ...] = ()


type Vector3 = tuple[float, float, float]
type Matrix3 = tuple[Vector3, Vector3, Vector3]


class CarveKind(StrEnum):
    GENCYL = "gencyl"
    BLOB = "blob"
    BOX = "box"


CARVE_KINDS = tuple(kind.value for kind in CarveKind)


class CarveRule(StrEnum):
    CARVES_ARRAY = "carves_array"
    OBJECT = "object"
    KNOWN_FIELD = "known_field"
    IDENTIFIER = "identifier"
    KIND = "kind"
    FINITE_NUMBER = "finite_number"
    FINITE_VECTOR3 = "finite_vector3"
    FINITE_POSITIVE_VECTOR3 = "finite_positive_vector3"
    FINITE_NON_NEGATIVE_NUMBER = "finite_non_negative_number"
    FINITE_STRICT_PAIR = "finite_strict_pair"
    FINITE_POSITIVE_ARRAY = "finite_positive_array"
    FINITE_STRICT_ARRAY = "finite_strict_array"
    MATCHING_ARITY = "matching_arity"
    STATIONS_MATCH_SPAN = "stations_match_span"


@dataclass(frozen=True)
class MalformedCarveObstruction:
    address: str
    rule: CarveRule
    authored: object
    required: object


@dataclass(frozen=True)
class DuplicateCarveIdObstruction:
    address: str
    carve_id: str


@dataclass(frozen=True)
class CarvePartCollisionObstruction:
    address: str
    part_id: str


type CarveObstruction = (
    MalformedCarveObstruction
    | DuplicateCarveIdObstruction
    | CarvePartCollisionObstruction
)


@dataclass(frozen=True)
class AcceptedCarves:
    carves: tuple[dict, ...]
    provenance: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RejectedCarves:
    obstructions: tuple[CarveObstruction, ...]


type CarveCompileResult = AcceptedCarves | RejectedCarves


class WebRule(StrEnum):
    WEBS_ARRAY = "webs_array"
    OBJECT = "object"
    KNOWN_FIELD = "known_field"
    IDENTIFIER = "identifier"
    ANCHORS_ARRAY = "anchors_array"
    ANCHOR_ARITY = "anchor_arity"
    ANCHOR_OBJECT = "anchor_object"
    BOOLEAN = "boolean"
    FINITE_NON_NEGATIVE_NUMBER = "finite_non_negative_number"
    OPERATOR = "operator"
    ROLE = "role"
    FINITE_STRICT_PAIR = "finite_strict_pair"
    FINITE_POSITIVE_ARRAY = "finite_positive_array"
    FINITE_STRICT_ARRAY = "finite_strict_array"
    MATCHING_ARITY = "matching_arity"
    STATIONS_MATCH_SPAN = "stations_match_span"
    STATION_ARITY = "station_arity"


@dataclass(frozen=True)
class MalformedWebObstruction:
    address: str
    rule: WebRule
    authored: object
    required: object


@dataclass(frozen=True)
class DuplicateWebIdObstruction:
    address: str
    web_id: str


@dataclass(frozen=True)
class WebPartCollisionObstruction:
    address: str
    part_id: str


@dataclass(frozen=True)
class UnknownWebAnchorObstruction:
    address: str
    bone: str
    closest_valid_candidates: tuple[str, ...]


type WebObstruction = (
    MalformedWebObstruction
    | DuplicateWebIdObstruction
    | WebPartCollisionObstruction
    | UnknownWebAnchorObstruction
)


@dataclass(frozen=True)
class AcceptedWebs:
    webs: tuple[dict, ...]
    provenance: tuple[tuple[str, str], ...]
    anchors: tuple[tuple[str, tuple[str, ...]], ...]
    non_carrier: tuple[str, ...]


@dataclass(frozen=True)
class RejectedWebs:
    obstructions: tuple[WebObstruction, ...]


type WebCompileResult = AcceptedWebs | RejectedWebs


@dataclass(frozen=True)
class EyeHostFrame:
    origin: Vector3
    rotation: Matrix3
    length: float


@dataclass(frozen=True)
class EyeSocket:
    radius: float
    depth: float


@dataclass(frozen=True)
class BrowRidge:
    offset: Vector3
    size: Vector3
    round_radius: float
    blend: float | None


@dataclass(frozen=True)
class EyeSpec:
    eye_id: str
    host_bone_id: str
    parameter: float
    offset: Vector3
    radius: float
    mirror: bool
    appearance_material: AppearanceMaterialId
    socket: EyeSocket
    brow_ridge: BrowRidge | None


@dataclass(frozen=True)
class EyeGlobe:
    eye_id: str
    center: Vector3
    radius: float
    mirror: bool
    appearance_material: AppearanceMaterialId


class EyeRule(StrEnum):
    EYES_ARRAY = "eyes_array"
    OBJECT = "object"
    KNOWN_FIELD = "known_field"
    IDENTIFIER = "identifier"
    FINITE_NUMBER = "finite_number"
    FINITE_VECTOR3 = "finite_vector3"
    BOOLEAN = "boolean"
    CANONICAL_MIRROR_SIDE = "canonical_mirror_side"
    FINITE_POSITIVE_NUMBER = "finite_positive_number"
    FINITE_POSITIVE_VECTOR3 = "finite_positive_vector3"
    FINITE_NON_NEGATIVE_NUMBER = "finite_non_negative_number"
    SOCKET_RADIUS_EXCEEDS_EYE_RADIUS = "socket_radius_exceeds_eye_radius"
    SOCKET_DEPTH_BELOW_RADIUS = "socket_depth_below_radius"


@dataclass(frozen=True)
class MalformedEyeObstruction:
    address: str
    rule: EyeRule
    authored: object
    required: object


@dataclass(frozen=True)
class DuplicateEyeIdObstruction:
    address: str
    eye_id: str


@dataclass(frozen=True)
class UnknownEyeHostObstruction:
    address: str
    host_bone_id: str
    closest_valid_candidates: tuple[str, ...]


@dataclass(frozen=True)
class UnknownEyeMaterialObstruction:
    address: str
    appearance_material: object


@dataclass(frozen=True)
class NonGlossyEyeMaterialObstruction:
    address: str
    appearance_material: str
    roughness: float
    maximum_roughness: float


@dataclass(frozen=True)
class EyePartCollisionObstruction:
    address: str
    part_id: str


type EyeObstruction = (
    MalformedEyeObstruction
    | DuplicateEyeIdObstruction
    | UnknownEyeHostObstruction
    | UnknownEyeMaterialObstruction
    | NonGlossyEyeMaterialObstruction
    | EyePartCollisionObstruction
)


@dataclass(frozen=True)
class AcceptedEyes:
    globes: tuple[EyeGlobe, ...]
    carves: tuple[dict, ...]
    brow_parts: tuple[dict, ...]
    provenance: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RejectedEyes:
    obstructions: tuple[EyeObstruction, ...]


type EyeCompileResult = AcceptedEyes | RejectedEyes
type BodyFeatureObstruction = CarveObstruction | EyeObstruction | WebObstruction


@dataclass(frozen=True)
class SkinFormationAnatomyObstruction:
    obstructions: tuple["AnatomyObstruction", ...]


type BodyObstruction = (
    BodySpecDecodeObstruction
    | BodyRelationObstruction
    | BodyFeatureObstruction
    | MuscleObstruction
    | SkinFormationAnatomyObstruction
)


@dataclass(frozen=True)
class SolvedLocalPose:
    """Solved parent-frame placement of one bone; kinematics rebuilds the table from it."""

    bone_id: str
    local_translation: tuple[float, float, float]
    rest_direction: tuple[float, float, float]
    twist: float


@dataclass(frozen=True)
class RelationalSolveReceipt:
    sections: tuple[LocalSolveReceipt, ...]
    declarations: tuple[RelationDeclaration, ...]


@dataclass(frozen=True)
class RejectedBody:
    obstructions: tuple[BodyObstruction, ...]


type BodyCompileResult = CompiledBody | RejectedBody
