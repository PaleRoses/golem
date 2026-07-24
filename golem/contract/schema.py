from __future__ import annotations

import json

from golem.contract.model import ContractSection
from golem.contract.vocabulary import (
    appearance_material_values,
    assembly_stratum_values,
    element_role_values,
)
from golem.kernel.anatomy.vocabulary import BodyRegionKind, PerfusedTissueKind
from golem.kernel.body.relations import (
    RELATION_ALIASES,
    RelationKind,
    RelationSolvePolicy,
)
from golem.kernel.body.types import DIALECT as BODY_DIALECT, _DOF
from golem.kernel.engine.types import (
    CompositionOperator,
    MuscleFormationKind,
    SkinFormationKind,
)
from golem.materials.surface import SurfaceColorVariationKind
from golem.plates import PlateLayout


def _tokens_for(kinds: tuple[RelationKind, ...]) -> list[str]:
    return [kind.value for kind in kinds] + [
        alias for alias, kind in RELATION_ALIASES.items() if kind in kinds
    ]


def _single_reference_kind_tokens() -> list[str]:
    return _tokens_for(
        tuple(
            kind
            for kind in RelationKind
            if kind is not RelationKind.ATTACH_AT_NAMED_SITE
            and kind is not RelationKind.BETWEEN
        )
    )


def _between_kind_tokens() -> list[str]:
    return _tokens_for((RelationKind.BETWEEN,))


def body_schema() -> dict[str, object]:
    vector2 = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 2,
        "maxItems": 2,
    }
    vector3 = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }
    quaternion = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
    }
    number_series = {
        "oneOf": [
            {"type": "number"},
            {"type": "array", "items": {"type": "number"}, "minItems": 1},
        ]
    }
    profile = {
        "type": "object",
        "not": {"required": ["aspect", "depth"]},
        "properties": {
            "n": number_series,
            "aspect": number_series,
            "depth": number_series,
            "roll": number_series,
            "offset": {
                "oneOf": [
                    {"$ref": "#/$defs/vector2"},
                    {
                        "type": "array",
                        "items": {"$ref": "#/$defs/vector2"},
                        "minItems": 1,
                    },
                ]
            },
            "up": {"$ref": "#/$defs/vector3"},
        },
        "additionalProperties": True,
    }
    common_flesh = {
        "name": {"type": "string", "minLength": 1},
        "t": {"type": "number"},
        "offset": {"$ref": "#/$defs/vector3"},
        "mirror": {"type": "boolean"},
        "blend": {
            "type": "number",
            "minimum": 0,
            "description": "Optional per-part composition-radius override. Absent uses the document-level blend value. The selected non-negative radius belongs to this part's incoming union edge: blend smooths with it, chamfer bevels with it, and crease ignores it for a hard union.",
        },
        "role": {
            "enum": ["non_carrier"],
            "description": "Ontology role. Absent = vascular carrier (default law): the record joins carrier cross-section sampling against the circulation floor and carries provenance. 'non_carrier' = membrane: authored dimensions are kept exactly (no carrier-floor inflation) and the record carries no carrier provenance; a route bone left with only non-carrier flesh is honestly rejected (MissingProvenance), never patched by inflation.",
        },
        "operator": {
            "enum": [operator.value for operator in CompositionOperator],
            "description": "Closed incoming-edge composition vocabulary: blend = legacy smooth union, chamfer = bevel-like union, crease = hard union, local_blend = certified overlap-local gradient blend with exact clean-union parity outside its support. The later incoming part owns the operator at its junction; optional per-part blend overrides the document-level radius for blend, chamfer, and local_blend.",
        },
    }
    loft_section = {
        "type": "object",
        "required": [
            "station",
            "width",
            "depth",
            "exponent",
            "roll",
        ],
        "properties": {
            "station": {"type": "number"},
            "width": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Half-extent (radius) across the section, not full width: width 0.1 emits a 0.2-wide section.",
            },
            "depth": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Half-extent (radius) along the section, not full depth: depth 0.1 emits a 0.2-deep section.",
            },
            "exponent": {
                "type": "number",
                "minimum": 2,
                "maximum": 12,
            },
            "roll": {"type": "number"},
        },
        "additionalProperties": False,
    }
    flesh = {
        "oneOf": [
            {
                "type": "object",
                "required": ["kind", "radii"],
                "properties": {
                    **common_flesh,
                    "kind": {"const": "gencyl"},
                    "span": {"$ref": "#/$defs/vector2"},
                    "stations": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                    },
                    "radii": {
                        "type": "array",
                        "items": {"type": "number", "exclusiveMinimum": 0},
                        "minItems": 2,
                    },
                    "profile": profile,
                },
                "additionalProperties": True,
            },
            {
                "type": "object",
                "required": ["kind", "sections"],
                "properties": {
                    **common_flesh,
                    "kind": {"const": "loft"},
                    "sections": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"$ref": "#/$defs/loftSection"},
                    },
                },
                "additionalProperties": True,
            },
            {
                "type": "object",
                "required": ["kind", "size"],
                "properties": {
                    **common_flesh,
                    "kind": {"const": "blob"},
                    "size": {
                        "$ref": "#/$defs/vector3",
                        "description": "Half-extents [hx, hy, hz] in the host bone's frame: the mass spans +/-size per axis, so size [0.1, 0.2, 0.3] emits a 0.20 x 0.40 x 0.60 extent.",
                    },
                    "rot": {"$ref": "#/$defs/quaternion"},
                },
                "additionalProperties": True,
            },
            {
                "type": "object",
                "required": ["kind", "size"],
                "properties": {
                    **common_flesh,
                    "kind": {"const": "box"},
                    "size": {
                        "$ref": "#/$defs/vector3",
                        "description": "Half-extents [hx, hy, hz] in the host bone's frame: the prism spans +/-size per axis, so size [0.1, 0.2, 0.3] emits a 0.20 x 0.40 x 0.60 extent.",
                    },
                    "round": {"type": "number", "minimum": 0},
                    "rot": {"$ref": "#/$defs/quaternion"},
                },
                "additionalProperties": True,
            },
        ]
    }
    common_muscle = {
        "kind": {"const": "muscle"},
        "id": {"type": "string", "minLength": 1},
        "origin": {"type": "string", "minLength": 1},
        "insertion": {"type": "string", "minLength": 1},
        "blend": {"type": "number", "minimum": 0},
        "operator": {
            "enum": [operator.value for operator in CompositionOperator]
        },
        "formation": {
            "enum": [formation.value for formation in MuscleFormationKind]
        },
    }
    muscle_bulk_dimensions = {
        "type": "object",
        "required": ["width", "depth"],
        "properties": {
            "width": {"type": "number", "exclusiveMinimum": 0},
            "depth": {"type": "number", "exclusiveMinimum": 0},
        },
        "additionalProperties": False,
    }
    relative_muscle_bulk = {
        "type": "object",
        "required": ["to", "width", "depth"],
        "properties": {
            "to": {"enum": ["origin", "insertion", "span"]},
            "width": {"type": "number", "exclusiveMinimum": 0},
            "depth": {"type": "number", "exclusiveMinimum": 0},
        },
        "additionalProperties": False,
    }
    muscle_bulk = {
        "oneOf": [
            {
                "type": "object",
                "required": ["relative"],
                "properties": {
                    "relative": {"$ref": "#/$defs/relativeMuscleBulk"},
                },
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": ["absolute"],
                "properties": {
                    "absolute": {"$ref": "#/$defs/muscleBulkDimensions"},
                },
                "additionalProperties": False,
            },
        ]
    }
    muscle = {
        "oneOf": [
            {
                "type": "object",
                "required": [
                    "kind",
                    "id",
                    "origin",
                    "insertion",
                    "sections",
                ],
                "properties": {
                    **common_muscle,
                    "sections": {
                        "type": "array",
                        "minItems": 3,
                        "items": {"$ref": "#/$defs/loftSection"},
                    },
                },
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": [
                    "kind",
                    "id",
                    "origin",
                    "insertion",
                    "bulk",
                    "definition",
                ],
                "properties": {
                    **common_muscle,
                    "bulk": {"$ref": "#/$defs/muscleBulk"},
                    "definition": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": [
                    "kind",
                    "id",
                    "origin",
                    "insertion",
                    "mirror_of",
                ],
                "properties": {
                    **common_muscle,
                    "mirror_of": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            },
        ]
    }
    joint = {
        "type": "object",
        "required": ["dof"],
        "properties": {
            "dof": {"enum": list(_DOF)},
            "limits": {
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/vector2"},
            },
        },
        "additionalProperties": True,
    }
    root = {
        "type": "object",
        "required": ["id", "world"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "world": {"$ref": "#/$defs/vector3"},
            "role": {
                "enum": ["handle"],
                "description": "Ontology role. Absent = perfused organ (default law). 'handle' = transform handle: frame composition is unchanged, but the bone is contracted out of the perfused forest — it neither demands a capillary exchange bed as a leaf nor revokes its parent's terminality.",
            },
            "flesh": {"type": "array", "items": {"$ref": "#/$defs/flesh"}},
            "ports": {"type": "array", "items": {"type": "object"}},
        },
        "additionalProperties": True,
    }
    bone = {
        "type": "object",
        "required": ["id", "parent", "length", "rest_dir", "joint"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "parent": {"type": "string", "minLength": 1},
            "attach": {"$ref": "#/$defs/attach"},
            "length": {"type": "number", "exclusiveMinimum": 0},
            "rest_dir": {
                "$ref": "#/$defs/vector3",
                "description": "Unit direction composed in the PARENT bone's frame, never in world: the child's world direction is parent_R @ rest_dir and the child's own frame (which orients its children's rest_dirs and its flesh axes) is derived from that composition. A chain continuing straight along the parent's axis authors local +z at every generation; authoring the world direction into each child folds the chain, because a -z local inverts direction at each generation. Solve exact locals as parent_R^T @ desired_world.",
            },
            "twist_deg": {"type": "number"},
            "mirror": {
                "type": "boolean",
                "description": "Reflects the whole emitted subtree across world x=0: parts gain '_m' mates and vascular node ids name the authored lane ':positive:' and its reflected mate ':negative:' (an unmirrored circuit is ':center:').",
            },
            "role": {
                "enum": ["handle"],
                "description": "Ontology role. Absent = perfused organ (default law). 'handle' = transform handle: frame composition is unchanged, but the bone is contracted out of the perfused forest — it neither demands a capillary exchange bed as a leaf nor revokes its parent's terminality.",
            },
            "joint": joint,
            "flesh": {"type": "array", "items": {"$ref": "#/$defs/flesh"}},
            "arrays": {"type": "array", "items": {"type": "object"}},
            "ports": {"type": "array", "items": {"type": "object"}},
            "bone_radii": {
                "type": "array",
                "items": {"type": "number", "exclusiveMinimum": 0},
                "minItems": 2,
            },
        },
        "additionalProperties": True,
    }
    anatomy = {
        "type": "object",
        "required": ["dialect", "overall"],
        "properties": {
            "dialect": {"const": "anatomy/0.1"},
            "overall": {
                "type": "object",
                "required": ["regions", "circulation"],
                "properties": {
                    "regions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["region_id", "kind", "host_bone_id"],
                            "properties": {
                                "region_id": {"type": "string"},
                                "kind": {
                                    "enum": [kind.value for kind in BodyRegionKind]
                                },
                                "host_bone_id": {"type": "string"},
                            },
                            "additionalProperties": True,
                        },
                    },
                    "circulation": {
                        "type": "object",
                        "description": "Closed vascular intent. Exchange-bed corridors follow the flesh topology of their own route and region host bones and may leave the bone centerline; every capsule must stay buried in flesh owned by those same bones (cover from unrelated flesh, even the pump's, does not count). The FEASIBLE ENVELOPE contract section states the full cover law.",
                        "required": [
                            "kind",
                            "pump_region_id",
                            "exchange_beds",
                            "carrier_radius_scale",
                            "distance_decay",
                        ],
                        "properties": {
                            "kind": {"const": "closed_vascular"},
                            "pump_region_id": {"type": "string"},
                            "carrier_radius_scale": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "distance_decay": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "exchange_beds": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "required": ["region_id", "tissue", "demand"],
                                    "properties": {
                                        "region_id": {"type": "string"},
                                        "tissue": {
                                            "enum": [
                                                kind.value
                                                for kind in PerfusedTissueKind
                                            ]
                                        },
                                        "demand": {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                        },
                                    },
                                    "additionalProperties": True,
                                },
                            },
                        },
                        "additionalProperties": True,
                    },
                    "myotendinous_units": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                    "integument_layers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "formation": {
                                    "enum": [
                                        formation.value
                                        for formation in SkinFormationKind
                                    ]
                                }
                            },
                            "additionalProperties": True,
                        },
                    },
                },
                "additionalProperties": True,
            },
        },
        "additionalProperties": True,
    }
    plate_policy = {
        "type": "object",
        "required": ["cell_count"],
        "properties": {
            "layout": {"enum": [layout.value for layout in PlateLayout]},
            "cell_count": {"type": "integer", "minimum": 4, "maximum": 512},
            "random_seed": {"type": "integer"},
            "relief": {"type": "number", "minimum": 0},
            "groove": {"type": "number", "minimum": 0},
            "seam_lift": {"type": "number", "minimum": 0},
            "seam_material": {"$ref": "#/$defs/appearanceMaterial"},
            "axis_weight": {"type": "number", "exclusiveMinimum": 0},
            "circulation_emission": {"type": "boolean"},
        },
        "additionalProperties": True,
    }
    surface_color = {
        "type": "object",
        "description": (
            "Vertex-color recipe for an assembly element's solid exterior; "
            "derived diagnostic strata retain their declared materials."
        ),
        "required": ["tint_linear_rgb"],
        "properties": {
            "tint_linear_rgb": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": 3,
                "maxItems": 3,
            },
            "variation": {
                "type": "object",
                "required": ["kind", "wavelength_world", "amplitude"],
                "properties": {
                    "kind": {
                        "const": SurfaceColorVariationKind.TRIPLANAR_VALUE_NOISE.value
                    },
                    "wavelength_world": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                    },
                    "amplitude": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }
    appearance_palette = {
        "type": "object",
        "description": (
            "Opt-in material-indexed colors for every matching assembly "
            "record, including vascular supply, return, and exchange strata. "
            "Keys are the closed appearanceMaterial enum: a palette re-tints an "
            "existing id, it never mints a new one. Both GLB/mesh export and "
            "`look` orthographic rendering honor the tint. `tint_linear_rgb` "
            "stays linear through derivation and is converted exactly once to "
            "display sRGB at the shared face-color projection seam."
        ),
        "minProperties": 1,
        "propertyNames": {"$ref": "#/$defs/appearanceMaterial"},
        "additionalProperties": {"$ref": "#/$defs/surfaceColor"},
    }
    attach = {
        "oneOf": [
            {
                "type": "object",
                "required": ["at"],
                "properties": {"at": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "t": {"type": "number"},
                    "offset": {"$ref": "#/$defs/vector3"},
                },
                "additionalProperties": False,
            },
        ]
    }
    between_solve = [
        policy.value
        for policy in RelationSolvePolicy
        if policy is not RelationSolvePolicy.REFERENCE
    ]
    relation = {
        "oneOf": [
            {
                "type": "object",
                "required": ["id", "kind", "subject", "reference", "solve"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "kind": {"enum": _single_reference_kind_tokens()},
                    "subject": {"type": "string", "minLength": 1},
                    "reference": {"type": "string", "minLength": 1},
                    "solve": {
                        "enum": [policy.value for policy in RelationSolvePolicy]
                    },
                    "distance": {"type": "number", "minimum": 0},
                },
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": ["id", "kind", "subject", "references", "solve"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "kind": {"enum": _between_kind_tokens()},
                    "subject": {"type": "string", "minLength": 1},
                    "references": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "solve": {"enum": between_solve},
                },
                "additionalProperties": False,
            },
        ]
    }
    pose = {
        "type": "object",
        "properties": {
            "relations": {
                "type": "array",
                "items": {"$ref": "#/$defs/relation"},
            }
        },
        "additionalProperties": True,
    }
    eye_socket = {
        "type": "object",
        "required": ["radius"],
        "properties": {
            "radius": {"type": "number", "exclusiveMinimum": 0},
            "depth": {"type": "number", "minimum": 0},
        },
        "additionalProperties": False,
    }
    brow_ridge = {
        "type": "object",
        "required": ["offset", "size"],
        "properties": {
            "offset": {"$ref": "#/$defs/vector3"},
            "size": {"$ref": "#/$defs/vector3"},
            "round": {"type": "number", "minimum": 0},
            "blend": {"type": "number", "minimum": 0},
        },
        "additionalProperties": False,
    }
    eye = {
        "type": "object",
        "required": [
            "id",
            "host_bone_id",
            "offset",
            "radius",
            "socket",
        ],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "host_bone_id": {"type": "string", "minLength": 1},
            "t": {"type": "number"},
            "offset": {"$ref": "#/$defs/vector3"},
            "radius": {"type": "number", "exclusiveMinimum": 0},
            "mirror": {"type": "boolean"},
            "appearance_material": {"$ref": "#/$defs/appearanceMaterial"},
            "socket": {"$ref": "#/$defs/eyeSocket"},
            "brow_ridge": {"$ref": "#/$defs/browRidge"},
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://golem.local/schema/body-0.3.json",
        "title": "GOLEM body/0.3 authoring document",
        "type": "object",
        "required": ["name", "dialect", "skeleton"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "dialect": {"const": BODY_DIALECT},
            "emit_target": {"enum": ["v02", "v03"]},
            "blend": {"type": "number", "minimum": 0},
            "ground_y": {"type": "number"},
            "ground_tol": {"type": "number", "minimum": 0},
            "appearance_material": {"$ref": "#/$defs/appearanceMaterial"},
            "appearance_palette": {"$ref": "#/$defs/appearancePalette"},
            "plate_policy": {"$ref": "#/$defs/platePolicy"},
            "surface_color": {"$ref": "#/$defs/surfaceColor"},
            "anatomy": {"$ref": "#/$defs/anatomy"},
            "skeleton": {
                "type": "object",
                "required": ["root", "bones"],
                "properties": {
                    "root": {"$ref": "#/$defs/root"},
                    "bones": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/bone"},
                    },
                },
                "additionalProperties": True,
            },
            "mounts": {"type": "array", "items": {"type": "object"}},
            "props": {"type": "array", "items": {"type": "object"}},
            "pose": {"$ref": "#/$defs/pose"},
            "contacts": {"type": "array", "items": {"type": "object"}},
            "appendages": {"type": "array", "items": {"type": "object"}},
            "conduits": {"type": "array", "items": {"type": "object"}},
            "eyes": {"type": "array", "items": {"$ref": "#/$defs/eye"}},
            "muscles": {
                "type": "array",
                "items": {"$ref": "#/$defs/muscle"},
            },
            "contract": {
                "oneOf": [{"type": "string"}, {"type": "object"}]
            },
            "schematic": {"type": "object"},
        },
        "additionalProperties": True,
        "$defs": {
            "vector2": vector2,
            "vector3": vector3,
            "quaternion": quaternion,
            "profile": profile,
            "loftSection": loft_section,
            "flesh": flesh,
            "muscleBulkDimensions": muscle_bulk_dimensions,
            "relativeMuscleBulk": relative_muscle_bulk,
            "muscleBulk": muscle_bulk,
            "muscle": muscle,
            "joint": joint,
            "root": root,
            "bone": bone,
            "attach": attach,
            "relation": relation,
            "pose": pose,
            "eyeSocket": eye_socket,
            "browRidge": brow_ridge,
            "eye": eye,
            "anatomy": anatomy,
            "platePolicy": plate_policy,
            "appearancePalette": appearance_palette,
            "surfaceColor": surface_color,
            "elementRole": {"enum": list(element_role_values())},
            "appearanceMaterial": {
                "enum": list(appearance_material_values()),
                "description": (
                    "Closed vocabulary of exactly these ids; no other material "
                    "name decodes. Eyes additionally require a glossy id, and "
                    "'eye_gloss' is the only glossy member."
                ),
            },
            "assemblyStratum": {"enum": list(assembly_stratum_values())},
        },
    }


def render() -> str:
    return json.dumps(body_schema(), indent=2, sort_keys=True) + "\n"


SECTION = ContractSection("schema", render)
