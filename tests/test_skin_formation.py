from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from golem import assembly
from golem.assembly.compile import (
    _apply_plate_policy,
    _apply_surface_detail,
    _expand_appendages,
)
from golem.assembly.mounts import concretize_mirrors, mount_graph
from golem.assembly.obstructions import (
    ElementSurfaceDetailObstruction,
    ElementSurfaceFormationObstruction,
    PlateElementObstruction,
)
from golem.assembly.project import project_assembly_obstructions
from golem.conduits import apply_conduits
from golem.conduits.surface.types import EmissionPitchObstruction
from golem.kernel import anatomy, body, engine
from golem.kernel.engine import compile as engine_compile
from golem.kernel.engine.compile import (
    SEALED_SKIN_POLICY,
    _skin_adjacency,
    _skin_formation_sections,
    _skin_segment_crossing_distances,
    _skin_segment_crossings,
    _solve_skin_layer_with_policy,
)
from golem.kernel.engine.types import (
    CompositionSectionId,
    Rejected,
    decode_graph,
    require_accepted,
)


_ROOT = Path(__file__).resolve().parents[1]
_KNIGHT = _ROOT / "specs" / "knight_body.json"
_KNIGHT_GRAPH_SHA256 = (
    "e0414c3001a1647ed6630bd4703b6daf9309cf4f93011b62c8b9159db3a463af"
)
_KNIGHT_RECEIPT_SHA256 = (
    "35f6cbd44979bbd9edfc473f35c540e03e3705616b4914ed902a60ba91d6414b"
)
_KNIGHT_RES32_ARRAY_SHA256 = {
    "vertices": "7c443412e2a6d3b30354942db964f9cdb28a5f21c9458b2f293c0266611187c5",
    "faces": "cf9ba346a66758ac64d0b5fcae76868eedb893e63fc15678bb03b955d516fa9d",
    "normals": "e1182bf93009af754d64f17bbc0ca75393e691842ecea8899f928676dc4976b3",
    "field": "b23f4d93cb34b37f24136fc2e2c134a7590a7a97ba4a8673d29158ec0b250db5",
}


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _compiled_knight() -> body.CompiledBody:
    result = body.Compiler(
        json.loads(_KNIGHT.read_text(encoding="utf-8")),
        spec_dir=_KNIGHT.parent,
    ).compile()

    assert isinstance(result, body.CompiledBody)
    return result


def _knight_with_formed_integument(
    region_ids: tuple[str, ...],
) -> dict[str, object]:
    spec = json.loads(_KNIGHT.read_text(encoding="utf-8"))
    overall = spec["anatomy"]["overall"]
    return {
        **spec,
        "anatomy": {
            **spec["anatomy"],
            "overall": {
                **overall,
                "integument_layers": [
                    {
                        "region_id": region_id,
                        "thickness": 0.02,
                        "formation": "static_implicit_relaxation",
                    }
                    for region_id in region_ids
                ],
            },
        },
    }


def _legacy_flesh_graph() -> dict[str, object]:
    return {
        "name": "legacy-flesh",
        "blend": 0.0,
        "parts": [
            {
                "id": "host",
                "type": "blob",
                "center": (0.0, 0.0, 0.0),
                "size": (0.5, 0.5, 0.5),
            }
        ],
    }


def _separated_limb_quadruped_graph() -> dict[str, object]:
    return {
        "name": "skin-siege-quadruped",
        "blend": 0.03,
        "parts": [
            {
                "id": "torso",
                "type": "blob",
                "center": (0.0, 0.5, 0.0),
                "size": (0.45, 0.12, 0.22),
            },
            *[
                {
                    "id": identifier,
                    "type": "gencyl",
                    "spine": ((x, 0.5, z), (x, 0.05, z)),
                    "radii": (0.05, 0.05),
                }
                for identifier, x, z in (
                    ("leg_fl", -0.35, 0.15),
                    ("leg_fr", -0.35, -0.15),
                    ("leg_bl", 0.35, 0.15),
                    ("leg_br", 0.35, -0.15),
                )
            ],
        ],
    }


def _skin_problem(
    thickness: float = 0.02,
) -> engine.SkinLayerProblem:
    return engine.SkinLayerProblem(
        engine.SkinFormationKind.STATIC_IMPLICIT_RELAXATION,
        thickness,
        ("host",),
    )


def _skinned(
    graph: dict[str, object],
    thickness: float = 0.02,
) -> dict[str, object]:
    return {
        **graph,
        "skin": {
            "formation": "static_implicit_relaxation",
            "thickness": thickness,
            "region_ids": ("host",),
        },
    }


def test_legacy_integument_omission_is_byte_exact() -> None:
    compiled = _compiled_knight()
    canonical_graph = body._canonical_json(compiled.graph)
    canonical_receipt = body._canonical_json(compiled.receipt)
    evaluated = engine.evaluate(compiled.graph, res=32)

    assert isinstance(evaluated, engine.EvaluatedMorphology)
    assert hashlib.sha256(canonical_graph.encode()).hexdigest() == (
        _KNIGHT_GRAPH_SHA256
    )
    assert hashlib.sha256(canonical_receipt.encode()).hexdigest() == (
        _KNIGHT_RECEIPT_SHA256
    )
    assert {
        name: _array_sha256(getattr(evaluated, name))
        for name in _KNIGHT_RES32_ARRAY_SHA256
    } == _KNIGHT_RES32_ARRAY_SHA256
    assert evaluated.lower == (-0.650286, -1.455559, -0.41316465)
    assert evaluated.upper == (0.650286, 2.102281, 0.51985965)
    assert "skin" not in compiled.graph
    assert "skin_formation" not in compiled.graph
    assert '"formation"' not in canonical_graph
    assert '"formation"' not in canonical_receipt
    assert evaluated.skin_correspondence is None
    assert evaluated.skin_evidence is None


def test_body_emits_skin_for_uniform_total_anatomy_region_cover() -> None:
    spec = json.loads(_KNIGHT.read_text(encoding="utf-8"))
    region_ids = tuple(
        region["region_id"]
        for region in spec["anatomy"]["overall"]["regions"]
    )
    result = body.Compiler(
        _knight_with_formed_integument(region_ids),
        spec_dir=_KNIGHT.parent,
    ).compile()

    assert isinstance(result, body.CompiledBody)
    assert result.graph["skin"] == {
        "formation": "static_implicit_relaxation",
        "thickness": 0.02,
        "region_ids": region_ids,
    }


def _duplicate_positive_id_graph() -> dict[str, object]:
    return {
        "blend": 0.0,
        "parts": (
            {
                "id": "duplicate",
                "type": "blob",
                "center": (-0.5, 0.0, 0.0),
                "size": (0.25, 0.25, 0.25),
            },
            {
                "id": "duplicate",
                "type": "blob",
                "center": (0.5, 0.0, 0.0),
                "size": (0.25, 0.25, 0.25),
            },
        ),
    }


def test_legacy_graph_decode_preserves_duplicate_positive_section_ids() -> None:
    decoded = decode_graph(_duplicate_positive_id_graph())

    assert require_accepted(decoded).skin is None


def test_skin_graph_decode_rejects_duplicate_positive_section_ids() -> None:
    decoded = decode_graph(_skinned(_duplicate_positive_id_graph()))

    assert isinstance(decoded, Rejected)
    assert decoded.obstructions[-1].detail == (
        "positive composition section ids must be unique: duplicate"
    )


def test_body_rejects_formed_skin_with_incomplete_anatomy_region_cover() -> None:
    result = body.Compiler(
        _knight_with_formed_integument(("forelimb", "hindlimb")),
        spec_dir=_KNIGHT.parent,
    ).compile()

    assert isinstance(result, body.RejectedBody)
    assert len(result.obstructions) == 1
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, body.SkinFormationAnatomyObstruction)
    assert len(obstruction.obstructions) == 1
    incompatibility = obstruction.obstructions[0]
    assert isinstance(
        incompatibility,
        anatomy.IncompatibleIntegumentFormationObstruction,
    )
    assert incompatibility.formed_region_ids == ("forelimb", "hindlimb")
    assert incompatibility.legacy_region_ids == ()
    assert incompatibility.uncovered_region_ids == ("axial_core", "cranial")
    assert incompatibility.formed_thicknesses == (0.02, 0.02)
    assert body.project_obstruction(obstruction)["obstructions"][0][
        "uncovered_region_ids"
    ] == ("axial_core", "cranial")


def test_fully_covered_knight_rejects_local_skin_ambiguity() -> None:
    spec = json.loads(_KNIGHT.read_text(encoding="utf-8"))
    region_ids = tuple(
        region["region_id"]
        for region in spec["anatomy"]["overall"]["regions"]
    )
    compiled = body.Compiler(
        _knight_with_formed_integument(region_ids),
        spec_dir=_KNIGHT.parent,
    ).compile()

    assert isinstance(compiled, body.CompiledBody)
    evaluated = engine.evaluate(compiled.graph, res=60)
    assert isinstance(evaluated, engine.RejectedSurfaceFormation)
    assert len(evaluated.obstructions) == 1
    obstruction = evaluated.obstructions[0]
    assert isinstance(obstruction, engine.SkinRelaxationObstruction)
    assert isinstance(obstruction.failure, engine.SkinProjectionUnsatisfied)
    multiplicity = obstruction.failure.failure
    assert isinstance(multiplicity, engine.SkinRootMultiplicity)
    assert multiplicity.minimum_root_count >= 2
    assert multiplicity.root_distances_world


def test_formed_skin_thickness_is_withheld_from_flesh_exactly_once() -> None:
    spec = json.loads(_KNIGHT.read_text(encoding="utf-8"))
    region_ids = tuple(
        region["region_id"]
        for region in spec["anatomy"]["overall"]["regions"]
    )
    legacy_layers = [
        {"region_id": region_id, "thickness": 0.02}
        for region_id in region_ids
    ]
    legacy_spec = {
        **spec,
        "anatomy": {
            **spec["anatomy"],
            "overall": {
                **spec["anatomy"]["overall"],
                "integument_layers": legacy_layers,
            },
        },
    }
    legacy = body.Compiler(
        legacy_spec,
        spec_dir=_KNIGHT.parent,
    ).compile()
    formed = body.Compiler(
        _knight_with_formed_integument(region_ids),
        spec_dir=_KNIGHT.parent,
    ).compile()

    assert isinstance(legacy, body.CompiledBody)
    assert isinstance(formed, body.CompiledBody)
    legacy_envelopes = {
        envelope["shape_address"]: envelope
        for envelope in legacy.receipt["anatomy"]["tissue_envelopes"]
    }
    formed_envelopes = {
        envelope["shape_address"]: envelope
        for envelope in formed.receipt["anatomy"]["tissue_envelopes"]
    }
    assert legacy_envelopes.keys() == formed_envelopes.keys()
    assert all(
        np.allclose(
            np.asarray(
                tuple(
                    station["half_size"]
                    for station in legacy_envelopes[address]["stations"]
                )
            )
            - np.asarray(
                tuple(
                    station["half_size"]
                    for station in formed_envelopes[address]["stations"]
                )
            ),
            0.02,
            rtol=0.0,
            atol=1.0e-12,
        )
        for address in legacy_envelopes
    )
    assert formed.graph["skin"]["thickness"] == 0.02


def test_static_skin_relaxation_accepts_legacy_flesh_with_scalar_proof() -> None:
    flesh_graph = _legacy_flesh_graph()
    skin_graph = _skinned(flesh_graph, thickness=0.04)
    flesh = engine.evaluate(flesh_graph, res=36)
    skin = engine.evaluate(skin_graph, res=36)

    assert isinstance(flesh, engine.EvaluatedMorphology)
    assert isinstance(skin, engine.EvaluatedMorphology)
    assert skin.skin_correspondence is not None
    assert skin.skin_evidence is not None
    correspondence = skin.skin_correspondence
    evidence = skin.skin_evidence
    displacement = skin.vertices - correspondence.flesh_vertices

    assert np.array_equal(skin.faces, flesh.faces)
    assert np.array_equal(skin.field, flesh.field - 0.04)
    assert skin.lower == flesh.lower
    assert skin.upper == flesh.upper
    assert np.array_equal(correspondence.flesh_vertices, flesh.vertices)
    assert correspondence.flesh_vertices.flags.writeable is False
    assert skin.vertices.flags.writeable is False
    assert skin.normals.flags.writeable is False
    assert not np.array_equal(skin.vertices, flesh.vertices)
    assert not np.array_equal(skin.normals, flesh.normals)
    assert np.all(
        np.einsum("ij,ij->i", displacement, skin.normals) > 0.0
    )
    np.testing.assert_allclose(
        np.linalg.norm(skin.normals, axis=1),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert evidence.formation is (
        engine.SkinFormationKind.STATIC_IMPLICIT_RELAXATION
    )
    assert evidence.region_ids == ("host",)
    assert evidence.thickness == 0.04
    assert evidence.source_vertex_count == len(flesh.vertices)
    assert evidence.source_face_count == len(flesh.faces)
    assert evidence.formed_correspondence_count == 0
    assert evidence.legacy_correspondence_count == len(flesh.vertices)
    assert evidence.maximum_owner_multiplicity == 0
    assert evidence.minimum_root_multiplicity == 1
    assert evidence.maximum_root_multiplicity == 1
    assert evidence.minimum_projection_gradient > 0.0
    assert evidence.minimum_containment_margin >= 0.0
    assert evidence.maximum_offset_residual <= (
        SEALED_SKIN_POLICY.offset_tolerance_pitch
        * skin.maximum_pitch
    )
    assert (
        SEALED_SKIN_POLICY.stretch_lower_ratio
        <= evidence.minimum_edge_stretch
        <= evidence.maximum_edge_stretch
        <= SEALED_SKIN_POLICY.stretch_upper_ratio
    )
    assert evidence.minimum_face_orientation > (
        SEALED_SKIN_POLICY.minimum_face_orientation
    )
    assert evidence.minimum_face_area_ratio >= (
        SEALED_SKIN_POLICY.minimum_face_area_ratio
    )
    assert evidence.iteration_budget == (
        SEALED_SKIN_POLICY.iteration_budget
    )
    assert evidence.projection_iteration_budget == (
        SEALED_SKIN_POLICY.projection_iteration_budget
    )
    assert evidence.maximum_tangential_displacement > 0.0
    assert evidence.fixed_point_residual <= (
        SEALED_SKIN_POLICY.convergence_tolerance_pitch
        * skin.maximum_pitch
    )
    assert all(section == () for section in correspondence.formation_sections)


def test_authoring_skin_memo_reuses_unchanged_governing_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOLEM_CIRCUIT_CACHE_COLD", raising=False)
    engine_compile._SKIN_MEMO.clear()
    calls = 0
    original = engine_compile._skin_local_scaffold

    def count_scaffold(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        engine_compile,
        "_skin_local_scaffold",
        count_scaffold,
    )
    graph = _legacy_flesh_graph()
    first = engine.evaluate(_skinned(graph, thickness=0.02), res=36)
    second = engine.evaluate(_skinned(graph, thickness=0.02), res=36)
    changed = engine.evaluate(_skinned(graph, thickness=0.03), res=36)
    engine_compile._SKIN_MEMO.clear()

    assert isinstance(first, engine.EvaluatedMorphology)
    assert isinstance(second, engine.EvaluatedMorphology)
    assert isinstance(changed, engine.EvaluatedMorphology)
    assert np.array_equal(second.vertices, first.vertices)
    assert np.array_equal(second.normals, first.normals)
    assert second.skin_evidence == first.skin_evidence
    assert calls == 2


def test_skin_memo_key_distinguishes_formation_sections() -> None:
    flesh = engine.evaluate(_legacy_flesh_graph(), res=20)

    assert isinstance(flesh, engine.EvaluatedMorphology)
    problem = _skin_problem(0.02)
    empty = tuple(() for _vertex in flesh.vertices)
    owned = (
        (CompositionSectionId("host", False),),
        *tuple(() for _vertex in flesh.vertices[1:]),
    )

    assert engine_compile._skin_memo_key(
        flesh, problem, SEALED_SKIN_POLICY, empty
    ) == engine_compile._skin_memo_key(
        flesh, problem, SEALED_SKIN_POLICY, empty
    )
    assert engine_compile._skin_memo_key(
        flesh, problem, SEALED_SKIN_POLICY, owned
    ) != engine_compile._skin_memo_key(
        flesh, problem, SEALED_SKIN_POLICY, empty
    )


def test_static_skin_accepts_separated_limb_quadruped() -> None:
    result = engine.evaluate(
        _skinned(_separated_limb_quadruped_graph(), thickness=0.004),
        res=64,
    )

    assert isinstance(result, engine.EvaluatedMorphology)
    assert result.skin_evidence is not None
    evidence = result.skin_evidence
    assert evidence.minimum_root_multiplicity == 1
    assert evidence.maximum_root_multiplicity == 1
    assert (
        SEALED_SKIN_POLICY.stretch_lower_ratio
        <= evidence.minimum_edge_stretch
        <= evidence.maximum_edge_stretch
        <= SEALED_SKIN_POLICY.stretch_upper_ratio
    )
    assert evidence.minimum_face_orientation > (
        SEALED_SKIN_POLICY.minimum_face_orientation
    )
    assert evidence.maximum_projection_displacement <= max(
        SEALED_SKIN_POLICY.locality_radius_thickness * evidence.thickness,
        SEALED_SKIN_POLICY.locality_radius_pitch * result.maximum_pitch,
    )


def test_skin_point_and_raster_fields_are_identical_on_the_evaluation_grid() -> None:
    graph = _skinned(_legacy_flesh_graph(), thickness=0.03)
    evaluated = engine.evaluate(graph, res=24)

    assert isinstance(evaluated, engine.EvaluatedMorphology)
    axes = tuple(
        np.linspace(
            evaluated.lower[axis],
            evaluated.upper[axis],
            evaluated.resolution,
        )
        for axis in range(3)
    )
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.stack(grid, axis=-1).reshape((-1, 3))
    sampled = engine.sample_graph_field(graph, points)

    assert isinstance(sampled, np.ndarray)
    assert np.array_equal(
        sampled.reshape(evaluated.field.shape),
        evaluated.field,
    )


def test_skin_accepts_the_certified_local_blend_legacy_union() -> None:
    graph = {
        "blend": 0.0,
        "parts": [
            {
                "id": "left",
                "type": "blob",
                "center": (-0.2, 0.0, 0.0),
                "size": (0.5, 0.5, 0.5),
            },
            {
                "id": "right",
                "type": "blob",
                "center": (0.2, 0.0, 0.0),
                "size": (0.5, 0.5, 0.5),
                "operator": "local_blend",
                "blend": 0.02,
            },
        ],
        "skin": {
            "formation": "static_implicit_relaxation",
            "thickness": 0.01,
            "region_ids": ("joined",),
        },
    }
    evaluated = engine.evaluate(graph, res=32)

    assert isinstance(evaluated, engine.EvaluatedMorphology)
    assert len(evaluated.composition_evidence) == 1
    assert evaluated.skin_evidence is not None
    assert evaluated.skin_evidence.minimum_root_multiplicity == 1
    assert evaluated.skin_evidence.maximum_root_multiplicity == 1


def test_static_skin_relaxation_preserves_phase_two_owner_correspondence() -> None:
    graph = {
        "name": "formed-flesh",
        "blend": 0.0,
        "parts": [
            {
                "id": "muscle",
                "type": "gencyl",
                "spine": ((0.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
                "radii": (0.4, 0.4),
                "formation": "skeleton_integral",
            }
        ],
        "skin": {
            "formation": "static_implicit_relaxation",
            "thickness": 0.02,
            "region_ids": ("host",),
        },
    }
    evaluated = engine.evaluate(graph, res=32)

    assert isinstance(evaluated, engine.EvaluatedMorphology)
    assert evaluated.formation_evidence
    assert evaluated.skin_correspondence is not None
    assert evaluated.skin_evidence is not None
    assert evaluated.skin_evidence.formed_correspondence_count == len(
        evaluated.vertices
    )
    assert evaluated.skin_evidence.legacy_correspondence_count == 0
    assert evaluated.skin_evidence.maximum_owner_multiplicity == 1
    assert frozenset(
        evaluated.skin_correspondence.formation_sections
    ) == frozenset(
        ((engine.CompositionSectionId("muscle", False),),)
    )


def test_static_skin_correspondence_preserves_formed_field_ties() -> None:
    graph = {
        "name": "formed-field-ties",
        "blend": 0.01,
        "parts": (
            {
                "id": "left",
                "type": "gencyl",
                "spine": ((-0.15, 0.0, 0.0), (-0.15, 1.5, 0.0)),
                "radii": (0.3, 0.3),
                "formation": "skeleton_integral",
            },
            {
                "id": "right",
                "type": "gencyl",
                "spine": ((0.15, 0.0, 0.0), (0.15, 1.5, 0.0)),
                "radii": (0.3, 0.3),
                "formation": "skeleton_integral",
                "operator": "local_blend",
            },
        ),
        "skin": {
            "formation": "static_implicit_relaxation",
            "thickness": 0.01,
            "region_ids": ("host",),
        },
    }
    evaluated = engine.evaluate(graph, res=25)

    assert isinstance(evaluated, engine.EvaluatedMorphology)
    assert evaluated.skin_correspondence is not None
    assert evaluated.skin_evidence is not None
    left = engine.CompositionSectionId("left", False)
    right = engine.CompositionSectionId("right", False)
    equivalence_classes = frozenset(
        map(frozenset, evaluated.skin_correspondence.formation_sections)
    )
    assert frozenset((left,)) in equivalence_classes
    assert frozenset((right,)) in equivalence_classes
    assert frozenset((left, right)) in equivalence_classes
    assert evaluated.skin_evidence.maximum_owner_multiplicity == 2


def test_skin_correspondence_excludes_contained_formed_sections() -> None:
    graph = {
        "name": "contained-formed-section",
        "blend": 0.0,
        "parts": [
            {
                "id": "legacy",
                "type": "blob",
                "center": (0.0, 0.0, 0.0),
                "size": (1.0, 1.0, 1.0),
            },
            {
                "id": "formed",
                "type": "gencyl",
                "spine": ((0.5, -0.05, 0.0), (0.5, 0.05, 0.0)),
                "radii": (0.45, 0.45),
                "formation": "skeleton_integral",
                "operator": "crease",
            }
        ],
    }
    flesh = engine.evaluate(graph, res=32)

    assert isinstance(flesh, engine.EvaluatedMorphology)
    formation_sections = _skin_formation_sections(
        require_accepted(decode_graph(graph)),
        flesh,
    )
    assert isinstance(formation_sections, tuple)
    assert frozenset(map(frozenset, formation_sections)) == frozenset(
        (frozenset(),)
    )


@pytest.mark.parametrize(
    (
        "thickness",
        "policy_overrides",
        "expected_failure",
        "expected_projection_failure",
    ),
    (
        (
            0.3,
            {},
            engine.SkinProjectionUnsatisfied,
            engine.SkinDomainEscape,
        ),
        (
            0.02,
            {"minimum_gradient": 10.0},
            engine.SkinProjectionUnsatisfied,
            engine.SkinGradientDegeneracy,
        ),
        (
            0.02,
            {"projection_descent_budget": 0},
            engine.SkinProjectionUnsatisfied,
            engine.SkinRootAbsent,
        ),
        (
            0.02,
            {
                "projection_descent_budget": 1,
                "descent_step_pitch": 1.0e-3,
            },
            engine.SkinProjectionUnsatisfied,
            engine.SkinRootAbsent,
        ),
        (
            0.02,
            {"minimum_containment_margin": 0.03},
            engine.SkinContainmentUnsatisfied,
            None,
        ),
        (
            0.02,
            {
                "projection_iteration_budget": 0,
                "offset_tolerance_pitch": 0.0,
            },
            engine.SkinOffsetUnsatisfied,
            None,
        ),
        (
            0.02,
            {
                "stretch_lower_ratio": 2.0,
                "stretch_upper_ratio": 3.0,
            },
            engine.SkinStretchUnsatisfied,
            None,
        ),
        (
            0.02,
            {
                "stretch_lower_ratio": -1.0e9,
                "stretch_upper_ratio": 1.0e9,
                "minimum_face_orientation": 1.0,
                "minimum_face_area_ratio": 0.0,
            },
            engine.SkinFoldover,
            None,
        ),
        (
            0.02,
            {
                "stretch_lower_ratio": -1.0e9,
                "stretch_upper_ratio": 1.0e9,
                "convergence_tolerance_pitch": 0.0,
                "projection_tolerance_pitch": 10.0,
            },
            engine.SkinConvergenceUnsatisfied,
            None,
        ),
    ),
    ids=(
        "domain-escape",
        "gradient-degeneracy",
        "root-absent",
        "root-absent-sampled",
        "containment",
        "offset",
        "stretch",
        "foldover",
        "convergence",
    ),
)
def test_skin_relaxation_failures_are_typed_obstructions(
    thickness: float,
    policy_overrides: dict[str, float | int],
    expected_failure: type[engine.SkinRelaxationFailure],
    expected_projection_failure: type[engine.SkinProjectionFailure] | None,
) -> None:
    flesh = engine.evaluate(_legacy_flesh_graph(), res=20)

    assert isinstance(flesh, engine.EvaluatedMorphology)
    result = _solve_skin_layer_with_policy(
        flesh,
        _skin_problem(thickness),
        replace(SEALED_SKIN_POLICY, **policy_overrides),
        tuple(() for _vertex in flesh.vertices),
    )

    assert isinstance(result, engine.SkinRelaxationObstruction)
    assert result.region_ids == ("host",)
    assert isinstance(result.failure, expected_failure)
    if expected_projection_failure is not None:
        assert isinstance(result.failure, engine.SkinProjectionUnsatisfied)
        assert isinstance(
            result.failure.failure,
            expected_projection_failure,
        )


def test_local_skin_projection_reports_multiple_crossings_without_tie_breaking() -> None:
    shape = (9, 9, 9)
    coordinates = np.indices(shape)
    interior = (
        (coordinates[0] >= 2)
        & (coordinates[0] < 7)
        & (coordinates[1] >= 2)
        & (coordinates[1] < 7)
    )
    axial_values = np.asarray(
        (1.0, -1.0, -0.1, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
    )
    field = np.where(
        interior,
        axial_values[coordinates[2]],
        1.0,
    )
    flesh_vertices = np.asarray(
        (
            (0.375, 0.375, 0.25),
            (0.625, 0.375, 0.25),
            (0.375, 0.625, 0.25),
        )
    )
    skin_vertices = np.array(flesh_vertices, copy=True)
    skin_vertices[:, 2] = 0.75
    lower = np.zeros(3)
    pitch = np.full(3, 0.125)

    counts = _skin_segment_crossings(
        flesh_vertices,
        skin_vertices,
        field,
        lower,
        pitch,
        minimum_gradient=SEALED_SKIN_POLICY.minimum_gradient,
        root_tolerance=1.0e-12,
    )
    distances = _skin_segment_crossing_distances(
        flesh_vertices,
        skin_vertices,
        field,
        lower,
        pitch,
        minimum_gradient=SEALED_SKIN_POLICY.minimum_gradient,
        root_tolerance=1.0e-12,
    )

    assert np.array_equal(counts, np.full(3, 3))
    assert tuple(map(len, distances)) == (3, 3, 3)
    assert all(
        all(left < right for left, right in zip(row, row[1:]))
        for row in distances
    )
    assert all(distance > 0.0 for row in distances for distance in row)


def test_skin_segment_exact_crossings_detect_thin_off_grid_slab() -> None:
    axial_indices = np.arange(65)
    axial_values = np.where(
        (axial_indices == 13) | (axial_indices >= 50),
        -1.0,
        1.0,
    )
    field = np.broadcast_to(axial_values, (2, 2, 65))
    flesh_vertices = np.asarray(((0.5, 0.5, 0.0),))
    skin_vertices = np.asarray(((0.5, 0.5, 1.0),))
    lower = np.zeros(3)
    pitch = np.asarray((1.0, 1.0, 1.0 / 64.0))
    historical_parameters = np.linspace(0.0, 1.0, 17)
    historical_points = (
        flesh_vertices[:, None, :]
        + historical_parameters[None, :, None]
        * (skin_vertices - flesh_vertices)[:, None, :]
    )
    historical_values = engine_compile._trilinear_samples(
        field,
        lower,
        pitch,
        historical_points.reshape((-1, 3)),
    ).values

    assert (
        np.count_nonzero(
            historical_values[:-1] * historical_values[1:] < 0.0
        )
        == 1
    )
    counts = _skin_segment_crossings(
        flesh_vertices,
        skin_vertices,
        field,
        lower,
        pitch,
        minimum_gradient=SEALED_SKIN_POLICY.minimum_gradient,
        root_tolerance=1.0e-12,
    )
    distances = _skin_segment_crossing_distances(
        flesh_vertices,
        skin_vertices,
        field,
        lower,
        pitch,
        minimum_gradient=SEALED_SKIN_POLICY.minimum_gradient,
        root_tolerance=1.0e-12,
    )

    assert np.array_equal(counts, np.asarray((3,)))
    np.testing.assert_allclose(
        distances[0],
        np.asarray((25.0, 27.0, 99.0)) / 128.0,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_skin_segment_exact_crossings_count_tangency_once() -> None:
    touch_parameter = 13.0 / 32.0
    segment_start = 0.39491505825194057
    segment_end = 0.4228164533236382
    coordinates = np.indices((2, 2, 2), dtype=np.float64)
    field = (
        (coordinates[0] - touch_parameter)
        * (coordinates[1] - touch_parameter)
    )
    flesh_vertices = np.asarray(
        ((segment_start, segment_start, 0.5),)
    )
    skin_vertices = np.asarray(
        ((segment_end, segment_end, 0.5),)
    )
    lower = np.zeros(3)
    pitch = np.ones(3)
    historical_parameters = np.linspace(0.0, 1.0, 17)
    historical_points = (
        flesh_vertices[:, None, :]
        + historical_parameters[None, :, None]
        * (skin_vertices - flesh_vertices)[:, None, :]
    )
    historical_values = engine_compile._trilinear_samples(
        field,
        lower,
        pitch,
        historical_points.reshape((-1, 3)),
    ).values

    assert np.all(historical_values > 1.0e-12)
    counts = _skin_segment_crossings(
        flesh_vertices,
        skin_vertices,
        field,
        lower,
        pitch,
        minimum_gradient=SEALED_SKIN_POLICY.minimum_gradient,
        root_tolerance=1.0e-12,
    )
    distances = _skin_segment_crossing_distances(
        flesh_vertices,
        skin_vertices,
        field,
        lower,
        pitch,
        minimum_gradient=SEALED_SKIN_POLICY.minimum_gradient,
        root_tolerance=1.0e-12,
    )

    assert np.array_equal(counts, np.asarray((1,)))
    tangent_segment_parameter = (
        (touch_parameter - segment_start)
        / (segment_end - segment_start)
    )
    np.testing.assert_allclose(
        distances[0],
        tangent_segment_parameter
        * np.linalg.norm(skin_vertices[0] - flesh_vertices[0]),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_skin_segment_exact_crossings_glue_shared_cell_boundary_once() -> None:
    coordinates = np.indices((3, 3, 3), dtype=np.float64)
    field = (
        (coordinates[0] - 1.0)
        * (coordinates[1] + 10.0)
        * (coordinates[2] + 20.0)
    )
    segment_start = 0.8611386105249752
    segment_end = 1.1388613894750248
    flesh_vertices = np.full((1, 3), segment_start)
    skin_vertices = np.full((1, 3), segment_end)
    lower = np.zeros(3)
    pitch = np.ones(3)

    counts = _skin_segment_crossings(
        flesh_vertices,
        skin_vertices,
        field,
        lower,
        pitch,
        minimum_gradient=SEALED_SKIN_POLICY.minimum_gradient,
        root_tolerance=1.0e-12,
    )
    distances = _skin_segment_crossing_distances(
        flesh_vertices,
        skin_vertices,
        field,
        lower,
        pitch,
        minimum_gradient=SEALED_SKIN_POLICY.minimum_gradient,
        root_tolerance=1.0e-12,
    )

    assert np.array_equal(counts, np.asarray((1,)))
    np.testing.assert_allclose(
        distances[0],
        0.5 * np.linalg.norm(skin_vertices[0] - flesh_vertices[0]),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_skin_thickness_covaries_under_uniform_mount_scale() -> None:
    graph = _skinned(_legacy_flesh_graph(), thickness=0.04)
    mounted = assembly.mount_graph(graph, {"scale": 2.5})

    assert graph["skin"]["thickness"] == 0.04
    assert mounted["skin"]["thickness"] == 0.1


def test_skin_geometry_covaries_under_scale_and_mirror() -> None:
    graph = _skinned(
        {
            "name": "skin-transform",
            "blend": 0.0,
            "parts": [
                {
                    "id": "host",
                    "type": "blob",
                    "center": (0.2, 0.1, -0.05),
                    "size": (0.5, 0.5, 0.5),
                }
            ],
        },
        thickness=0.02,
    )
    baseline = engine.evaluate(graph, res=28)
    scaled = engine.evaluate(
        assembly.mount_graph(graph, {"scale": 3.0}),
        res=28,
    )
    mirrored = engine.evaluate(
        assembly.mount_graph(graph, {"mirror_x": True}),
        res=28,
    )

    assert isinstance(baseline, engine.EvaluatedMorphology)
    assert isinstance(scaled, engine.EvaluatedMorphology)
    assert isinstance(mirrored, engine.EvaluatedMorphology)
    assert scaled.skin_evidence is not None
    assert scaled.skin_evidence.thickness == 0.06
    assert np.array_equal(scaled.faces, baseline.faces)
    scale_residual = np.max(
        np.linalg.norm(scaled.vertices - 3.0 * baseline.vertices, axis=1)
    )
    assert scale_residual <= 1.0e-6 * scaled.maximum_pitch
    np.testing.assert_allclose(
        scaled.field,
        3.0 * baseline.field,
        rtol=0.0,
        atol=2.0e-15,
    )

    reflected = baseline.vertices * np.asarray((-1.0, 1.0, 1.0))
    mirror_residual = max(
        float(np.max(cKDTree(mirrored.vertices).query(reflected)[0])),
        float(np.max(cKDTree(reflected).query(mirrored.vertices)[0])),
    )
    assert mirror_residual <= 2.0e-6 * baseline.maximum_pitch


def test_skin_geometric_neighborhood_preserves_source_components() -> None:
    scaffold = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.0, 0.1, 0.0),
            (0.0, 0.0, 0.05),
            (0.1, 0.0, 0.05),
            (0.0, 0.1, 0.05),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    normals = np.tile((0.0, 0.0, 1.0), (len(scaffold), 1))
    adjacency = _skin_adjacency(
        faces,
        scaffold,
        normals,
        0.1,
        SEALED_SKIN_POLICY,
    ).toarray()

    assert np.count_nonzero(adjacency[:3, 3:]) == 0
    assert np.count_nonzero(adjacency[3:, :3]) == 0
    assert np.count_nonzero(adjacency[:3, :3]) > 0
    assert np.count_nonzero(adjacency[3:, 3:]) > 0
    np.testing.assert_allclose(
        np.sum(adjacency, axis=1),
        np.ones(len(scaffold)),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_legacy_assembly_report_omits_skin_formation(tmp_path: Path) -> None:
    result = assembly.compile_assembly(
        {
            "name": "legacy-assembly",
            "elements": [
                {
                    "id": "body",
                    "role": "creature",
                    "graph": _legacy_flesh_graph(),
                }
            ],
        },
        tmp_path,
        resolution_policy=assembly.PinnedResolution(60),
    )

    assert isinstance(result, assembly.AcceptedAssembly)
    assert "skin_formation" not in result.records[0].report


def test_conduits_embed_against_flesh_correspondence_and_deform_skin() -> None:
    graph = {
        "name": "skin-conduit-correspondence",
        "blend": 0.0,
        "parts": [
            {
                "id": "host",
                "type": "gencyl",
                "spine": ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                "radii": (0.3, 0.3),
            }
        ],
    }
    evaluated = engine.evaluate(graph, res=36)

    assert isinstance(evaluated, engine.EvaluatedMorphology)
    flesh_vertices = evaluated.vertices
    skin_vertices = flesh_vertices + 0.05 * evaluated.normals
    result = apply_conduits(
        graph,
        skin_vertices,
        evaluated.faces,
        (
            {
                "id": "waist",
                "kind": "axial_loop",
                "part": "host",
                "t": 0.5,
                "width": 0.12,
                "depth": 0.005,
                "emit": ("groove", "band"),
            },
        ),
        sdf_tol=0.02,
        pitch=0.02,
        flesh_host_vertices=flesh_vertices,
    )

    assert not isinstance(result, EmissionPitchObstruction)
    displaced, bands = result
    displacement = np.linalg.norm(displaced - skin_vertices, axis=1)
    assert np.count_nonzero(displacement > 1.0e-12) > 0
    assert float(np.max(displacement)) <= 0.005 + 1.0e-12
    assert len(bands) == 1
    assert not bands[0]["empty"]
    assert bands[0]["verts"] is not None
    assert bands[0]["faces"] is not None


def test_assembly_orders_skin_before_fixed_host_conduits(
    tmp_path: Path,
) -> None:
    graph = {
        "name": "skin-conduit-assembly",
        "blend": 0.0,
        "parts": [
            {
                "id": "host",
                "type": "gencyl",
                "spine": ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                "radii": (0.3, 0.3),
            }
        ],
        "skin": {
            "formation": "static_implicit_relaxation",
            "thickness": 0.04,
            "region_ids": ("host",),
        },
        "conduits": [
            {
                "id": "waist",
                "kind": "axial_loop",
                "part": "host",
                "t": 0.5,
                "width": 0.12,
                "depth": 0.005,
                "emit": ("groove", "band"),
            }
        ],
    }
    evaluated = engine.evaluate(graph, res=60)

    assert isinstance(evaluated, engine.EvaluatedMorphology)
    assert evaluated.skin_correspondence is not None
    assert evaluated.skin_evidence is not None
    entry = {
        "id": "body",
        "surface_detail": {
            "kind": "curvature",
            "amplitude": 0.001,
            "scale": 0.1,
        },
        "plate_policy": {
            "layout": "surface_voronoi",
            "cell_count": 8,
            "random_seed": 7,
            "relief": 0.001,
            "groove": 0.001,
            "seam_lift": 0.001,
            "seam_material": "emissive_seam",
        },
    }
    detail_result = _apply_surface_detail(
        entry,
        assembly.ElementRole.CREATURE,
        evaluated.vertices,
        evaluated.faces,
        evaluated.normals,
        evaluated.maximum_pitch,
    )
    assert not isinstance(
        detail_result,
        ElementSurfaceDetailObstruction,
    )
    detailed_vertices, detailed_normals, detail_report = detail_result
    plate_result = _apply_plate_policy(
        entry,
        graph,
        detailed_vertices,
        evaluated.faces,
        detailed_normals,
        None,
    )
    assert not isinstance(plate_result, PlateElementObstruction)
    plated_vertices, plate_surface, plate_report = plate_result
    assert plate_surface is not None
    assert plate_report is not None
    assert np.array_equal(
        plate_surface.source_vertices,
        detailed_vertices,
    )
    fixed_host_result = apply_conduits(
        graph,
        plated_vertices,
        evaluated.faces,
        graph["conduits"],
        pitch=evaluated.maximum_pitch,
        flesh_host_vertices=evaluated.skin_correspondence.flesh_vertices,
    )
    displaced_host_result = apply_conduits(
        graph,
        evaluated.vertices,
        evaluated.faces,
        graph["conduits"],
        pitch=evaluated.maximum_pitch,
    )

    assert not isinstance(fixed_host_result, EmissionPitchObstruction)
    assert not isinstance(displaced_host_result, EmissionPitchObstruction)
    fixed_vertices, fixed_bands = fixed_host_result
    displaced_vertices, displaced_bands = displaced_host_result
    assert not fixed_bands[0]["empty"]
    assert fixed_bands[0]["verts"] is not None
    assert fixed_bands[0]["faces"] is not None
    assert np.count_nonzero(
        np.linalg.norm(fixed_vertices - plated_vertices, axis=1)
        > 1.0e-12
    ) > 0
    assert displaced_bands[0]["empty"]
    assert np.array_equal(displaced_vertices, evaluated.vertices)

    assembly_result = assembly.compile_assembly(
        {
            "name": "skin-conduit-order",
            "elements": [
                {
                    "id": "body",
                    "role": "creature",
                    "graph": graph,
                    "surface_detail": entry["surface_detail"],
                    "plate_policy": entry["plate_policy"],
                }
            ],
        },
        tmp_path,
        resolution_policy=assembly.PinnedResolution(60),
    )

    assert isinstance(assembly_result, assembly.AcceptedAssembly)
    assert tuple(
        record.record_id for record in assembly_result.records
    ) == ("body", "body__waist", "body__plate_seams")
    solid, band, plate = assembly_result.records
    assert solid.stratum is assembly.AssemblyStratum.SOLID
    assert band.stratum is assembly.AssemblyStratum.CONDUIT
    assert plate.stratum is assembly.AssemblyStratum.PLATE_SEAM
    assert np.array_equal(solid.vertices, fixed_vertices)
    assert np.array_equal(band.vertices, fixed_bands[0]["verts"])
    assert np.array_equal(band.faces, fixed_bands[0]["faces"])
    assert solid.report["surface_detail"] == detail_report
    assert solid.report["plates"] == plate_report
    assert dict(solid.report["skin_formation"]) == asdict(
        evaluated.skin_evidence
    )


def test_formation_rejection_carries_unmasked_vocabulary_evidence(
    tmp_path: Path,
) -> None:
    spec = json.loads(_KNIGHT.read_text(encoding="utf-8"))
    region_ids = tuple(
        region["region_id"]
        for region in spec["anatomy"]["overall"]["regions"]
    )
    compiled = body.Compiler(
        _knight_with_formed_integument(region_ids),
        spec_dir=_KNIGHT.parent,
    ).compile()

    assert isinstance(compiled, body.CompiledBody)
    result = assembly.compile_assembly(
        {
            "name": "masked-evidence-probe",
            "elements": [
                {
                    "id": "body",
                    "role": "creature",
                    "graph": compiled.graph,
                }
            ],
        },
        tmp_path,
        resolution_policy=assembly.PinnedResolution(60),
    )

    assert isinstance(result, assembly.RejectedAssembly)
    assert len(result.obstructions) == 1
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, ElementSurfaceFormationObstruction)
    evaluated_graph = mount_graph(
        concretize_mirrors(_expand_appendages(compiled.graph)),
        None,
    )
    expected = tuple(engine.vocab_violations(evaluated_graph, res=60))

    assert obstruction.vocabulary_violations == expected
    assert expected
    assert obstruction.masked_integrity is not None
    assert obstruction.masked_integrity.checks == (
        "component_count",
        "watertight",
    )
    assert "not computable" in obstruction.masked_integrity.reason

    (projection,) = project_assembly_obstructions(obstruction)

    assert projection["vocabulary_violations"] == expected
    assert projection["masked_integrity"] == {
        "checks": ("component_count", "watertight"),
        "reason": obstruction.masked_integrity.reason,
    }
