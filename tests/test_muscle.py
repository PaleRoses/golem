from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError

from golem.assembly import AcceptedAssembly, PinnedResolution
from golem.cli import main
from golem.cli.compile import compile_spec
from golem.addressing.scope import ScopeKind
from golem.contract import render_section
from golem.contract.schema import body_schema
from golem.kernel import body, engine
from golem.kernel.body.types import MuscleBulkRule, MuscleEndpoint
from golem.kernel.engine.types import (
    Accepted,
    CompositionOperator,
    GencylPart,
    decode_part,
)


_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "specs" / "quadruped_muscle_demo.json"
_MINIMAL_FIXTURE = _ROOT / "specs" / "quadruped_minimal_muscle_demo.json"
_KNIGHT = _ROOT / "specs" / "knight_body.json"


def _fixture_spec() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _minimal_fixture_spec() -> dict:
    return json.loads(_MINIMAL_FIXTURE.read_text(encoding="utf-8"))


def _compile(spec: dict) -> body.BodyCompileResult:
    return body.Compiler(spec, spec_dir=_ROOT / "specs").compile()


def _replace_muscle(spec: dict, index: int, replacement: dict) -> dict:
    return {
        **spec,
        "muscles": tuple(
            replacement if candidate_index == index else muscle
            for candidate_index, muscle in enumerate(spec["muscles"])
        ),
    }


def test_schema_accepts_base_and_mirrored_muscle_declarations() -> None:
    schema = body_schema()
    Draft202012Validator(schema).validate(_fixture_spec())
    Draft202012Validator(schema).validate(_minimal_fixture_spec())
    variants = schema["$defs"]["muscle"]["oneOf"]
    assert tuple(
        tuple(variant["required"])
        for variant in variants
    ) == (
        ("kind", "id", "origin", "insertion", "sections"),
        ("kind", "id", "origin", "insertion", "bulk", "definition"),
        ("kind", "id", "origin", "insertion", "mirror_of"),
    )
    assert (
        variants[0]["properties"]["sections"]["items"]
        == schema["$defs"]["flesh"]["oneOf"][1]["properties"]["sections"]["items"]
        == {"$ref": "#/$defs/loftSection"}
    )
    assert all(
        tuple(variant["properties"]["operator"]["enum"])
        == tuple(operator.value for operator in CompositionOperator)
        for variant in variants
    )
    assert schema["$defs"]["muscleBulk"]["oneOf"] == [
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


def test_minimal_muscles_derive_fusiform_profiles_from_host_scale() -> None:
    result = _compile(_minimal_fixture_spec())
    assert isinstance(result, body.CompiledBody)
    rows = {row["id"]: row for row in result.receipt["myology"]}
    quadriceps = rows["left_quadriceps"]["sections"]
    hamstrings = rows["left_hamstrings"]["sections"]
    assert tuple(section["station"] for section in quadriceps) == pytest.approx(
        (0.0, 0.2632, 0.5, 0.7368, 1.0)
    )
    assert quadriceps[2]["width"] == pytest.approx(2.0 * 0.065 * 1.05)
    assert quadriceps[2]["depth"] == pytest.approx(2.0 * 0.065 * 0.7)
    assert hamstrings[2]["width"] == pytest.approx(0.72 * 0.16)
    assert hamstrings[2]["depth"] == pytest.approx(0.72 * 0.1)
    assert quadriceps[0]["width"] < quadriceps[1]["width"] < quadriceps[2]["width"]
    assert quadriceps[-1]["depth"] < quadriceps[-2]["depth"] < quadriceps[2]["depth"]


def test_absolute_minimal_bulk_uses_authored_belly_dimensions() -> None:
    spec = _minimal_fixture_spec()
    quadriceps = spec["muscles"][0]
    result = _compile(
        _replace_muscle(
            spec,
            0,
            {
                **quadriceps,
                "bulk": {"absolute": {"width": 0.14, "depth": 0.09}},
            },
        )
    )
    assert isinstance(result, body.CompiledBody)
    sections = result.receipt["myology"][0]["sections"]
    assert sections[2]["width"] == pytest.approx(0.14)
    assert sections[2]["depth"] == pytest.approx(0.09)


def test_relative_bulk_requires_a_positive_resolved_host_scale() -> None:
    spec = _minimal_fixture_spec()
    quadriceps = spec["muscles"][0]
    result = _compile(
        _replace_muscle(
            spec,
            0,
            {**quadriceps, "origin": "bone:pelvis"},
        )
    )
    assert isinstance(result, body.RejectedBody)
    obstruction = next(
        obstruction
        for obstruction in result.obstructions
        if isinstance(obstruction, body.UnresolvableMuscleBulkBasisObstruction)
    )
    assert obstruction.address == "muscles/0/bulk/relative/to"
    assert obstruction.selector == "bone:pelvis"
    assert obstruction.observed_value == 0.0


def test_relative_and_absolute_bulk_are_a_typed_schema_and_check_obstruction(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _minimal_fixture_spec()
    quadriceps = spec["muscles"][0]
    contradictory = _replace_muscle(
        spec,
        0,
        {
            **quadriceps,
            "bulk": {
                "relative": {"to": "origin", "width": 1.05, "depth": 0.7},
                "absolute": {"width": 0.14, "depth": 0.09},
            },
        },
    )
    with pytest.raises(ValidationError):
        Draft202012Validator(body_schema()).validate(contradictory)
    result = _compile(contradictory)
    assert isinstance(result, body.RejectedBody)
    obstruction = next(
        obstruction
        for obstruction in result.obstructions
        if isinstance(obstruction, body.MalformedMuscleBulkObstruction)
    )
    assert obstruction.address == "muscles/0/bulk"
    assert obstruction.rule is MuscleBulkRule.RELATIVE_XOR_ABSOLUTE
    path = tmp_path / "contradictory-muscle.json"
    path.write_text(json.dumps(contradictory), encoding="utf-8")
    assert main(("check", str(path))) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "MalformedMuscleBulkObstruction" in captured.out
    assert "relative_xor_absolute" in captured.out


def test_minimal_muscle_demo_compiles_to_accepted_assembly() -> None:
    result = compile_spec(_MINIMAL_FIXTURE, PinnedResolution(60))
    assert isinstance(result, AcceptedAssembly), getattr(result, "obstructions", ())


def test_primer_teaches_minimal_intent_before_profile_refinement() -> None:
    rendered = render_section("primer")
    myology = rendered.split("MYOLOGY", 1)[1].split("MOUNTS", 1)[0]
    assert myology.index('"bulk":{"relative"') < myology.index("Detailed profile")
    assert all(
        obstruction_name in rendered
        for obstruction_name in (
            "MalformedMuscleBulkObstruction",
            "UnresolvableMuscleBulkBasisObstruction",
            "LoftSectionObstruction",
            "AppearancePaletteObstruction",
            "MalformedEyeObstruction",
            "SurfaceDetailObstruction",
        )
    )
    assert all(
        heading in rendered
        for heading in ("LOFT", "OPERATORS", "PALETTE", "EYES", "FINISH")
    )


def test_quadruped_muscles_lower_to_profiled_union_parts() -> None:
    result = _compile(_fixture_spec())
    assert isinstance(result, body.CompiledBody)
    muscle_ids = tuple(row["id"] for row in result.receipt["myology"])
    assert muscle_ids == (
        "left_quadriceps",
        "right_quadriceps",
        "left_hamstrings",
        "left_gastrocnemius",
    )
    parts = {
        part["id"]: part
        for part in result.graph["parts"]
        if part["id"] in muscle_ids
    }
    assert tuple(parts) == muscle_ids
    assert all(
        part["type"] == "gencyl"
        and len(part["spine"]) == len(part["radii"])
        and len(part["profile"]["depth"]) == len(part["radii"])
        and max(part["radii"][1:-1]) > max(part["radii"][0], part["radii"][-1])
        and max(part["profile"]["depth"][1:-1])
        > max(part["profile"]["depth"][0], part["profile"]["depth"][-1])
        for part in parts.values()
    )
    hamstrings = parts["left_hamstrings"]
    decoded = decode_part(hamstrings)
    assert isinstance(decoded, Accepted)
    assert isinstance(decoded.value, GencylPart)
    assert decoded.value.operator is CompositionOperator.CREASE
    assert result.receipt["provenance"]["left_hamstrings"] == "muscles/2"


def test_legacy_explicit_profile_compilation_is_byte_identical() -> None:
    result = _compile(_fixture_spec())
    assert isinstance(result, body.CompiledBody)
    assert hashlib.sha256(body._canonical_json(result.graph).encode()).hexdigest() == (
        "4d07b882c7059028ac942ae453bd7c629a14758d672e67f3459e7fa4cf461aeb"
    )
    assert hashlib.sha256(body._canonical_json(result.receipt).encode()).hexdigest() == (
        "ec819ae1e5d11353b85f58b099b8c536b9ddad4c2edfce50fb5cd61d122ca1e2"
    )


def test_mirrored_muscle_is_the_exact_reflected_profile() -> None:
    result = _compile(_fixture_spec())
    assert isinstance(result, body.CompiledBody)
    parts = {part["id"]: part for part in result.graph["parts"]}
    left = parts["left_quadriceps"]
    right = parts["right_quadriceps"]
    reflection = np.asarray((-1.0, 1.0, 1.0))
    np.testing.assert_allclose(
        np.asarray(right["spine"]),
        np.asarray(left["spine"]) * reflection,
    )
    np.testing.assert_allclose(
        np.asarray(right["profile"]["up"]),
        np.asarray(left["profile"]["up"]) * reflection,
    )
    np.testing.assert_allclose(
        np.asarray(right["profile"]["roll"]),
        -np.asarray(left["profile"]["roll"]),
    )
    sample_points = np.asarray(
        ((0.18, 1.2, 0.02), (0.26, 0.96, 0.08), (0.12, 0.72, -0.03))
    )
    np.testing.assert_allclose(
        engine.part_sdf(sample_points * reflection, right),
        engine.part_sdf(sample_points, left),
        atol=1.0e-12,
    )


def test_mirror_true_skeleton_sites_anchor_an_explicit_muscle_partner() -> None:
    spec = {
        "name": "mirrored-muscle-sites",
        "dialect": "body/0.3",
        "emit_target": "v03",
        "skeleton": {
            "root": {
                "id": "pelvis",
                "world": [0.0, 1.0, 0.0],
                "flesh": [
                    {
                        "kind": "blob",
                        "name": "pelvis_mass",
                        "t": 0.0,
                        "size": [0.18, 0.12, 0.16],
                    }
                ],
            },
            "bones": [
                {
                    "id": "leg",
                    "parent": "pelvis",
                    "attach": {"t": 0.0, "offset": [0.22, 0.0, 0.0]},
                    "length": 0.8,
                    "rest_dir": [0.0, -1.0, 0.0],
                    "mirror": True,
                    "joint": {"dof": "fixed"},
                    "flesh": [
                        {
                            "kind": "gencyl",
                            "name": "leg_mass",
                            "span": [0.0, 1.0],
                            "radii": [0.05, 0.04],
                        }
                    ],
                }
            ],
        },
        "muscles": [
            {
                "kind": "muscle",
                "id": "leg_extensor",
                "origin": "part:leg_mass",
                "insertion": "landmark:leg/tail",
                "sections": [
                    {
                        "station": 0.0,
                        "width": 0.015,
                        "depth": 0.012,
                        "exponent": 2.2,
                        "roll": 0.0,
                    },
                    {
                        "station": 0.45,
                        "width": 0.07,
                        "depth": 0.05,
                        "exponent": 3.4,
                        "roll": 12.0,
                    },
                    {
                        "station": 1.0,
                        "width": 0.01,
                        "depth": 0.008,
                        "exponent": 2.2,
                        "roll": 0.0,
                    },
                ],
            },
            {
                "kind": "muscle",
                "id": "leg_extensor_m",
                "origin": "part:leg_mass_m",
                "insertion": "landmark:leg_m/tail",
                "mirror_of": "leg_extensor",
            },
        ],
    }
    result = _compile(spec)
    assert isinstance(result, body.CompiledBody)
    parts = {part["id"]: part for part in result.graph["parts"]}
    reflection = np.asarray((-1.0, 1.0, 1.0))
    np.testing.assert_allclose(
        np.asarray(parts["leg_extensor_m"]["spine"]),
        np.asarray(parts["leg_extensor"]["spine"]) * reflection,
    )


def test_unresolvable_and_degenerate_anchors_are_typed_obstructions() -> None:
    spec = _fixture_spec()
    hamstrings = spec["muscles"][2]
    unresolved = _compile(
        _replace_muscle(
            spec,
            2,
            {**hamstrings, "origin": "landmark:left_thigh/ghost"},
        )
    )
    assert isinstance(unresolved, body.RejectedBody)
    missing = next(
        obstruction
        for obstruction in unresolved.obstructions
        if isinstance(obstruction, body.UnresolvableMuscleAnchorObstruction)
    )
    assert missing.muscle_id == "left_hamstrings"
    assert missing.endpoint is MuscleEndpoint.ORIGIN
    assert missing.selector == "landmark:left_thigh/ghost"
    assert missing.address == "muscles/2/origin"
    degenerate = _compile(
        _replace_muscle(
            spec,
            2,
            {
                **hamstrings,
                "insertion": hamstrings["origin"],
            },
        )
    )
    assert isinstance(degenerate, body.RejectedBody)
    span = next(
        obstruction
        for obstruction in degenerate.obstructions
        if isinstance(obstruction, body.DegenerateMuscleSpanObstruction)
    )
    assert span.muscle_id == "left_hamstrings"
    assert span.observed_span == 0.0
    assert span.minimum_span > span.observed_span
    unsupported = _compile(
        _replace_muscle(
            spec,
            2,
            {**hamstrings, "origin": "world"},
        )
    )
    assert isinstance(unsupported, body.RejectedBody)
    scope = next(
        obstruction
        for obstruction in unsupported.obstructions
        if isinstance(obstruction, body.UnsupportedMuscleAnchorObstruction)
    )
    assert scope.scope_kind is ScopeKind.WORLD
    assert scope.supported_scope_kinds == (
        ScopeKind.BONE,
        ScopeKind.PART,
        ScopeKind.LANDMARK,
    )


def test_profile_and_mirror_invariants_are_typed_obstructions() -> None:
    spec = _fixture_spec()
    quadriceps = spec["muscles"][0]
    flat_sections = tuple(
        {
            **section,
            "width": 0.04,
            "depth": 0.03,
        }
        for section in quadriceps["sections"]
    )
    flat = _compile(
        _replace_muscle(
            spec,
            0,
            {**quadriceps, "sections": flat_sections},
        )
    )
    assert isinstance(flat, body.RejectedBody)
    profile = next(
        obstruction
        for obstruction in flat.obstructions
        if isinstance(obstruction, body.InvalidMuscleProfileObstruction)
    )
    assert profile.maximum_interior_width == profile.endpoint_widths[0]
    mirrored = spec["muscles"][1]
    mismatched = _compile(
        _replace_muscle(
            spec,
            1,
            {**mirrored, "insertion": "landmark:right_shin/tail"},
        )
    )
    assert isinstance(mismatched, body.RejectedBody)
    mismatch = next(
        obstruction
        for obstruction in mismatched.obstructions
        if isinstance(obstruction, body.MuscleMirrorAnchorMismatchObstruction)
    )
    assert mismatch.muscle_id == "right_quadriceps"
    assert mismatch.mirror_of == "left_quadriceps"
    assert mismatch.endpoint is MuscleEndpoint.INSERTION
    assert mismatch.observed_distance > mismatch.maximum_distance


def test_body_without_muscle_declarations_has_no_myology_view() -> None:
    result = _compile(json.loads(_KNIGHT.read_text(encoding="utf-8")))
    assert isinstance(result, body.CompiledBody)
    assert "myology" not in result.receipt
    assert all(
        not address.startswith("muscles/")
        for address in result.receipt["provenance"].values()
    )
