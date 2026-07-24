from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError
from PIL import Image

import golem.cli.check as check_cli
from golem.assembly import RES_MIN, AssemblyStratum, ElementRole
from golem.cli import main
from golem.cli.check import CHECK_DIALECTS
from golem.cli.compile import COMPILE_DIALECTS
from golem.cli.registry import COMMANDS, command_by_name
from golem.cli.spec import (
    UnrecognizedDocumentObstruction,
    classify_document,
    load_spec,
)
from golem.contract import SECTIONS, render_section
from golem.kernel.body.cli import main as body_main
from golem.kernel.engine import CompositionOperator
from golem.materials import MATERIAL_CATALOGUE
from golem.paths import SPECS
from golem.senses.proprio import build_senses
from golem.senses.proprio.anomaly import Anomaly, AnomalyKind


def _smallest_graph_payload() -> dict[str, object]:
    return {
        "name": "raw-probe",
        "blend": 0.02,
        "parts": [
            {
                "id": "core",
                "type": "blob",
                "center": [0.0, 0.0, 0.0],
                "size": [0.3, 0.4, 0.2],
            }
        ],
    }


def _body_contract_payload(
    clause: dict[str, object],
    *,
    joint: dict[str, object] | None = None,
    pose: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "name": "cli-verdict-probe",
        "dialect": "body/0.3",
        "emit_target": "v03",
        "blend": 0.05,
        "ground_y": 0.02,
        "skeleton": {
            "root": {"id": "root", "world": [0, 0, 0]},
            "bones": [
                {
                    "id": "b0",
                    "parent": "root",
                    "attach": {"t": 0},
                    "length": 1.0,
                    "rest_dir": [0, 1, 0],
                    "joint": {"dof": "ball"} if joint is None else joint,
                    "flesh": [
                        {
                            "kind": "gencyl",
                            "name": "b0",
                            "span": [0, 1],
                            "radii": [0.1, 0.1],
                        }
                    ],
                }
            ],
        },
        "contract": {
            "contract": "cli.verdict",
            "clauses": [clause],
        },
        **({"pose": pose} if pose is not None else {}),
    }


def _write_body_contract(
    tmp_path: Path,
    clause: dict[str, object],
    *,
    joint: dict[str, object] | None = None,
    pose: dict[str, object] | None = None,
) -> Path:
    path = tmp_path / f"{clause['id']}.json"
    path.write_text(
        json.dumps(
            _body_contract_payload(
                clause,
                joint=joint,
                pose=pose,
            )
        ),
        encoding="utf-8",
    )
    return path


def _png_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _subject_frame_fraction(path: Path) -> float:
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("RGB"), dtype=np.int16)
        body = np.max(np.abs(pixels - 244), axis=2) > 8
        occupied_rows, occupied_columns = np.nonzero(body)
        assert occupied_rows.size > 0
        width = int(np.ptp(occupied_columns)) + 1
        height = int(np.ptp(occupied_rows)) + 1
        return max(width, height) / max(source.size)


def test_command_registry_dispatch_is_total() -> None:
    assert tuple(command.name for command in COMMANDS) == (
        "check",
        "compile",
        "contract",
        "eval",
        "forge",
        "look",
        "pose-prior",
        "session",
    )
    assert all(command_by_name(command.name) is command for command in COMMANDS)
    assert command_by_name("absent") is None


def test_check_knight_body_is_mesh_free_and_structural(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(("check", str(SPECS / "knight_body.json")))
    captured = capsys.readouterr()
    # The structural check remains mesh-free, but Wall 006 now rejects its
    # 25 undeclared anomalies instead of letting warning severity ride.
    assert exit_code == 1
    assert captured.err == ""
    assert "PROPRIO " in captured.out
    assert "GLOBAL  bbox" in captured.out
    assert "ASSERT  " in captured.out
    assert "ANOMALY " in captured.out
    assert "no mesh" in captured.out


def test_check_accepts_a_raw_part_graph(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "raw_graph.json"
    path.write_text(
        json.dumps(_smallest_graph_payload()),
        encoding="utf-8",
    )
    exit_code = main(("check", str(path)))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "PROPRIO raw-probe" in captured.out
    assert "no mesh" in captured.out


def test_check_help_exposes_complete_receipt_rendering(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(("check", "--help"))
    captured = capsys.readouterr()

    assert stopped.value.code == 0
    assert captured.err == ""
    assert "--all" in captured.out
    assert "all failed/unmeasurable assertions" in captured.out
    assert "unexpected anomalies" in captured.out
    assert "suggested carrier scales" in captured.out


def test_check_all_removes_every_proprio_presentation_cap() -> None:
    base_senses = build_senses(_smallest_graph_payload(), {})
    carrier_rows = tuple(
        {
            "controlling_constraint": "circulation_clearance",
            "interface_address": f"skeleton/bone-{index}/joint",
            "measured_minimum_radius": 0.01,
            "target_minimum_radius": 0.04,
            "shape_address": f"skeleton/bone-{index}/flesh[0]",
            "suggested_scale": 4.0,
        }
        for index in range(30)
    )
    senses = replace(
        base_senses,
        anatomy={
            "status": "accepted",
            "overall": {"pump": {"host_bone_id": "root"}},
            "circuits": (),
            "carrier_rows": carrier_rows,
            "terminal_coverage": 1.0,
            "authoring_leverage": 1.0,
        },
    )
    assertion_records = tuple(
        {
            "id": f"assert-{index}",
            "status": "fail",
            "human": f"assertion {index}",
            "part_scopes": (),
        }
        for index in range(26)
    )
    anomalies = tuple(
        Anomaly(
            AnomalyKind.blend_ambiguity,
            (f"left-{index}", f"right-{index}"),
            0.01,
            0.01,
            f"anomaly {index}",
            f"left-{index}~right-{index}",
        )
        for index in range(44)
    )

    capped = check_cli._render_check_receipt(
        senses,
        assertion_records,
        anomalies,
        pack="cap-probe",
        all_records=False,
    )
    complete = check_cli._render_check_receipt(
        senses,
        assertion_records,
        anomalies,
        pack="cap-probe",
        all_records=True,
    )

    assert "+2 more -- proprio(asserts)" in capped
    assert "+32 more -- proprio(anomalies)" in capped
    assert "skeleton/bone-29/flesh[0]" not in capped
    assert "more -- proprio" not in complete
    assert "assertion 25" in complete
    assert "ANOMALY 44" in complete
    assert "N44 blend_ambiguity" in complete
    assert "skeleton/bone-29/flesh[0]" in complete
    assert complete.count("suggest scale skeleton/bone-") == 30


def test_failing_assert_exits_one_from_check_and_body_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_body_contract(
        tmp_path,
        {
            "id": "cli_failure",
            "scope": ["part:b0"],
            "metric": "height",
            "operator": "minimum",
            "value": 5.0,
            "unit": "world_unit",
            "tolerance": 0,
        },
    )

    check_exit_code = main(("check", str(path)))
    check_output = capsys.readouterr()
    body_exit_code = body_main([str(path)])
    body_output = capsys.readouterr()

    assert (check_exit_code, body_exit_code) == (1, 1)
    assert (check_output.err, body_output.err) == ("", "")
    assert "cli_failure height @ b0" in check_output.out
    assert "cli_failure height @ b0" in body_output.out


def test_unmeasurable_assert_exits_one_with_a_distinct_rendered_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_body_contract(
        tmp_path,
        {
            "id": "cli_unmeasurable",
            "scope": ["whole"],
            "metric": "vibe",
            "operator": "maximum",
            "value": 1.0,
        },
    )

    check_exit_code = main(("check", str(path)))
    check_output = capsys.readouterr()
    body_exit_code = body_main([str(path)])
    body_output = capsys.readouterr()
    rendered_line = (
        "  cli_unmeasurable vibe: "
        "UNMEASURABLE (unknown metric vibe)\n"
    )

    assert (check_exit_code, body_exit_code) == (1, 1)
    assert (check_output.err, body_output.err) == ("", "")
    assert rendered_line in check_output.out
    assert rendered_line in body_output.out


def test_passing_assert_exits_zero_from_check_and_body_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_body_contract(
        tmp_path,
        {
            "id": "cli_pass",
            "scope": ["part:b0"],
            "metric": "height",
            "operator": "minimum",
            "value": 0.0,
            "unit": "world_unit",
            "tolerance": 0,
        },
    )

    check_exit_code = main(("check", str(path)))
    check_output = capsys.readouterr()
    body_exit_code = body_main([str(path)])
    body_output = capsys.readouterr()

    assert (check_exit_code, body_exit_code) == (0, 0)
    assert (check_output.err, body_output.err) == ("", "")
    assert "0 FAIL / 1 pass" in check_output.out
    assert "0 FAIL / 1 pass" in body_output.out


def test_check_reports_nonfatal_geometry_obstructions_without_rejecting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _smallest_graph_payload()
    path = tmp_path / "nonunit-rotation.json"
    path.write_text(
        json.dumps(
            {
                **payload,
                "parts": [
                    {
                        "id": "core",
                        "type": "blob",
                        "center": [0.0, 0.0, 0.0],
                        "size": [0.3, 0.4, 0.2],
                        "rot": [2.0, 0.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(("check", str(path)))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "OBSTRUCTIONS 1" in captured.out
    assert "[R-quat-nonunit]" in captured.out


def test_check_rejects_fatal_geometry_obstructions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "missing-parts.json"
    path.write_text(
        json.dumps({"name": "fatal-probe", "parts": []}),
        encoding="utf-8",
    )

    exit_code = main(("check", str(path)))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err == ""
    assert "OBSTRUCTIONS 1" in captured.out
    assert "[R-malformed-graph]" in captured.out


@pytest.mark.parametrize(
    ("joint", "pose", "check_line", "body_line"),
    (
        (
            {"dof": "slider"},
            None,
            "[bad_dof]",
            "VIOLATION [bad_dof]",
        ),
        (
            {"dof": "ball", "limits": {"pitch": [-10, 10]}},
            {"joints": {"b0": {"pitch": 40}}},
            "[joint_limit]",
            "limit_violations 1",
        ),
    ),
)
def test_body_and_check_exit_one_for_body_or_joint_limit_obstructions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    joint: dict[str, object],
    pose: dict[str, object] | None,
    check_line: str,
    body_line: str,
) -> None:
    path = _write_body_contract(
        tmp_path,
        {
            "id": "cli_pass_with_obstruction",
            "scope": ["part:b0"],
            "metric": "height",
            "operator": "minimum",
            "value": 0.0,
            "unit": "world_unit",
            "tolerance": 0,
        },
        joint=joint,
        pose=pose,
    )

    check_exit_code = main(("check", str(path)))
    check_output = capsys.readouterr()
    body_exit_code = body_main([str(path)])
    body_output = capsys.readouterr()

    assert (check_exit_code, body_exit_code) == (1, 1)
    assert (check_output.err, body_output.err) == ("", "")
    assert "OBSTRUCTIONS 1" in check_output.out
    assert check_line in check_output.out
    assert body_line in body_output.out


def test_look_writes_reopenable_pngs_at_the_lowest_lawful_resolution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "smallest.json"
    output_directory = tmp_path / "views"
    spec_path.write_text(
        json.dumps(_smallest_graph_payload()),
        encoding="utf-8",
    )
    exit_code = main(
        (
            "look",
            str(spec_path),
            "--res",
            str(RES_MIN),
            "--out",
            str(output_directory),
        )
    )
    captured = capsys.readouterr()
    paths = tuple(map(Path, filter(None, captured.out.splitlines())))
    assert exit_code == 0
    assert captured.err == ""
    assert tuple(path.name for path in paths) == (
        "smallest_front.png",
        "smallest_side.png",
        "smallest_top.png",
    )
    assert all(path.suffix == ".png" and path.is_file() for path in paths)
    assert all(
        width > 0 and height > 0
        for width, height in map(_png_size, paths)
    )


def test_look_quality_options_preserve_omitted_defaults_and_fit_explicit_size(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "smallest.json"
    default_directory = tmp_path / "default"
    explicit_default_directory = tmp_path / "explicit-default"
    resized_directory = tmp_path / "resized"
    shaded_directory = tmp_path / "shaded"
    spec_path.write_text(
        json.dumps(_smallest_graph_payload()),
        encoding="utf-8",
    )
    default_exit_code = main(
        (
            "look",
            str(spec_path),
            "--view",
            "front",
            "--res",
            str(RES_MIN),
            "--out",
            str(default_directory),
        )
    )
    default_capture = capsys.readouterr()
    explicit_default_exit_code = main(
        (
            "look",
            str(spec_path),
            "--view",
            "front",
            "--res",
            str(RES_MIN),
            "--size",
            "256",
            "--out",
            str(explicit_default_directory),
        )
    )
    explicit_default_capture = capsys.readouterr()
    resized_exit_code = main(
        (
            "look",
            str(spec_path),
            "--view",
            "front",
            "--res",
            str(RES_MIN),
            "--size",
            "96",
            "--out",
            str(resized_directory),
        )
    )
    resized_capture = capsys.readouterr()
    shaded_exit_code = main(
        (
            "look",
            str(spec_path),
            "--view",
            "front",
            "--res",
            str(RES_MIN),
            "--shaded",
            "--out",
            str(shaded_directory),
        )
    )
    shaded_capture = capsys.readouterr()
    default_path = default_directory / "smallest_front.png"
    explicit_default_path = explicit_default_directory / "smallest_front.png"
    resized_path = resized_directory / "smallest_front.png"
    shaded_path = shaded_directory / "smallest_front.png"
    assert (
        default_exit_code,
        explicit_default_exit_code,
        resized_exit_code,
        shaded_exit_code,
    ) == (0, 0, 0, 0)
    assert (
        default_capture.err,
        explicit_default_capture.err,
        resized_capture.err,
        shaded_capture.err,
    ) == ("", "", "", "")
    assert default_path.read_bytes() != explicit_default_path.read_bytes()
    assert 0.70 <= _subject_frame_fraction(explicit_default_path) <= 0.75
    assert _png_size(resized_path) == (96, 96)
    assert 0.70 <= _subject_frame_fraction(resized_path) <= 0.75
    assert default_path.read_bytes() != shaded_path.read_bytes()


def test_look_rejects_a_nonpositive_canvas_size_as_a_typed_obstruction(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "smallest.json"
    spec_path.write_text(
        json.dumps(_smallest_graph_payload()),
        encoding="utf-8",
    )
    exit_code = main(
        (
            "look",
            str(spec_path),
            "--view",
            "front",
            "--res",
            str(RES_MIN),
            "--size",
            "0",
            "--out",
            str(tmp_path / "rejected"),
        )
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "OrthographicRenderObstruction" in captured.err
    assert "render/image_size" in captured.err


def test_eval_runs_the_applicable_knight_body_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(("eval", str(SPECS / "knight_body.json")))
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert report["status"] == "accepted"
    assert "anatomy_guided_construction_score" in report


def test_compile_rejects_a_hand_subgrammar_with_a_typed_obstruction(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = SPECS / "hand_v2.json"
    source = load_spec(spec_path)
    obstruction = classify_document(source, COMPILE_DIALECTS)
    assert isinstance(obstruction, UnrecognizedDocumentObstruction)
    assert obstruction.path == spec_path
    assert obstruction.accepted_dialects == COMPILE_DIALECTS

    exit_code = main(("compile", str(spec_path), "--res", "60"))
    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "KeyError" not in captured.err
    assert "UnrecognizedDocument" in captured.err


def test_check_rejects_a_hand_subgrammar_with_a_typed_obstruction(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = SPECS / "hand_v2.json"
    source = load_spec(spec_path)
    obstruction = classify_document(source, CHECK_DIALECTS)
    assert isinstance(obstruction, UnrecognizedDocumentObstruction)
    assert obstruction.path == spec_path
    assert obstruction.accepted_dialects == CHECK_DIALECTS

    exit_code = main(("check", str(spec_path)))
    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "UnrecognizedDocument" in captured.err


def test_contract_without_section_lists_the_registry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(("contract",))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert all(section.name in captured.out for section in SECTIONS)


def test_contract_sections_are_total_and_nonempty() -> None:
    rendered = tuple(render_section(section.name) for section in SECTIONS)
    assert len(rendered) == len(SECTIONS)
    assert all(text.strip() for text in rendered)


def test_vocabulary_section_projects_every_live_kind() -> None:
    payload = json.loads((SPECS / "vocabulary.json").read_text(encoding="utf-8"))
    kind_names = tuple(entry["kind_name"] for entry in payload["vocab"])
    rendered = render_section("vocabulary")
    assert all(kind_name in rendered for kind_name in kind_names)


def test_schema_is_valid_and_closed_enums_are_live() -> None:
    schema = json.loads(render_section("schema"))
    Draft202012Validator.check_schema(schema)
    body = json.loads((SPECS / "knight_body.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(body)
    Draft202012Validator(schema).validate(
        {
            **body,
            "surface_color": {
                "tint_linear_rgb": [0.62, 0.22, 0.1],
                "variation": {
                    "kind": "triplanar_value_noise",
                    "wavelength_world": 0.55,
                    "amplitude": 0.12,
                },
            },
        }
    )
    Draft202012Validator(schema).validate(
        {
            **body,
            "appearance_material": "obsidian_warden",
            "appearance_palette": {
                "obsidian_warden": {"tint_linear_rgb": [0.035, 0.03, 0.05]},
                "vascular_supply_gold": {"tint_linear_rgb": [0.9, 0.25, 0.08]},
                "vascular_return_violet": {"tint_linear_rgb": [0.22, 0.08, 0.65]},
                "vascular_exchange_cyan": {"tint_linear_rgb": [0.1, 0.75, 0.82]},
            },
        }
    )
    definitions = schema["$defs"]
    assert tuple(definitions["elementRole"]["enum"]) == tuple(
        role.value for role in ElementRole
    )
    assert tuple(definitions["appearanceMaterial"]["enum"]) == tuple(
        material.material_id.value
        for material in MATERIAL_CATALOGUE.appearance_materials
    )
    assert len(definitions["appearanceMaterial"]["enum"]) == 10
    assert tuple(definitions["assemblyStratum"]["enum"]) == tuple(
        stratum.value for stratum in AssemblyStratum
    )
    assert definitions["appearancePalette"]["propertyNames"] == {
        "$ref": "#/$defs/appearanceMaterial"
    }
    palette_description = definitions["appearancePalette"]["description"]
    assert "`look` orthographic rendering honor" in palette_description
    assert "linear" in palette_description
    assert "exactly once" in palette_description
    assert "display sRGB" in palette_description


def test_schema_accepts_exact_loft_sections_and_live_union_operators() -> None:
    schema = json.loads(render_section("schema"))
    validator = Draft202012Validator(schema)
    sections = [
        {
            "station": 0.0,
            "width": 0.3,
            "depth": 0.18,
            "exponent": 6.0,
            "roll": 35.0,
        },
        {
            "station": 1.0,
            "width": 0.14,
            "depth": 0.08,
            "exponent": 4.0,
            "roll": 0.0,
        },
    ]
    loft = {
        "kind": "loft",
        "name": "keeled",
        "operator": "crease",
        "sections": sections,
    }
    document = {
        "name": "loft-schema-probe",
        "dialect": "body/0.3",
        "skeleton": {
            "root": {"id": "root", "world": [0.0, 0.0, 0.0]},
            "bones": [
                {
                    "id": "chest",
                    "parent": "root",
                    "length": 1.0,
                    "rest_dir": [0.0, 1.0, 0.0],
                    "joint": {"dof": "fixed"},
                    "flesh": [loft],
                }
            ],
        },
    }
    validator.validate(document)
    conflicting_profile_document = {
        **document,
        "skeleton": {
            **document["skeleton"],
            "bones": [
                {
                    **document["skeleton"]["bones"][0],
                    "flesh": [
                        {
                            "kind": "gencyl",
                            "radii": [0.3, 0.2],
                            "profile": {
                                "aspect": [0.6, 0.5],
                                "depth": [0.18, 0.1],
                            },
                        }
                    ],
                }
            ],
        },
    }
    with pytest.raises(ValidationError):
        validator.validate(conflicting_profile_document)
    flesh_branches = schema["$defs"]["flesh"]["oneOf"]
    assert all(
        tuple(branch["properties"]["operator"]["enum"])
        == tuple(operator.value for operator in CompositionOperator)
        for branch in flesh_branches
    )
    assert all(
        branch["properties"]["blend"]["minimum"] == 0
        and "per-part" in branch["properties"]["blend"]["description"]
        and "incoming" in branch["properties"]["operator"]["description"]
        for branch in flesh_branches
    )
    negative_blend = {
        **loft,
        "blend": -0.01,
    }
    negative_blend_document = {
        **document,
        "skeleton": {
            **document["skeleton"],
            "bones": [
                {
                    **document["skeleton"]["bones"][0],
                    "flesh": [negative_blend],
                }
            ],
        },
    }
    with pytest.raises(ValidationError):
        validator.validate(negative_blend_document)
    invalid_section = {**sections[0], "ornamental_guess": 1.0}
    invalid_loft = {**loft, "sections": [invalid_section, sections[1]]}
    invalid_document = {
        **document,
        "skeleton": {
            **document["skeleton"],
            "bones": [
                {
                    **document["skeleton"]["bones"][0],
                    "flesh": [invalid_loft],
                }
            ],
        },
    }
    with pytest.raises(ValidationError):
        validator.validate(invalid_document)


def test_contract_teaches_the_current_hidden_authoring_laws() -> None:
    schema = json.loads(render_section("schema"))
    primer = render_section("primer")
    envelope = render_section("envelope")
    receipts = render_section("receipts")
    flesh_branches = schema["$defs"]["flesh"]["oneOf"]
    blob = next(
        branch
        for branch in flesh_branches
        if branch["properties"]["kind"].get("const") == "blob"
    )
    bone = schema["$defs"]["bone"]["properties"]
    anatomy = schema["$defs"]["anatomy"]
    circulation = anatomy["properties"]["overall"]["properties"]["circulation"]

    assert "Half-extents" in blob["properties"]["size"]["description"]
    assert all(
        "Half-extent" in schema["$defs"]["loftSection"]["properties"][axis]["description"]
        for axis in ("width", "depth")
    )
    assert "PARENT" in bone["rest_dir"]["description"]
    assert "-z local inverts" in bone["rest_dir"]["description"]
    assert all(
        token in bone["mirror"]["description"]
        for token in ("_m", ":positive:", ":negative:")
    )
    assert bone["role"]["enum"] == ["handle"]
    assert "frame composition is unchanged" in bone["role"]["description"]
    assert all(
        branch["properties"]["role"]["enum"] == ["non_carrier"]
        for branch in flesh_branches
    )
    assert "may leave the bone centerline" in circulation["description"]
    assert "AERIAL COVER" in envelope
    assert "blend | chamfer | crease | local_blend" in primer
    assert "N suggested carrier scales" in receipts
    assert "witness.expected" in receipts
    assert "authored circuit lane `:positive:`" in receipts


def test_compile_receipt_renders_vocabulary_violation_repairs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(("compile", str(SPECS / "knight_body.json"), "--res", "60"))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    advisory_lines = tuple(
        line
        for line in captured.out.splitlines()
        if line.lstrip().startswith("advisory [")
    )
    assert advisory_lines
    assert (
        f"advisories vocabulary_violations={len(advisory_lines)}"
        in captured.out
    )
    assert any(
        "[R-8-feature-size]" in line and " -> " in line
        for line in advisory_lines
    )


def test_check_knight_reports_accepted_vascular_feasibility(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(("check", str(SPECS / "knight_body.json")))
    captured = capsys.readouterr()
    # Vascular feasibility remains accepted; the command rejects only because
    # Wall 006 now treats the knight's 25 undeclared anomalies as blocking.
    assert exit_code == 1
    assert captured.err == ""
    assert "VASCULAR feasibility: ACCEPTED" in captured.out
    assert "  vascular nodes=" in captured.out
    assert "  allocation " in captured.out


def test_check_distinguishes_closed_intent_from_infeasible_vasculature(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_path = (
        Path(__file__).parent / "fixtures" / "quadruped_trial_body.json"
    )
    exit_code = main(("check", str(fixture_path)))
    captured = capsys.readouterr()
    circulation_lines = tuple(
        line
        for line in captured.out.splitlines()
        if line.startswith("CIRCULATION ")
    )
    _, vascular_heading, vascular_block = captured.out.partition(
        "VASCULAR feasibility: REJECTED"
    )
    rejection_lines = tuple(
        line
        for line in vascular_block.splitlines()
        if line.startswith("REJECTED [VascularIntersectionObstruction]")
    )
    rejection_payloads = tuple(
        json.loads(line.partition("] ")[2])
        for line in rejection_lines
    )
    assert exit_code != 0
    assert captured.err == ""
    assert len(circulation_lines) == 1
    assert "3/3 closed" in circulation_lines[0]
    assert "coverage 1.000" in circulation_lines[0]
    assert vascular_heading
    assert rejection_payloads
    assert all(
        set(payload)
        == {"left_edge_id", "right_edge_id", "clearance", "failing_segments"}
        and float(payload["clearance"]) < 0.0
        for payload in rejection_payloads
    )
    assert all(
        {segment["edge_id"] for segment in payload["failing_segments"]}
        == {payload["left_edge_id"], payload["right_edge_id"]}
        and all(
            {"source_position", "target_position"} <= set(segment)
            for segment in payload["failing_segments"]
        )
        for payload in rejection_payloads
    )
    assert {
        str(payload["left_edge_id"]).partition(":")[0]
        for payload in rejection_payloads
    } == {"return", "supply"}


def _sibling_overlap_body_payload(
    contract: dict[str, object] | None,
) -> dict[str, object]:
    def bone(bone_id: str, flesh_name: str, offset: float) -> dict[str, object]:
        return {
            "id": bone_id,
            "parent": "root",
            "attach": {"t": 0, "offset": [offset, 0.0, 0.0]},
            "length": 0.4,
            "rest_dir": [0, 1, 0],
            "joint": {"dof": "fixed"},
            "flesh": [
                {
                    "kind": "blob",
                    "name": flesh_name,
                    "t": 0.5,
                    "size": [0.3, 0.3, 0.3],
                }
            ],
        }

    return {
        "name": "contract-attach-probe",
        "dialect": "body/0.3",
        "emit_target": "v03",
        "blend": 0.02,
        "ground_y": -1.0,
        "skeleton": {
            "root": {"id": "root", "world": [0, 0, 0]},
            "bones": [bone("ba", "mass_a", -0.1), bone("bb", "mass_b", 0.1)],
        },
        **({"contract": contract} if contract is not None else {}),
    }


def test_contract_attach_declares_intentional_fusion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    undeclared = tmp_path / "undeclared.json"
    undeclared.write_text(
        json.dumps(_sibling_overlap_body_payload(None)), encoding="utf-8"
    )
    declared = tmp_path / "declared.json"
    declared.write_text(
        json.dumps(
            _sibling_overlap_body_payload(
                {
                    "contract": "body.contract",
                    "clauses": [],
                    "attach": [["mass_a", "mass_b"]],
                }
            )
        ),
        encoding="utf-8",
    )

    undeclared_code = main(("check", str(undeclared)))
    undeclared_out = capsys.readouterr()
    declared_code = main(("check", str(declared)))
    declared_out = capsys.readouterr()

    assert "unintended_fusion @ mass_a~mass_b" in undeclared_out.out
    assert (declared_code, declared_out.err) == (0, "")
    assert "ANOMALY 0" in declared_out.out
    assert "unintended_fusion" not in declared_out.out


def test_contract_attach_rejects_unknown_part_ids(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "unknown-attach.json"
    path.write_text(
        json.dumps(
            _sibling_overlap_body_payload(
                {
                    "contract": "body.contract",
                    "clauses": [],
                    "attach": [["mass_a", "mass_missing"]],
                }
            )
        ),
        encoding="utf-8",
    )

    exit_code = main(("check", str(path)))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err == ""
    assert "[bad_contract_attach]" in captured.out
    assert "mass_missing" in captured.out


def _midline_crossing_body_payload(
    contract: dict[str, object] | None,
) -> dict[str, object]:
    """Mirrored flesh blob straddling x=0: the .L/.R instances overlap across
    the plane, which proprio reads as cross_plane_fusion unless the part is
    declared on the midline channel."""
    return {
        "name": "contract-midline-probe",
        "dialect": "body/0.3",
        "emit_target": "v03",
        "blend": 0.02,
        "ground_y": -1.0,
        "skeleton": {
            "root": {"id": "root", "world": [0, 0, 0]},
            "bones": [
                {
                    "id": "haunch",
                    "parent": "root",
                    "attach": {"t": 0, "offset": [0.05, 0.0, 0.0]},
                    "length": 0.4,
                    "rest_dir": [0, 1, 0],
                    "joint": {"dof": "fixed"},
                    "mirror": True,
                    "flesh": [
                        {
                            "kind": "blob",
                            "name": "haunch_cap",
                            "t": 0.5,
                            "size": [0.3, 0.3, 0.3],
                        }
                    ],
                }
            ],
        },
        **({"contract": contract} if contract is not None else {}),
    }


def test_contract_midline_declares_intentional_plane_crossing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    undeclared = tmp_path / "undeclared.json"
    undeclared.write_text(
        json.dumps(_midline_crossing_body_payload(None)), encoding="utf-8"
    )
    declared = tmp_path / "declared.json"
    declared.write_text(
        json.dumps(
            _midline_crossing_body_payload(
                {
                    "contract": "body.contract",
                    "clauses": [],
                    "midline": ["haunch_cap"],
                }
            )
        ),
        encoding="utf-8",
    )

    undeclared_code = main(("check", str(undeclared)))
    undeclared_out = capsys.readouterr()
    declared_code = main(("check", str(declared)))
    declared_out = capsys.readouterr()

    # honest negative: the undeclared crossing stays a visible anomaly.
    assert "cross_plane_fusion @ haunch_cap.L~haunch_cap.R" in undeclared_out.out
    assert (declared_code, declared_out.err) == (0, "")
    assert "ANOMALY 0" in declared_out.out
    assert "cross_plane_fusion" not in declared_out.out


def test_contract_midline_rejects_unknown_part_ids(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "unknown-midline.json"
    path.write_text(
        json.dumps(
            _midline_crossing_body_payload(
                {
                    "contract": "body.contract",
                    "clauses": [],
                    "midline": ["haunch_missing"],
                }
            )
        ),
        encoding="utf-8",
    )

    exit_code = main(("check", str(path)))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err == ""
    assert "[bad_contract_midline]" in captured.out
    assert "haunch_missing" in captured.out
