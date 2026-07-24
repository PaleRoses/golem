"""Typed body-relation rejection at every production consumer boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from golem.assembly import RejectedAssembly, compile_assembly
from golem.assembly.__main__ import _assembly_obstruction_lines
from golem.assembly.obstructions import ElementBodyObstruction
from golem.cli import main as cli_main
from golem.evals import eval_anatomy, eval_plate_authoring
from golem.evals.harness import EvaluationObstruction
from golem.kernel.body import RejectedBody, project_rejected_body, rejected_body_text
from golem.kernel.body.cli import main as body_main
from golem.kernel.body.relations import (
    BodyRelationObstruction,
    FixedPlacementConflictObstruction,
)
from golem.senses import silhouette
from golem.session.effect import compile_authored
from golem.session.state import AuthoredState, BodyCompileObstruction, CompileObstructed


def _rejected_body_spec() -> dict:
    return {
        "dialect": "body/0.3",
        "name": "boundary-rejection",
        "skeleton": {
            "root": {"id": "root", "world": [0, 0, 0]},
            "bones": [
                {
                    "id": "chest",
                    "parent": "root",
                    "attach": {"t": 0},
                    "length": 0.6,
                    "rest_dir": [0, 1, 0],
                    "flesh": [
                        {
                            "kind": "blob",
                            "name": "chest",
                            "size": [0.2, 0.2, 0.2],
                            "t": 0.5,
                        }
                    ],
                },
                {
                    "id": "foot",
                    "parent": "root",
                    "attach": {"t": 0},
                    "length": 0.1,
                    "rest_dir": [0, 1, 0],
                    "flesh": [
                        {
                            "kind": "blob",
                            "name": "footf",
                            "size": [0.05, 0.05, 0.05],
                            "t": 0.5,
                        }
                    ],
                },
            ],
        },
        "pose": {
            "relations": [
                {
                    "id": "fc",
                    "kind": "above",
                    "subject": "part:footf",
                    "reference": "part:chest",
                    "distance": 0.05,
                    "solve": "subject",
                }
            ]
        },
    }


def _write_rejected_body(tmp_path: Path) -> Path:
    path = tmp_path / "rejected-body.json"
    path.write_text(json.dumps(_rejected_body_spec()), encoding="utf-8")
    return path


def _fixed_conflict(
    obstructions: tuple[BodyRelationObstruction, ...],
) -> FixedPlacementConflictObstruction:
    assert len(obstructions) == 1
    obstruction = obstructions[0]
    assert isinstance(obstruction, FixedPlacementConflictObstruction)
    assert obstruction.required == 0.05
    assert obstruction.observed == -0.5
    return obstruction


def _canonical_rejection(
    obstructions: tuple[BodyRelationObstruction, ...],
) -> str:
    return json.dumps(
        project_rejected_body(RejectedBody(obstructions)),
        separators=(",", ":"),
        sort_keys=True,
    )


def test_check_renders_typed_body_rejection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_rejected_body(tmp_path)

    exit_code = cli_main(("check", str(path)))
    captured = capsys.readouterr()

    assert exit_code != 0
    assert captured.err == ""
    assert "[FixedPlacementConflictObstruction]" in captured.out
    assert "required 0.05, observed -0.5" in captured.out
    assert "Traceback" not in captured.out


def test_body_cli_rejection_writes_no_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_rejected_body(tmp_path)
    graph_path = tmp_path / "graph.json"
    receipt_path = tmp_path / "receipt.json"

    exit_code = body_main(
        [str(path), "--out", str(graph_path), "--receipt", str(receipt_path)]
    )
    captured = capsys.readouterr()

    assert exit_code != 0
    assert captured.out == ""
    assert "required 0.05, observed -0.5" in captured.err
    assert not graph_path.exists()
    assert not receipt_path.exists()


def test_assembly_descent_and_renderer_preserve_nested_body_evidence(
    tmp_path: Path,
) -> None:
    path = _write_rejected_body(tmp_path)
    result = compile_assembly(
        {
            "name": "boundary-assembly",
            "pitch": 0.1,
            "elements": [
                {
                    "id": "creature",
                    "role": "creature",
                    "body": path.name,
                }
            ],
        },
        tmp_path,
    )

    assert isinstance(result, RejectedAssembly)
    assert len(result.obstructions) == 1
    element_obstruction = result.obstructions[0]
    assert isinstance(element_obstruction, ElementBodyObstruction)
    assert element_obstruction.element_id == "creature"
    obstruction = _fixed_conflict(element_obstruction.obstructions)
    (line,) = _assembly_obstruction_lines(result)
    assert "[ElementBodyObstruction]" in line
    projected = json.loads(line.split("] ", 1)[1])
    assert projected["element_id"] == "creature"
    assert projected["obstructions"] == [
        {
            "observed": obstruction.observed,
            "reference_address": obstruction.reference_address,
            "relation_id": obstruction.relation_id,
            "required": obstruction.required,
            "subject_address": obstruction.subject_address,
        }
    ]


def test_session_retains_body_obstruction_and_derives_human_text(
    tmp_path: Path,
) -> None:
    outcome = compile_authored(AuthoredState(_rejected_body_spec(), tmp_path))

    assert isinstance(outcome, CompileObstructed)
    assert len(outcome.obstructions) == 1
    body_obstruction = outcome.obstructions[0]
    assert isinstance(body_obstruction, BodyCompileObstruction)
    _fixed_conflict(body_obstruction.obstructions)
    rejected = RejectedBody(body_obstruction.obstructions)
    assert outcome.error == rejected_body_text(rejected)


def test_silhouette_preserves_body_obstruction_through_main(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_rejected_body(tmp_path)
    missing_reference = tmp_path / "missing.glb"

    loaded = silhouette.load_geometry(path, mesh_resolution=60)
    assert isinstance(loaded, silhouette.BodyGeometryLoadObstruction)
    _fixed_conflict(loaded.obstructions)
    evaluated = silhouette.evaluate_sources(
        "boundary",
        path,
        missing_reference,
        mesh_resolution=60,
        image_size=32,
    )
    assert evaluated == loaded

    exit_code = silhouette.main(
        [
            "silhouette",
            str(path),
            str(missing_reference),
            "--brief-id",
            "boundary",
            "--res",
            "60",
            "--size",
            "32",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    assert "required 0.05, observed -0.5" in captured.err


def test_anatomy_evaluation_uses_canonical_body_projection() -> None:
    result = eval_anatomy._compile_variant(_rejected_body_spec(), "boundary")

    assert isinstance(result, EvaluationObstruction)
    projected = json.loads(result.reason)
    assert projected["rejected"] == "body"
    obstruction = projected["obstructions"][0]
    assert obstruction["obstruction"] == "FixedPlacementConflictObstruction"
    assert obstruction["required"] == 0.05
    assert obstruction["observed"] == -0.5


def test_plate_evaluation_uses_canonical_body_projection(tmp_path: Path) -> None:
    body_path = _write_rejected_body(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}", encoding="utf-8")

    result = eval_plate_authoring.evaluate_plate_authoring(
        body_path,
        policy_path,
        resolution=60,
        image_size=32,
    )

    assert isinstance(result, EvaluationObstruction)
    assert result.address == str(body_path)
    expected = FixedPlacementConflictObstruction(
        "fc",
        "part:footf",
        "part:chest",
        0.05,
        -0.5,
    )
    assert result.reason == _canonical_rejection((expected,))
