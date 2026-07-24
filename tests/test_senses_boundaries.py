from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from golem.cli import check as check_cli
from golem.cli.spec import LoadedSpec
from golem.contracts import cli as contracts_cli
from golem.evals import eval_anatomy
from golem.kernel.blame import verdict as blame_verdict
from golem.kernel.body import cli as body_cli
from golem.kernel.body import report as body_report
from golem.kernel.body.types import CompiledBody
from golem.senses.model import RejectedSenses
from golem.senses.project import project_rejected_senses
from golem.senses.proprio import build_senses
from golem.senses.proprio import cli as proprio_cli


def _rejected_graph() -> dict[str, object]:
    return {
        "name": "rejected-senses",
        "blend": 0.01,
        "parts": [
            {
                "id": "formed",
                "type": "gencyl",
                "spine": (
                    (0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ),
                "radii": (0.2, 0.2),
                "formation": "numerical_guesswork",
            }
        ],
    }


def _rejected_senses() -> RejectedSenses:
    result = build_senses(_rejected_graph(), None)

    assert isinstance(result, RejectedSenses)
    return result


def _unsatisfied_formation_graph() -> dict[str, object]:
    return {
        "name": "unsatisfied-formation-senses",
        "blend": 0.01,
        "parts": [
            {
                "id": "formed",
                "type": "gencyl",
                "spine": (
                    (0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 2.0, 0.0),
                ),
                "radii": (0.2, 0.3, 0.2),
                "formation": "skeleton_integral",
                "profile": {
                    "n": 2.0,
                    "depth": (0.3, 0.3, 0.3),
                    "up": (1.0, 0.0, 0.0),
                },
            }
        ],
    }


def test_body_report_propagates_rejected_senses_without_contract_evaluation() -> None:
    intent = {"asserts": {"clauses": ()}}
    typed_result = body_report.run_typed_asserts(_rejected_graph(), intent)
    projected_result = body_report.run_asserts(_rejected_graph(), intent)

    assert isinstance(typed_result, RejectedSenses)
    assert isinstance(projected_result, RejectedSenses)
    assert typed_result == projected_result


def test_rejected_senses_projection_preserves_typed_obstruction() -> None:
    projected = project_rejected_senses(_rejected_senses())

    assert projected["status"] == "senses_obstructed"
    assert len(projected["obstructions"]) == 1
    assert projected["obstructions"][0]["obstruction"] == (
        "GeometryObstruction"
    )
    assert projected["obstructions"][0]["part_id"] == "formed"
    assert projected["obstructions"][0]["fatal"] is True


def test_proprio_cli_returns_rejected_senses_with_obstruction_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    graph_path = tmp_path / "rejected-graph.json"
    graph_path.write_text(json.dumps(_rejected_graph()), encoding="utf-8")

    run_result = proprio_cli.run(_rejected_graph(), None)
    exit_code = proprio_cli.main([str(graph_path)])
    captured = capsys.readouterr()

    assert isinstance(run_result, RejectedSenses)
    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["status"] == "senses_obstructed"


def test_contract_cli_stops_before_clause_evaluation_on_rejected_senses(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    graph_path = tmp_path / "rejected-graph.json"
    intent_path = tmp_path / "intent.json"
    graph_path.write_text(json.dumps(_rejected_graph()), encoding="utf-8")
    intent_path.write_text(
        json.dumps({"asserts": {"clauses": []}}),
        encoding="utf-8",
    )

    exit_code = contracts_cli.main(
        [
            str(graph_path),
            "--intent",
            str(intent_path),
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["status"] == "senses_obstructed"


def test_body_cli_stops_on_rejected_senses(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "body.json"
    spec_path.write_text("{}", encoding="utf-8")
    compiled = CompiledBody(
        {
            "name": "rejected-body-senses",
            "parts": (),
            "intent": {},
        },
        {},
        None,
    )
    rejected = _rejected_senses()
    compiler = SimpleNamespace(compile=lambda: compiled)
    monkeypatch.setattr(body_cli, "Compiler", lambda *_args, **_kwargs: compiler)
    monkeypatch.setattr(
        body_cli,
        "run_asserts",
        lambda _graph, _intent: rejected,
    )

    exit_code = body_cli.main([str(spec_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["status"] == "senses_obstructed"


def test_check_command_projects_senses_obstruction_before_rendering(
    tmp_path: Path,
) -> None:
    result = check_cli._check_loaded(
        LoadedSpec(
            tmp_path / "unsatisfied-formation.json",
            _unsatisfied_formation_graph(),
        ),
        False,
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["status"] == "senses_obstructed"


def test_blame_verdict_preserves_senses_obstruction_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = CompiledBody(
        _unsatisfied_formation_graph(),
        {
            "violations": (),
            "solved": {"limit_violations": ()},
        },
        None,
    )
    compiler = SimpleNamespace(compile=lambda: compiled)
    monkeypatch.setattr(
        blame_verdict,
        "Compiler",
        lambda *_args, **_kwargs: compiler,
    )

    verdict = blame_verdict.check_verdict({}, tmp_path)

    assert verdict.stage == "senses"
    assert verdict.rejected is True
    assert verdict.resolved is True
    assert verdict.raw_obstructions
    assert verdict.obstructions[0].kind == "MuscleFormationObstruction"


def test_anatomy_evaluation_preserves_rejected_senses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "anatomy.json"
    spec_path.write_text("{}", encoding="utf-8")
    compiled = SimpleNamespace(graph={"intent": {}})
    accepted_compilation = eval_anatomy._AcceptedCompilation(
        compiled,
        object(),
    )
    reference = eval_anatomy.silhouette.Geometry(
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.int64),
    )
    rejected = _rejected_senses()
    monkeypatch.setattr(eval_anatomy, "_BODY_PATH", spec_path)
    monkeypatch.setattr(
        eval_anatomy,
        "_without_authored_tissue",
        lambda _spec: {},
    )
    monkeypatch.setattr(
        eval_anatomy,
        "_compile_variant",
        lambda _spec, _variant: accepted_compilation,
    )
    monkeypatch.setattr(
        eval_anatomy.silhouette,
        "load_geometry",
        lambda _path, _resolution: reference,
    )
    monkeypatch.setattr(
        eval_anatomy,
        "_score_graph",
        lambda _graph, _reference, _variant: (
            eval_anatomy._GraphEvaluation(1.0, 1.0)
        ),
    )
    monkeypatch.setattr(
        eval_anatomy.proprio,
        "build_senses",
        lambda _graph, _intent: rejected,
    )

    result = eval_anatomy.evaluate_anatomy()
    rendered = eval_anatomy._render_anatomy_evaluation(result)

    assert isinstance(result, eval_anatomy.AnatomySensesObstruction)
    assert result.obstructions == rejected.obstructions
    assert rendered.exit_code == 2
    assert rendered.stdout == ""
    assert json.loads(rendered.stderr) == json.loads(
        json.dumps(
            {
                "address": "senses/anatomy",
                **project_rejected_senses(rejected),
            }
        )
    )
