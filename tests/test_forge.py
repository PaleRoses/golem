from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

from golem.cli.forge import AuthoringChannel, ForgeArguments, forge
from golem.cli.parser import build_parser


@dataclass(frozen=True)
class _Session:
    document: dict[str, object]
    txn: int


@dataclass(frozen=True)
class _Verdict:
    accepted: bool
    record: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return self.record


@dataclass(frozen=True)
class _Submission:
    session: _Session
    verdict: _Verdict


def _scripted_author(replies: tuple[str, ...]) -> tuple[Mock, AuthoringChannel]:
    author = Mock(side_effect=replies)
    return author, author


def _reply(revision: int) -> str:
    return f"authoring result\n```json\n{{\"revision\": {revision}}}\n```\n"


def _verdict(
    *,
    accepted: bool,
    code: str | None,
    revision: int,
) -> _Verdict:
    return _Verdict(
        accepted,
        {
            "status": "accepted" if accepted else "rejected",
            "diagnostics": (
                []
                if code is None
                else [
                    {
                        "code": code,
                        "address": "skeleton/wing",
                        "predicate": "wing_pair",
                        "required": 2,
                        "observed": 1,
                    }
                ]
            ),
            "margins": [
                {
                    "predicate": "wing_clearance",
                    "margin": 0.25 + revision,
                }
            ],
            "binding_constraint": {
                "predicate": "wing_clearance",
                "margin": 0.25 + revision,
            },
            "deltas": {
                "margins_moved": ["wing_clearance"],
                "newly_active": ["wing_pair"] if not accepted else [],
                "newly_inactive": ["wing_pair"] if accepted else [],
            },
            "contract_refs": ["relations"],
            "changed_addresses": ["skeleton/wing"],
            "transaction": {"txn": revision},
        },
    )


def _normalized_submission(
    session: object,
    request: dict[str, object],
    *,
    accepted_revision: int | None,
    rejection_code: str,
) -> _Submission:
    document = request["document"]
    assert isinstance(document, dict)
    revision = document["revision"]
    assert isinstance(revision, int)
    txn = getattr(session, "txn") + 1
    accepted = revision == accepted_revision
    return _Submission(
        _Session({**document, "normalized": True}, txn),
        _verdict(
            accepted=accepted,
            code=None if accepted else rejection_code,
            revision=revision,
        ),
    )


def test_scripted_author_converges_through_session_protocol(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "specs" / "serpent.json"
    namespace = build_parser().parse_args(
        ("forge", "a winged serpent", "--out", str(output_path))
    )
    assert namespace.output_path == output_path
    assert namespace.rounds == 12
    assert namespace.model == "gpt-5.6-sol"
    author_mock, author = _scripted_author((_reply(1), _reply(2)))
    submission = Mock(
        side_effect=lambda session, request: _normalized_submission(
            session,
            request,
            accepted_revision=2,
            rejection_code="MissingWingPair",
        )
    )

    result = forge(
        ForgeArguments(
            brief="a winged serpent",
            output_path=output_path,
            rounds=4,
            model="test-model",
        ),
        author,
        submission_channel=submission,
    )

    assert result.exit_code == 0
    assert output_path.read_text(encoding="utf-8") == (
        '{\n  "revision": 2,\n  "normalized": true\n}\n'
    )
    assert "round 1: session REJECTED - MissingWingPair" in result.stdout
    assert "round 2: session ACCEPTED" in result.stdout
    assert "final tally: ACCEPTED; rounds=2" in result.stdout
    prompts = tuple(call.args[0] for call in author_mock.call_args_list)
    assert len(prompts) == 2
    assert "LATEST TRANSACTION VERDICT FEEDBACK\nnull" in prompts[0]
    assert "python -m golem contract SECTION" in prompts[0]
    assert "MissingWingPair" in prompts[1]
    assert '"margins"' in prompts[1]
    assert '"binding_constraint"' in prompts[1]
    assert '"deltas"' in prompts[1]
    assert '"contract_refs"' in prompts[1]
    assert '"changed_addresses"' in prompts[1]
    assert '"normalized": true' in prompts[1]
    assert all("LIVE AUTHORING CONTRACT" not in prompt for prompt in prompts)
    assert all("OBSTRUCTIONS" not in prompt for prompt in prompts)
    assert all("assembly verdict:" not in prompt for prompt in prompts)
    assert all("ROUND 1 CHECK" not in prompt for prompt in prompts)
    requests = tuple(call.args[1] for call in submission.call_args_list)
    assert requests == (
        {"base_txn": 0, "document": {"revision": 1}},
        {"base_txn": 1, "document": {"revision": 2}},
    )


def test_round_exhaustion_is_an_honest_protocol_failure(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "serpent.json"
    _, author = _scripted_author((_reply(1), _reply(2)))
    submission = Mock(
        side_effect=lambda session, request: _normalized_submission(
            session,
            request,
            accepted_revision=None,
            rejection_code="StillMalformed",
        )
    )

    result = forge(
        ForgeArguments(
            brief="a winged serpent",
            output_path=output_path,
            rounds=2,
            model="test-model",
        ),
        author,
        submission_channel=submission,
    )

    assert result.exit_code == 1
    assert "round 1: session REJECTED - StillMalformed" in result.stdout
    assert "round 2: session REJECTED - StillMalformed" in result.stdout
    assert "final tally: REJECTED; rounds=2" in result.stdout
    assert "session_rejections=2" in result.stdout
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "revision": 2,
        "normalized": True,
    }


def test_malformed_reply_becomes_a_typed_round_entry(tmp_path: Path) -> None:
    output_path = tmp_path / "serpent.json"
    _, author = _scripted_author(("I decline to provide JSON.",))
    submission = Mock(side_effect=AssertionError("malformed replies are not submitted"))

    result = forge(
        ForgeArguments(
            brief="a winged serpent",
            output_path=output_path,
            rounds=1,
            model="test-model",
        ),
        author,
        submission_channel=submission,
    )

    assert result.exit_code == 1
    assert "round 1: author REJECTED" in result.stdout
    assert "[MalformedAuthorReply]" in result.stdout
    assert "malformed_replies=1" in result.stdout
    assert not output_path.exists()
    submission.assert_not_called()


def test_nonfinite_existing_document_is_rejected_by_the_shared_source_loader(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "serpent.json"
    output_path.write_text('{"blend": NaN}', encoding="utf-8")
    author = Mock(side_effect=AssertionError("invalid bases never reach the author"))

    result = forge(
        ForgeArguments(
            brief="a winged serpent",
            output_path=output_path,
            rounds=1,
            model="test-model",
        ),
        author,
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "[MalformedInitialDocument]" in result.stderr
    assert "NonFiniteJsonNumber" in result.stderr
    author.assert_not_called()
