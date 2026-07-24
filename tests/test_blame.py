"""Semantic-structure tests for the minimal-blame (WHY) layer.

These assert the *shape* of blame — subset containment, verdict-flip on removal,
minimality against a stub oracle, determinism, budget accounting, the probe
guard, and near-miss aggregation derived from the live obstructions — never
pinned float regression values. The integration fixtures are the fast
``probe_legs`` rejecting spec; every test caps its work well under ten seconds.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from golem.kernel.anatomy import (
    DuplicateAnatomyRegionObstruction,
    MissingTerminalRegionObstruction,
    RejectedAnatomy,
    RejectedVasculature,
    VascularSolverResidualObstruction,
    VascularBridgeWitness,
    VascularBudgetHypothesis,
    VascularConditioningHypothesis,
    VascularConditionSectionWitness,
    VascularDemandHypothesis,
    VascularGluingInterrogation,
    VascularGluingObstruction,
    VascularResidualSectionWitness,
    VascularTopologyHypothesis,
    realize_vasculature,
)
from golem.kernel.blame import (
    AuthoredUnitKind,
    CheckVerdict,
    DeclarationFamily,
    ObstructionSignature,
    check_verdict,
    extract_blame,
    guarded_verdict,
    interrogate_blame,
    near_miss_distributions,
    obstruction_signature,
)
from golem.kernel.blame.addresses import authored_addresses, authored_units, restrict
from golem.kernel.body import Compiler, CompiledBody, RejectedBody
from golem.kernel.engine.types import Rejected, decode_graph
from golem.kernel.anatomy.project import project_vascular_obstruction

SPECS = Path(__file__).resolve().parent.parent / "specs"


def _load(name: str) -> dict:
    return json.loads((SPECS / f"{name}.json").read_text())


def _rejection_obstructions(spec: dict) -> tuple[object, ...]:
    """The raw obstructions a caller would hold from the first rejecting stage."""
    compiled = Compiler(copy.deepcopy(spec), spec_dir=SPECS).compile()
    if isinstance(compiled, RejectedBody):
        return tuple(compiled.obstructions)
    assert isinstance(compiled, CompiledBody)
    decoded = decode_graph(compiled.graph)
    if isinstance(decoded, Rejected):
        fatal = tuple(o for o in decoded.obstructions if getattr(o, "fatal", True))
        if fatal:
            return fatal
    if isinstance(compiled.anatomy, RejectedAnatomy):
        return tuple(compiled.anatomy.obstructions)
    vascular = realize_vasculature(compiled.anatomy, compiled.graph)
    assert isinstance(vascular, (RejectedVasculature, RejectedAnatomy))
    return tuple(vascular.obstructions)


def _address_index(spec: dict) -> dict[str, object]:
    return {address.address: address for address in authored_addresses(spec)}


@dataclass(frozen=True)
class _LegsBlame:
    spec: dict
    obstructions: tuple[object, ...]
    result: object
    unmutated: bool


@pytest.fixture(scope="module")
def legs_blame() -> _LegsBlame:
    """Compute probe_legs blame once; most integration tests reuse this result."""
    spec = _load("probe_legs")
    guard = copy.deepcopy(spec)
    obstructions = _rejection_obstructions(spec)
    result = extract_blame(spec, obstructions, spec_dir=SPECS)
    return _LegsBlame(spec, obstructions, result, unmutated=(spec == guard))


# --------------------------------------------------------------------------- #
# Address enumeration and the pure restriction fork                           #
# --------------------------------------------------------------------------- #
def test_authored_addresses_cover_declared_families() -> None:
    spec = _load("probe_legs")
    labels = {address.address for address in authored_addresses(spec)}
    for index in range(len(spec["skeleton"]["bones"])):
        assert f"skeleton/bones/{index}" in labels
    assert "anatomy/overall/regions/0" in labels
    assert "anatomy/overall/circulation/exchange_beds/0" in labels
    assert any(label.startswith("skeleton/bones/0/flesh/") for label in labels)


def test_restrict_is_pure_and_removes_by_original_index() -> None:
    spec = _load("probe_legs")
    before = copy.deepcopy(spec)
    index = _address_index(spec)
    forked = restrict(spec, frozenset({index["anatomy/overall/regions/1"]}))
    assert spec == before  # the source document is never mutated
    assert len(forked["anatomy"]["overall"]["regions"]) == len(
        spec["anatomy"]["overall"]["regions"]
    ) - 1
    assert forked["skeleton"]["bones"] == spec["skeleton"]["bones"]


def test_obstruction_signature_excludes_quantitative_fields() -> None:
    (obstruction, *_rest) = _rejection_obstructions(_load("probe_legs"))
    signature = obstruction_signature(obstruction)
    identity_fields = {name for name, _value in signature.identity}
    assert "region_id" in identity_fields
    assert "observed" not in identity_fields
    assert "required" not in identity_fields
    assert "attempted_candidate_count" not in identity_fields
    assert "failing_segments" not in identity_fields


def test_check_verdict_retains_raw_obstructions_for_interrogation() -> None:
    verdict = check_verdict(_load("probe_legs"), SPECS)
    assert verdict.raw_obstructions
    assert tuple(map(obstruction_signature, verdict.raw_obstructions)) == verdict.obstructions


def test_declaration_family_vocabulary_is_closed() -> None:
    families = {address.family for address in authored_addresses(_load("probe_legs"))}
    assert families <= set(DeclarationFamily)


def test_group_mus_cover_is_limb_region_and_bed_units() -> None:
    units = authored_units(_load("probe_legs"))
    assert {kind for unit in units for kind in unit.kinds} == set(AuthoredUnitKind)
    territories = {
        unit.identifier: (
            unit.kinds,
            {address.address for address in unit.addresses},
        )
        for unit in units
    }
    assert set(territories) == {"thoracic", "fore_paw", "hind_paw"}
    assert territories["fore_paw"][0] == set(AuthoredUnitKind)
    assert "skeleton/bones/2" in territories["fore_paw"][1]
    assert "anatomy/overall/regions/1" in territories["fore_paw"][1]
    assert "anatomy/overall/circulation/exchange_beds/0" in territories["fore_paw"][1]
    assert "skeleton/bones/5" in territories["hind_paw"][1]


# --------------------------------------------------------------------------- #
# Shrinker minimality against a synthetic oracle (no solver cost)             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _SyntheticCause:
    cause: str = "forefoot"


def _forefoot_oracle(spec, _spec_dir) -> CheckVerdict:
    """Rejects exactly while the forefoot bone is present: a known 1-element MUS."""
    bones = spec.get("skeleton", {}).get("bones", [])
    present = any(isinstance(b, dict) and b.get("id") == "forefoot" for b in bones)
    signature = obstruction_signature(_SyntheticCause())
    return CheckVerdict(
        "stub",
        rejected=present,
        resolved=True,
        obstructions=(signature,) if present else (),
    )


def test_shrinker_isolates_the_exact_minimal_cause() -> None:
    # With a synthetic oracle whose rejection is caused by a single declaration,
    # the deletion shrinker must blame exactly that declaration — proving both
    # minimality and obstruction-class agnosticism without any vascular cost.
    spec = _load("probe_legs")
    forefoot_index = next(
        i for i, b in enumerate(spec["skeleton"]["bones"]) if b["id"] == "forefoot"
    )
    result = extract_blame(
        spec, [_SyntheticCause()], spec_dir=SPECS, oracle=_forefoot_oracle
    )
    assert len(result.subsets) == 1
    assert result.subsets[0].blamed_addresses == (f"skeleton/bones/{forefoot_index}",)


def test_group_prepass_localizes_with_budget_below_declaration_scan() -> None:
    spec = _load("probe_legs")
    result = extract_blame(
        spec,
        [_SyntheticCause()],
        spec_dir=SPECS,
        budget=15,
        oracle=_forefoot_oracle,
    )
    assert result.check_invocations == 15
    assert not result.exhausted_budget
    assert result.subsets[0].blamed_addresses == ("skeleton/bones/2",)


def test_probe_guard_reports_overrun_as_unresolved() -> None:
    # A probe that would run long is cut by the wall-clock guard and reported as
    # unresolved, never a hang — the robustness contract for pathological forks.
    def _slow_oracle(_spec, _spec_dir) -> CheckVerdict:
        time.sleep(3.0)
        return CheckVerdict("stub", rejected=True, resolved=True, obstructions=())

    start = time.perf_counter()
    verdict = guarded_verdict(_slow_oracle, {}, SPECS, probe_timeout=0.2)
    elapsed = time.perf_counter() - start
    assert verdict.resolved is False
    assert verdict.stage == "timeout"
    assert elapsed < 1.5


# --------------------------------------------------------------------------- #
# Core blame semantics on the live check pipeline                            #
# --------------------------------------------------------------------------- #
def test_blamed_addresses_are_a_subset_of_authored(legs_blame: _LegsBlame) -> None:
    result = legs_blame.result
    authored = set(result.authored_addresses)
    assert result.subsets  # the rejection is genuinely reproduced and explained
    for subset in result.subsets:
        assert set(subset.blamed_addresses) <= authored
        assert subset.blamed_addresses  # a non-empty cause


def test_shrinking_exonerates_some_declarations(legs_blame: _LegsBlame) -> None:
    # deletion-based shrinking must clear at least one declaration, otherwise it
    # is doing no work; probe_legs' non-load-bearing flesh is exonerable.
    result = legs_blame.result
    for subset in result.subsets:
        assert len(subset.blamed_addresses) < len(result.authored_addresses)


def test_removing_the_blamed_set_flips_the_verdict(legs_blame: _LegsBlame) -> None:
    index = _address_index(legs_blame.spec)
    for subset in legs_blame.result.subsets:
        blamed = frozenset(index[label] for label in subset.blamed_addresses)
        verdict = check_verdict(restrict(legs_blame.spec, blamed), SPECS)
        assert not verdict.reproduces(subset.obstruction)


def test_distinct_obstructions_get_distinct_blame(legs_blame: _LegsBlame) -> None:
    # probe_legs rejects two symmetric regions (fore_paw, hind_paw); their blame
    # must localize to different limb flesh rather than collapsing to one answer.
    result = legs_blame.result
    assert len(result.subsets) == 2
    flesh_blame = [
        {label for label in subset.blamed_addresses if "/flesh/" in label}
        for subset in result.subsets
    ]
    assert flesh_blame[0] != flesh_blame[1]


def test_input_spec_is_not_mutated(legs_blame: _LegsBlame) -> None:
    assert legs_blame.unmutated


def test_blame_is_deterministic_across_runs(legs_blame: _LegsBlame) -> None:
    again = extract_blame(legs_blame.spec, legs_blame.obstructions, spec_dir=SPECS)
    assert again.subsets == legs_blame.result.subsets
    assert again.near_miss == legs_blame.result.near_miss
    assert again.authored_addresses == legs_blame.result.authored_addresses


# --------------------------------------------------------------------------- #
# Budget accounting                                                           #
# --------------------------------------------------------------------------- #
def test_budget_caps_real_check_invocations() -> None:
    spec = _load("probe_legs")
    obstructions = _rejection_obstructions(spec)
    result = extract_blame(spec, obstructions, spec_dir=SPECS, budget=6)
    assert result.check_invocations <= 6
    assert result.exhausted_budget
    authored = set(result.authored_addresses)
    for subset in result.subsets:
        assert set(subset.blamed_addresses) <= authored


def test_interrogation_serializes_localization_and_budget() -> None:
    section = interrogate_blame(
        _load("probe_legs"),
        (_SyntheticCause(),),
        spec_dir=SPECS,
        budget=15,
        oracle=_forefoot_oracle,
    ).to_json()
    entry = section["_SyntheticCause(cause=forefoot)"]
    assert entry == {
        "blamed_addresses": ("skeleton/bones/2",),
        "budget": {"spent": 15, "limit": 15, "exhausted": False},
        "disposition": "localized",
    }


def test_interrogation_declines_structurally_ill_posed_taxa_without_probing() -> None:
    def forbidden_oracle(_spec, _spec_dir) -> CheckVerdict:
        raise AssertionError("declined obstructions must not invoke the oracle")

    section = interrogate_blame(
        _load("probe_legs"),
        (
            DuplicateAnatomyRegionObstruction("fore_paw"),
            MissingTerminalRegionObstruction("forefoot"),
            VascularSolverResidualObstruction(0.2, 0.1),
        ),
        spec_dir=SPECS,
        budget=23,
        oracle=forbidden_oracle,
    ).to_json()
    assert {entry["disposition"] for entry in section.values()} == {
        "declined: coupled-unit",
        "declined: absence-degenerate",
        "declined: global-solve",
    }
    assert all(entry["blamed_addresses"] == () for entry in section.values())
    assert all(
        entry["budget"] == {"spent": 0, "limit": 23, "exhausted": False}
        for entry in section.values()
    )


def test_interrogation_consumes_gluing_decomposition_without_mus_probes() -> None:
    def forbidden_oracle(_spec, _spec_dir) -> CheckVerdict:
        raise AssertionError("solve-level decomposition must not invoke MUS")

    obstruction = VascularGluingObstruction(
        "closed-vascular-field",
        "NonConvergentBalance @closed-vascular-balance: solver_info=40",
        gluing_interrogation=VascularGluingInterrogation(
            VascularTopologyHypothesis(
                1, (VascularBridgeWitness("edge-0", 1.0e-6),)
            ),
            VascularConditioningHypothesis(
                2.0e6,
                1.0e12,
                (
                    VascularConditionSectionWitness(
                        "wing-node", "wing_tips", 1.0e6, 1.0e12
                    ),
                ),
            ),
            VascularDemandHypothesis(
                maximum_stop_residual=3.0e-11,
                residual_tolerance=1.0e-10,
                relative_residual=3.0e-13,
                relative_tolerance=1.0e-13,
                iteration_cap=40,
                iterations_used=40,
                exhausted=True,
                worst_sections=(
                    VascularResidualSectionWitness(
                        "wing-node", "wing_tips", -3.0e-11
                    ),
                ),
            ),
            VascularBudgetHypothesis(40, 40, 1.0e-13, True),
        ),
    )
    projection = project_vascular_obstruction(obstruction)
    assert projection["predicate"] == (
        "relative free-system residual at solver stop"
    )
    assert projection["required"] == 1.0e-13
    assert projection["observed"] == 3.0e-13
    assert projection["unit"] == "relative_residual"
    section = interrogate_blame(
        _load("probe_legs"),
        (obstruction,),
        spec_dir=SPECS,
        oracle=forbidden_oracle,
    ).to_json()
    entry = next(iter(section.values()))

    assert entry["disposition"] == "decomposed: global-solve"
    assert entry["blamed_addresses"] == ("anatomy/regions/wing_tips",)
    assert tuple(entry["witness"]["hypotheses"]) == (
        "topology",
        "conditioning",
        "demand",
        "budget",
    )
    demand_witness = entry["witness"]["hypotheses"]["demand"]["witness"]
    assert demand_witness["maximum_stop_residual"] == 3.0e-11
    assert demand_witness["residual_tolerance"] == 1.0e-10
    assert demand_witness["relative_residual"] == 3.0e-13
    assert demand_witness["relative_tolerance"] == 1.0e-13
    assert demand_witness["iteration_cap"] == 40
    assert demand_witness["iterations_used"] == 40
    assert demand_witness["exhausted"] is True
    assert entry["budget"] == {
        "spent": 0,
        "limit": 512,
        "exhausted": False,
    }


def test_pathological_fork_declines_instead_of_emitting_a_subset() -> None:
    original_bone_count = len(_load("probe_legs")["skeleton"]["bones"])

    def timeout_on_fork(spec, _spec_dir) -> CheckVerdict:
        signature = obstruction_signature(_SyntheticCause())
        bone_count = len(spec.get("skeleton", {}).get("bones", ()))
        return (
            CheckVerdict("stub", True, True, (signature,))
            if bone_count == original_bone_count
            else CheckVerdict("timeout", False, False, ())
        )

    section = interrogate_blame(
        _load("probe_legs"),
        (_SyntheticCause(),),
        spec_dir=SPECS,
        oracle=timeout_on_fork,
    ).to_json()
    entry = section["_SyntheticCause(cause=forefoot)"]
    assert entry["blamed_addresses"] == ()
    assert entry["disposition"] == "declined: pathological-fork"


# --------------------------------------------------------------------------- #
# Near-miss aggregation                                                       #
# --------------------------------------------------------------------------- #
def test_near_miss_distribution_summarizes_candidate_exhaustion(
    legs_blame: _LegsBlame,
) -> None:
    obstructions = legs_blame.obstructions
    bifurcations = tuple(
        o for o in obstructions if type(o).__name__ == "InfeasibleBifurcationObstruction"
    )
    assert bifurcations  # this fixture exhausts candidate search fronts
    distributions = legs_blame.result.near_miss
    assert distributions
    grouped_total = sum(d.total_attempted_candidates for d in distributions)
    assert grouped_total == sum((o.attempted_candidate_count or 0) for o in bifurcations)
    assert all(obstruction.failing_segments for obstruction in bifurcations)
    assert all(
        segment.source_host_part_addresses
        and segment.source_host_bone_addresses
        and segment.target_host_part_addresses
        and segment.target_host_bone_addresses
        for obstruction in bifurcations
        for segment in obstruction.failing_segments
    )
    for dist in distributions:
        members = tuple(
            o
            for o in bifurcations
            if str(o.predicate) == dist.predicate and str(o.lane) == dist.lane
        )
        observeds = tuple(o.observed for o in members)
        assert dist.nearest_miss_observed == max(observeds)
        assert dist.worst_observed == min(observeds)
        assert dist.nearest_miss_margin == dist.nearest_miss_observed - dist.required


def test_near_miss_is_deterministically_ordered() -> None:
    distributions = near_miss_distributions(_rejection_obstructions(_load("probe_legs")))
    keys = [(d.predicate, d.lane) for d in distributions]
    assert keys == sorted(keys)
