"""The minimal-blame entry point: spec + rejection obstructions -> blame result."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

from golem.kernel.blame.addresses import authored_addresses
from golem.kernel.blame.aggregate import NearMissDistribution, near_miss_distributions
from golem.kernel.blame.core import AuthoredAddress, ObstructionSignature
from golem.kernel.blame.shrink import (
    ShrinkBudget,
    VerdictOracle,
    guarded_verdict,
    shrink_target,
)
from golem.kernel.blame.verdict import check_verdict, obstruction_signature

DEFAULT_BUDGET = 512


@dataclass(frozen=True)
class BlameSubset:
    """The approximately-minimal authored subset that reproduces one obstruction."""

    obstruction: ObstructionSignature
    blamed_addresses: tuple[str, ...]
    pathological_fork: bool = False


@dataclass(frozen=True)
class BlameResult:
    """The WHY layer: what authored declarations a rejection is minimally blamed on."""

    authored_addresses: tuple[str, ...]
    subsets: tuple[BlameSubset, ...]
    near_miss: tuple[NearMissDistribution, ...]
    check_invocations: int
    budget_limit: int
    exhausted_budget: bool


def _distinct_targets(obstructions: Sequence[object]) -> tuple[ObstructionSignature, ...]:
    return tuple(sorted(frozenset(map(obstruction_signature, obstructions))))


@dataclass(frozen=True)
class _ExtractionState:
    subsets: tuple[BlameSubset, ...]
    budget: ShrinkBudget
    exhausted_budget: bool = False


def _extract_target(
    spec: Mapping[str, object],
    addresses: tuple[AuthoredAddress, ...],
    resolved_dir: Path,
    oracle: VerdictOracle,
    probe_timeout: float | None,
    state: _ExtractionState,
    target: ObstructionSignature,
) -> _ExtractionState:
    outcome = shrink_target(
        spec,
        addresses,
        target,
        state.budget,
        spec_dir=resolved_dir,
        oracle=oracle,
        probe_timeout=probe_timeout,
    )
    return _ExtractionState(
        (
            *state.subsets,
            BlameSubset(
                obstruction=target,
                blamed_addresses=tuple(address.address for address in outcome.blamed),
                pathological_fork=outcome.pathological_fork,
            ),
        ),
        outcome.budget,
        state.exhausted_budget or outcome.exhausted_budget,
    )


def extract_blame(
    spec: Mapping[str, object],
    obstructions: Sequence[object],
    *,
    spec_dir: Path | str | None = None,
    budget: int = DEFAULT_BUDGET,
    probe_timeout: float | None = None,
    oracle: VerdictOracle = check_verdict,
) -> BlameResult:
    """Blame a body-document rejection on a minimal subset of authored declarations.

    ``obstructions`` are the rejection the caller already observed; each distinct
    obstruction signature becomes a blame target. For every target that the
    pristine spec genuinely reproduces, a deletion-based shrink isolates the
    authored declarations whose presence still forces that rejection. Candidate
    -exhaustion obstructions are folded into near-miss distributions in parallel.

    Deterministic: addresses, subsets, and distributions are all canonically
    ordered, and the underlying check pipeline is itself deterministic. The one
    quarantined impurity is the kernel's content-addressed vasculature cache,
    which is keyed by canonical body and never perturbs the returned value.
    """
    resolved_dir = Path(spec_dir) if spec_dir is not None else Path.cwd()
    addresses: tuple[AuthoredAddress, ...] = authored_addresses(spec)
    address_labels = tuple(address.address for address in addresses)
    targets = _distinct_targets(obstructions)

    baseline = guarded_verdict(oracle, spec, resolved_dir, probe_timeout)
    live_targets = tuple(target for target in targets if baseline.reproduces(target))

    budget_limit = max(budget, 0)
    extraction = reduce(
        lambda state, target: _extract_target(
            spec,
            addresses,
            resolved_dir,
            oracle,
            probe_timeout,
            state,
            target,
        ),
        live_targets,
        _ExtractionState((), ShrinkBudget(limit=budget_limit)),
    )

    return BlameResult(
        authored_addresses=address_labels,
        subsets=extraction.subsets,
        near_miss=near_miss_distributions(obstructions),
        check_invocations=extraction.budget.spent,
        budget_limit=budget_limit,
        exhausted_budget=extraction.exhausted_budget,
    )
