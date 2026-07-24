"""Deletion-based greedy 1-minimization of the blamed authored subset."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import reduce
from pathlib import Path

from golem.kernel.blame.addresses import authored_units, restrict
from golem.kernel.blame.core import AuthoredAddress, CheckVerdict, ObstructionSignature
from golem.kernel.blame.verdict import check_verdict

VerdictOracle = Callable[[Mapping[str, object], Path], CheckVerdict]

_TIMEOUT_VERDICT = CheckVerdict("timeout", rejected=False, resolved=False, obstructions=())


def guarded_verdict(
    oracle: VerdictOracle,
    spec: Mapping[str, object],
    spec_dir: Path,
    probe_timeout: float | None,
) -> CheckVerdict:
    """Run one probe under an optional wall-clock guard.

    A forked spec can drive the vascular candidate search into a pathological
    blow-up far costlier than the original — the sharpest ill-posedness of
    deletion-based shrinking. When ``probe_timeout`` is set (and we hold the main
    thread, where ``SIGALRM`` is deliverable), an overrun is reported as an
    unresolved verdict rather than a hang, so the declaration stays blamed. The
    guard is opt-in: with ``None`` the check runs to completion, keeping the
    common path free of signal machinery and any risk to the kernel's caches.
    """
    if probe_timeout is None or threading.current_thread() is not threading.main_thread():
        return oracle(spec, spec_dir)

    def _raise(_signum: int, _frame: object) -> None:
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, probe_timeout)
    try:
        return oracle(spec, spec_dir)
    except TimeoutError:
        return _TIMEOUT_VERDICT
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@dataclass(frozen=True)
class ShrinkBudget:
    """Accounting for the number of *real* (non-memoized) check probes spent."""

    limit: int
    spent: int = 0

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.limit


@dataclass(frozen=True)
class ShrinkOutcome:
    blamed: tuple[AuthoredAddress, ...]
    budget: ShrinkBudget
    exhausted_budget: bool
    pathological_fork: bool = False


@dataclass(frozen=True)
class _ShrinkState:
    removed: frozenset[AuthoredAddress]
    budget: ShrinkBudget
    exhausted_budget: bool = False
    pathological_fork: bool = False


def _probe_candidate(
    spec: Mapping[str, object],
    target: ObstructionSignature,
    oracle: VerdictOracle,
    spec_dir: Path,
    probe_timeout: float | None,
    state: _ShrinkState,
    candidate: frozenset[AuthoredAddress],
) -> _ShrinkState:
    if state.budget.exhausted:
        return replace(state, exhausted_budget=True)
    removed = state.removed | candidate
    verdict = guarded_verdict(oracle, restrict(spec, removed), spec_dir, probe_timeout)
    next_budget = replace(state.budget, spent=state.budget.spent + 1)
    return _ShrinkState(
        removed if verdict.reproduces(target) else state.removed,
        next_budget,
        state.exhausted_budget,
        state.pathological_fork or verdict.stage == "timeout",
    )


def _shrink_candidates(
    spec: Mapping[str, object],
    target: ObstructionSignature,
    oracle: VerdictOracle,
    spec_dir: Path,
    probe_timeout: float | None,
    state: _ShrinkState,
    candidates: tuple[frozenset[AuthoredAddress], ...],
) -> _ShrinkState:
    return reduce(
        lambda current, candidate: _probe_candidate(
            spec,
            target,
            oracle,
            spec_dir,
            probe_timeout,
            current,
            candidate,
        ),
        candidates,
        state,
    )


def shrink_target(
    spec: Mapping[str, object],
    addresses: tuple[AuthoredAddress, ...],
    target: ObstructionSignature,
    budget: ShrinkBudget,
    *,
    spec_dir: Path,
    oracle: VerdictOracle = check_verdict,
    probe_timeout: float | None = None,
) -> ShrinkOutcome:
    """Greedily remove declarations that preserve ``target``; blame what remains.

    Addresses are visited leaf-first (see ``DeclarationFamily`` ranking) so the
    separable declarations are exonerated before the structural spine is tried.
    A declaration is dropped only when its cumulative removal still reproduces
    the target, so the surviving blamed set is 1-minimal up to visit order: no
    remaining declaration can additionally be removed without losing the target.
    """
    ordered = tuple(sorted(addresses, key=lambda address: address.shrink_key))
    coarse = _shrink_candidates(
        spec,
        target,
        oracle,
        spec_dir,
        probe_timeout,
        _ShrinkState(frozenset(), budget),
        tuple(frozenset(unit.addresses) for unit in authored_units(spec, addresses)),
    )
    refinement = _shrink_candidates(
        spec,
        target,
        oracle,
        spec_dir,
        probe_timeout,
        coarse,
        tuple(
            frozenset((address,))
            for address in ordered
            if address not in coarse.removed
        ),
    )
    blamed = tuple(
        sorted(
            (address for address in addresses if address not in refinement.removed),
            key=lambda address: address.display_key,
        )
    )
    return ShrinkOutcome(
        blamed=blamed,
        budget=refinement.budget,
        exhausted_budget=refinement.exhausted_budget,
        pathological_fork=refinement.pathological_fork,
    )
