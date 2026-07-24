"""Aggregate candidate-exhaustion obstructions into near-miss distributions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import groupby


@dataclass(frozen=True)
class NearMissDistribution:
    """One (predicate, lane) group of candidate-exhaustion evidence.

    ``nearest_miss_observed`` is the least-infeasible margin the solver reached
    across the group (the observed value closest to ``required``); ``margin`` is
    its signed distance to the feasibility threshold. Per-candidate points are
    never carried — the obstructions already summarize whole search fronts.
    """

    predicate: str
    lane: str
    obstruction_count: int
    total_attempted_candidates: int
    required: float
    nearest_miss_observed: float
    worst_observed: float
    nearest_miss_margin: float
    region_ids: tuple[str, ...]

    def render(self) -> str:
        return (
            f"all {self.total_attempted_candidates} candidates failed "
            f"{self.predicate} lane={self.lane} across "
            f"{len(self.region_ids)} region(s), nearest miss "
            f"{self.nearest_miss_observed:.4g} (margin {self.nearest_miss_margin:.4g})"
        )


def _candidate_records(obstructions: Iterable[object]) -> tuple[dict, ...]:
    return tuple(
        {
            "predicate": str(getattr(obstruction, "predicate")),
            "lane": str(getattr(obstruction, "lane")),
            "required": float(getattr(obstruction, "required")),
            "observed": float(getattr(obstruction, "observed")),
            "attempted": int(getattr(obstruction, "attempted_candidate_count") or 0),
            "region_id": str(getattr(obstruction, "region_id", "")),
        }
        for obstruction in obstructions
        if hasattr(obstruction, "predicate")
        and hasattr(obstruction, "observed")
        and hasattr(obstruction, "required")
    )


def near_miss_distributions(obstructions: Iterable[object]) -> tuple[NearMissDistribution, ...]:
    """Group candidate-exhaustion obstructions by (predicate, lane), summarized.

    The result is ordered deterministically by (predicate, lane); within a group
    the nearest miss is the maximum observed margin and the worst is the minimum.
    """
    records = sorted(
        _candidate_records(obstructions),
        key=lambda record: (record["predicate"], record["lane"]),
    )
    distributions = []
    for (predicate, lane), group in groupby(
        records, key=lambda record: (record["predicate"], record["lane"])
    ):
        members = tuple(group)
        observeds = tuple(member["observed"] for member in members)
        nearest = max(observeds)
        required = next(
            member["required"] for member in members if member["observed"] == nearest
        )
        distributions.append(
            NearMissDistribution(
                predicate=predicate,
                lane=lane,
                obstruction_count=len(members),
                total_attempted_candidates=sum(member["attempted"] for member in members),
                required=required,
                nearest_miss_observed=nearest,
                worst_observed=min(observeds),
                nearest_miss_margin=nearest - required,
                region_ids=tuple(
                    sorted({member["region_id"] for member in members if member["region_id"]})
                ),
            )
        )
    return tuple(distributions)
