# Skin-siege authoring-loop benchmark.
#
# Keeps the ordinary field cache warm in every arm. "before" clears only the
# in-process skin memo before each transaction, forcing the formed-skin solve;
# "after" retains the accepted sheet for identical governing inputs; "legacy"
# evaluates the same quadruped geometry without the skin token. Writes one JSON
# receipt beside this script. Cold acceptance oracles are outside this probe:
# GOLEM_CIRCUIT_CACHE_COLD=1 bypasses the memo and always seals from cold.
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable

from golem.kernel import engine
from golem.kernel.engine import compile as engine_compile

RECEIPT = Path(__file__).with_name("measure-warm-start.json")
RESOLUTION = 64
SAMPLES = 7


def quadruped_graph(*, skin: bool) -> dict:
    graph = {
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
    return (
        {
            **graph,
            "skin": {
                "formation": "static_implicit_relaxation",
                "thickness": 0.004,
                "region_ids": ("host",),
            },
        }
        if skin
        else graph
    )


def timed(thunk: Callable[[], object]) -> float:
    start = time.perf_counter()
    result = thunk()
    elapsed = time.perf_counter() - start
    assert isinstance(result, engine.EvaluatedMorphology), result
    return elapsed


def main() -> None:
    os.environ.pop("GOLEM_CIRCUIT_CACHE_COLD", None)
    formed = quadruped_graph(skin=True)
    legacy = quadruped_graph(skin=False)

    # Prime the field cache without retaining a skin memo.
    engine_compile._SKIN_MEMO.clear()
    timed(lambda: engine.evaluate(formed, res=RESOLUTION))
    engine_compile._SKIN_MEMO.clear()
    timed(lambda: engine.evaluate(legacy, res=RESOLUTION))

    def cold_skin_transaction() -> object:
        engine_compile._SKIN_MEMO.clear()
        return engine.evaluate(formed, res=RESOLUTION)

    before = [timed(cold_skin_transaction) for _ in range(SAMPLES)]
    engine_compile._SKIN_MEMO.clear()
    timed(lambda: engine.evaluate(formed, res=RESOLUTION))
    after = [
        timed(lambda: engine.evaluate(formed, res=RESOLUTION))
        for _ in range(SAMPLES)
    ]
    legacy_times = [
        timed(lambda: engine.evaluate(legacy, res=RESOLUTION))
        for _ in range(SAMPLES)
    ]
    engine_compile._SKIN_MEMO.clear()

    before_median = statistics.median(before)
    after_median = statistics.median(after)
    legacy_median = statistics.median(legacy_times)
    receipt = {
        "resolution": RESOLUTION,
        "samples": SAMPLES,
        "geometry": "separated-limb quadruped",
        "before_skin_cold_seconds": before,
        "after_skin_memo_seconds": after,
        "legacy_seconds": legacy_times,
        "before_median_seconds": before_median,
        "after_median_seconds": after_median,
        "legacy_median_seconds": legacy_median,
        "warm_speedup": before_median / after_median,
        "formed_over_legacy_before": before_median / legacy_median,
        "formed_over_legacy_after": after_median / legacy_median,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
