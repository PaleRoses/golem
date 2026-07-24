# Skin-siege measurement: local nearest-surface projection distribution on
# the Sable Stride formed-flesh field. For every flesh vertex it runs the
# proposed Newton descent on Psi = F_flesh - thickness starting at the
# vertex, and records:
#   (a) total Euclidean displacement / thickness and / pitch  -> locality R
#   (b) terminal |Psi| residual                               -> convergence
#   (c) sign crossings on the segment v -> p*                 -> local
#       exactly-one-crossing survivors
#   (d) gradient-degeneracy / domain-escape counts
# Read-only w.r.t. the sealed formation-trial tree; writes only its own JSON
# receipt beside this script. This is the pre-implementation evidence the
# journal's design entry (010) owes before the locality radius is sealed.
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from golem.assembly.compile import pitch_res
from golem.kernel import body
from golem.kernel.engine.compile import (
    _evaluate_certified_checked,
    _symmetric_trilinear_gradients,
    _trilinear_samples,
)
from golem.kernel.engine.types import decode_graph, require_accepted

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "specs" / "sable_stride.json"
RECEIPT = Path(__file__).with_name("measure-local-projection.json")

THICKNESS = 0.004
DESCENT_BUDGET = 16
SEGMENT_SAMPLES = 17
STEP_PITCH_CAP = 0.5
MINIMUM_GRADIENT = 1.0e-8


def formed_spec() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    layers = spec["anatomy"]["overall"]["integument_layers"]
    for layer in layers:
        layer["formation"] = "static_implicit_relaxation"
        assert abs(layer["thickness"] - THICKNESS) < 1e-15
    return spec


def _bounded(vectors: np.ndarray, maximum: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    factors = np.minimum(
        1.0,
        np.divide(
            maximum, norms, out=np.ones_like(norms), where=norms > 0.0
        ),
    )
    return vectors * factors[:, None]


def _descend(
    skin_field: np.ndarray,
    lower: np.ndarray,
    pitch: np.ndarray,
    vertices: np.ndarray,
    maximum_pitch: float,
    budget: int,
    step_pitch_cap: float,
) -> dict:
    step_cap = step_pitch_cap * maximum_pitch
    root_tolerance = max(1.0e-12, MINIMUM_GRADIENT * maximum_pitch)
    current = vertices.copy()
    domain_escape = np.zeros(len(vertices), dtype=bool)
    degenerate = np.zeros(len(vertices), dtype=bool)
    start = time.perf_counter()
    for _ in range(budget):
        samples = _trilinear_samples(skin_field, lower, pitch, current)
        gradients = _symmetric_trilinear_gradients(
            skin_field, lower, pitch, current
        )
        magnitudes = np.linalg.norm(gradients, axis=1)
        squared = magnitudes * magnitudes
        corrections = _bounded(
            np.divide(
                samples.values[:, None] * gradients,
                squared[:, None],
                out=np.zeros_like(gradients),
                where=squared[:, None] > 0.0,
            ),
            step_cap,
        )
        current = current - corrections
        domain_escape |= ~samples.inside
        degenerate |= magnitudes < MINIMUM_GRADIENT
    descent_seconds = time.perf_counter() - start

    final_samples = _trilinear_samples(skin_field, lower, pitch, current)
    residual = np.abs(final_samples.values)
    displacement = np.linalg.norm(current - vertices, axis=1)

    parameters = np.linspace(0.0, 1.0, SEGMENT_SAMPLES)
    segment = (
        vertices[:, None, :]
        + parameters[None, :, None] * (current - vertices)[:, None, :]
    )
    segment_values = _trilinear_samples(
        skin_field, lower, pitch, segment.reshape((-1, 3))
    ).values.reshape((len(vertices), SEGMENT_SAMPLES))
    signs = np.signbit(segment_values)
    crossings = np.count_nonzero(signs[:, :-1] != signs[:, 1:], axis=1)

    converged = (residual <= root_tolerance) & ~domain_escape & ~degenerate
    disp_t = displacement[converged] / THICKNESS
    disp_p = displacement[converged] / maximum_pitch
    return {
        "budget": budget,
        "step_pitch_cap": step_pitch_cap,
        "descent_seconds": descent_seconds,
        "converged_vertices": int(np.count_nonzero(converged)),
        "domain_escape_vertices": int(np.count_nonzero(domain_escape)),
        "degenerate_vertices": int(np.count_nonzero(degenerate)),
        "unconverged_vertices": int(
            np.count_nonzero(~converged & ~domain_escape & ~degenerate)
        ),
        "overshoot_vertices": int(np.count_nonzero(crossings >= 2)),
        "residual_over_pitch_max": float(np.max(residual) / maximum_pitch),
        "displacement_over_thickness": {
            "p50": float(np.percentile(disp_t, 50)),
            "p95": float(np.percentile(disp_t, 95)),
            "p99": float(np.percentile(disp_t, 99)),
            "p999": float(np.percentile(disp_t, 99.9)),
            "max": float(np.max(disp_t)),
        },
        "displacement_over_pitch": {
            "p999": float(np.percentile(disp_p, 99.9)),
            "max": float(np.max(disp_p)),
        },
        "segment_crossing_histogram": {
            str(count): int(np.count_nonzero(crossings == count))
            for count in sorted(set(crossings.tolist()))
        },
    }


def main() -> None:
    compiled = body.Compiler(formed_spec(), spec_dir=SPEC.parent).compile()
    assert isinstance(compiled, body.CompiledBody), "formed body must compile"
    graph = compiled.graph
    assert graph.get("skin") is not None, "formed token must reach the graph"

    resolution = pitch_res(graph, 0.017)
    decoded = require_accepted(decode_graph(graph))

    start = time.perf_counter()
    flesh = _evaluate_certified_checked(decoded, resolution)
    flesh_seconds = time.perf_counter() - start
    assert not hasattr(flesh, "obstructions"), f"flesh rejected: {flesh!r}"

    vertices = np.asarray(flesh.vertices, dtype=np.float64)
    lower = np.asarray(flesh.lower, dtype=np.float64)
    pitch = np.asarray(flesh.world_pitch, dtype=np.float64)
    maximum_pitch = float(flesh.maximum_pitch)
    skin_field = np.asarray(flesh.field, dtype=np.float64) - THICKNESS

    configs = (
        (16, 0.5),
        (24, 0.25),
        (32, 0.125),
        (48, 0.0625),
    )
    sweeps = [
        _descend(
            skin_field, lower, pitch, vertices, maximum_pitch, budget, cap
        )
        for budget, cap in configs
    ]
    receipt = {
        "resolution": int(resolution),
        "vertex_count": int(len(vertices)),
        "thickness": THICKNESS,
        "maximum_pitch": maximum_pitch,
        "segment_samples": SEGMENT_SAMPLES,
        "flesh_eval_seconds": flesh_seconds,
        "sweeps": sweeps,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
