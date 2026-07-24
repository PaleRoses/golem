"""Rival vascular constructors judged on identical terms, side by side.

For each circulation circuit of a body spec, the greedy CCO constructor and the
flow-relaxation grower each propose a candidate network for the SAME admissible
domain.  Both proposals face the identical geometry judgment; the runner reports
feasibility verdicts, worst clearance margins, wall-clock, and network shape,
then a delta-response probe: perturb one region's skeletal landmark slightly,
re-run both, and measure how much each constructor's output moved.  The grower
warm-starts from its previous converged conductance field; CCO recomputes cold.
That asymmetry is the point of the experiment.

The final stdout line is stable JSON.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from golem import paths as _paths
from golem.kernel import body
from golem.kernel.anatomy.graph import RejectedVasculature
from golem.kernel.anatomy.realize.allocation import allocate_terminal_pairs
from golem.kernel.anatomy.realize.carriers import _CircuitGeometry
from golem.kernel.anatomy.realize.clearance import _validate_vascular_geometry
from golem.kernel.anatomy.realize.constructor import (
    DEFAULT_CIRCUIT_CONSTRUCTOR,
    realize_circuit,
)
from golem.kernel.anatomy.realize.corridor import _shared_pump_interface_points
from golem.kernel.anatomy.realize.grow import (
    FlowRelaxationGrower,
    last_grow_stats,
    last_grown_geometry,
    reset_warm_field,
)
from golem.kernel.anatomy.realize.materialize import _glue_circuit_sections
from golem.kernel.anatomy.vocabulary import SealedVascularConfig

if TYPE_CHECKING:
    from golem.kernel.anatomy.realize.constructor import CircuitConstructor
    from golem.kernel.anatomy.vocabulary import AcceptedAnatomy, CirculationCircuit


@dataclass(frozen=True)
class _CircuitInputs:
    circuit: CirculationCircuit
    terminal_pair_count: int
    accepted: AcceptedAnatomy
    part_by_id: dict[str, dict]
    provenance: dict[str, str]
    landmarks: dict[str, object]
    bone_use_count: dict[str, int]
    shared_pump_interfaces: tuple[tuple[float, float, float], ...]

    def with_landmarks(self, landmarks: dict[str, object]) -> "_CircuitInputs":
        return replace(self, landmarks=landmarks)


@dataclass(frozen=True)
class _Verdict:
    feasible: bool
    min_wall_margin: float
    obstruction_summary: tuple[str, ...]
    node_count: int
    edge_count: int
    total_length: float
    bifurcation_count: int
    radius_min: float
    radius_max: float
    radius_mean: float
    wall_seconds: float


def evaluate_rivals(
    spec_name: str, only_region: str | None = None
) -> dict[str, object]:
    spec = json.loads((_paths.SPECS / spec_name).read_text(encoding="utf-8"))
    compiled = body.Compiler(spec, spec_dir=_paths.SPECS).compile()
    accepted = compiled.anatomy
    body_graph = compiled.graph
    whole_parts = tuple(body_graph.get("parts", ()))
    config = SealedVascularConfig()
    inputs_by_region = {
        region: inputs
        for region, inputs in _circuit_inputs(accepted, body_graph).items()
        if only_region is None or region == only_region
    }
    grower = FlowRelaxationGrower()
    warm_grower = FlowRelaxationGrower(warm_start=True)
    circuits = tuple(
        _evaluate_circuit(
            region,
            inputs,
            whole_parts,
            config,
            grower,
            warm_grower,
        )
        for region, inputs in inputs_by_region.items()
    )
    return {
        "spec": spec_name,
        "circuits": circuits,
    }


def _circuit_inputs(
    accepted: AcceptedAnatomy, body_graph: dict
) -> dict[str, _CircuitInputs]:
    allocation = dict(allocate_terminal_pairs(accepted))
    provenance = body_graph.get("intent", {}).get("provenance", {})
    landmarks = body_graph.get("intent", {}).get("landmarks", {})
    part_by_id = {part["id"]: part for part in body_graph.get("parts", ())}
    bone_use_count = {
        bone: sum(bone in circuit.bones for circuit in accepted.circuits)
        for bone in {
            bone for circuit in accepted.circuits for bone in circuit.bones
        }
    }
    shared = _shared_pump_interface_points(accepted, landmarks)
    return {
        circuit.capillary_bed.region.region_id: _CircuitInputs(
            circuit=circuit,
            terminal_pair_count=allocation[
                circuit.capillary_bed.region.region_id
            ],
            accepted=accepted,
            part_by_id=part_by_id,
            provenance=provenance,
            landmarks=landmarks,
            bone_use_count=bone_use_count,
            shared_pump_interfaces=shared,
        )
        for circuit in accepted.circuits
    }


def _evaluate_circuit(
    region: str,
    inputs: _CircuitInputs,
    whole_parts: tuple[dict, ...],
    config: SealedVascularConfig,
    grower: FlowRelaxationGrower,
    warm_grower: FlowRelaxationGrower,
) -> dict[str, object]:
    cco = _run(DEFAULT_CIRCUIT_CONSTRUCTOR, inputs, whole_parts, config)
    grown = _run(grower, inputs, whole_parts, config)
    delta = _delta_response(inputs, whole_parts, config, warm_grower)
    return {
        "region": region,
        "terminal_pairs": inputs.terminal_pair_count,
        "cco": _verdict_dict(cco),
        "grower": _verdict_dict(grown),
        "grower_stats": _stats_dict(last_grow_stats(region)),
        "delta_response": delta,
    }


def _run(
    constructor: CircuitConstructor,
    inputs: _CircuitInputs,
    whole_parts: tuple[dict, ...],
    config: SealedVascularConfig,
) -> _Verdict:
    start = time.perf_counter()
    result = realize_circuit(
        constructor,
        inputs.circuit,
        inputs.terminal_pair_count,
        inputs.accepted,
        inputs.part_by_id,
        inputs.provenance,
        inputs.landmarks,
        inputs.bone_use_count,
        inputs.shared_pump_interfaces,
        config,
        None,
    )
    elapsed = time.perf_counter() - start
    return _judge(result, whole_parts, config, elapsed)


def _judge(
    result: _CircuitGeometry | RejectedVasculature,
    whole_parts: tuple[dict, ...],
    config: SealedVascularConfig,
    elapsed: float,
) -> _Verdict:
    if isinstance(result, RejectedVasculature):
        return _Verdict(
            feasible=False,
            min_wall_margin=float("nan"),
            obstruction_summary=tuple(
                _obstruction_label(obstruction)
                for obstruction in result.obstructions
            ),
            node_count=0,
            edge_count=0,
            total_length=0.0,
            bifurcation_count=0,
            radius_min=0.0,
            radius_max=0.0,
            radius_mean=0.0,
            wall_seconds=elapsed,
        )
    glued = _glue_circuit_sections((result,))
    if isinstance(glued, RejectedVasculature):
        return _Verdict(
            feasible=False,
            min_wall_margin=float("nan"),
            obstruction_summary=tuple(
                _obstruction_label(obstruction)
                for obstruction in glued.obstructions
            ),
            node_count=len(result.nodes),
            edge_count=len(result.edges),
            total_length=_total_length(result),
            bifurcation_count=_bifurcation_count(result),
            radius_min=min(edge.radius for edge in result.edges),
            radius_max=max(edge.radius for edge in result.edges),
            radius_mean=_mean(tuple(edge.radius for edge in result.edges)),
            wall_seconds=elapsed,
        )
    validation = _validate_vascular_geometry(
        glued, whole_parts, config
    )
    radii = tuple(edge.radius for edge in result.edges)
    return _Verdict(
        feasible=not validation.obstructions,
        min_wall_margin=validation.minimum_capsule_margin,
        obstruction_summary=tuple(
            _obstruction_label(obstruction)
            for obstruction in validation.obstructions
        ),
        node_count=len(result.nodes),
        edge_count=len(result.edges),
        total_length=_total_length(result),
        bifurcation_count=_bifurcation_count(result),
        radius_min=min(radii),
        radius_max=max(radii),
        radius_mean=_mean(radii),
        wall_seconds=elapsed,
    )


def _delta_response(
    inputs: _CircuitInputs,
    whole_parts: tuple[dict, ...],
    config: SealedVascularConfig,
    warm_grower: FlowRelaxationGrower,
) -> dict[str, object]:
    region = inputs.circuit.capillary_bed.region.region_id
    perturbed_landmarks = _perturb_landmarks(
        inputs.landmarks, inputs.circuit.bones[-1], 2.0e-4
    )
    perturbed_inputs = inputs.with_landmarks(perturbed_landmarks)
    cco_base = _geometry(DEFAULT_CIRCUIT_CONSTRUCTOR, inputs, config)
    cco_perturbed = _geometry(
        DEFAULT_CIRCUIT_CONSTRUCTOR, perturbed_inputs, config
    )
    reset_warm_field()
    _geometry(warm_grower, inputs, config)
    grower_base = last_grown_geometry(region)
    base_iterations = _iteration_total(region)
    _geometry(warm_grower, perturbed_inputs, config)
    grower_perturbed = last_grown_geometry(region)
    warm_iterations = _iteration_total(region)
    return {
        "cco_topology_change": _topology_change(cco_base, cco_perturbed),
        "grower_topology_change": _topology_change(
            grower_base, grower_perturbed
        ),
        "cco_node_displacement": _node_displacement(cco_base, cco_perturbed),
        "grower_node_displacement": _node_displacement(
            grower_base, grower_perturbed
        ),
        "grower_cold_iterations": base_iterations,
        "grower_warm_iterations": warm_iterations,
    }


def _geometry(
    constructor: CircuitConstructor,
    inputs: _CircuitInputs,
    config: SealedVascularConfig,
) -> _CircuitGeometry | None:
    result = realize_circuit(
        constructor,
        inputs.circuit,
        inputs.terminal_pair_count,
        inputs.accepted,
        inputs.part_by_id,
        inputs.provenance,
        inputs.landmarks,
        inputs.bone_use_count,
        inputs.shared_pump_interfaces,
        config,
        None,
    )
    return result if isinstance(result, _CircuitGeometry) else None


def _iteration_total(region: str) -> int | None:
    stats = last_grow_stats(region)
    return (
        None
        if stats is None
        else stats.supply_iterations + stats.return_iterations
    )


def _perturb_landmarks(
    landmarks: dict[str, object], bone_id: str, delta: float
) -> dict[str, object]:
    return {
        key: (
            [value[0] + delta, value[1], value[2]]
            if key.startswith(f"{bone_id}/")
            and isinstance(value, (list, tuple))
            and len(value) == 3
            else value
        )
        for key, value in landmarks.items()
    }


def _topology_change(
    base: _CircuitGeometry | None, other: _CircuitGeometry | None
) -> float:
    if base is None or other is None:
        return float("nan")
    base_edges = frozenset(
        (edge.source_node_id, edge.target_node_id) for edge in base.edges
    )
    other_edges = frozenset(
        (edge.source_node_id, edge.target_node_id) for edge in other.edges
    )
    union = base_edges | other_edges
    return (
        0.0
        if not union
        else len(base_edges ^ other_edges) / len(union)
    )


def _node_displacement(
    base: _CircuitGeometry | None, other: _CircuitGeometry | None
) -> float:
    if base is None or other is None:
        return float("nan")
    other_by_id = {node.node_id: node.position for node in other.nodes}
    shared = tuple(
        _distance(node.position, other_by_id[node.node_id])
        for node in base.nodes
        if node.node_id in other_by_id
    )
    return _mean(shared) if shared else float("nan")


def _total_length(geometry: _CircuitGeometry) -> float:
    by_id = {node.node_id: node.position for node in geometry.nodes}
    return sum(
        _distance(by_id[edge.source_node_id], by_id[edge.target_node_id])
        for edge in geometry.edges
    )


def _bifurcation_count(geometry: _CircuitGeometry) -> int:
    from golem.kernel.anatomy.graph import VascularNodeKind

    return sum(
        node.kind is VascularNodeKind.BIFURCATION for node in geometry.nodes
    )


def _distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _obstruction_label(obstruction: object) -> str:
    predicate = getattr(obstruction, "predicate", None)
    if predicate is not None:
        return (
            f"{type(obstruction).__name__}:{predicate.value}:"
            f"req={getattr(obstruction, 'required', 0.0):.3e}:"
            f"obs={getattr(obstruction, 'observed', 0.0):.3e}"
        )
    margin = getattr(obstruction, "margin", None)
    if margin is not None:
        return f"{type(obstruction).__name__}:margin={margin:.3e}"
    return type(obstruction).__name__


def _verdict_dict(verdict: _Verdict) -> dict[str, object]:
    return {
        "feasible": verdict.feasible,
        "min_wall_margin": verdict.min_wall_margin,
        "obstructions": list(verdict.obstruction_summary),
        "nodes": verdict.node_count,
        "edges": verdict.edge_count,
        "total_length": verdict.total_length,
        "bifurcations": verdict.bifurcation_count,
        "radius_min": verdict.radius_min,
        "radius_max": verdict.radius_max,
        "radius_mean": verdict.radius_mean,
        "wall_seconds": verdict.wall_seconds,
    }


def _stats_dict(stats: object) -> dict[str, object] | None:
    if stats is None:
        return None
    return {
        "supply_iterations": stats.supply_iterations,
        "return_iterations": stats.return_iterations,
        "supply_residual": stats.supply_residual,
        "return_residual": stats.return_residual,
    }


def _render(report: dict[str, object]) -> str:
    lines = [f"vascular rivals: {report['spec']}", ""]
    for circuit in report["circuits"]:
        cco = circuit["cco"]
        grower = circuit["grower"]
        delta = circuit["delta_response"]
        lines.append(
            f"[{circuit['region']}] terminal_pairs={circuit['terminal_pairs']}"
        )
        lines.append(
            f"  CCO    feasible={cco['feasible']} "
            f"nodes={cco['nodes']} edges={cco['edges']} "
            f"bif={cco['bifurcations']} len={cco['total_length']:.4f} "
            f"wall_margin={cco['min_wall_margin']:.3e} "
            f"t={cco['wall_seconds']:.2f}s"
        )
        if cco["obstructions"]:
            lines.append(f"         obstructions={cco['obstructions']}")
        lines.append(
            f"  GROWER feasible={grower['feasible']} "
            f"nodes={grower['nodes']} edges={grower['edges']} "
            f"bif={grower['bifurcations']} len={grower['total_length']:.4f} "
            f"wall_margin={grower['min_wall_margin']:.3e} "
            f"t={grower['wall_seconds']:.2f}s"
        )
        if grower["obstructions"]:
            lines.append(f"         obstructions={grower['obstructions']}")
        lines.append(
            f"  DELTA  cco_topo={delta['cco_topology_change']:.3f} "
            f"grower_topo={delta['grower_topology_change']:.3f} "
            f"cco_disp={delta['cco_node_displacement']:.2e} "
            f"grower_disp={delta['grower_node_displacement']:.2e} "
            f"iters cold={delta['grower_cold_iterations']} "
            f"warm={delta['grower_warm_iterations']}"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    spec_name = sys.argv[1] if len(sys.argv) > 1 else "knight_body.json"
    only_region = sys.argv[2] if len(sys.argv) > 2 else None
    report = evaluate_rivals(spec_name, only_region)
    print(_render(report))
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
