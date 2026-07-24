"""Pure coalgebra from an authored graph to immutable senses."""

from __future__ import annotations

from collections.abc import Mapping
from functools import reduce
from itertools import chain

import numpy as np

from golem.kernel import engine as ev
from golem.kernel.engine.algebra import solve_muscle_formation
from golem.kernel.engine.types import (
    Accepted,
    GencylPart,
    GeometryObstruction,
    GeometryRule,
    MuscleFormationEvidence,
    MuscleFormationObstruction,
    Rejected,
    decode_part,
)
from golem.senses.model import (
    GlobalDims,
    InstanceDatum,
    PartDatum,
    RejectedSenses,
    Senses,
    SensesObstruction,
    SensesResult,
)
from golem.senses.proprio.fusion import compute_fusion
from golem.senses.proprio.geometry import (
    PartShape,
    _part_shape,
    instance_bbox,
    part_min_radius,
    sample_instance,
)
from golem.senses.proprio.posture import (
    _balance,
    _bands,
    compute_centroid,
    compute_ground,
)


type _Instance = tuple[str, dict, bool]
type _PreparedInstance = tuple[str, dict, bool, PartShape]
type _FormationEvidenceIndex = Mapping[
    tuple[str, bool],
    MuscleFormationEvidence,
]


def _instance_list(graph: dict) -> tuple[_Instance, ...]:
    # Spanning webs (R10) are flesh topology: they join the instance fold in
    # authored order after the parts, exactly as the engine unions them.
    return tuple(
        chain.from_iterable(
            (
                ((f"{part['id']}.L", part, False), (f"{part['id']}.R", part, True))
                if part.get("mirror")
                else ((part["id"], part, False),)
            )
            for part in (*graph["parts"], *graph.get("webs", ()))
        )
    )


def _formation_preflight(
    instance: _Instance,
) -> MuscleFormationEvidence | tuple[SensesObstruction, ...]:
    _address, part, mirrored = instance
    match decode_part(part):
        case Rejected(obstructions=obstructions):
            return obstructions
        case Accepted(value=GencylPart() as decoded):
            match solve_muscle_formation(decoded, mirrored=mirrored):
                case MuscleFormationEvidence() as evidence:
                    return evidence
                case MuscleFormationObstruction() as obstruction:
                    return (obstruction,)
        case Accepted(value=decoded):
            return (
                GeometryObstruction(
                    part_id=str(part["id"]),
                    kind=str(part["type"]),
                    rule=GeometryRule.MALFORMED_PART,
                    detail=(
                        "muscle formation requires a generalized cylinder, "
                        f"got {type(decoded).__name__}"
                    ),
                    fatal=True,
                ),
            )


def _formation_evidence_index(
    instances: tuple[_Instance, ...],
) -> _FormationEvidenceIndex | RejectedSenses:
    declared_instances = tuple(
        instance
        for instance in instances
        if instance[1].get("formation") is not None
    )
    results = tuple(map(_formation_preflight, declared_instances))
    obstructions = tuple(
        obstruction
        for result in results
        if isinstance(result, tuple)
        for obstruction in result
    )
    return (
        RejectedSenses(obstructions)
        if obstructions
        else dict(
            (
                (evidence.part_id, evidence.mirrored),
                evidence,
            )
            for evidence in results
            if isinstance(evidence, MuscleFormationEvidence)
        )
    )


def _prepared_instances(
    instances: tuple[_Instance, ...],
    evidence: _FormationEvidenceIndex,
) -> tuple[_PreparedInstance, ...]:
    def prepare(instance: _Instance) -> _PreparedInstance:
        address, part, mirrored = instance
        shape = (
            evidence[(str(part["id"]), mirrored)]
            if part.get("formation") is not None
            else _part_shape(part, mirrored)
        )
        return address, part, mirrored, shape

    return tuple(map(prepare, instances))


def _instance_data(
    insts: tuple[_PreparedInstance, ...],
    graph: dict,
    diagonal: float,
) -> tuple[dict[str, InstanceDatum], np.ndarray, np.ndarray]:
    def datum(entry: _PreparedInstance) -> tuple[str, InstanceDatum]:
        address, part, mirrored, shape = entry
        lower, upper = instance_bbox(shape)
        return address, InstanceDatum(
            part=part,
            mirrored=mirrored,
            samples=sample_instance(shape),
            lo=lower,
            hi=upper,
            blend_r=part.get("blend", graph.get("blend", 0.05)) * diagonal,
            min_r=part_min_radius(shape),
            pid=part["id"],
        )

    records = tuple(map(datum, insts))
    data = dict(records)
    raw_lower = reduce(
        np.minimum,
        map(lambda item: item[1].lo, records),
        np.full(3, np.inf),
    )
    raw_upper = reduce(
        np.maximum,
        map(lambda item: item[1].hi, records),
        np.full(3, -np.inf),
    )
    return data, raw_lower, raw_upper


def _part_data(
    graph: dict,
    insts: tuple[_PreparedInstance, ...],
    inst_data: dict[str, InstanceDatum],
) -> dict[str, PartDatum]:
    def datum(part: dict) -> tuple[str, PartDatum]:
        addresses = tuple(
            address
            for address, candidate, _mirrored, _shape in insts
            if candidate is part
        )
        return part["id"], PartDatum(
            bbox_lo=np.min(tuple(inst_data[address].lo for address in addresses), axis=0),
            bbox_hi=np.max(tuple(inst_data[address].hi for address in addresses), axis=0),
            mirror=bool(part.get("mirror")),
            instances=addresses,
            min_r=min(inst_data[address].min_r for address in addresses),
        )

    return dict(map(datum, (*graph["parts"], *graph.get("webs", ()))))


def build_senses(graph: dict, intent: dict | None) -> SensesResult:
    declared_intent = intent or {}
    instances = _instance_list(graph)
    formation_evidence = _formation_evidence_index(instances)
    if isinstance(formation_evidence, RejectedSenses):
        return formation_evidence
    prepared_instances = _prepared_instances(instances, formation_evidence)
    padded_lower, padded_upper = ev.graph_bounds(graph)
    diagonal = float(np.linalg.norm(padded_upper - padded_lower))
    global_blend = graph.get("blend", 0.05) * diagonal
    inst_data, raw_lower, raw_upper = _instance_data(
        prepared_instances, graph, diagonal
    )
    parts = _part_data(graph, prepared_instances, inst_data)
    contract = (declared_intent.get("asserts") or {}).get("clauses", ())
    fusion = compute_fusion(inst_data, parts, contract)
    ground_decl = tuple(
        (declared_intent.get("ground") or {}).get("parts", ())
    )
    ground_tol = float(
        (declared_intent.get("ground") or {}).get("tolerance", 0.02)
    )
    ground, near_ground = compute_ground(parts, inst_data, ground_decl)
    centroid = compute_centroid(prepared_instances)
    balance = _balance(inst_data, parts, ground_decl, ground_tol, centroid)
    bands = tuple(_bands(parts, raw_lower, raw_upper))
    width, height, depth = (raw_upper - raw_lower).tolist()
    return Senses(
        name=graph.get("name", "?"),
        txn=graph.get("txn", 1),
        n_parts=len(graph["parts"]) + len(graph.get("webs", ())),
        n_mirror=sum(
            map(
                lambda part: int(bool(part.get("mirror"))),
                (*graph["parts"], *graph.get("webs", ())),
            )
        ),
        n_instances=len(instances),
        global_dims=GlobalDims(
            W=width,
            H=height,
            D=depth,
            HW=(height / width) if width else 0.0,
            centroid=centroid,
            k=global_blend,
        ),
        raw_lo=raw_lower,
        raw_hi=raw_upper,
        parts=parts,
        inst_data=inst_data,
        inst_gaps=fusion.instance_gaps,
        declared_pair_gap=fusion.declared_pair_gaps,
        fusion_rows=fusion.rows,
        cross_plane=fusion.cross_plane,
        n_components=fusion.component_count,
        fused_adj=fusion.adjacency,
        ground=ground,
        near_ground=near_ground,
        ground_decl=ground_decl,
        ground_tol=ground_tol,
        balance=balance,
        bands=bands,
        landmarks=declared_intent.get("landmarks", {}),
        schematic=declared_intent.get("schematic", {}),
        attach=tuple(
            tuple(pair) for pair in declared_intent.get("attach", ())
        ),
        expected_articulations=tuple(
            tuple(pair)
            for pair in declared_intent.get("expected_articulations", ())
        ),
        midline=tuple(
            str(part_id) for part_id in declared_intent.get("midline", ())
        ),
        roles={
            str(role): tuple(map(str, declared))
            for role, declared in (declared_intent.get("roles") or {}).items()
        },
        provenance=declared_intent.get("provenance", {}),
        anatomy=declared_intent.get("anatomy"),
        graph=graph,
    )
