"""Pure construction of session edit programs from relational observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from golem.addressing.anchor import Anchor, RejectedAnchor
from golem.addressing.anchor_grammar import parse_anchor
from golem.addressing.session_grammar import encode_address_segment
from golem.session import ops
from golem.session.ops import _obstruct


@dataclass(frozen=True)
class EditProgram:
    operations: tuple[ops.Op, ...]


type ConstructionResult = EditProgram | ops.EditObstructed


def _program(*raw_operations: dict) -> ConstructionResult:
    decoded = tuple(ops.decode_op(raw) for raw in raw_operations)
    obstruction = next(
        (result for result in decoded if isinstance(result, ops.EditObstructed)),
        None,
    )
    return (
        obstruction
        if obstruction is not None
        else EditProgram(tuple(result for result in decoded if not isinstance(result, ops.EditObstructed)))
    )


def _anchor(source: str) -> Anchor | ops.EditObstructed:
    result = parse_anchor(source)
    match result:
        case RejectedAnchor(obstructions):
            return _obstruct(source, obstructions[0].reason)
        case Anchor():
            return result


def attach_bone(
    spec: dict,
    frames: dict | None,
    name: str,
    anchor_source: str,
    params: dict,
) -> ConstructionResult:
    anchor = _anchor(anchor_source)
    if isinstance(anchor, ops.EditObstructed):
        return anchor
    forbidden = next(
        (key for key in ("world", "center", "spine") if key in params),
        None,
    )
    if forbidden is not None:
        return _obstruct(
            f"skeleton/{encode_address_segment(name)}",
            f"attach forbids world-coordinate field {forbidden!r}",
        )
    rest_dir = params.get("dir", [0, 0, 1])
    if "world_dir" in params:
        root = spec.get("skeleton", {}).get("root", {})
        parent_id = root.get("id", "root") if anchor.bone == "root" else anchor.bone
        if frames is None or parent_id not in frames:
            return _obstruct(
                f"skeleton/{encode_address_segment(anchor.bone)}",
                f"no bone {anchor.bone!r}",
            )
        world_dir = params["world_dir"]
        if not (
            isinstance(world_dir, list)
            and len(world_dir) == 3
            and all(ops.is_number(value) for value in world_dir)
        ):
            return _obstruct(
                f"skeleton/{encode_address_segment(name)}",
                "world_dir must be a 3-vector",
            )
        vector = np.asarray(world_dir, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return _obstruct(
                f"skeleton/{encode_address_segment(name)}",
                "world_dir is zero",
            )
        rest_dir = [
            round(float(value), 6)
            for value in frames[parent_id]["R"].T @ (vector / norm)
        ]
    bone = {
        "id": name,
        "attach": (
            {"t": anchor.t, "offset": params["offset"]}
            if "offset" in params
            else {"t": anchor.t}
        ),
        "length": params.get("length", 0.3),
        "rest_dir": rest_dir,
        "joint": {
            "dof": params.get("dof", "fixed"),
            **({"limits": params["limits"]} if "limits" in params else {}),
        },
        "flesh": params.get("flesh", []),
        **({"mirror": True} if params.get("mirror") else {}),
    }
    target = (
        "skeleton/root"
        if anchor.bone == "root"
        else f"skeleton/{encode_address_segment(anchor.bone)}"
    )
    return _program({"op": "add", "kind": "bone", "addr": target, "params": bone})


def attach_flesh(bone: str, name: str, params: dict) -> ConstructionResult:
    if "center" in params or "spine" in params:
        return _obstruct(
            f"skeleton/{encode_address_segment(bone)}/"
            f"{encode_address_segment(name)}",
            "attach forbids world-coordinate fields",
        )
    return _program(
        {
            "op": "add",
            "kind": "flesh",
            "addr": f"skeleton/{encode_address_segment(bone)}",
            "params": {"name": name, **params},
        }
    )


def reflect(bone: str) -> ConstructionResult:
    return _program(
        {
            "op": "set",
            "addr": f"skeleton/{encode_address_segment(bone)}@mirror",
            "value": True,
        }
    )


def array(
    bone: str,
    name: str,
    count: int,
    along: list,
    template: dict,
) -> ConstructionResult:
    return _program(
        {
            "op": "add",
            "kind": "array",
            "addr": f"skeleton/{encode_address_segment(bone)}",
            "params": {
                "name": name,
                "n": count,
                "along": along,
                "template": template,
            },
        }
    )


def _contract_edit(spec: dict, clauses: tuple[dict, ...]) -> ops.Op | ops.EditObstructed:
    contract_reference = spec.get("contract")
    if isinstance(contract_reference, str):
        return _obstruct(
            "meta@contract",
            "contract is a frozen file reference; inline clauses are for session-authored specs",
        )
    contract = (
        contract_reference
        if isinstance(contract_reference, dict)
        else {
            "contract": f"{spec.get('name', 'session')}-envelope",
            "clauses": [],
        }
    )
    replaced = frozenset(clause["id"] for clause in clauses)
    retained = tuple(
        clause
        for clause in contract.get("clauses", [])
        if clause.get("id") not in replaced
    )
    decoded = ops.decode_op(
        {
            "op": "set",
            "addr": "meta@contract",
            "value": {**contract, "clauses": [*retained, *clauses]},
        }
    )
    return decoded


def squeeze(
    spec: dict,
    frames: dict,
    name: str,
    anchor_a_source: str,
    anchor_b_source: str,
    params: dict,
) -> ConstructionResult:
    anchor_a = _anchor(anchor_a_source)
    anchor_b = _anchor(anchor_b_source)
    obstruction = next(
        (
            result
            for result in (anchor_a, anchor_b)
            if isinstance(result, ops.EditObstructed)
        ),
        None,
    )
    if obstruction is not None:
        return obstruction
    if not isinstance(anchor_a, Anchor) or not isinstance(anchor_b, Anchor):
        return _obstruct(f"{anchor_a_source} ~ {anchor_b_source}", "invalid anchors")
    missing = next(
        (bone for bone in (anchor_a.bone, anchor_b.bone) if bone not in frames),
        None,
    )
    if missing is not None:
        return _obstruct(
            f"{anchor_a_source} ~ {anchor_b_source}",
            f"no compiled bone {missing!r}",
        )
    if frames[anchor_a.bone]["mirrored"] or frames[anchor_b.bone]["mirrored"]:
        return _obstruct(
            f"{anchor_a.bone} ~ {anchor_b.bone}",
            "squeeze between mirrored bones is not solvable one-sided; anchor the unmirrored parents instead",
        )

    def station(anchor: Anchor) -> np.ndarray:
        record = frames[anchor.bone]
        return record["head"] + record["R"] @ np.asarray(
            [0.0, 0.0, anchor.t * record["length"]]
        )

    span = station(anchor_b) - station(anchor_a)
    distance = float(np.linalg.norm(span))
    if distance < 1e-6:
        return _obstruct(
            f"{anchor_a_source} ~ {anchor_b_source}",
            "anchors coincide; nothing to squeeze",
        )
    rest_dir = [
        round(float(value), 6)
        for value in frames[anchor_a.bone]["R"].T @ (span / distance)
    ]
    bone_raw = {
        "op": "add",
        "kind": "bone",
        "addr": f"skeleton/{encode_address_segment(anchor_a.bone)}",
        "params": {
            "id": name,
            "attach": {"t": anchor_a.t},
            "length": round(distance, 6),
            "rest_dir": rest_dir,
            "joint": {"dof": params.get("dof", "fixed")},
            "flesh": params.get("flesh", []),
        },
    }
    bone_op = ops.decode_op(bone_raw)
    if isinstance(bone_op, ops.EditObstructed):
        return bone_op
    landmark = {0.0: "head", 0.5: "center", 1.0: "tail"}.get(round(anchor_b.t, 3))
    if landmark is None:
        return EditProgram((bone_op,))
    clause = {
        "id": f"squeeze_{name}",
        "scope": [f"landmark:{name}/tail", f"landmark:{anchor_b.bone}/{landmark}"],
        "metric": "landmark_distance",
        "operator": "maximum",
        "value": params.get("tol", 0.02),
        "unit": "world_unit",
    }
    contract_op = _contract_edit(spec, (clause,))
    return contract_op if isinstance(contract_op, ops.EditObstructed) else EditProgram((bone_op, contract_op))


def envelope(spec: dict, params: dict) -> ConstructionResult:
    heads = params.get("heads")
    if heads is not None and not ops.is_number(heads):
        return _obstruct("meta@contract", "envelope heads must be a number")
    head = params.get("head", "head")
    if not isinstance(head, str):
        return _obstruct("meta@contract", "envelope head must be a name")
    clauses = (
        *(
            (
                {
                    "id": "env_heads_tall",
                    "scope": [f"part:{head}"],
                    "metric": "heads_tall",
                    "operator": "within",
                    "value": [heads - 0.5, heads + 0.5],
                    "unit": "heads",
                },
            )
            if heads is not None
            else ()
        ),
        {
            "id": "env_balanced",
            "scope": ["whole", "contact:support"],
            "metric": "centroid_support_margin",
            "operator": "minimum",
            "value": params.get("margin", 0.03),
            "unit": "world_unit",
        },
        {
            "id": "env_integrity",
            "scope": ["whole"],
            "metric": "post_dust_components",
            "operator": "equal",
            "value": 1,
            "unit": "count",
        },
    )
    contract_op = _contract_edit(spec, clauses)
    return contract_op if isinstance(contract_op, ops.EditObstructed) else EditProgram((contract_op,))
