from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from golem.contract.model import ContractSection
from golem.paths import SPECS


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, list) else ()


def render() -> str:
    path = SPECS / "knight_body.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("specs/knight_body.json must contain an object")
    skeleton = _mapping(payload.get("skeleton"))
    root = _mapping(skeleton.get("root"))
    bones = _sequence(skeleton.get("bones"))
    anatomy = _mapping(payload.get("anatomy"))
    overall = _mapping(anatomy.get("overall"))
    circulation = _mapping(overall.get("circulation"))
    thigh = next(
        (
            _mapping(bone)
            for bone in bones
            if _mapping(bone).get("id") == "thigh"
        ),
        {},
    )
    if not thigh:
        raise TypeError("specs/knight_body.json requires the thigh exemplar bone")
    thigh_attach = _mapping(thigh.get("attach"))
    excerpt = {
        "name": payload.get("name"),
        "dialect": payload.get("dialect"),
        "anatomy": {
            "dialect": anatomy.get("dialect"),
            "overall": {
                "regions": list(_sequence(overall.get("regions"))[:1]),
                "circulation": {
                    "kind": circulation.get("kind"),
                    "pump_region_id": circulation.get("pump_region_id"),
                    "exchange_beds": list(
                        _sequence(circulation.get("exchange_beds"))[:1]
                    ),
                    "carrier_radius_scale": circulation.get(
                        "carrier_radius_scale"
                    ),
                    "distance_decay": circulation.get("distance_decay"),
                },
            },
        },
        "skeleton": {
            "root": root,
            "bones": [*bones[:2], thigh],
        },
        "mounts": payload.get("mounts", []),
        "pose": payload.get("pose", {}),
        "contacts": payload.get("contacts", []),
    }
    first_bone = _mapping(next(iter(bones), {})).get("id", "unknown")
    guide = "\n".join(
        (
            "GUIDED READING",
            f"Source: specs/{path.name}",
            f"Root `{root.get('id', 'unknown')}` fixes the world frame.",
            f"The ordered tree contains {len(bones)} bones; `{first_bone}` is the first restriction from the root cover.",
            f"Anatomy declares {len(_sequence(overall.get('regions')))} hosted regions and {len(_sequence(circulation.get('exchange_beds')))} exchange beds.",
            f"Its circulation block sets required finite-positive carrier_radius_scale={circulation.get('carrier_radius_scale')} and distance_decay={circulation.get('distance_decay')}.",
            f"Bone `{thigh.get('id')}` demonstrates the fixed escape-hatch `attach` form `{{t, offset: [x, y, z]}}`: t={thigh_attach.get('t')} plus parent-frame offset {json.dumps(thigh_attach.get('offset'))}. The canonical named-site and pose-relation surface is taught in RELATIONAL PLACEMENT; this legacy exemplar deliberately uses fixed coordinates.",
            "Each flesh record predicts emitted geometry in its bone frame; pose, mounts, and contacts remain separate intent.",
            "The excerpt is loaded from the live exemplar and deliberately stops before repeating the whole document.",
        )
    )
    return guide + "\n\nSTRUCTURAL EXCERPT\n" + json.dumps(
        excerpt,
        indent=2,
        sort_keys=True,
    )


SECTION = ContractSection("exemplar", render)
