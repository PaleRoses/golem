"""Pure mirror concretization and rigid assembly placement."""

from __future__ import annotations

import json
import numpy as np

from dataclasses import replace
from itertools import chain

from golem.kernel import engine as E
from golem.kernel.anatomy import ClosedVascularGraph


def _copied(value):
    return json.loads(json.dumps(value))


def _reflected_quaternion(rotation: object) -> list[float]:
    raw = np.asarray(rotation, dtype=np.float64)
    canonical = (
        np.array([1.0, 0.0, 0.0, 0.0])
        if float(np.linalg.norm(raw)) < 1.0e-12
        else -raw
        if raw[0] < 0.0
        else raw
    )
    w, x, y, z = canonical
    return [float(w), float(x), float(-y), float(-z)]


def _reflect_part(part: dict) -> dict:
    copied = _copied(part)
    reflected_geometry = (
        {
            **copied,
            "spine": [[-point[0], point[1], point[2]] for point in copied["spine"]],
        }
        if copied["type"] == "gencyl"
        else {
            **copied,
            "center": [
                -copied["center"][0],
                copied["center"][1],
                copied["center"][2],
            ],
        }
    )
    return (
        {
            **reflected_geometry,
            "rot": _reflected_quaternion(reflected_geometry["rot"]),
        }
        if reflected_geometry.get("rot") is not None
        else reflected_geometry
    )


def _concrete_parts(part: dict) -> tuple[dict, ...]:
    base = {key: value for key, value in part.items() if key != "mirror"}
    return (
        (
            base,
            {
                **_reflect_part(base),
                "id": f"{part['id']}_m",
            },
        )
        if part.get("mirror")
        else (base,)
    )


def concretize_mirrors(graph: dict) -> dict:
    return {
        **graph,
        **{
            key: [
                concrete
                for part in graph[key]
                for concrete in _concrete_parts(part)
            ]
            for key in ("parts", "carves")
            if key in graph
        },
    }


def _xform_point(point, rotation, scale, translation):
    return (
        rotation @ (scale * np.asarray(point, dtype=np.float64))
    ) + translation


def _mounted_profile(
    profile: dict,
    rotation: np.ndarray,
    identity_rotation: bool,
) -> dict:
    return (
        profile
        if identity_rotation
        else {
            **profile,
            "up": [
                float(coordinate)
                for coordinate in (
                    rotation
                    @ np.asarray(profile.get("up", [0.0, 0.0, 1.0]), dtype=np.float64)
                )
            ],
        }
    )


def _mounted_gencyl(
    part: dict,
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
    identity_rotation: bool,
) -> dict:
    return {
        **part,
        "spine": [
            [float(coordinate) for coordinate in _xform_point(point, rotation, scale, translation)]
            for point in part["spine"]
        ],
        "radii": [float(radius) * scale for radius in part["radii"]],
        **(
            {
                "profile": _mounted_profile(
                    part["profile"],
                    rotation,
                    identity_rotation,
                )
            }
            if part.get("profile") is not None
            else {}
        ),
    }


def _mounted_primitive(
    part: dict,
    quaternion: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
    identity_rotation: bool,
) -> dict:
    scaled = {
        **part,
        "center": [
            float(coordinate)
            for coordinate in _xform_point(part["center"], rotation, scale, translation)
        ],
        "size": [float(value) * scale for value in part["size"]],
        **(
            {"round": float(part["round"]) * scale}
            if part.get("round") is not None
            else {}
        ),
    }
    return (
        scaled
        if identity_rotation
        else {
            **scaled,
            "rot": [
                float(coordinate)
                for coordinate in E.read_quat(
                    E.quat_mul(
                        quaternion,
                        E.read_quat(part.get("rot", [1.0, 0.0, 0.0, 0.0])),
                    )
                )
            ],
        }
    )


def _mounted_part(
    part: dict,
    quaternion: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
    identity_rotation: bool,
    mirror_x: bool,
) -> dict:
    reflected = _reflect_part(part) if mirror_x else _copied(part)
    return (
        _mounted_gencyl(
            reflected,
            rotation,
            scale,
            translation,
            identity_rotation,
        )
        if reflected["type"] == "gencyl"
        else _mounted_primitive(
            reflected,
            quaternion,
            rotation,
            scale,
            translation,
            identity_rotation,
        )
    )


def _mounted_skin(skin: dict, scale: float) -> dict:
    return {
        **skin,
        "thickness": float(skin["thickness"]) * scale,
    }


def mount_graph(graph: dict, mount: dict | None) -> dict:
    if not mount:
        return graph
    concrete = (
        graph
        if not any(
            part.get("mirror")
            for part in chain(graph["parts"], graph.get("carves", ()))
        )
        else concretize_mirrors(graph)
    )
    translation = np.asarray(
        mount.get("translate", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    quaternion = E.read_quat(mount.get("rotate", [1.0, 0.0, 0.0, 0.0]))
    rotation = E.quat_to_matrix(quaternion)
    scale = float(mount.get("scale", 1.0))
    identity_rotation = bool(
        np.allclose(quaternion, [1.0, 0.0, 0.0, 0.0])
    )
    mirror_x = bool(mount.get("mirror_x", False))
    return {
        **concrete,
        **{
            key: [
                _mounted_part(
                    part,
                    quaternion,
                    rotation,
                    scale,
                    translation,
                    identity_rotation,
                    mirror_x,
                )
                for part in concrete[key]
            ]
            for key in ("parts", "carves")
            if key in concrete
        },
        **(
            {"skin": _mounted_skin(concrete["skin"], scale)}
            if "skin" in concrete
            else {}
        ),
    }


def mount_vascular_graph(
    graph: ClosedVascularGraph,
    mount: dict | None,
) -> ClosedVascularGraph:
    if not mount:
        return graph
    translation = np.asarray(
        mount.get("translate", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    rotation = E.quat_to_matrix(
        E.read_quat(mount.get("rotate", [1.0, 0.0, 0.0, 0.0]))
    )
    scale = float(mount.get("scale", 1.0))
    reflection = np.array(
        [-1.0, 1.0, 1.0]
        if bool(mount.get("mirror_x", False))
        else [1.0, 1.0, 1.0]
    )
    return replace(
        graph,
        nodes=tuple(
            replace(
                node,
                position=tuple(
                    map(
                        float,
                        rotation
                        @ (scale * (np.asarray(node.position) * reflection))
                        + translation,
                    )
                ),
            )
            for node in graph.nodes
        ),
        edges=tuple(
            replace(edge, radius=edge.radius * scale) for edge in graph.edges
        ),
    )
