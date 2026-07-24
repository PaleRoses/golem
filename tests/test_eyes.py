from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from trimesh.visual.material import PBRMaterial

from golem import assembly
from golem import paths as _paths
from golem.kernel import body
from golem.kernel import engine


_DEMO_SPEC = _paths.SPECS / "eye_demo_head.json"
_RECORD_IDS = (
    "head",
    "head__eye_watcher",
    "head__eye_watcher_m",
)
_EYE_RECORD_IDS = _RECORD_IDS[1:]


def _read_demo_spec() -> dict:
    return json.loads(_DEMO_SPEC.read_text(encoding="utf-8"))


def _require_compiled_body(spec: dict) -> body.CompiledBody:
    result = body.Compiler(spec, spec_dir=_DEMO_SPEC.parent).compile()
    assert isinstance(result, body.CompiledBody), getattr(
        result, "obstructions", ()
    )
    return result


def _canonical_body_product(compiled: body.CompiledBody) -> bytes:
    return json.dumps(
        {"graph": compiled.graph, "receipt": compiled.receipt},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_absent_and_explicit_empty_eyes_preserve_body_product_bytes() -> None:
    demo_spec = _read_demo_spec()
    absent = {key: value for key, value in demo_spec.items() if key != "eyes"}
    explicit_empty = {**absent, "eyes": []}
    assert _canonical_body_product(
        _require_compiled_body(absent)
    ) == _canonical_body_product(_require_compiled_body(explicit_empty))


def test_bilateral_eye_projection_is_exact_sagittal_reflection() -> None:
    compiled = _require_compiled_body(_read_demo_spec())
    projected = assembly.concretize_mirrors(
        body.eye_globe_graph(compiled.eye_globes[0])
    )
    authored, mirrored = projected["parts"]
    assert (authored["id"], mirrored["id"]) == ("watcher", "watcher_m")
    assert mirrored["center"] == [
        -authored["center"][0],
        authored["center"][1],
        authored["center"][2],
    ]
    assert mirrored["size"] == authored["size"]


def test_eye_sections_descend_to_globe_socket_and_optional_brow() -> None:
    compiled = _require_compiled_body(_read_demo_spec())
    assert tuple(part["id"] for part in compiled.graph["parts"]) == (
        "skull",
        "eye.watcher.brow",
    )
    assert tuple(carve["id"] for carve in compiled.graph["carves"]) == (
        "eye.watcher.socket",
    )
    assert compiled.graph["carves"][0]["mirror"] is True
    assert compiled.graph["intent"]["provenance"]["eye.watcher.brow"] == (
        "eyes[0]/brow_ridge"
    )


def test_socket_recess_changes_skull_interior_to_exterior() -> None:
    compiled = _require_compiled_body(_read_demo_spec())
    carve_center = np.asarray((compiled.graph["carves"][0]["center"],))
    baseline_graph = {
        key: value for key, value in compiled.graph.items() if key != "carves"
    }
    assert float(engine.sample_graph_field(baseline_graph, carve_center)[0]) < 0.0
    assert float(engine.sample_graph_field(compiled.graph, carve_center)[0]) > 0.0


@pytest.mark.parametrize(
    ("eye_override", "obstruction_type"),
    (
        ({"host_bone_id": "missing-head"}, body.UnknownEyeHostObstruction),
        (
            {"appearance_material": "obsidian_warden"},
            body.NonGlossyEyeMaterialObstruction,
        ),
    ),
)
def test_invalid_eye_context_is_a_typed_rejection(
    eye_override: dict,
    obstruction_type: type,
) -> None:
    demo_spec = _read_demo_spec()
    rejected_spec = {
        **demo_spec,
        "eyes": [{**demo_spec["eyes"][0], **eye_override}],
    }
    result = body.Compiler(rejected_spec, spec_dir=_DEMO_SPEC.parent).compile()
    assert isinstance(result, body.RejectedBody)
    assert tuple(map(type, result.obstructions)) == (obstruction_type,)


def test_unknown_eye_materials_are_reported_as_one_batch() -> None:
    demo_spec = _read_demo_spec()
    authored_eye = demo_spec["eyes"][0]
    rejected_spec = {
        **demo_spec,
        "eyes": [
            {
                **authored_eye,
                "id": "watcher_left",
                "mirror": False,
                "appearance_material": "verdigris_glass",
            },
            {
                **authored_eye,
                "id": "watcher_right",
                "mirror": False,
                "appearance_material": "moonstone",
            },
        ],
    }
    result = body.Compiler(
        rejected_spec,
        spec_dir=_DEMO_SPEC.parent,
    ).compile()

    assert isinstance(result, body.RejectedBody)
    assert result.obstructions == (
        body.UnknownEyeMaterialObstruction(
            "/eyes/0/appearance_material",
            "verdigris_glass",
        ),
        body.UnknownEyeMaterialObstruction(
            "/eyes/1/appearance_material",
            "moonstone",
        ),
    )


def test_mirrored_eye_uses_world_sagittal_side_and_reserves_mirrored_ids() -> None:
    demo_spec = _read_demo_spec()
    off_sagittal_host = {
        **demo_spec,
        "skeleton": {
            **demo_spec["skeleton"],
            "root": {
                **demo_spec["skeleton"]["root"],
                "world": [-0.2, 0.22, 0.0],
            },
        },
    }
    sagittal_rejection = body.Compiler(
        off_sagittal_host,
        spec_dir=_DEMO_SPEC.parent,
    ).compile()
    assert isinstance(sagittal_rejection, body.RejectedBody)
    assert tuple(map(type, sagittal_rejection.obstructions)) == (
        body.MalformedEyeObstruction,
    )

    colliding_spec = {
        **demo_spec,
        "skeleton": {
            **demo_spec["skeleton"],
            "root": {
                **demo_spec["skeleton"]["root"],
                "flesh": [
                    *demo_spec["skeleton"]["root"]["flesh"],
                    {
                        "kind": "blob",
                        "name": "eye.watcher.brow_m",
                        "t": 0.0,
                        "offset": [0.0, 0.0, 0.0],
                        "size": [0.01, 0.01, 0.01],
                    },
                ],
            },
        },
    }
    collision_rejection = body.Compiler(
        colliding_spec,
        spec_dir=_DEMO_SPEC.parent,
    ).compile()
    assert isinstance(collision_rejection, body.RejectedBody)
    assert collision_rejection.obstructions == (
        body.EyePartCollisionObstruction(
            "/eyes/0",
            "eye.watcher.brow_m",
        ),
    )


def test_eye_records_export_as_named_glossy_pbr_geometries(
    tmp_path: Path,
) -> None:
    assembly_spec = {
        "name": "eye-demo",
        "elements": [
            {
                "id": "head",
                "role": "creature",
                "body": _DEMO_SPEC.name,
                "appearance_material": "obsidian_warden",
            }
        ],
    }
    result = assembly.compile_assembly(
        assembly_spec,
        _DEMO_SPEC.parent,
        assembly.PinnedResolution(60),
    )
    assert isinstance(result, assembly.AcceptedAssembly), getattr(
        result, "obstructions", ()
    )
    assert tuple(map(lambda record: record.record_id, result.records)) == _RECORD_IDS
    assert tuple(map(lambda record: record.stratum, result.records)) == (
        assembly.AssemblyStratum.SOLID,
        assembly.AssemblyStratum.EYE,
        assembly.AssemblyStratum.EYE,
    )
    assert tuple(map(lambda record: record.appearance_material, result.records)) == (
        "obsidian_warden",
        "eye_gloss",
        "eye_gloss",
    )
    assert result.records[0].resolution == 60
    assert result.records[0].violations == ()

    output = tmp_path / "eye_demo_head.glb"
    assembly.export_scene(result, str(output))
    scene = trimesh.load(output, process=False)
    assert isinstance(scene, trimesh.Scene)
    assert frozenset(scene.geometry) == frozenset(_RECORD_IDS)
    eye_materials = tuple(
        map(lambda record_id: scene.geometry[record_id].visual.material, _EYE_RECORD_IDS)
    )
    assert all(map(lambda material: isinstance(material, PBRMaterial), eye_materials))
    assert tuple(map(lambda material: material.name, eye_materials)) == (
        "eye_gloss",
        "eye_gloss",
    )
    assert tuple(map(lambda material: material.roughnessFactor, eye_materials)) == (
        0.08,
        0.08,
    )

    sidecar = json.loads(
        Path(f"{output}.materials.json").read_text(encoding="utf-8")
    )
    assert sidecar["elements"] == {
        "head": "obsidian_warden",
        "head__eye_watcher": "eye_gloss",
        "head__eye_watcher_m": "eye_gloss",
    }
    assert sidecar["records"]["eye_gloss"]["roughness"] == 0.08
