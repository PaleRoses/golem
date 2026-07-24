"""Plate policy descent: frozen reproduction, typed failure, and assembly glue."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from golem import paths
from golem.assembly import AcceptedAssembly, RejectedAssembly, compile_assembly
from golem.assembly.core import PlateElementObstruction
from golem.kernel import engine
from golem.plates import (
    AcceptedPlatePolicy,
    AcceptedPlateSurface,
    PlatePolicyRule,
    RejectedPlatePolicy,
    decode_plate_policy,
    derive_plate_surface,
)


_POLICY = {
    "layout": "surface_voronoi",
    "cell_count": 12,
    "random_seed": 7,
    "relief": 0.006,
    "groove": 0.003,
    "seam_lift": 0.001,
    "seam_material": "emissive_seam",
}


def _probe_surface():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.7)
    decoded = decode_plate_policy(_POLICY)
    assert isinstance(decoded, AcceptedPlatePolicy)
    result = derive_plate_surface(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        np.asarray(mesh.vertex_normals),
        decoded.policy,
    )
    assert isinstance(result, AcceptedPlateSurface)
    return result.surface


def test_policy_decoder_accumulates_typed_local_obstructions() -> None:
    result = decode_plate_policy(
        {
            "layout": "decorative-string-soup",
            "cell_count": 2,
            "random_seed": 1.5,
            "relief": -1.0,
            "groove": float("nan"),
            "seam_lift": "high",
            "seam_material": "unknown-purple",
        }
    )
    assert isinstance(result, RejectedPlatePolicy)
    assert tuple(obstruction.address for obstruction in result.obstructions) == (
        "/layout",
        "/cell_count",
        "/random_seed",
        "/relief",
        "/groove",
        "/seam_lift",
        "/seam_material",
    )
    assert tuple(obstruction.rule for obstruction in result.obstructions) == (
        PlatePolicyRule.LAYOUT,
        PlatePolicyRule.CELL_COUNT,
        PlatePolicyRule.RANDOM_SEED,
        PlatePolicyRule.FINITE_NON_NEGATIVE,
        PlatePolicyRule.FINITE_NON_NEGATIVE,
        PlatePolicyRule.FINITE_NON_NEGATIVE,
        PlatePolicyRule.APPEARANCE_MATERIAL,
    )
    assert tuple(
        obstruction.authored
        for obstruction in (
            *result.obstructions[:4],
            *result.obstructions[5:],
        )
    ) == (
        "decorative-string-soup",
        2,
        1.5,
        -1.0,
        "high",
        "unknown-purple",
    )
    assert np.isnan(result.obstructions[4].authored)
    assert result.obstructions[1].required == {
        "integer": True,
        "minimum": 4,
        "maximum": 512,
    }


def test_surface_cover_is_deterministic_immutable_and_signed() -> None:
    first = _probe_surface()
    second = _probe_surface()
    assert np.array_equal(first.seed_indices, second.seed_indices)
    assert np.array_equal(first.vertex_cells, second.vertex_cells)
    assert np.array_equal(first.vertices, second.vertices)
    assert not first.vertices.flags.writeable
    assert len(first.adjacency) > 0
    signed_displacement = np.einsum(
        "ij,ij->i",
        first.vertices - first.source_vertices,
        np.asarray(
            trimesh.Trimesh(
                first.source_vertices, first.faces, process=False
            ).vertex_normals
        ),
    )
    assert float(np.max(signed_displacement[first.seam_vertices])) < 0.0
    assert float(np.min(signed_displacement[~first.seam_vertices])) > 0.0


def test_assembly_glues_plate_surface_as_derived_strata(tmp_path: Path) -> None:
    graph = {
        "name": "plate_probe",
        "blend": 0.02,
        "parts": [
            {
                "id": "body",
                "type": "gencyl",
                "spine": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "radii": [0.2, 0.25],
            }
        ],
    }
    result = compile_assembly(
        {
            "name": "plate_probe",
            "pitch": 0.03,
            "elements": [
                {
                    "id": "body",
                    "role": "creature",
                    "graph": graph,
                    "appearance_material": "obsidian_deep",
                    "plate_policy": _POLICY,
                }
            ],
        },
        tmp_path,
    )
    assert isinstance(result, AcceptedAssembly)
    solid = next(record for record in result.records if record.stratum == "solid")
    seams = next(
        record for record in result.records if record.stratum == "plate_seam"
    )
    assert solid.report["components"] == 1
    assert solid.report["watertight_main"] is True
    assert solid.report["plates"]["cells"] == _POLICY["cell_count"]
    assert seams.appearance_material == "emissive_seam"
    assert seams.report["faces"] > 0


def test_assembly_rejects_malformed_plate_policy_as_a_value(tmp_path: Path) -> None:
    result = compile_assembly(
        {
            "elements": [
                {
                    "id": "body",
                    "role": "creature",
                    "graph": {
                        "name": "body",
                        "parts": [
                            {
                                "id": "body",
                                "type": "blob",
                                "center": [0.0, 0.0, 0.0],
                                "size": [0.2, 0.3, 0.2],
                            }
                        ],
                    },
                    "plate_policy": {"cell_count": "many"},
                }
            ]
        },
        tmp_path,
    )
    assert isinstance(result, RejectedAssembly)
    assert len(result.obstructions) == 1
    assert isinstance(result.obstructions[0], PlateElementObstruction)


@pytest.mark.slow
def test_frozen_plate_policy_reproduces_the_write_once_cell_complex() -> None:
    graph = json.loads((paths.PILOTS / "golem_v4.json").read_text())
    evaluated = engine.evaluate(graph, res=190)
    processed = trimesh.Trimesh(
        evaluated.vertices, evaluated.faces, process=True
    )
    cleaned = trimesh.Trimesh(
        processed.vertices,
        processed.faces[processed.nondegenerate_faces()],
        process=False,
    )
    mesh = max(cleaned.split(only_watertight=False), key=lambda item: len(item.faces))
    decoded = decode_plate_policy(
        {
            "cell_count": 110,
            "random_seed": 7,
            "relief": 0.045,
            "groove": 0.02,
            "seam_lift": 0.0,
            "seam_material": "emissive_seam",
        }
    )
    assert isinstance(decoded, AcceptedPlatePolicy)
    result = derive_plate_surface(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        np.asarray(mesh.vertex_normals),
        decoded.policy,
    )
    assert isinstance(result, AcceptedPlateSurface)
    golden = json.loads((paths.OUTPUTS / "plate_complex.json").read_text())
    expected_adjacency = {
        tuple(entry["plates"]): entry["seam_edges"]
        for entry in golden["adjacency"]
    }
    actual_adjacency = {
        (entry.left_cell, entry.right_cell): entry.seam_edge_count
        for entry in result.surface.adjacency
    }
    assert np.array_equal(
        np.round(result.surface.seed_points, 3),
        np.asarray(golden["plate_centers"]),
    )
    assert actual_adjacency == expected_adjacency
