"""Tests for the GOLEM assembly plane (``golem.assembly``): mirror
concretization, rigid mounts, per-element pitch resolution, the role/dialect
guard, and end-to-end multi-element compilation + GLB export.

The assembly plane composes *elements* (each a separate solid in its own field)
by rigid *mounts* -- never smooth-min. Two invariants anchor the suite:

  * concretization reproduces the engine's mirror EXACTLY -- a graph with
    ``mirror: true`` instances, expanded through ``concretize_mirrors`` and
    evaluated with no mirror flags, yields a field ``np.array_equal`` to the
    same graph run through ``engine.evaluate`` (which expands mirror itself);
  * a mount is a rigid isometry with uniform scale -- identity is a no-op,
    translation shifts bounds by exactly t, rotation is an SDF conjugation
    (composing per-part ``rot`` and rotating profile ``up``), and uniform scale
    is SDF-homogeneous (d(s*p) = s*d(p)).

Plus the D12 uniform-pitch rule (``pitch_res``), the D14 lean-subset guard
(equipment takes no body dialect), and a tiny two-element assembly meshed and
exported to a GLB scene.

Meshing resolutions are fixed and noted inline; every test runs in a few
seconds (the end-to-end elements stay at res <= 100).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

import engine  # noqa: F401  (frozen pilots bootstrap parity with the suite)
from golem import paths as _paths
from golem.kernel import engine as ev
from golem import assembly as asm
from golem.assembly.__main__ import _solid_receipt_lines
from golem.assembly.core import (
    AcceptedAssembly,
    ElementClearanceObstruction,
    ElementFitLaw,
    ElementFitObstruction,
    EquipmentBodyDialectObstruction,
    MalformedAssemblyObstruction,
    RejectedAssembly,
    _assembly_source_digest,
)


# --------------------------------------------------------------------------- #
# Small deterministic point cloud for analytic / property SDF checks.          #
# --------------------------------------------------------------------------- #
def _sample_points(n=3000, seed=0, scale=1.2):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, 3))


def _quat_about_y(theta):
    """Unit quaternion [w,x,y,z] for a rotation of ``theta`` about +y."""
    return [float(np.cos(theta / 2)), 0.0, float(np.sin(theta / 2)), 0.0]


# --------------------------------------------------------------------------- #
# 1. Concretization reproduces the engine's mirror EXACTLY.                    #
# --------------------------------------------------------------------------- #
def test_concretize_matches_engine_mirror():
    """A graph carrying a mirrored profiled gencyl AND a mirrored rot-blob,
    expanded via ``concretize_mirrors`` and evaluated with no mirror flags,
    yields a field ``np.array_equal`` to the original mirrored graph run through
    ``engine.evaluate`` (which expands mirror itself). Bounds, instance order,
    and per-point SDFs all coincide -- concretization IS the engine's mirror.
    (res 64.)"""
    graph = {
        "name": "mirror_probe",
        "blend": 0.05,
        "parts": [
            {
                "id": "wing", "type": "gencyl", "mirror": True,
                "spine": [[0.3, 1.0, 0.0], [0.7, 0.6, 0.2]], "radii": [0.16, 0.10],
                "profile": {"n": 6, "aspect": [0.5, 0.4], "up": [0, 0, 1]},
            },
            {
                "id": "pod", "type": "blob", "mirror": True,
                "center": [0.45, 0.5, 0.1], "size": [0.30, 0.22, 0.26],
                "rot": [0.8446231020639574, 0.19134171618254486,
                        0.4619397662556434, 0.19134171618254486],
            },
        ],
    }
    res = 64
    original = ev.evaluate(graph, res=res)

    concrete = asm.concretize_mirrors(graph)
    # The mirror is now two explicit parts, no mirror flags remain.
    assert [p["id"] for p in concrete["parts"]] == ["wing", "wing_m", "pod", "pod_m"]
    assert not any(p.get("mirror") for p in concrete["parts"])

    concretized = ev.evaluate(concrete, res=res)
    assert original.lower == concretized.lower
    assert original.upper == concretized.upper
    assert np.array_equal(original.field, concretized.field)


def test_carve_mirrors_concretize_before_rigid_mounts() -> None:
    graph = {
        "name": "carve_mirror_probe",
        "blend": 0.0,
        "parts": [
            {
                "id": "host",
                "type": "blob",
                "center": [0.0, 0.5, 0.0],
                "size": [0.7, 0.7, 0.7],
            },
        ],
        "carves": [
            {
                "id": "socket",
                "type": "blob",
                "mirror": True,
                "center": [0.3, 0.55, 0.55],
                "size": [0.18, 0.16, 0.2],
                "rot": [0.9238795325112867, 0.0, 0.3826834323650898, 0.0],
            },
        ],
    }
    concrete = asm.concretize_mirrors(graph)
    points = _sample_points(n=512, seed=23)
    mount = {
        "translate": [0.2, -0.1, 0.3],
        "rotate": _quat_about_y(np.pi / 2),
        "scale": 1.0,
    }

    assert [part["id"] for part in concrete["parts"]] == ["host"]
    assert [part["id"] for part in concrete["carves"]] == [
        "socket",
        "socket_m",
    ]
    assert np.array_equal(
        ev.sample_graph_field(graph, points),
        ev.sample_graph_field(concrete, points),
    )
    assert asm.mount_graph(graph, mount) == asm.mount_graph(concrete, mount)
    assert "carves" not in asm.concretize_mirrors(
        {"parts": graph["parts"]}
    )


# --------------------------------------------------------------------------- #
# 2. Identity mount is a no-op (numeric content preserved).                    #
# --------------------------------------------------------------------------- #
def test_mount_identity_is_noop():
    """A mount of translate [0,0,0], rotate [1,0,0,0], scale 1 leaves the
    graph's numeric content unchanged (compared through a JSON round-trip)."""
    graph = {
        "name": "g", "blend": 0.04,
        "parts": [
            {"id": "torso", "type": "gencyl",
             "spine": [[0.0, 0.4, 0.0], [0.0, 1.2, 0.1]], "radii": [0.40, 0.45],
             "profile": {"n": 6.0, "aspect": [0.6, 0.5], "up": [0.0, 0.0, 1.0]}},
            {"id": "head", "type": "blob", "center": [0.0, 1.4, 0.0],
             "size": [0.25, 0.25, 0.25],
             "rot": [0.9238795325112867, 0.0, 0.3826834323650898, 0.0]},
            {"id": "slab", "type": "box", "center": [0.1, 0.5, 0.0],
             "size": [0.30, 0.10, 0.18], "round": 0.03},
        ],
    }
    mounted = asm.mount_graph(
        graph, {"translate": [0.0, 0.0, 0.0], "rotate": [1.0, 0.0, 0.0, 0.0], "scale": 1.0}
    )
    assert json.loads(json.dumps(mounted)) == json.loads(json.dumps(graph))


# --------------------------------------------------------------------------- #
# 3. Translation-only mount shifts bounds by exactly t.                        #
# --------------------------------------------------------------------------- #
def test_mount_translation_moves_bounds():
    """A translate-only mount shifts ``graph_bounds`` (lo and hi) by exactly the
    translation vector; the padded span is unchanged (to 1e-12)."""
    graph = {
        "name": "g", "blend": 0.03,
        "parts": [
            {"id": "arm", "type": "gencyl",
             "spine": [[0.2, 1.0, 0.0], [0.6, 0.5, 0.2]], "radii": [0.15, 0.10]},
            {"id": "hub", "type": "blob", "center": [0.0, 0.6, 0.0],
             "size": [0.25, 0.20, 0.22]},
        ],
    }
    t = np.array([0.13, -0.27, 0.40])
    lo0, hi0 = ev.graph_bounds(graph)
    mounted = asm.mount_graph(
        graph, {"translate": t.tolist(), "rotate": [1, 0, 0, 0], "scale": 1.0}
    )
    lo1, hi1 = ev.graph_bounds(mounted)
    assert np.allclose(lo1 - lo0, t, atol=1e-12, rtol=0.0)
    assert np.allclose(hi1 - hi0, t, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# 4. Rotation mount composes rot and rotates profile up (SDF conjugation).     #
# --------------------------------------------------------------------------- #
def test_mount_rotation_composes_rot_and_up():
    """A pure-rotation mount is a rigid isometry about the origin: the mounted
    SDF equals the unmounted SDF at the inverse-rotated point.

      * box (with an existing base ``rot``): mounted_sdf(p) == unmounted_sdf(R^T p)
        -- proving the mount composed the mount quaternion onto the part's rot.
      * profiled gencyl (``up`` vector): mounted_sdf(R p) == unmounted_sdf(p)
        -- proving ``up`` was carried through the rotation.

    Both to 1e-9."""
    q = _quat_about_y(np.pi / 2)  # 90 deg about +y
    R = ev.quat_to_matrix(ev.read_quat(q))
    P = _sample_points(seed=4)

    # (a) box with an existing rot, rotated 90 deg about y.
    box = {"id": "b", "type": "box", "center": [0.2, 0.5, -0.1],
           "size": [0.30, 0.40, 0.20], "round": 0.05,
           "rot": [0.9238795325112867, 0.0, 0.3826834323650898, 0.0]}
    g_box = {"name": "g", "blend": 0.03, "parts": [box]}
    mbox = asm.mount_graph(
        g_box, {"translate": [0, 0, 0], "rotate": q, "scale": 1.0}
    )["parts"][0]
    d_mounted = ev.part_sdf(P, mbox)
    d_unmounted_inv = ev.part_sdf(P @ R, box)  # rows R^T p (inverse-rotated probe)
    assert np.allclose(d_mounted, d_unmounted_inv, atol=1e-9, rtol=0.0)

    # (b) profiled gencyl up vector, rotated 90 deg about y.
    gc = {"id": "blade", "type": "gencyl",
          "spine": [[0.0, 0.2, 0.0], [0.1, 0.9, 0.05]], "radii": [0.12, 0.06],
          "profile": {"n": 6, "aspect": [0.5, 0.4], "up": [0, 0, 1]}}
    g_gc = {"name": "g", "blend": 0.02, "parts": [gc]}
    mgc = asm.mount_graph(
        g_gc, {"translate": [0, 0, 0], "rotate": q, "scale": 1.0}
    )["parts"][0]
    d_mounted_gc = ev.part_sdf(P @ R.T, mgc)  # rows R p
    d_unmounted_gc = ev.part_sdf(P, gc)
    assert np.allclose(d_mounted_gc, d_unmounted_gc, atol=1e-9, rtol=0.0)


# --------------------------------------------------------------------------- #
# 5. Uniform-scale mount scales geometry (SDF homogeneity).                    #
# --------------------------------------------------------------------------- #
def test_mount_scale_scales_geometry():
    """A scale-2 mount doubles the padded bounds span and the gencyl radii, and
    the field is SDF-homogeneous about the origin: for a simple circular gencyl,
    d(2p) == 2*d(p) to 1e-6."""
    gc = {"id": "c", "type": "gencyl",
          "spine": [[0.1, 0.2, 0.0], [0.1, 0.9, 0.0]], "radii": [0.15, 0.15]}
    graph = {"name": "g", "blend": 0.0, "parts": [gc]}
    mounted = asm.mount_graph(
        graph, {"translate": [0, 0, 0], "rotate": [1, 0, 0, 0], "scale": 2.0}
    )
    mp = mounted["parts"][0]

    # radii doubled.
    assert np.allclose(mp["radii"], [0.30, 0.30], atol=1e-12, rtol=0.0)

    # padded bounds span doubled.
    lo0, hi0 = ev.graph_bounds(graph)
    lo1, hi1 = ev.graph_bounds(mounted)
    assert np.allclose(hi1 - lo1, 2.0 * (hi0 - lo0), atol=1e-12, rtol=0.0)

    # SDF homogeneity: d_scaled(2p) == 2 * d_base(p).
    P = _sample_points(seed=5)
    d_scaled = ev.part_sdf(2.0 * P, mp)
    d_base = ev.part_sdf(P, gc)
    assert np.allclose(d_scaled, 2.0 * d_base, atol=1e-6, rtol=0.0)


# --------------------------------------------------------------------------- #
# 6. Equipment rejects the body dialect (D14 lean-subset guard).              #
# --------------------------------------------------------------------------- #
def test_equipment_rejects_body_dialect():
    """An equipment element carrying a ``body`` key is rejected -- the lean
    subset needs no skeleton (D14). Compiling it raises a ValueError naming the
    doctrine ('D14' / 'lean')."""
    spec = {
        "name": "kit", "pitch": 0.02,
        "elements": [{"id": "x", "role": "equipment", "body": "whatever.json"}],
    }
    result = asm.compile_assembly(spec, Path("."))
    assert isinstance(result, RejectedAssembly)
    assert result.obstructions == (EquipmentBodyDialectObstruction("x"),)


def _fit_probe_spec(
    equipment_translate: list[float],
    fit: dict | None = None,
) -> dict:
    body = {
        "name": "fit-body",
        "blend": 0.0,
        "parts": [
            {
                "id": "body",
                "type": "blob",
                "center": [0.0, 0.0, 0.0],
                "size": [0.4, 0.4, 0.4],
            }
        ],
    }
    equipment = {
        "name": "fit-equipment",
        "blend": 0.0,
        "parts": [
            {
                "id": "equipment",
                "type": "blob",
                "center": [0.0, 0.0, 0.0],
                "size": [0.25, 0.25, 0.25],
            }
        ],
    }
    equipment_entry = {
        "id": "equipment",
        "role": "equipment",
        "graph": equipment,
        "mount": {"translate": equipment_translate},
        **({"fit": fit} if fit is not None else {}),
    }
    return {
        "name": "fit-probe",
        "pitch": 0.025,
        "elements": [
            {"id": "body", "role": "creature", "graph": body},
            equipment_entry,
        ],
    }


def test_default_fit_law_rejects_interpenetrating_elements(tmp_path) -> None:
    result = asm.compile_assembly(_fit_probe_spec([0.3, 0.0, 0.0]), tmp_path)

    assert isinstance(result, RejectedAssembly)
    obstruction = next(
        obstruction
        for obstruction in result.obstructions
        if isinstance(obstruction, ElementFitObstruction)
    )
    assert obstruction.left_element_id == "body"
    assert obstruction.right_element_id == "equipment"
    assert obstruction.fit_law is ElementFitLaw.DISJOINT
    assert obstruction.outside_contact_sample_count > 0
    assert obstruction.maximum_penetration_world > obstruction.tolerance_world


def test_default_fit_law_accepts_separated_elements(tmp_path) -> None:
    result = asm.compile_assembly(_fit_probe_spec([1.0, 0.0, 0.0]), tmp_path)

    assert isinstance(result, AcceptedAssembly)


def test_bounded_mount_contact_accepts_only_its_local_overlap(tmp_path) -> None:
    accepted = asm.compile_assembly(
        _fit_probe_spec(
            [0.45, 0.0, 0.0],
            {
                "law": "bounded_mount_contact",
                "with": "body",
                "radius": 0.4,
            },
        ),
        tmp_path,
    )
    rejected = asm.compile_assembly(
        _fit_probe_spec(
            [0.45, 0.0, 0.0],
            {
                "law": "bounded_mount_contact",
                "with": "body",
                "radius": 0.05,
            },
        ),
        tmp_path,
    )

    assert isinstance(accepted, AcceptedAssembly)
    assert isinstance(rejected, RejectedAssembly)
    obstruction = next(
        obstruction
        for obstruction in rejected.obstructions
        if isinstance(obstruction, ElementFitObstruction)
    )
    assert obstruction.fit_law is ElementFitLaw.BOUNDED_MOUNT_CONTACT
    assert obstruction.contact_element_id == "equipment"
    assert obstruction.outside_contact_sample_count > 0


def test_bounded_mount_contact_rejects_a_floating_element(tmp_path) -> None:
    result = asm.compile_assembly(
        _fit_probe_spec(
            [1.0, 0.0, 0.0],
            {
                "law": "bounded_mount_contact",
                "with": "body",
                "radius": 0.4,
            },
        ),
        tmp_path,
    )

    assert isinstance(result, RejectedAssembly)
    obstruction = next(
        obstruction
        for obstruction in result.obstructions
        if isinstance(obstruction, ElementClearanceObstruction)
    )
    assert obstruction.fit_law is ElementFitLaw.BOUNDED_MOUNT_CONTACT
    assert (
        obstruction.minimum_clearance_world
        > obstruction.maximum_clearance_world + obstruction.tolerance_world
    )


def test_surface_clearance_requires_nearby_disjoint_surfaces(tmp_path) -> None:
    declaration = {
        "law": "surface_clearance",
        "with": "body",
        "maximum_gap": 0.06,
    }
    fitted = asm.compile_assembly(
        _fit_probe_spec([0.66, 0.0, 0.0], declaration), tmp_path
    )
    floating = asm.compile_assembly(
        _fit_probe_spec([0.8, 0.0, 0.0], declaration), tmp_path
    )
    point_contact = asm.compile_assembly(
        _fit_probe_spec(
            [0.65, 0.0, 0.0],
            {
                "law": "surface_clearance",
                "with": "body",
                "maximum_gap": 0.001,
            },
        ),
        tmp_path,
    )

    assert isinstance(fitted, AcceptedAssembly)
    assert len(fitted.receipt.fit_receipts) == 1
    assert fitted.receipt.fit_receipts[0].fit_law is ElementFitLaw.SURFACE_CLEARANCE
    assert fitted.receipt.fit_receipts[0].minimum_clearance_world is not None
    assert fitted.receipt.fit_receipts[0].fitted_surface_fraction is not None
    assert isinstance(floating, RejectedAssembly)
    obstruction = next(
        obstruction
        for obstruction in floating.obstructions
        if isinstance(obstruction, ElementClearanceObstruction)
    )
    assert obstruction.fit_law is ElementFitLaw.SURFACE_CLEARANCE
    assert obstruction.declaring_element_id == "equipment"
    assert isinstance(point_contact, RejectedAssembly)
    point_obstruction = next(
        obstruction
        for obstruction in point_contact.obstructions
        if isinstance(obstruction, ElementClearanceObstruction)
    )
    assert (
        point_obstruction.fitted_surface_fraction
        < point_obstruction.required_fitted_surface_fraction
    )


def test_bounded_mount_contact_requires_an_explicit_mount_translation(
    tmp_path,
) -> None:
    spec = _fit_probe_spec(
        [0.45, 0.0, 0.0],
        {
            "law": "bounded_mount_contact",
            "with": "body",
            "radius": 0.4,
        },
    )
    del spec["elements"][1]["mount"]

    result = asm.compile_assembly(spec, tmp_path)

    assert isinstance(result, RejectedAssembly)
    assert result.obstructions == (
        MalformedAssemblyObstruction(
            "/elements/1/mount/translate",
            "bounded mount contact requires a finite three-coordinate translation",
        ),
    )


def test_cache_digest_tracks_body_owned_file_dependencies(tmp_path):
    body_path = tmp_path / "body.json"
    contract_path = tmp_path / "contract.json"
    body_path.write_text(
        json.dumps(
            {
                "dialect": "body/0.3",
                "contract": contract_path.name,
                "mounts": [],
            }
        )
    )
    spec = {
        "elements": [
            {"id": "body", "role": "creature", "body": body_path.name}
        ]
    }
    missing_digest = _assembly_source_digest(spec, tmp_path)
    contract_path.write_text('{"contract":"probe","clauses":[]}')
    present_digest = _assembly_source_digest(spec, tmp_path)
    contract_path.write_text('{"contract":"revised","clauses":[]}')
    revised_digest = _assembly_source_digest(spec, tmp_path)
    assert len({missing_digest, present_digest, revised_digest}) == 3


# --------------------------------------------------------------------------- #
# 7. Per-element pitch resolution (D12 uniform-pitch rule).                    #
# --------------------------------------------------------------------------- #
def test_pitch_res_uniform_pitch():
    """``pitch_res`` sizes each element so its implied world pitch (max padded
    span / (res - 1)) tracks the target pitch to within 15%, and the larger
    element gets the larger res. Both stay inside the [RES_MIN, RES_MAX] clamp."""
    pitch = 0.02
    small = {"name": "s", "blend": 0.03,
             "parts": [{"id": "p", "type": "gencyl",
                        "spine": [[0, 0, 0], [0, 1.0, 0]], "radii": [0.10, 0.10]}]}
    big = {"name": "b", "blend": 0.03,
           "parts": [{"id": "p", "type": "gencyl",
                      "spine": [[0, 0, 0], [0, 2.4, 0]], "radii": [0.25, 0.25]}]}

    res_small = asm.pitch_res(small, pitch)
    res_big = asm.pitch_res(big, pitch)

    for graph, res in ((small, res_small), (big, res_big)):
        lo, hi = ev.graph_bounds(graph)
        span = float(np.max(hi - lo))
        implied = span / (res - 1)
        assert abs(implied - pitch) <= 0.15 * pitch, (
            f"implied pitch {implied} off target {pitch} by > 15%"
        )
        assert asm.RES_MIN <= res <= asm.RES_MAX  # clamp respected

    assert res_big > res_small


# --------------------------------------------------------------------------- #
# 8. End-to-end: two-element assembly compiles and exports a GLB scene.        #
# --------------------------------------------------------------------------- #
def test_assembly_end_to_end(tmp_path):
    """A tiny two-element assembly -- a creature 'body' (blob + gencyl) and an
    equipment 'dagger' (a profiled gencyl authored in its canonical frame,
    placed by a translate mount) -- compiles to per-element reports that are
    each a single watertight component, and ``export_scene`` writes a GLB that
    trimesh reloads as a Scene with exactly the two named geometries. Pitch is
    chosen so each element's res stays <= 100 (runs in seconds)."""
    body = {"name": "body", "blend": 0.06, "parts": [
        {"id": "torso", "type": "blob", "center": [0, 0.6, 0],
         "size": [0.28, 0.34, 0.24]},
        {"id": "neck", "type": "gencyl",
         "spine": [[0, 0.9, 0], [0, 1.15, 0]], "radii": [0.10, 0.09]},
    ]}
    # Dagger authored in its own canonical frame (grip base at origin, blade
    # down +y), placed onto the body's flank by a pure translate mount.
    dagger = {"name": "dagger", "blend": 0.02, "parts": [
        {"id": "blade", "type": "gencyl",
         "spine": [[0, 0.5, 0], [0, 0.02, 0]], "radii": [0.05, 0.02],
         "profile": {"n": 6, "aspect": [0.8, 0.8], "up": [0, 0, 1]}},
    ]}
    spec = {"name": "kit", "pitch": 0.02, "elements": [
        {"id": "body", "role": "creature", "graph": body,
         "appearance_material": "obsidian_warden"},
        {"id": "dagger", "role": "equipment", "graph": dagger,
         "appearance_material": "steel_violet",
         "mount": {"translate": [0.30, 0.60, 0.15], "rotate": [1, 0, 0, 0],
                   "scale": 1.0},
         "fit": {"law": "surface_clearance", "with": "body",
                 "maximum_gap": 0.05}},
    ]}

    assembly = asm.compile_assembly(spec, tmp_path)
    assert isinstance(assembly, AcceptedAssembly)
    assert tuple(record.record_id for record in assembly.records) == (
        "body",
        "dagger",
    )
    assert all(record.resolution <= 100 for record in assembly.records)
    assert all(
        record.report["components"] == 1
        and record.report["watertight_main"] is True
        for record in assembly.records
    )
    assert assembly.receipt.physical_evidence.value == "not_requested"

    out = tmp_path / "scene.glb"
    asm.export_scene(assembly, str(out))
    assert out.exists()

    payload = out.read_bytes()
    document_length = int.from_bytes(payload[12:16], byteorder="little")
    document = json.loads(payload[20 : 20 + document_length])
    attributes = tuple(
        primitive["attributes"]
        for mesh in document["meshes"]
        for primitive in mesh["primitives"]
    )
    assert all("NORMAL" in attribute for attribute in attributes)

    scene = trimesh.load(str(out), process=False)
    assert isinstance(scene, trimesh.Scene)
    assert set(scene.geometry.keys()) == {"body", "dagger"}
    assert len(scene.geometry) == 2
    record_by_id = {record.record_id: record for record in assembly.records}
    assert all(
        np.array_equal(geometry.faces, record_by_id[name].faces)
        for name, geometry in scene.geometry.items()
    )
    assert all(
        np.array_equal(
            geometry.vertices,
            record_by_id[name].vertices.astype(np.float32),
        )
        for name, geometry in scene.geometry.items()
    )
    assert all(
        np.allclose(
            geometry.vertex_normals,
            trimesh.Trimesh(
                vertices=record_by_id[name].vertices,
                faces=record_by_id[name].faces,
                process=False,
            ).vertex_normals,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        for name, geometry in scene.geometry.items()
    )


def test_body_entry_emits_deterministic_closed_vascular_strata():
    """The canonical body entry keeps typed anatomy through assembly descent.

    Two compiles produce identical supported skin and three diagnostic strata;
    the authoritative receipt clears flow, demand, symmetry, and displacement
    bounds without treating those strata as separate solids.
    """
    base_dir = _paths.KERNEL_ROOT / "rehearsal" / "reforge"
    body_entry = {
        "id": "body",
        "role": "creature",
        "body": "../../specs/knight_body.json",
        "appearance_material": "gambeson_dark",
    }
    spec = {"name": "vascular-body", "pitch": 0.0171, "elements": [body_entry]}
    first = asm.compile_assembly(spec, base_dir)
    second = asm.compile_assembly(spec, base_dir)
    assert isinstance(first, AcceptedAssembly)
    assert isinstance(second, AcceptedAssembly)
    assert tuple(record.record_id for record in first.records) == (
        "body",
        "body__vascular_supply",
        "body__vascular_return",
        "body__vascular_exchange",
    )
    assert all(
        left.report == right.report
        and np.array_equal(left.vertices, right.vertices)
        and np.array_equal(left.faces, right.faces)
        for left, right in zip(first.records, second.records)
    )
    assert all(
        record.vertices.flags.writeable is False for record in first.records
    )
    original_face_count = second.records[0].report["faces"]
    with pytest.raises(TypeError):
        first.records[0].report["faces"] = -1
    third = asm.compile_assembly(spec, base_dir)
    assert isinstance(third, AcceptedAssembly)
    assert third.records[0].report["faces"] == original_face_count
    with pytest.raises(ValueError):
        first.records[0].vertices[0, 0] = 0.0
    body_record = third.records[0]
    report = body_record.report
    vascular = report["vasculature"]
    assert report["components"] == 1
    assert report["watertight_main"] is True
    assert body_record.violations == ()
    assert vascular["topology"] == {
        "nodes": 1060,
        "edges": 1314,
        "terminal_pairs": 256,
    }
    assert vascular["maximum_free_cell_residual"] <= 1.0e-10
    assert vascular["boundary_balance_error"] <= 1.0e-10
    assert vascular["maximum_delivery_relative_error"] <= 1.0e-3
    assert vascular["maximum_symmetry_error"] == 0.0
    assert vascular["checked_nonincident_pair_count"] > 0
    assert tuple(
        record.appearance_material for record in third.records[1:]
    ) == (
        "vascular_supply_gold",
        "vascular_return_violet",
        "vascular_exchange_cyan",
    )
    receipt_text = "\n".join(_solid_receipt_lines(body_record))
    assert all(
        label in receipt_text
        for label in (
            "allocation",
            "stages",
            "capsule_margin",
            (
                "intersections=0 "
                f"({vascular['checked_nonincident_pair_count']} pairs checked)"
            ),
            "delivery",
            "conservation",
        )
    )


def test_unknown_appendage_intent_populates_closest_candidates():
    from golem.assembly.compile import _resolve_appendage_intents
    from golem.assembly.obstructions import UnknownAppendageIntentObstruction

    close = _resolve_appendage_intents(
        "elem",
        {
            "_intents": (
                {"id": "grasp"},
                {"id": "brace"},
                {"id": "reach"},
            ),
            "appendages": (
                {"id": "left_hand", "intent_refs": ("grsap",)},
            ),
        },
    )
    assert isinstance(close, UnknownAppendageIntentObstruction)
    assert close.intent_id == "grsap"
    assert close.closest_valid_candidates == ("grasp",)

    fallback = _resolve_appendage_intents(
        "elem",
        {
            "_intents": ({"id": "alpha"}, {"id": "beta"}),
            "appendages": (
                {"id": "left_hand", "intent_refs": ("zzzzzzzz",)},
            ),
        },
    )
    assert isinstance(fallback, UnknownAppendageIntentObstruction)
    assert fallback.closest_valid_candidates == ("alpha", "beta")
