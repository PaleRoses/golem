"""Conduit subsystem v1: embedding determinism, groove safety, band
extraction, and the named first-attack risk -- re-embedding stability across
meshing resolutions (the v1 embeddings are analytic in the authored
part/station, so bands must persist and stay comparable when the mesh under
them changes)."""

import numpy as np

import engine as _e
from golem.conduits import apply_conduits
from golem.kernel import engine as E

_BLADE = {
    "name": "blade",
    "blend": 0.01,
    "parts": [
        {
            "id": "blade",
            "type": "gencyl",
            "spine": [[0, 1.3, 0], [0, 0.1, 0]],
            "radii": [0.085, 0.04],
            "profile": {"n": 7, "aspect": [0.30, 0.55], "up": [0, 0, 1]},
        }
    ],
}

_BURIED_HOST = {
    "name": "buried-host",
    "blend": 0.01,
    "parts": [
        {
            "id": "shell",
            "type": "blob",
            "center": [0.0, 0.0, 0.0],
            "size": [0.3, 0.3, 0.3],
        },
        {
            "id": "host",
            "type": "blob",
            "center": [0.0, 0.0, 0.0],
            "size": [0.08, 0.08, 0.08],
        },
    ],
}

_DECLS = [
    {"id": "rim", "kind": "axial_loop", "part": "blade", "t": 0.25,
     "width": 0.05, "depth": 0.006, "emit": ["groove", "band"],
     "material": "emissive_seam"},
    {"id": "rune", "kind": "face_line", "part": "blade", "axis": "x",
     "width": 0.03, "face_normal": [0, 0, 1], "emit": ["band"],
     "material": "emissive_seam"},
]


def _mesh(res):
    evaluated = E.evaluate(_BLADE, res=res)
    return evaluated.vertices, evaluated.faces


def test_groove_preserves_watertight_single_component():
    v, f = _mesh(120)
    v2, bands = apply_conduits(_BLADE, v, f, _DECLS)
    rep = _e.coherence_report(v2, f)
    assert rep["components"] == 1 and rep["watertight_main"]
    assert not np.allclose(v, v2)  # the groove actually displaced something


def test_bands_exist_and_are_lifted():
    v, f = _mesh(120)
    v2, bands = apply_conduits(_BLADE, v, f, _DECLS)
    assert len(bands) == 2 and not any(b["empty"] for b in bands)
    rim = next(b for b in bands if b["id"] == "rim")
    assert rim["faces"].shape[0] > 50
    # the inlay floats off the parent surface: nearest parent vertex is
    # strictly farther than zero for essentially all band vertices
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(v2).query(rim["verts"])
    assert float(np.median(dist)) > 1e-4


def test_face_line_respects_face_normal():
    v, f = _mesh(120)
    _v2, bands = apply_conduits(_BLADE, v, f, _DECLS)
    rune = next(b for b in bands if b["id"] == "rune")
    tri = rune["verts"][rune["faces"]]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    assert float((fn @ np.array([0.0, 0.0, 1.0])).min()) > 0.25


def test_face_line_projects_buried_host_along_declared_normal():
    evaluated = E.evaluate(_BURIED_HOST, res=100)

    def band_for(normal):
        _vertices, bands = apply_conduits(
            _BURIED_HOST,
            evaluated.vertices,
            evaluated.faces,
            [
                {
                    "id": "buried-line",
                    "kind": "face_line",
                    "part": "host",
                    "axis": "x",
                    "width": 0.04,
                    "face_normal": normal,
                    "emit": ["band"],
                }
            ],
        )
        return bands[0]

    positive = band_for([0.0, 0.0, 1.0])
    negative = band_for([0.0, 0.0, -1.0])

    assert not positive["empty"] and not negative["empty"]
    assert float(positive["verts"][:, 2].mean()) > 0.2
    assert float(negative["verts"][:, 2].mean()) < -0.2


def test_apply_is_deterministic():
    v, f = _mesh(100)
    a_v, a_b = apply_conduits(_BLADE, v, f, _DECLS)
    b_v, b_b = apply_conduits(_BLADE, v, f, _DECLS)
    assert np.array_equal(a_v, b_v)
    assert all(
        np.array_equal(x["verts"], y["verts"]) and np.array_equal(x["faces"], y["faces"])
        for x, y in zip(a_b, b_b)
    )


def test_reembedding_stable_across_resolution():
    """The first-attack risk: the same declarations on the same spec meshed
    at different resolutions must yield bands in the same place with
    comparable extent (band AABB centers within one coarse cell; areas within
    35 percent)."""

    def band_stats(res):
        v, f = _mesh(res)
        v2, bands = apply_conduits(_BLADE, v, f, _DECLS)
        rim = next(b for b in bands if b["id"] == "rim")
        tri = rim["verts"][rim["faces"]]
        area = float(
            np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
            / 2.0
        )
        center = rim["verts"].mean(axis=0)
        return center, area

    c1, a1 = band_stats(100)
    c2, a2 = band_stats(150)
    assert float(np.linalg.norm(c1 - c2)) < 0.02
    assert abs(a1 - a2) / max(a1, a2) < 0.35


def test_unknown_part_raises():
    v, f = _mesh(80)
    try:
        apply_conduits(_BLADE, v, f, [{"id": "x", "kind": "axial_loop",
                                       "part": "nope", "t": 0.5, "width": 0.02,
                                       "emit": ["band"]}])
        raise AssertionError("expected ValueError for unknown part")
    except ValueError as exc:
        assert "nope" in str(exc)
