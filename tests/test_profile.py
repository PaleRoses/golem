"""Tests for the GOLEM v0.5 ``profile`` vocabulary (swept-superellipse gencyl
cross-section) added to the ``engine`` wrapper.

The profile word extends the ``gencyl`` cross-section from a circle to a
superellipse (per-anchor exponent ``n``, ``aspect`` ratio, ``up`` reference).
These tests pin the three invariants that mattered during development:

  * delegation is untouched -- a no-profile v0.2 graph is still bit-identical
    through the wrapper as through the frozen engine;
  * the mirror rule (x-reflection of the spine) is *exact* for a profiled part
    whose ``up`` respects the mirror symmetry (up_x = 0);
  * a straight ``n=2, aspect=1`` profile reproduces the frozen circular gencyl
    on the segment sides (where the flat-cap combination cannot reach).

Plus the golden flat-blade artifact (the box-free flatness proof), the
interior-sign regression, and the ``vocab_violations`` rules the word adds.

Meshing resolutions are fixed and noted inline; every test runs in a few
seconds (the golden mesh is a single part at res 140).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from golem import paths as _paths
import engine  # frozen (pilots), the delegation / circular-gencyl reference
from golem.kernel import engine as ev

_GOLDEN = _paths.GOLDEN

# vocab_violations rules that the profile word owns; a clean profiled part must
# raise none of these.
_PROFILE_RULES = frozenset(
    {
        "FENCE-profile-on-nongencyl",
        "R-profile-aspect",
        "R-profile-exponent",
        "R-profile-anchor-arity",
        "R-profile-offset",
        "R-8-feature-size",
    }
)


def _sample_points(n=5000, seed=0, scale=1.5):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, 3))


def _rules(graph, res):
    """The ``(part, rule)`` pairs reported by ``vocab_violations``."""
    return {(v["part"], v["rule"]) for v in ev.vocab_violations(graph, res=res)}


def _golden_blade():
    return json.loads((_GOLDEN / "profile_blade_v1.json").read_text())


# --------------------------------------------------------------------------- #
# 1. Delegation is untouched -- a no-profile v0.2 graph stays bit-identical.   #
# --------------------------------------------------------------------------- #
def test_profile_delegation_untouched():
    """A v0.2 corpus graph (no ``profile`` anywhere) compiles to a field that is
    ``np.array_equal`` through the wrapper as through the frozen engine -- the
    profile extension must not perturb the delegated fast path. (res 90; the
    full-corpus check is the ``slow`` test in test_engine.py.)"""
    graph = json.loads((_paths.PILOTS / "golem_v4.json").read_text())
    res = 90
    _, _, _, F_ref, _, _ = engine.evaluate(graph, res=res)
    evaluated = ev.evaluate(graph, res=res)
    assert np.array_equal(F_ref, evaluated.field)


# --------------------------------------------------------------------------- #
# 2. Mirror exactness -- the spine x-reflection matches point x-reflection.    #
# --------------------------------------------------------------------------- #
def test_profile_mirror_exactness():
    """For a profiled gencyl whose ``up`` respects the mirror (up_x = 0),
    ``part_sdf(pts, part, mirrored=True)`` is *bit-exactly* equal to
    ``part_sdf(M pts, part, mirrored=False)`` with M = diag(-1,1,1) -- the
    mirrored instance is the exact geometric reflection of the original."""
    part = {
        "id": "m", "type": "gencyl", "mirror": True,
        "spine": [[0.4, 1.0, 0.1], [0.7, 0.4, 0.3]], "radii": [0.15, 0.1],
        "profile": {"n": 6, "aspect": [0.5, 0.4], "up": [0, 0, 1]},
    }
    pts = _sample_points(n=5000, seed=7)
    d_mirror = ev.part_sdf(pts, part, mirrored=True)
    d_reflected = ev.part_sdf(pts * np.array([-1.0, 1.0, 1.0]), part, mirrored=False)
    assert np.max(np.abs(d_mirror - d_reflected)) == 0.0


def test_offset_profile_mirror_exactness_for_an_arbitrary_bone_frame() -> None:
    """Derived tissue offsets reflect in their local width coordinate."""
    part = {
        "id": "offset-mirror",
        "type": "gencyl",
        "spine": [[0.3, 0.1, -0.2], [0.7, 0.9, 0.4]],
        "radii": [0.21, 0.16],
        "profile": {
            "n": [2.0, 2.0],
            "aspect": [0.62, 0.74],
            "offset": [[0.08, -0.03], [-0.04, 0.05]],
            "up": [0.31, 0.82, -0.48],
        },
    }
    points = _sample_points(n=5000, seed=71)
    reflected_points = points * np.asarray((-1.0, 1.0, 1.0))
    np.testing.assert_allclose(
        ev.part_sdf(points, part),
        ev.part_sdf(reflected_points, part, mirrored=True),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_offset_profile_bounds_cover_the_displaced_section() -> None:
    part = {
        "id": "offset-bounds",
        "type": "gencyl",
        "spine": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "radii": [0.1, 0.1],
        "profile": {
            "n": 2.0,
            "aspect": 0.5,
            "offset": [0.4, 0.0],
            "up": [0.0, 0.0, 1.0],
        },
    }
    lower, upper = ev.graph_bounds(
        {"name": "offset-bounds", "parts": [part]}, pad=0.0
    )
    assert lower[0] <= -0.5
    assert upper[0] >= -0.3


# --------------------------------------------------------------------------- #
# 3. Golden flat blade -- the box-free flatness proof (res 140).              #
# --------------------------------------------------------------------------- #
def test_profile_blade_golden_mesh():
    """The golden ``profile_blade_v1`` meshes to a single watertight component
    with no grid dust, and its compiled extents are the flat blade: wide in x,
    thin in z, long in y -- flatness a round gencyl cannot express, now bought
    on the swept spine with no box part."""
    graph = _golden_blade()
    evaluated = ev.evaluate(graph, res=140)
    verts, faces = evaluated.vertices, evaluated.faces

    rep = ev.coherence_report(verts, faces)
    assert rep["components"] == 1
    assert rep["grid_dust_slivers"] == 0
    assert rep["watertight_main"] is True

    ext = verts.max(axis=0) - verts.min(axis=0)
    assert 0.16 < ext[0] < 0.18, f"x extent {ext[0]}"
    assert 0.045 < ext[2] < 0.06, f"z extent {ext[2]}"
    assert 1.15 < ext[1] < 1.25, f"y extent {ext[1]}"


# --------------------------------------------------------------------------- #
# 4. Interior sign -- the regression that bit during development.             #
# --------------------------------------------------------------------------- #
def test_profile_interior_is_negative():
    """The SDF at the blade's spine midpoint is strictly negative (interior).
    The flat-cap axial combination once flipped this sign for profiled parts."""
    blade = next(p for p in _golden_blade()["parts"] if p["id"] == "blade")
    d = ev.part_sdf(np.array([[0.0, 0.7, 0.0]]), blade)
    assert d[0] < 0.0, f"interior SDF {d[0]} not negative"


def test_profile_internal_stations_are_glued_interior_sections() -> None:
    """Non-uniform axial anchors shape one sweep, not stacked capped solids."""
    part = {
        "id": "stationed",
        "type": "gencyl",
        "spine": [[0, 0, 0], [0, 0.2, 0], [0, 0.65, 0], [0, 1, 0]],
        "radii": [0.1, 0.12, 0.08, 0.11],
        "profile": {
            "n": [2.0, 2.4, 2.8, 3.0],
            "aspect": [0.8, 0.6, 0.7, 0.75],
            "up": [0, 0, 1],
        },
    }
    fields = ev.part_sdf(np.asarray(part["spine"][1:-1], dtype=float), part)
    np.testing.assert_allclose(fields, (-0.072, -0.056), atol=1.0e-12)


def test_profile_stations_preserve_straight_constant_sweep_and_outer_caps() -> None:
    """Subdividing a straight profile changes neither its field nor end caps."""
    stationed = {
        "id": "stationed",
        "type": "gencyl",
        "spine": [[0, 0, 0], [0, 0.13, 0], [0, 0.71, 0], [0, 1, 0]],
        "radii": [0.1, 0.1, 0.1, 0.1],
        "profile": {"n": 2.0, "aspect": 1.0, "up": [0, 0, 1]},
    }
    unsectioned = {
        **stationed,
        "id": "unsectioned",
        "spine": [[0, 0, 0], [0, 1, 0]],
        "radii": [0.1, 0.1],
    }
    points = np.asarray(
        (
            (0.02, -0.03, 0.01),
            (0.04, 0.13, -0.02),
            (-0.03, 0.42, 0.01),
            (0.01, 0.71, 0.02),
            (0.02, 1.04, -0.01),
        ),
        dtype=float,
    )
    np.testing.assert_allclose(
        ev.part_sdf(points, stationed),
        ev.part_sdf(points, unsectioned),
        atol=1.0e-12,
    )


def test_profile_bent_station_glues_to_a_finite_junction() -> None:
    part = {
        "id": "bent",
        "type": "gencyl",
        "spine": [[0, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]],
        "radii": [0.1, 0.09, 0.08],
        "profile": {
            "n": [2.0, 2.4, 2.8],
            "aspect": [0.8, 0.7, 0.75],
            "up": [0, 0, 1],
        },
    }
    points = np.asarray(
        ((0.0, 0.5, 0.0), (-0.03, 0.53, 0.01), (0.03, 0.47, -0.01)),
        dtype=float,
    )
    fields = ev.part_sdf(points, part)
    assert np.isfinite(fields).all()
    assert fields[0] < 0.0


def test_profile_raster_localization_preserves_the_authoritative_zero_band() -> None:
    part = {
        "id": "localized",
        "type": "gencyl",
        "spine": [[0, 0, 0], [0, 0.2, 0], [0, 0.72, 0], [0, 1, 0]],
        "radii": [0.1, 0.13, 0.08, 0.1],
        "profile": {
            "n": [2.0, 2.4, 2.8, 3.0],
            "aspect": [0.8, 0.65, 0.7, 0.75],
            "up": [0, 0, 1],
        },
    }
    evaluated = ev.evaluate({"name": "localized", "blend": 0.03, "parts": [part]}, res=48)
    axes = tuple(
        np.linspace(lower, upper, evaluated.resolution)
        for lower, upper in zip(evaluated.lower, evaluated.upper)
    )
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.stack(tuple(axis.ravel() for axis in grid), axis=1)
    full_field = ev.part_sdf(points, part)
    localized_field = evaluated.field.ravel()
    near_surface = np.abs(full_field) <= 2.0 * evaluated.maximum_pitch
    assert np.array_equal(localized_field <= 0.0, full_field <= 0.0)
    np.testing.assert_allclose(
        localized_field[near_surface], full_field[near_surface], atol=0.0
    )


def test_absolute_depth_profile_preserves_authored_section_axes() -> None:
    part = {
        "id": "loft-section",
        "type": "gencyl",
        "spine": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "radii": [0.3, 0.3],
        "profile": {
            "n": [6.0, 6.0],
            "depth": [0.1, 0.1],
            "roll": [0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
        },
    }
    fields = ev.part_sdf(
        np.asarray(((0.2, 0.5, 0.0), (0.0, 0.5, 0.2))),
        part,
    )

    assert fields[0] < 0.0
    assert fields[1] > 0.0


def test_profile_roll_transports_width_into_the_depth_axis() -> None:
    part = {
        "id": "rolled-loft-section",
        "type": "gencyl",
        "spine": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "radii": [0.3, 0.3],
        "profile": {
            "n": [6.0, 6.0],
            "depth": [0.1, 0.1],
            "roll": [90.0, 90.0],
            "up": [0.0, 0.0, 1.0],
        },
    }
    fields = ev.part_sdf(
        np.asarray(((0.2, 0.5, 0.0), (0.0, 0.5, 0.2))),
        part,
    )

    assert fields[0] > 0.0
    assert fields[1] < 0.0


def test_absolute_depth_roll_mirrors_exactly() -> None:
    part = {
        "id": "rolled-mirror",
        "type": "gencyl",
        "spine": [[0.3, 0.1, -0.2], [0.5, 0.6, 0.1], [0.7, 0.9, 0.4]],
        "radii": [0.24, 0.2, 0.16],
        "profile": {
            "n": [4.0, 6.0, 8.0],
            "depth": [0.13, 0.09, 0.12],
            "roll": [-25.0, 15.0, 55.0],
            "up": [0.31, 0.82, -0.48],
        },
    }
    points = _sample_points(n=2000, seed=711)
    reflected_points = points * np.asarray((-1.0, 1.0, 1.0))

    np.testing.assert_allclose(
        ev.part_sdf(points, part),
        ev.part_sdf(reflected_points, part, mirrored=True),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_absolute_depth_profile_validates_without_aspect_advisories() -> None:
    graph = {
        "name": "deep-loft",
        "blend": 0.03,
        "parts": [
            {
                "id": "deep",
                "type": "gencyl",
                "spine": [[0, 0, 0], [0, 1, 0]],
                "radii": [0.1, 0.1],
                "profile": {
                    "n": [4.0, 4.0],
                    "depth": [0.3, 0.3],
                    "roll": [0.0, 0.0],
                },
            }
        ],
    }

    assert not {
        violation["rule"]
        for violation in ev.vocab_violations(graph, res=64)
    } & {"R-profile-aspect", "R-profile-depth", "R-profile-roll"}


# --------------------------------------------------------------------------- #
# 5. Vocabulary violations owned by the profile word.                         #
# --------------------------------------------------------------------------- #
def test_profile_violations():
    res = 64

    def gencyl(profile, radii=(0.2, 0.2)):
        return {
            "name": "g", "blend": 0.03,
            "parts": [{"id": "p", "type": "gencyl",
                       "spine": [[0, 0, 0], [0, 1, 0]], "radii": list(radii),
                       "profile": profile}],
        }

    # FENCE: profile on a non-gencyl (a blob) is reported, never applied.
    blob = {"name": "g", "blend": 0.03,
            "parts": [{"id": "b", "type": "blob", "center": [0, 0.5, 0],
                       "size": [0.3, 0.3, 0.3], "profile": {"n": 4, "aspect": 0.5}}]}
    assert ("b", "FENCE-profile-on-nongencyl") in _rules(blob, res)

    # aspect outside (0, 1]: > 1 and <= 0 both trip R-profile-aspect.
    assert ("p", "R-profile-aspect") in _rules(gencyl({"n": 4, "aspect": 1.4}), res)
    assert ("p", "R-profile-aspect") in _rules(gencyl({"n": 4, "aspect": 0.0}), res)

    # exponent outside [2, 12]: below the circle and past the aliasing corner.
    assert ("p", "R-profile-exponent") in _rules(gencyl({"n": 1.5, "aspect": 0.5}), res)
    assert ("p", "R-profile-exponent") in _rules(gencyl({"n": 13, "aspect": 0.5}), res)

    # anchor arity: a 3-entry per-anchor list on a 2-anchor spine.
    assert ("p", "R-profile-anchor-arity") in _rules(
        gencyl({"n": [4, 5, 6], "aspect": 0.5}), res
    )

    assert ("p", "R-profile-anchor-arity") in _rules(
        gencyl({"n": 4, "aspect": 0.5, "offset": [[0.0, 0.0]]}), res
    )
    assert ("p", "R-profile-offset") in _rules(
        gencyl(
            {
                "n": 4,
                "aspect": 0.5,
                "offset": [[0.0, 0.0], [float("nan"), 0.0]],
            }
        ),
        res,
    )

    # feature size: 2*min(r*aspect) below 2*pitch at this res (thin z axis).
    assert ("p", "R-8-feature-size") in _rules(
        gencyl({"n": 4, "aspect": [0.02, 0.02]}, radii=(0.3, 0.3)), res
    )

    # A well-formed profiled part raises NONE of the profile-owned rules.
    clean = gencyl({"n": 6, "aspect": [0.5, 0.5], "up": [0, 0, 1]})
    clean_rules = {r for _, r in _rules(clean, res)}
    assert clean_rules & _PROFILE_RULES == set(), f"unexpected {clean_rules}"


# --------------------------------------------------------------------------- #
# 6. Circle limit -- n=2, aspect=1 matches the frozen gencyl on the sides.    #
# --------------------------------------------------------------------------- #
def test_profile_circle_matches_frozen_on_sides():
    """A straight vertical profiled gencyl with ``n=2, aspect=1`` is a circle:
    beside the segment interior (y strictly inside, radial distance > 0, where
    the flat-vs-round cap difference cannot reach) it agrees with the frozen
    circular gencyl SDF to within 1e-9."""
    part = {"id": "c", "type": "gencyl", "spine": [[0, 0, 0], [0, 1, 0]],
            "radii": [0.1, 0.1], "profile": {"n": 2.0, "aspect": 1.0, "up": [0, 0, 1]}}

    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.3, 0.3, size=(8000, 3))
    pts[:, 1] = rng.uniform(0.2, 0.8, size=8000)      # strictly beside the interior
    pts = pts[np.hypot(pts[:, 0], pts[:, 2]) > 1e-6]  # off the spine axis

    d_profile = ev.part_sdf(pts, part)
    d_frozen = engine.sdf_gencyl(
        pts, np.asarray(part["spine"], float), np.asarray(part["radii"], float)
    )
    assert np.max(np.abs(d_profile - d_frozen)) < 1e-9
