"""Tests for the presentation FALLBACK backend (``presentation/native.py``)
and the D3 fence that isolates the whole presentation plane.

Presentation is evaluation infrastructure OUTSIDE the sealed environment: per
plan D3 it consumes exchange artifacts (a GLB scene + its ``.materials.json``
sidecar) and imports nothing from ``golem``. The native backend is the judged
sense wherever Blender is absent (CI, sandboxes) -- a painter's-algorithm
rasterizer that shades each element from its material record. These tests:

  * build a tiny two-element assembly (a creature blob 'body' + an equipment
    profiled-gencyl 'dagger' on a translate mount), compile it, and export the
    exchange GLB + sidecar -- the ONLY things presentation is allowed to see;
  * reload the GLB (trimesh) and assert the two named geometries carry real
    glTF PBR materials whose roughness/metallic factors match their tags, and
    that the sidecar names both elements;
  * run the fallback ``render()`` (loaded BY PATH -- presentation/ is not a
    package) at size 128 and assert it returns paths whose PNGs decode at the
    contract sizes (128x128 hero views, 256x128 hero sheet), deterministically;
  * ENFORCE the D3 fence directly on source text: the fallback and the contract
    contain no ``import golem`` / ``from golem``;
  * assert the (here un-runnable) Blender backend at least PARSES -- ``ast``
    only, no ``bpy``, no execution.

Meshing pitch is chosen so each element clamps to ``RES_MIN`` (=60); the whole
file runs in a few seconds.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

import engine  # noqa: F401  (frozen pilots bootstrap parity with the suite)
from golem import assembly as asm


_REPO = Path(__file__).resolve().parents[1]
_PRESENTATION = _REPO / "presentation"

# Expected glTF PBR factors per element, pinned. The exchange plane resolves
# these from the element's material tag (golem.materials) into the GLB slots:
#   obsidian_warden -> roughness 0.24, metallic 0.05
#   steel_violet    -> roughness 0.38, metallic 0.75
_EXPECT = {
    "body": ("obsidian_warden", 0.24, 0.05),
    "dagger": ("steel_violet", 0.38, 0.75),
}


def _tiny_assembly_spec():
    """A creature blob 'body' + an equipment profiled-gencyl 'dagger' authored
    in its own canonical frame and placed by a pure translate mount -- mirrors
    the end-to-end pattern in test_assembly at a coarser pitch."""
    body = {"name": "body", "blend": 0.06, "parts": [
        {"id": "torso", "type": "blob", "center": [0, 0.6, 0],
         "size": [0.28, 0.34, 0.24]},
    ]}
    dagger = {"name": "dagger", "blend": 0.02, "parts": [
        {"id": "blade", "type": "gencyl",
         "spine": [[0, 0.5, 0], [0, 0.02, 0]], "radii": [0.05, 0.02],
         "profile": {"n": 6, "aspect": [0.8, 0.8], "up": [0, 0, 1]}},
    ]}
    return {"name": "kit", "pitch": 0.03, "elements": [
        {"id": "body", "role": "creature", "graph": body,
         "appearance_material": "obsidian_warden"},
        {"id": "dagger", "role": "equipment", "graph": dagger,
         "appearance_material": "steel_violet",
         "mount": {"translate": [0.30, 0.60, 0.15], "rotate": [1, 0, 0, 0],
                   "scale": 1.0}},
    ]}


def _export_scene(tmp_path):
    """Compile the tiny assembly and export the exchange GLB (+ sidecar).
    Returns the GLB path. Asserts the coarse pitch keeps res small."""
    assembly = asm.compile_assembly(_tiny_assembly_spec(), tmp_path)
    assert isinstance(assembly, asm.AcceptedAssembly)
    assert all(record.resolution <= 80 for record in assembly.records)
    glb = tmp_path / "scene.glb"
    asm.export_scene(assembly, str(glb))
    return glb


def _tinted_assembly_spec():
    base = _tiny_assembly_spec()
    body = base["elements"][0]
    return {
        **base,
        "elements": [
            {
                **body,
                "surface_color": {
                    "tint_linear_rgb": [0.62, 0.22, 0.10],
                    "variation": {
                        "kind": "triplanar_value_noise",
                        "wavelength_world": 0.55,
                        "amplitude": 0.12,
                    },
                },
            }
        ],
    }


def _glb_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    json_length = int.from_bytes(payload[12:16], "little")
    return json.loads(payload[20 : 20 + json_length].decode("utf-8"))


def _load_fallback():
    """Load presentation/native.py by file path -- presentation/ is not a
    package (and imports nothing from golem, so it is loaded standalone)."""
    fp = _PRESENTATION / "native.py"
    spec = importlib.util.spec_from_file_location("golem_fallback_under_test", str(fp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1. Exchange GLB carries real PBR materials + a sidecar naming both elements. #
# --------------------------------------------------------------------------- #
def test_exchange_glb_pbr_and_sidecar(tmp_path):
    """The exported GLB reloads (trimesh) as a Scene with exactly the two named
    geometries, each carrying a real ``PBRMaterial`` whose roughness/metallic
    factors (and name) match its tag; the ``.materials.json`` sidecar exists and
    names both elements."""
    glb = _export_scene(tmp_path)
    assert glb.exists()
    sidecar = Path(str(glb) + ".materials.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert set(payload["elements"]) == {"body", "dagger"}
    assert "surface_colors" not in payload

    from trimesh.visual.material import PBRMaterial

    scene = trimesh.load(str(glb))
    assert isinstance(scene, trimesh.Scene)
    assert set(scene.geometry) == {"body", "dagger"}
    for name, geom in scene.geometry.items():
        tag, rough, metal = _EXPECT[name]
        mat = geom.visual.material
        assert isinstance(mat, PBRMaterial), f"{name} material {type(mat).__name__}"
        assert mat.name == tag
        assert float(mat.roughnessFactor) == pytest.approx(rough, abs=1e-6)
        assert float(mat.metallicFactor) == pytest.approx(metal, abs=1e-6)
    assert all(
        "color" not in geometry.visual.vertex_attributes
        for geometry in scene.geometry.values()
    )


def test_exchange_glb_bakes_deterministic_opt_in_vertex_colors(tmp_path):
    spec_path = tmp_path / "tinted.json"
    spec_path.write_text(json.dumps(_tinted_assembly_spec()), encoding="utf-8")
    first_glb = tmp_path / "first.glb"
    second_glb = tmp_path / "second.glb"
    command = (
        sys.executable,
        "-m",
        "golem",
        "compile",
        str(spec_path),
        "--res",
        "60",
    )
    first = subprocess.run(
        (*command, "--out", str(first_glb)),
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(
        (*command, "--out", str(second_glb)),
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert first_glb.read_bytes() == second_glb.read_bytes()
    first_sidecar = Path(f"{first_glb}.materials.json")
    second_sidecar = Path(f"{second_glb}.materials.json")
    assert first_sidecar.read_bytes() == second_sidecar.read_bytes()

    tree = _glb_json(first_glb)
    mesh = next(item for item in tree["meshes"] if item["name"] == "body")
    assert "COLOR_0" in mesh["primitives"][0]["attributes"]

    scene = trimesh.load(first_glb)
    geometry = scene.geometry["body"]
    colors = geometry.visual.vertex_attributes["color"]
    assert colors.shape == (len(geometry.vertices), 4)
    assert colors.dtype == np.uint8

    surface_color = json.loads(first_sidecar.read_text())["surface_colors"]["body"]
    assert surface_color["tint_linear_rgb"] == [0.62, 0.22, 0.10]
    assert surface_color["variation"]["kind"] == "triplanar_value_noise"
    assert len(surface_color["derived_seed"]) == 16
    assert int(surface_color["derived_seed"], 16) >= 0


# --------------------------------------------------------------------------- #
# 2. Fallback render (loaded by path) at size 128: paths + contract sizes.    #
# --------------------------------------------------------------------------- #
def test_fallback_render_hero_sizes(tmp_path):
    """``render(..., views='hero', size=128)`` returns paths; the two hero-view
    PNGs decode at 128x128 and the hero sheet at 256x128 (two 128-wide views
    side by side)."""
    glb = _export_scene(tmp_path)
    fb = _load_fallback()

    paths = fb.render(str(glb), str(tmp_path / "hero"), views="hero", size=128)
    assert paths and all(Path(p).exists() for p in paths)

    views = [p for p in paths if p.endswith("_az015.png") or p.endswith("_az055.png")]
    assert len(views) == 2, f"expected 2 hero views, got {paths}"
    for p in views:
        with Image.open(p) as im:
            assert im.size == (128, 128), f"{Path(p).name} is {im.size}"

    sheet = [p for p in paths if p.endswith("_sheet.png")]
    assert len(sheet) == 1
    with Image.open(sheet[0]) as im:
        assert im.size == (256, 128), f"sheet is {im.size}"


# --------------------------------------------------------------------------- #
# 3. Determinism: same GLB, two prefixes -> byte-identical az015 PNGs.        #
# --------------------------------------------------------------------------- #
def test_fallback_render_deterministic(tmp_path):
    """Rendering the same scene into two different prefixes yields byte-for-byte
    identical ``az015`` PNGs -- the fallback rasterizer is deterministic."""
    glb = _export_scene(tmp_path)
    fb = _load_fallback()

    fb.render(str(glb), str(tmp_path / "a"), views="hero", size=128)
    fb.render(str(glb), str(tmp_path / "b"), views="hero", size=128)

    a = (tmp_path / "a_az015.png").read_bytes()
    b = (tmp_path / "b_az015.png").read_bytes()
    assert a == b


# --------------------------------------------------------------------------- #
# 4. THE D3 FENCE: the presentation plane imports nothing from golem.         #
# --------------------------------------------------------------------------- #
def test_d3_fence_presentation_does_not_import_golem():
    """Source-text fence: neither the native backend nor the shared contract
    contains ``import golem`` or ``from golem`` -- presentation consumes
    exchange artifacts only (plan D3)."""
    for name in ("native.py", "contract.py"):
        src = (_PRESENTATION / name).read_text()
        assert "import golem" not in src, f"{name} imports golem"
        assert "from golem" not in src, f"{name} imports from golem"


# --------------------------------------------------------------------------- #
# 5. The (here un-runnable) Blender backend at least PARSES.                  #
# --------------------------------------------------------------------------- #
def test_blender_contract_parses():
    """``blender_contract.py`` parses as valid Python (ast only -- no bpy, no
    execution): the un-runnable backend is at least syntactically sound."""
    src = (_PRESENTATION / "blender_contract.py").read_text()
    ast.parse(src)
