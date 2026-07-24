#!/usr/bin/env python3
"""Render only already-accepted GOLEM coupled-substrate evidence.

The assembly, thermal, mechanics, and vascular owners remain authoritative.
This module merely projects their accepted meshes and scalar samples into five
deterministic image sheets.  Nearest-cell colour is an explicitly recorded
derived view; no pressure, temperature, displacement, stress, or safety value
is solved or inferred here.  A rejected or incomplete coupled result descends
to typed render obstructions before the output boundary is entered.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import assert_never

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "pilots"), str(ROOT / "presentation")]

import native  # noqa: E402
from render import render_view  # noqa: E402

from golem.assembly.core import (  # noqa: E402
    AcceptedAssembly,
    AssemblyRecord,
    CoupledPerformanceReceipt,
    RejectedAssembly,
)
from golem.assembly.service import PhysicalEvidenceStatus  # noqa: E402
from golem.kernel.mechanics import AcceptedMechanics  # noqa: E402
from golem.kernel.sheaf import ThermalSolution  # noqa: E402
from golem.materials import (  # noqa: E402
    appearance_material_to_dict,
    resolve_appearance_material,
)


_NETWORK_AZIMUTHS = (0, 40, 90, 180)
_HERO_AZIMUTHS = (15, 55)
_SCALAR_AZIMUTH = 35
_SCALAR_CROP_FRACTION = 0.56
_DEFAULT_IMAGE_SIZE = 560
_VASCULAR_STRATA = (
    "vascular_supply",
    "vascular_return",
    "vascular_exchange",
)
_THERMAL_PALETTE = np.asarray(
    (
        (18, 38, 126),
        (28, 172, 214),
        (245, 224, 91),
        (218, 53, 49),
    ),
    dtype=np.float64,
)
_STRESS_PALETTE = np.asarray(
    (
        (31, 28, 88),
        (91, 55, 177),
        (219, 72, 136),
        (251, 173, 52),
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class ScalarProjectionReceipt:
    quantity: str
    case_id: str
    source_cell_count: int
    rendered_face_count: int
    source_minimum: float
    source_maximum: float
    projection: str = "nearest accepted cell center"


@dataclass(frozen=True)
class CoupledAcceptancePaths:
    supply_return_exchange_xray: Path
    network_only_orthographics: Path
    temperature: Path
    stress: Path
    assembled_knight: Path

    @property
    def all_paths(self) -> tuple[Path, ...]:
        return (
            self.supply_return_exchange_xray,
            self.network_only_orthographics,
            self.temperature,
            self.stress,
            self.assembled_knight,
        )


@dataclass(frozen=True)
class AcceptedCoupledRender:
    paths: CoupledAcceptancePaths
    scalar_projections: tuple[ScalarProjectionReceipt, ...]


@dataclass(frozen=True)
class AssemblyRejectedRenderObstruction:
    assembly_obstructions: tuple[object, ...]


@dataclass(frozen=True)
class MissingCoupledEvidenceObstruction:
    actual_receipt: str


@dataclass(frozen=True)
class IncompleteCoupledFieldObstruction:
    quantity: str
    receipt_count: int
    solution_count: int


@dataclass(frozen=True)
class MissingRenderStratumObstruction:
    stratum: str


@dataclass(frozen=True)
class InvalidRenderRecordObstruction:
    record_id: str
    reason: str


@dataclass(frozen=True)
class MissingElementGeometryObstruction:
    quantity: str
    element_id: str


@dataclass(frozen=True)
class InvalidImageSizeObstruction:
    image_size: int


@dataclass(frozen=True)
class DerivedRenderFailureObstruction:
    reason: str


@dataclass(frozen=True)
class OutputDirectoryExistsObstruction:
    path: Path


@dataclass(frozen=True)
class CoupledRenderWriteObstruction:
    path: Path
    reason: str


type CoupledRenderObstruction = (
    AssemblyRejectedRenderObstruction
    | MissingCoupledEvidenceObstruction
    | IncompleteCoupledFieldObstruction
    | MissingRenderStratumObstruction
    | InvalidRenderRecordObstruction
    | MissingElementGeometryObstruction
    | InvalidImageSizeObstruction
    | DerivedRenderFailureObstruction
    | OutputDirectoryExistsObstruction
    | CoupledRenderWriteObstruction
)


@dataclass(frozen=True)
class RejectedCoupledRender:
    obstructions: tuple[CoupledRenderObstruction, ...]


type CoupledRenderResult = AcceptedCoupledRender | RejectedCoupledRender


@dataclass(frozen=True)
class _DerivedMesh:
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]


@dataclass(frozen=True)
class _ScalarPanel:
    image: Image.Image
    label: str
    receipt: ScalarProjectionReceipt


@dataclass(frozen=True)
class _CoupledImages:
    supply_return_exchange_xray: Image.Image
    network_only_orthographics: Image.Image
    temperature: Image.Image
    stress: Image.Image
    assembled_knight: Image.Image
    scalar_projections: tuple[ScalarProjectionReceipt, ...]


def render_coupled_acceptance(
    assembly: AcceptedAssembly | RejectedAssembly,
    output_directory: Path,
    image_size: int = _DEFAULT_IMAGE_SIZE,
) -> CoupledRenderResult:
    """Derive and atomically publish the five coupled acceptance sheets."""
    if output_directory.exists():
        return RejectedCoupledRender(
            (OutputDirectoryExistsObstruction(output_directory),)
        )
    images = derive_coupled_acceptance_images(assembly, image_size)
    if isinstance(images, RejectedCoupledRender):
        return images
    paths = CoupledAcceptancePaths(
        output_directory / "supply_return_exchange_xray_sheet.png",
        output_directory / "network_only_orthographic_sheet.png",
        output_directory / "temperature_sheet.png",
        output_directory / "stress_sheet.png",
        output_directory / "assembled_knight_sheet.png",
    )
    named_images = tuple(
        zip(
            map(lambda path: path.name, paths.all_paths),
            (
                images.supply_return_exchange_xray,
                images.network_only_orthographics,
                images.temperature,
                images.stress,
                images.assembled_knight,
            ),
            strict=True,
        )
    )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(
            prefix=f".{output_directory.name}-", dir=output_directory.parent
        ) as staging_name:
            staging = Path(staging_name)
            tuple(
                image.save(staging / name)
                for name, image in named_images
            )
            staging.replace(output_directory)
    except (OSError, ValueError) as failure:
        return RejectedCoupledRender(
            (CoupledRenderWriteObstruction(output_directory, str(failure)),)
        )
    return AcceptedCoupledRender(paths, images.scalar_projections)


def derive_coupled_acceptance_images(
    assembly: AcceptedAssembly | RejectedAssembly,
    image_size: int = _DEFAULT_IMAGE_SIZE,
) -> _CoupledImages | RejectedCoupledRender:
    """Pure descent from accepted fields and meshes to in-memory images."""
    obstructions = _preflight_obstructions(assembly, image_size)
    if obstructions:
        return RejectedCoupledRender(obstructions)
    if not isinstance(assembly, AcceptedAssembly):
        return RejectedCoupledRender(
            (AssemblyRejectedRenderObstruction(assembly.obstructions),)
        )
    receipt = assembly.receipt
    if not isinstance(receipt, CoupledPerformanceReceipt):
        return RejectedCoupledRender(
            (MissingCoupledEvidenceObstruction(type(receipt).__name__),)
        )
    records = tuple(filter(lambda record: not record.empty, assembly.records))
    creature_solids = tuple(
        record
        for record in records
        if record.role == "creature" and record.stratum == "solid"
    )
    vascular_groups = tuple(
        tuple(filter(lambda record: record.stratum == stratum, records))
        for stratum in _VASCULAR_STRATA
    )
    try:
        xray = _derive_xray_sheet(
            creature_solids, vascular_groups, image_size
        )
        network = _derive_network_sheet(vascular_groups, image_size)
        temperature_panels = tuple(
            _temperature_panel(
                tuple(
                    record
                    for record in creature_solids
                    if record.element_id == thermal_receipt.element_id
                ),
                thermal_solution,
                thermal_receipt.case_id,
                receipt.metres_per_world_unit,
                image_size,
            )
            for thermal_receipt, thermal_solution in zip(
                receipt.thermal_receipts,
                assembly.thermal_solutions,
                strict=True,
            )
        )
        stress_panels = tuple(
            _stress_panel(
                tuple(
                    record
                    for record in creature_solids
                    if record.element_id == mechanics_receipt.element_id
                ),
                mechanics_solution,
                mechanics_receipt.case_id,
                receipt.metres_per_world_unit,
                image_size,
            )
            for mechanics_receipt, mechanics_solution in zip(
                receipt.mechanics_receipts,
                assembly.mechanics_solutions,
                strict=True,
            )
        )
        assembled = _derive_assembled_sheet(records, image_size)
    except (ValueError, TypeError, IndexError) as failure:
        return RejectedCoupledRender(
            (DerivedRenderFailureObstruction(str(failure)),)
        )
    scalar_panels = (*temperature_panels, *stress_panels)
    return _CoupledImages(
        supply_return_exchange_xray=xray,
        network_only_orthographics=network,
        temperature=_horizontal_sheet(
            tuple(panel.image for panel in temperature_panels),
            tuple(panel.label for panel in temperature_panels),
        ),
        stress=_horizontal_sheet(
            tuple(panel.image for panel in stress_panels),
            tuple(panel.label for panel in stress_panels),
        ),
        assembled_knight=assembled,
        scalar_projections=tuple(panel.receipt for panel in scalar_panels),
    )


def render_obstruction(obstruction: CoupledRenderObstruction) -> str:
    match obstruction:
        case AssemblyRejectedRenderObstruction(values):
            return f"AssemblyRejectedRender: {len(values)} assembly obstruction(s)"
        case MissingCoupledEvidenceObstruction(actual):
            return f"MissingCoupledEvidence: got {actual}"
        case IncompleteCoupledFieldObstruction(quantity, receipts, solutions):
            return (
                f"IncompleteCoupledField @{quantity}: "
                f"{receipts} receipt(s), {solutions} solution(s)"
            )
        case MissingRenderStratumObstruction(stratum):
            return f"MissingRenderStratum @{stratum}"
        case InvalidRenderRecordObstruction(record_id, reason):
            return f"InvalidRenderRecord @{record_id}: {reason}"
        case MissingElementGeometryObstruction(quantity, element_id):
            return f"MissingElementGeometry @{quantity}/element:{element_id}"
        case InvalidImageSizeObstruction(image_size):
            return f"InvalidImageSize: {image_size}"
        case DerivedRenderFailureObstruction(reason):
            return f"DerivedRenderFailure: {reason}"
        case OutputDirectoryExistsObstruction(path):
            return f"OutputDirectoryExists @{path}"
        case CoupledRenderWriteObstruction(path, reason):
            return f"CoupledRenderWrite @{path}: {reason}"
        case _ as unreachable:
            assert_never(unreachable)


def _preflight_obstructions(
    assembly: AcceptedAssembly | RejectedAssembly,
    image_size: int,
) -> tuple[CoupledRenderObstruction, ...]:
    if isinstance(assembly, RejectedAssembly):
        return (AssemblyRejectedRenderObstruction(assembly.obstructions),)
    receipt = assembly.receipt
    if not isinstance(receipt, CoupledPerformanceReceipt) or (
        receipt.physical_evidence is not PhysicalEvidenceStatus.REQUESTED
    ):
        return (MissingCoupledEvidenceObstruction(type(receipt).__name__),)
    nonempty_records = tuple(
        filter(lambda record: not record.empty, assembly.records)
    )
    record_obstructions = tuple(
        obstruction
        for record in nonempty_records
        for obstruction in _record_obstructions(record)
    )
    creature_solids = tuple(
        record
        for record in nonempty_records
        if record.role == "creature" and record.stratum == "solid"
    )
    missing_strata = tuple(
        MissingRenderStratumObstruction(stratum)
        for stratum in ("creature_solid", *_VASCULAR_STRATA)
        if (
            not creature_solids
            if stratum == "creature_solid"
            else not any(
                record.stratum == stratum for record in nonempty_records
            )
        )
    )
    field_obstructions = (
        (
            IncompleteCoupledFieldObstruction(
                "temperature",
                len(receipt.thermal_receipts),
                len(assembly.thermal_solutions),
            ),
        )
        if (
            not receipt.thermal_receipts
            or len(receipt.thermal_receipts) != len(assembly.thermal_solutions)
        )
        else ()
    ) + (
        (
            IncompleteCoupledFieldObstruction(
                "stress",
                len(receipt.mechanics_receipts),
                len(assembly.mechanics_solutions),
            ),
        )
        if (
            not receipt.mechanics_receipts
            or len(receipt.mechanics_receipts)
            != len(assembly.mechanics_solutions)
        )
        else ()
    )
    missing_geometry = tuple(
        MissingElementGeometryObstruction(quantity, element_id)
        for quantity, element_ids in (
            (
                "temperature",
                tuple(
                    item.element_id for item in receipt.thermal_receipts
                ),
            ),
            (
                "stress",
                tuple(
                    item.element_id for item in receipt.mechanics_receipts
                ),
            ),
        )
        for element_id in element_ids
        if not any(record.element_id == element_id for record in creature_solids)
    )
    size_obstructions: tuple[CoupledRenderObstruction, ...] = (
        (InvalidImageSizeObstruction(image_size),)
        if not isinstance(image_size, int) or image_size <= 0
        else ()
    )
    return (
        *size_obstructions,
        *record_obstructions,
        *missing_strata,
        *field_obstructions,
        *missing_geometry,
    )


def _record_obstructions(
    record: AssemblyRecord,
) -> tuple[InvalidRenderRecordObstruction, ...]:
    vertices = record.vertices
    faces = record.faces
    return (
        (
            InvalidRenderRecordObstruction(
                record.record_id, "missing derived geometry"
            ),
        )
        if vertices is None or faces is None
        else (
            (
                InvalidRenderRecordObstruction(
                    record.record_id,
                    f"expected vertices (n,3), got {vertices.shape}",
                ),
            )
            if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0
            else ()
        )
        + (
            (
                InvalidRenderRecordObstruction(
                    record.record_id,
                    f"expected faces (n,3), got {faces.shape}",
                ),
            )
            if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0
            else ()
        )
    )


def _derive_xray_sheet(
    creature_solids: tuple[AssemblyRecord, ...],
    vascular_groups: tuple[tuple[AssemblyRecord, ...], ...],
    image_size: int,
) -> Image.Image:
    body_elements = _native_elements(creature_solids)
    vascular_elements = tuple(map(_native_elements, vascular_groups))
    frame_elements = (*body_elements, *(item for group in vascular_elements for item in group))
    body_layer = _faint_body_layer(
        native._shade(
            body_elements,
            15,
            image_size,
            frame_elements=frame_elements,
        )
    )
    vascular_layers = tuple(
        native._shade(
            group,
            15,
            image_size,
            frame_elements=frame_elements,
        )
        for group in vascular_elements
    )
    return _horizontal_sheet(
        (
            *tuple(
                _lighten_layers(body_layer, (layer,))
                for layer in vascular_layers
            ),
            _lighten_layers(body_layer, vascular_layers),
        ),
        ("supply", "return", "exchange", "closed system"),
    )


def _derive_network_sheet(
    vascular_groups: tuple[tuple[AssemblyRecord, ...], ...],
    image_size: int,
) -> Image.Image:
    network_elements = tuple(
        item
        for group in map(_native_elements, vascular_groups)
        for item in group
    )
    return _horizontal_sheet(
        tuple(
            native._shade(
                network_elements,
                azimuth,
                image_size,
                frame_elements=network_elements,
            )
            for azimuth in _NETWORK_AZIMUTHS
        ),
        ("front 0 deg", "three-quarter 40 deg", "side 90 deg", "back 180 deg"),
    )


def _derive_assembled_sheet(
    records: tuple[AssemblyRecord, ...], image_size: int
) -> Image.Image:
    elements = _native_elements(records)
    return _horizontal_sheet(
        tuple(native._shade(elements, azimuth, image_size) for azimuth in _HERO_AZIMUTHS),
        ("assembled 15 deg", "assembled 55 deg"),
    )


def _temperature_panel(
    records: tuple[AssemblyRecord, ...],
    thermal: ThermalSolution,
    case_id: str,
    metres_per_world_unit: float,
    image_size: int,
) -> _ScalarPanel:
    mesh = _combine_records(records)
    centers = np.asarray(
        thermal.balance.problem.domain.cell_centers, dtype=np.float64
    )
    values = np.asarray(
        tuple(map(lambda value: value.value, thermal.balance.field.values)),
        dtype=np.float64,
    )
    source_minimum = float(np.min(values))
    source_maximum = float(np.max(values))
    face_values = _nearest_face_values(
        mesh, centers, values, metres_per_world_unit
    )
    image = _tight_scalar_frame(
        render_view(
            mesh.vertices,
            mesh.faces,
            _SCALAR_AZIMUTH,
            size=image_size,
            face_colors=_scalar_colours(
                face_values,
                source_minimum,
                source_maximum,
                _THERMAL_PALETTE,
            ),
        )
    )
    return _ScalarPanel(
        image,
        f"T {case_id} | {source_minimum:.2f}–{source_maximum:.2f} K",
        ScalarProjectionReceipt(
            "temperature_kelvin",
            case_id,
            len(values),
            len(mesh.faces),
            source_minimum,
            source_maximum,
        ),
    )


def _stress_panel(
    records: tuple[AssemblyRecord, ...],
    mechanics: AcceptedMechanics,
    case_id: str,
    metres_per_world_unit: float,
    image_size: int,
) -> _ScalarPanel:
    mesh = _combine_records(records)
    centers = np.asarray(
        tuple(map(lambda cell: cell.center, mechanics.cells)), dtype=np.float64
    )
    values = np.asarray(
        tuple(map(lambda cell: cell.von_mises_stress, mechanics.cells)),
        dtype=np.float64,
    )
    source_minimum = float(np.min(values))
    source_maximum = float(np.max(values))
    face_values = _nearest_face_values(
        mesh, centers, values, metres_per_world_unit
    )
    image = _tight_scalar_frame(
        render_view(
            mesh.vertices,
            mesh.faces,
            _SCALAR_AZIMUTH,
            size=image_size,
            face_colors=_scalar_colours(
                face_values, 0.0, source_maximum, _STRESS_PALETTE
            ),
        )
    )
    return _ScalarPanel(
        image,
        f"von Mises {case_id} | 0–{source_maximum:.3e} Pa",
        ScalarProjectionReceipt(
            "von_mises_stress_pascal",
            case_id,
            len(values),
            len(mesh.faces),
            source_minimum,
            source_maximum,
        ),
    )


def _combine_records(records: tuple[AssemblyRecord, ...]) -> _DerivedMesh:
    vertices = tuple(
        np.asarray(record.vertices, dtype=np.float64) for record in records
    )
    faces = tuple(np.asarray(record.faces, dtype=np.int64) for record in records)
    offsets = np.cumsum(
        np.asarray((0, *tuple(len(value) for value in vertices[:-1])), dtype=np.int64)
    )
    return _DerivedMesh(
        np.concatenate(vertices, axis=0),
        np.concatenate(
            tuple(face + offset for face, offset in zip(faces, offsets, strict=True)),
            axis=0,
        ),
    )


def _nearest_face_values(
    mesh: _DerivedMesh,
    source_centers_metres: NDArray[np.float64],
    source_values: NDArray[np.float64],
    metres_per_world_unit: float,
) -> NDArray[np.float64]:
    face_centers_metres = (
        np.mean(mesh.vertices[mesh.faces], axis=1) * metres_per_world_unit
    )
    nearest = np.asarray(
        cKDTree(source_centers_metres).query(face_centers_metres, k=1)[1],
        dtype=np.int64,
    )
    return source_values[nearest]


def _scalar_colours(
    values: NDArray[np.float64],
    minimum: float,
    maximum: float,
    palette: NDArray[np.float64],
) -> NDArray[np.uint8]:
    span = maximum - minimum
    normalized = (
        np.clip((values - minimum) / span, 0.0, 1.0)
        if span > 0.0
        else np.zeros(values.shape, dtype=np.float64)
    )
    positions = normalized * (len(palette) - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(palette) - 1)
    fraction = (positions - lower)[:, None]
    rgb = np.rint(
        palette[lower] + fraction * (palette[upper] - palette[lower])
    ).astype(np.uint8)
    return np.concatenate(
        (rgb, np.full((len(rgb), 1), 255, dtype=np.uint8)), axis=1
    )


def _native_elements(
    records: tuple[AssemblyRecord, ...],
) -> tuple[tuple[str, NDArray, NDArray, dict[str, object]], ...]:
    return tuple(
        (
            record.record_id,
            np.asarray(record.vertices, dtype=np.float64),
            np.asarray(record.faces, dtype=np.int64),
            appearance_material_to_dict(
                resolve_appearance_material(record.appearance_material)
            ),
        )
        for record in records
    )


def _tight_scalar_frame(image: Image.Image) -> Image.Image:
    """Tighten only coupled scalar panels; the frozen pilot remains untouched."""
    crop_size = max(1, int(round(image.width * _SCALAR_CROP_FRACTION)))
    left = (image.width - crop_size) // 2
    upper = (image.height - crop_size) // 2
    return image.crop(
        (left, upper, left + crop_size, upper + crop_size)
    ).resize(image.size, Image.Resampling.NEAREST)


def _font(size: int = 22) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    available = next((path for path in candidates if Path(path).is_file()), None)
    return (
        ImageFont.truetype(available, size)
        if available is not None
        else ImageFont.load_default()
    )


def _label_panel(image: Image.Image, label: str) -> Image.Image:
    panel = image.convert("RGB").copy()
    draw = ImageDraw.Draw(panel, "RGBA")
    label_width = min(panel.width - 28, max(260, 13 * len(label)))
    draw.rounded_rectangle(
        (14, 14, 14 + label_width, 51), radius=9, fill=(5, 7, 12, 210)
    )
    font_size = min(
        22,
        max(12, int((panel.width - 48) / max(0.58 * len(label), 1.0))),
    )
    draw.text((25, 20), label, fill=(238, 241, 248), font=_font(font_size))
    return panel


def _horizontal_sheet(
    images: tuple[Image.Image, ...], labels: tuple[str, ...]
) -> Image.Image:
    return Image.fromarray(
        np.concatenate(
            tuple(
                np.asarray(_label_panel(image, label))
                for image, label in zip(images, labels, strict=True)
            ),
            axis=1,
        )
    )


def _faint_body_layer(image: Image.Image) -> Image.Image:
    stage = np.asarray(native.stage_background(image.width), dtype=np.float64)
    body = np.asarray(image, dtype=np.float64)
    return Image.fromarray(
        np.clip(stage + 0.22 * (body - stage), 0.0, 255.0).astype(np.uint8)
    )


def _lighten_layers(
    base: Image.Image, layers: tuple[Image.Image, ...]
) -> Image.Image:
    return Image.fromarray(
        reduce(
            np.maximum,
            (np.asarray(layer, dtype=np.uint8) for layer in (base, *layers)),
        )
    )
