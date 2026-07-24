from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from golem.assembly import AcceptedAssembly, PinnedResolution
from golem.cli.compile import (
    compilation_obstruction_result,
    compile_spec,
)
from golem.cli.model import (
    CommandDescriptor,
    CommandResult,
    CommandRunner,
    OptionArgument,
    PositionalArgument,
    SwitchArgument,
)
from golem.session.fault import (
    ASSEMBLY_COMPILE_SEAM,
    ORTHOGRAPHIC_RENDER_SEAM,
    ORTHOGRAPHIC_WRITE_SEAM,
    project_engine_fault,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from golem.assembly import AssemblyRecord
    from golem.materials.surface import SeededSurfaceColor, SurfaceClass
    from golem.senses.orthographic import (
        AcceptedOrthographicWrite,
        OrthographicImage,
        OrthographicRenderResult,
        OrthographicRenderObstruction,
        OrthographicView,
        OrthographicWriteObstruction,
        OrthographicWriteResult,
        _CombinedGeometry,
        _RenderableRecord,
    )


class LookView(StrEnum):
    FRONT = "front"
    SIDE = "side"
    TOP = "top"
    ALL = "all"


@dataclass(frozen=True)
class LookArguments:
    spec_path: Path
    view: LookView
    resolution: int | None
    image_size: int | None
    shaded: bool
    output_directory: Path


def _look_arguments(namespace: Namespace) -> LookArguments:
    return LookArguments(
        spec_path=namespace.spec,
        view=namespace.view,
        resolution=namespace.resolution,
        image_size=namespace.image_size,
        shaded=namespace.shaded,
        output_directory=namespace.output_directory,
    )


def _selected_views(view: LookView) -> tuple[OrthographicView, ...]:
    # Lazy: the orthographic stack (PIL) must stay optional for `--help`.
    from golem.senses.orthographic import OrthographicView

    match view:
        case LookView.FRONT:
            return (OrthographicView.FRONT,)
        case LookView.SIDE:
            return (OrthographicView.SIDE,)
        case LookView.TOP:
            return (OrthographicView.TOP,)
        case LookView.ALL:
            return tuple(OrthographicView)
        case _ as unreachable:
            assert_never(unreachable)


def _render_obstruction_result(
    obstruction: OrthographicRenderObstruction,
) -> CommandResult:
    return CommandResult(
        1,
        stderr=(
            "REJECTED [OrthographicRenderObstruction] "
            f"{obstruction.address}: {obstruction.reason}\n"
        ),
    )


def _engine_fault_refusal(
    arguments: LookArguments,
    seam: str,
    failure: Exception,
) -> CommandResult:
    fault = project_engine_fault(seam, failure)
    return CommandResult(
        1,
        stderr=(
            fault.render_refusal()
            + f"render refused -> {arguments.output_directory} "
            "(typed refusal above)\n"
        ),
    )


def _has_palette_surface_color(record: AssemblyRecord) -> bool:
    from golem.materials.surface import SurfaceColorOwner

    return (
        record.surface_color is not None
        and record.surface_color.owner is SurfaceColorOwner.APPEARANCE_PALETTE
    )


def _record_surface_class(record: AssemblyRecord) -> SurfaceClass:
    from golem.materials.surface import OpaqueSurfaceClass

    return (
        record.surface_color.surface_class
        if _has_palette_surface_color(record)
        else OpaqueSurfaceClass()
    )


def _has_surface_classes(records: tuple[AssemblyRecord, ...]) -> bool:
    from golem.materials.surface import OpaqueSurfaceClass

    return any(
        not isinstance(_record_surface_class(record), OpaqueSurfaceClass)
        for record in records
    )


def _palette_face_colors(
    renderable: _RenderableRecord,
    surface_color: SeededSurfaceColor,
) -> NDArray[np.uint8]:
    import numpy as np
    import trimesh

    from golem.materials.projection import (
        project_surface_color_to_face_srgb_bytes,
    )
    from golem.materials.surface import EmissiveSurfaceClass

    surface_class = surface_color.surface_class
    projected_color = (
        replace(
            surface_color,
            recipe=replace(
                surface_color.recipe,
                tint_linear_rgb=tuple(
                    component * surface_class.intensity
                    for component in surface_color.recipe.tint_linear_rgb
                ),
            ),
        )
        if isinstance(surface_class, EmissiveSurfaceClass)
        else surface_color
    )
    mesh = trimesh.Trimesh(
        renderable.vertices,
        renderable.faces,
        process=False,
    )
    return project_surface_color_to_face_srgb_bytes(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(mesh.vertex_normals, dtype=np.float64),
        projected_color,
    )


def _record_face_colors(
    renderable: _RenderableRecord,
    fallback: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    surface_color = renderable.record.surface_color
    return (
        _palette_face_colors(renderable, surface_color)
        if surface_color is not None
        and _has_palette_surface_color(renderable.record)
        else fallback
    )


def _surface_colored_geometry(
    records: tuple[AssemblyRecord, ...],
) -> _CombinedGeometry | OrthographicRenderObstruction:
    import numpy as np

    from golem.senses import orthographic

    geometry = orthographic._combine_records(records)
    if isinstance(geometry, orthographic.OrthographicRenderObstruction):
        return geometry
    visible = tuple(
        filter(
            lambda candidate: candidate is not None,
            map(orthographic._renderable_record, records),
        )
    )
    face_counts = tuple(map(lambda record: len(record.faces), visible))
    fallback_sections = tuple(
        np.split(
            geometry.face_colors,
            tuple(accumulate(face_counts))[:-1],
        )
    )
    return replace(
        geometry,
        face_colors=np.concatenate(
            tuple(
                map(
                    lambda pair: _record_face_colors(*pair),
                    zip(visible, fallback_sections, strict=True),
                )
            )
        ),
    )


@dataclass(frozen=True)
class _ClassedGeometry:
    geometry: _CombinedGeometry
    emissive_faces: NDArray[np.bool_]
    translucent_faces: NDArray[np.bool_]
    face_opacity: NDArray[np.float64]


_CLASS_COMPOSITING_NOTE = (
    "golem material classes (wave-3 R8), painter-z orthographic "
    "compositing: emissive faces render at authored display luminance "
    "(shading-exempt); translucent faces composite the nearest translucent "
    "layer over the shaded opaque backdrop at authored opacity -- "
    "overlapping translucent layers collapse to the nearest, and the "
    "backdrop behind a translucent layer excludes every translucent "
    "surface. The unshaded class path renders through the deterministic "
    "painter rasterizer rather than the pilot oracle."
)


def _classed_surface_geometry(
    records: tuple[AssemblyRecord, ...],
) -> _ClassedGeometry | OrthographicRenderObstruction:
    import numpy as np

    from golem.materials.surface import (
        EmissiveSurfaceClass,
        TranslucentSurfaceClass,
    )
    from golem.senses import orthographic

    geometry = _surface_colored_geometry(records)
    if isinstance(geometry, orthographic.OrthographicRenderObstruction):
        return geometry
    visible = tuple(
        filter(
            lambda candidate: candidate is not None,
            map(orthographic._renderable_record, records),
        )
    )
    face_classes = tuple(
        map(lambda renderable: _record_surface_class(renderable.record), visible)
    )
    face_counts = tuple(map(lambda renderable: len(renderable.faces), visible))
    return _ClassedGeometry(
        geometry,
        np.concatenate(
            tuple(
                np.full(
                    count,
                    isinstance(surface_class, EmissiveSurfaceClass),
                    dtype=np.bool_,
                )
                for surface_class, count in zip(
                    face_classes, face_counts, strict=True
                )
            )
        ),
        np.concatenate(
            tuple(
                np.full(
                    count,
                    isinstance(surface_class, TranslucentSurfaceClass),
                    dtype=np.bool_,
                )
                for surface_class, count in zip(
                    face_classes, face_counts, strict=True
                )
            )
        ),
        np.concatenate(
            tuple(
                np.full(
                    count,
                    (
                        surface_class.opacity
                        if isinstance(surface_class, TranslucentSurfaceClass)
                        else 1.0
                    ),
                    dtype=np.float64,
                )
                for surface_class, count in zip(
                    face_classes, face_counts, strict=True
                )
            )
        ),
    )


def _winner_face_indices(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    global_indices: NDArray[np.int64],
    azimuth_degrees: float,
    elevation_degrees: float,
    render_size: int,
) -> NDArray[np.int64]:
    """Per-pixel nearest global face index via a flat id-color pass (-1: none)."""
    import numpy as np

    from golem.senses import orthographic

    identifiers = global_indices + 1
    id_colors = np.stack(
        (
            identifiers & 0xFF,
            (identifiers >> 8) & 0xFF,
            (identifiers >> 16) & 0xFF,
        ),
        axis=1,
    ).astype(np.uint8)
    frame = orthographic.render_raster_view(
        vertices,
        faces,
        azimuth_degrees,
        image_size=render_size,
        elevation_degrees=elevation_degrees,
        face_colors=id_colors,
        shade_faces=False,
    )
    packed = frame.pixels.astype(np.int64) @ np.asarray(
        (1, 0x100, 0x10000), dtype=np.int64
    )
    winner = packed - 1
    if len(global_indices) == 0:
        return np.full(winner.shape, -1, dtype=np.int64)
    return np.where(
        (winner >= 0) & (winner <= int(global_indices.max())),
        winner,
        -1,
    )


def _render_class_aware_view(
    classed: _ClassedGeometry,
    view: OrthographicView,
    image_size: int,
    shaded: bool,
    fit_subject: bool,
) -> OrthographicImage:
    """Shading-exempt emissive and opacity-composited translucent faces.

    The renderer in `golem.senses` stays untouched: this composes the
    painter rasterizer, the shading stages, and two flat id-color passes
    from the outside, at the same render size and camera, so every mask
    aligns with the backdrop by construction.
    """
    import numpy as np

    from golem.senses import orthographic

    geometry = classed.geometry
    azimuth_degrees, elevation_degrees = orthographic._view_angles(view)
    enhanced = shaded or fit_subject
    render_size = (
        image_size * orthographic._SUPERSAMPLING_FACTOR
        if enhanced
        else image_size
    )
    has_translucent = bool(np.any(classed.translucent_faces))
    opaque_indices = np.nonzero(~classed.translucent_faces)[0]
    backdrop_faces = geometry.faces[opaque_indices]
    backdrop_colors = geometry.face_colors[opaque_indices]
    backdrop_geometry = replace(
        geometry,
        faces=backdrop_faces,
        face_colors=backdrop_colors,
    )
    frame = orthographic.render_raster_view(
        geometry.vertices,
        backdrop_faces,
        azimuth_degrees,
        image_size=render_size,
        elevation_degrees=elevation_degrees,
        face_colors=(
            orthographic._quality_face_colors(
                backdrop_geometry,
                azimuth_degrees,
                elevation_degrees,
            )
            if shaded
            else backdrop_colors
        ),
        shade_faces=False,
    )
    pixels = np.array(frame.pixels, dtype=np.uint8)
    if shaded:
        pixels = orthographic._enhance_shading(
            orthographic._smooth_surface_shading(
                pixels,
                frame.body,
                render_size,
            ),
            frame.body,
            render_size,
        )
    winner_opaque = _winner_face_indices(
        geometry.vertices,
        backdrop_faces,
        opaque_indices,
        azimuth_degrees,
        elevation_degrees,
        render_size,
    )
    body_full = frame.body
    if np.any(classed.emissive_faces):
        emissive_hits = (winner_opaque >= 0) & classed.emissive_faces[
            np.clip(winner_opaque, 0, None)
        ]
        pixels[emissive_hits] = geometry.face_colors[
            winner_opaque[emissive_hits]
        ]
    if has_translucent:
        all_indices = np.arange(len(geometry.faces), dtype=np.int64)
        winner_all = _winner_face_indices(
            geometry.vertices,
            geometry.faces,
            all_indices,
            azimuth_degrees,
            elevation_degrees,
            render_size,
        )
        body_full = winner_all >= 0
        translucent_hits = body_full & classed.translucent_faces[
            np.clip(winner_all, 0, None)
        ]
        winners = winner_all[translucent_hits]
        alpha = classed.face_opacity[winners][:, np.newaxis]
        pixels[translucent_hits] = np.rint(
            alpha * geometry.face_colors[winners].astype(np.float64)
            + (1.0 - alpha) * pixels[translucent_hits].astype(np.float64)
        ).astype(np.uint8)
    if fit_subject:
        pixels = orthographic._fit_subject(pixels, body_full)
    return orthographic.OrthographicImage(
        view,
        (
            orthographic._downsample(pixels, image_size)
            if enhanced
            else pixels
        ),
    )


def _render_class_aware_views(
    records: tuple[AssemblyRecord, ...],
    views: tuple[OrthographicView, ...],
    image_size: int,
    *,
    fit_subject: bool,
    shaded: bool,
) -> OrthographicRenderResult:
    from golem.senses import orthographic

    classed = _classed_surface_geometry(records)
    return (
        classed
        if isinstance(classed, orthographic.OrthographicRenderObstruction)
        else orthographic.AcceptedOrthographicRender(
            tuple(
                map(
                    lambda view: _render_class_aware_view(
                        classed,
                        view,
                        image_size,
                        shaded,
                        fit_subject,
                    ),
                    views,
                )
            )
        )
    )


def _write_class_aware_views(
    rendered: AcceptedOrthographicRender,
    output_directory: Path,
    output_stem: str,
) -> OrthographicWriteResult:
    """Write PNGs whose text chunks name the compositing approximation."""
    from PIL import Image, PngImagePlugin

    from golem.senses.orthographic import (
        AcceptedOrthographicWrite,
        OrthographicWriteObstruction,
    )

    def _write(image: OrthographicImage) -> Path:
        path = output_directory / f"{output_stem}_{image.view.value}.png"
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("golem.material_classes", _CLASS_COMPOSITING_NOTE)
        Image.fromarray(image.pixels).save(path, pnginfo=metadata)
        return path

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        return AcceptedOrthographicWrite(
            tuple(map(_write, rendered.images))
        )
    except (OSError, ValueError) as failure:
        return OrthographicWriteObstruction(
            output_directory,
            f"{type(failure).__name__}: {failure}",
        )


def _render_orthographic_views(
    records: tuple[AssemblyRecord, ...],
    views: tuple[OrthographicView, ...],
    image_size: int,
    *,
    fit_subject: bool,
    shaded: bool,
) -> OrthographicRenderResult:
    from golem.senses import orthographic

    if not any(map(_has_palette_surface_color, records)):
        return orthographic.render_orthographic_views(
            records,
            views,
            image_size,
            fit_subject=fit_subject,
            shaded=shaded,
        )
    if image_size < 1:
        return orthographic.OrthographicRenderObstruction(
            "render/image_size",
            f"image size must be positive, received {image_size}",
        )
    if _has_surface_classes(records):
        return _render_class_aware_views(
            records,
            views,
            image_size,
            fit_subject=fit_subject,
            shaded=shaded,
        )
    geometry = _surface_colored_geometry(records)
    return (
        geometry
        if isinstance(geometry, orthographic.OrthographicRenderObstruction)
        else orthographic.AcceptedOrthographicRender(
            tuple(
                map(
                    lambda view: orthographic._render_view(
                        geometry,
                        view,
                        image_size,
                        shaded,
                        fit_subject,
                    ),
                    views,
                )
            )
        )
    )


def _write_result(
    result: AcceptedOrthographicWrite | OrthographicWriteObstruction,
) -> CommandResult:
    from golem.senses.orthographic import (
        AcceptedOrthographicWrite,
        OrthographicWriteObstruction,
    )

    match result:
        case AcceptedOrthographicWrite(paths):
            return CommandResult(
                0,
                stdout="\n".join((*tuple(map(str, paths)), "")),
            )
        case OrthographicWriteObstruction(path, reason):
            return CommandResult(
                1,
                stderr=(
                    "REJECTED [OrthographicWriteObstruction] "
                    f"{path}: {reason}\n"
                ),
            )
        case _ as unreachable:
            assert_never(unreachable)


def run(namespace: Namespace) -> CommandResult:
    from golem.senses.orthographic import (
        DEFAULT_ORTHOGRAPHIC_IMAGE_SIZE,
        OrthographicRenderObstruction,
        write_orthographic_views,
    )

    arguments = _look_arguments(namespace)
    try:
        compiled = compile_spec(
            arguments.spec_path,
            (
                PinnedResolution(arguments.resolution)
                if arguments.resolution is not None
                else None
            ),
        )
    except Exception as failure:
        return _engine_fault_refusal(
            arguments,
            ASSEMBLY_COMPILE_SEAM,
            failure,
        )
    if not isinstance(compiled, AcceptedAssembly):
        return compilation_obstruction_result(
            compiled,
            (
                f"render refused -> {arguments.output_directory} "
                "(typed rejection above)"
            ),
        )
    try:
        rendered = _render_orthographic_views(
            compiled.records,
            _selected_views(arguments.view),
            (
                arguments.image_size
                if arguments.image_size is not None
                else DEFAULT_ORTHOGRAPHIC_IMAGE_SIZE
            ),
            fit_subject=arguments.image_size is not None,
            shaded=arguments.shaded,
        )
    except Exception as failure:
        return _engine_fault_refusal(
            arguments,
            ORTHOGRAPHIC_RENDER_SEAM,
            failure,
        )
    if isinstance(rendered, OrthographicRenderObstruction):
        return _render_obstruction_result(rendered)
    try:
        return _write_result(
            (
                _write_class_aware_views
                if _has_surface_classes(compiled.records)
                else write_orthographic_views
            )(
                rendered,
                arguments.output_directory,
                arguments.spec_path.stem,
            )
        )
    except Exception as failure:
        return _engine_fault_refusal(
            arguments,
            ORTHOGRAPHIC_WRITE_SEAM,
            failure,
        )


COMMAND = CommandDescriptor(
    name="look",
    help_line="compile and render orthographic PNG views",
    runner=CommandRunner(
        arguments=(
            PositionalArgument(
                "spec",
                "body/0.3 document, raw part graph, or assembly document",
                Path,
            ),
            OptionArgument(
                ("--view",),
                "view",
                "render front, side, top, or every orthographic view",
                LookView,
                "front|side|top|all",
                LookView.ALL,
            ),
            OptionArgument(
                ("--res",),
                "resolution",
                "pin the author's grid resolution before rendering",
                int,
                "N",
            ),
            OptionArgument(
                ("--size",),
                "image_size",
                "set the output canvas width and height in pixels",
                int,
                "N",
            ),
            SwitchArgument(
                ("--shaded",),
                "shaded",
                "enable enhanced diffuse contrast and rim lighting",
            ),
            OptionArgument(
                ("--out",),
                "output_directory",
                "directory for emitted PNG files",
                Path,
                "DIR",
                Path("."),
            ),
        ),
        evaluate=run,
    ),
)
