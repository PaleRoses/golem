"""Sealed rendered-output evaluator for GOLEM's elemental material lowering.

This instrument measures whether authoritative balance interfaces survive the
actual mesh -> material -> raster pipeline. It does not
pretend to score beauty.  A candidate earns ``runic_path_legibility`` only when
its pixel-level change from the same-geometry uniform-seam control is aligned with an
independently derived pulse on those exact seam overlaps and large enough to
read.  Merely coloring every plate adjacent to a conduit earns little.

The guard separately forbids metric gaming by same-geometry silhouette damage,
frozen-mesh outline drift, background replacement, plate-detail collapse,
broad unrelated recoloring, or saturation.  All measurements are in memory;
the command writes no artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from statistics import fmean

import numpy as np
from numpy.typing import NDArray

from golem.senses import elemental_render
from golem.senses.raster import (
    body_mask,
    intersection_over_union,
    render_raster_view,
    safe_mean,
)
from golem.evals.harness import (
    AcceptedEvaluation,
    render_outcome_text as render_evaluation_json,
)


_IMAGE_SIZE = 192
_SAMPLES = (
    (0.0, 0.00),
    (40.0, 0.25),
    (90.0, 0.50),
    (140.0, 0.70),
    (180.0, 0.88),
)
_ACTIVE_ORACLE_FRACTION = 0.30
_VISIBLE_DELTA = 52.0
_MAXIMUM_QUIET_RMSE = 22.0
_MINIMUM_DETAIL_RETENTION = 0.90
_MINIMUM_BODY_LUMINANCE_RATIO = 0.82
_MAXIMUM_BODY_LUMINANCE_RATIO = 1.20
_MAXIMUM_CLIPPED_FRACTION = 0.08
_MINIMUM_FROZEN_SILHOUETTE_IOU = 0.97


@dataclass(frozen=True)
class FrameMeasurement:
    alignment: float
    visibility: float
    quiet_rmse: float
    detail_retention: float
    body_luminance_ratio: float
    clipped_fraction: float
    frozen_silhouette_iou: float
    silhouette_exact: bool
    background_exact: bool

    @property
    def legibility(self) -> float:
        return self.alignment * self.visibility


@dataclass(frozen=True)
class ElementalVisualEvaluation(AcceptedEvaluation):
    frames: tuple[FrameMeasurement, ...]

    @property
    def runic_path_legibility(self) -> float:
        return fmean(map(lambda frame: frame.legibility, self.frames))

    @property
    def mean_alignment(self) -> float:
        return fmean(map(lambda frame: frame.alignment, self.frames))

    @property
    def mean_visibility(self) -> float:
        return fmean(map(lambda frame: frame.visibility, self.frames))

    @property
    def guard_passes(self) -> bool:
        return all(
            frame.silhouette_exact
            and frame.background_exact
            and frame.quiet_rmse <= _MAXIMUM_QUIET_RMSE
            and frame.detail_retention >= _MINIMUM_DETAIL_RETENTION
            and _MINIMUM_BODY_LUMINANCE_RATIO
            <= frame.body_luminance_ratio
            <= _MAXIMUM_BODY_LUMINANCE_RATIO
            and frame.clipped_fraction <= _MAXIMUM_CLIPPED_FRACTION
            and frame.frozen_silhouette_iou >= _MINIMUM_FROZEN_SILHOUETTE_IOU
            for frame in self.frames
        )

    def evaluation_payload(self) -> dict[str, object]:
        return {
            "background_exact": all(
                map(lambda frame: frame.background_exact, self.frames)
            ),
            "body_luminance_ratio_min": min(
                map(lambda frame: frame.body_luminance_ratio, self.frames)
            ),
            "clipped_fraction_max": max(
                map(lambda frame: frame.clipped_fraction, self.frames)
            ),
            "detail_retention_min": min(
                map(lambda frame: frame.detail_retention, self.frames)
            ),
            "runic_path_legibility": self.runic_path_legibility,
            "guard_passes": self.guard_passes,
            "frozen_silhouette_iou_min": min(
                map(lambda frame: frame.frozen_silhouette_iou, self.frames)
            ),
            "mean_alignment": self.mean_alignment,
            "mean_visibility": self.mean_visibility,
            "quiet_rmse_max": max(
                map(lambda frame: frame.quiet_rmse, self.frames)
            ),
            "silhouette_exact": all(
                map(lambda frame: frame.silhouette_exact, self.frames)
            ),
        }


def evaluate_scene(
    scene: elemental_render.ElementalRenderScene,
) -> ElementalVisualEvaluation:
    """Evaluate a fixed local cover of view/phase contexts and glue the scores."""
    return ElementalVisualEvaluation(
        tuple(
            _measure_frame(scene, azimuth_degrees, phase)
            for azimuth_degrees, phase in _SAMPLES
        )
    )


def _measure_frame(
    scene: elemental_render.ElementalRenderScene,
    azimuth_degrees: float,
    phase: float,
) -> FrameMeasurement:
    baseline_frame = render_raster_view(
        scene.render_vertices,
        scene.faces,
        azimuth_degrees,
        image_size=_IMAGE_SIZE,
        face_colors=scene.frozen_face_colors,
    )
    frozen_baseline_frame = render_raster_view(
        scene.frozen_vertices,
        scene.faces,
        azimuth_degrees,
        image_size=_IMAGE_SIZE,
        face_colors=scene.frozen_face_colors,
    )
    candidate = np.asarray(
        elemental_render.render_elemental_frame(
            scene,
            azimuth_degrees,
            phase,
            image_size=_IMAGE_SIZE,
        ),
        dtype=np.uint8,
    )
    baseline = baseline_frame.pixels
    frozen_baseline = frozen_baseline_frame.pixels
    body = baseline_frame.body
    frozen_body = frozen_baseline_frame.body
    candidate_body = body_mask(candidate)
    oracle = _render_oracle(scene, azimuth_degrees, phase, body)
    salience = np.linalg.norm(
        candidate.astype(np.float64) - baseline.astype(np.float64), axis=2
    ) / sqrt(3.0)
    active = body & (oracle >= _ACTIVE_ORACLE_FRACTION * float(oracle.max()))
    quiet = body & (oracle <= 0.01)
    baseline_luminance = _luminance(baseline)
    candidate_luminance = _luminance(candidate)
    return FrameMeasurement(
        alignment=_cosine_similarity(salience[body], oracle[body]),
        visibility=min(safe_mean(salience[active]) / _VISIBLE_DELTA, 1.0),
        quiet_rmse=float(sqrt(safe_mean(np.square(salience[quiet])))),
        detail_retention=min(
            _detail_energy(candidate_luminance, body)
            / max(_detail_energy(baseline_luminance, body), 1.0e-12),
            1.0,
        ),
        body_luminance_ratio=safe_mean(candidate_luminance[body])
        / max(safe_mean(baseline_luminance[body]), 1.0e-12),
        clipped_fraction=safe_mean(
            np.any((candidate[body] <= 1) | (candidate[body] >= 254), axis=1)
        ),
        frozen_silhouette_iou=intersection_over_union(body, frozen_body),
        silhouette_exact=bool(np.array_equal(candidate_body, body)),
        background_exact=bool(np.array_equal(candidate[~body], baseline[~body])),
    )


def _render_oracle(
    scene: elemental_render.ElementalRenderScene,
    azimuth_degrees: float,
    phase: float,
    body: NDArray[np.bool_],
) -> NDArray[np.float64]:
    path_seams = scene.balance_interface_faces
    normalized_seam_flow = scene.normalized_face_flux
    face_potential = scene.normalized_face_field
    travelling_wave = np.power(
        0.5 + 0.5 * np.cos(2.0 * pi * (face_potential + phase)), 4.0
    )
    face_signal = path_seams * (0.16 + 0.84 * travelling_wave) * (
        0.35 + 0.65 * np.sqrt(normalized_seam_flow)
    )
    face_colors = np.rint(255.0 * face_signal[:, None]).astype(np.uint8)
    rendered = render_raster_view(
        scene.render_vertices,
        scene.faces,
        azimuth_degrees,
        image_size=_IMAGE_SIZE,
        face_colors=np.repeat(face_colors, 3, axis=1),
    ).pixels.astype(np.float64)
    signal = np.zeros(body.shape, dtype=np.float64)
    signal[body] = rendered[body].mean(axis=1) / 255.0
    return signal


def _cosine_similarity(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else 0.0


def _detail_energy(luminance: NDArray[np.float64], body: NDArray[np.bool_]) -> float:
    horizontal = np.abs(np.diff(luminance, axis=1))
    vertical = np.abs(np.diff(luminance, axis=0))
    horizontal_body = body[:, 1:] & body[:, :-1]
    vertical_body = body[1:, :] & body[:-1, :]
    return safe_mean(
        np.concatenate((horizontal[horizontal_body], vertical[vertical_body]))
    )


def _luminance(pixels: NDArray[np.uint8]) -> NDArray[np.float64]:
    return pixels.astype(np.float64) @ np.asarray((0.2126, 0.7152, 0.0722))
