"""Controlled evaluation for anatomy-first GOLEM body construction.

The control removes only authored muscle and skin sections.  Skeleton,
circulation, renderer, reference, and resolution remain fixed.  Silhouette IoU
therefore measures a coarse construction effect, not beauty; structural gates
separately require the typed anatomy to close, root, cover, address, and emit a
coherent mesh.  Human blind preference remains the appeal authority.

The final stdout line is stable JSON for ``codex-autoresearch``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import prod
from typing import assert_never

import numpy as np

from golem import paths as _paths
from golem.evals.harness import (
    AcceptedEvaluation,
    EvaluationObstruction,
    RenderedEvaluation,
    emit_rendered_evaluation,
    render_outcome,
)
from golem.kernel import anatomy as anatomy_kernel
from golem.kernel import body, engine
from golem.kernel.engine.types import SurfaceFormationObstruction
from golem.senses import proprio, silhouette
from golem.senses.model import (
    RejectedSenses,
    Senses,
    SensesObstruction,
)
from golem.senses.project import project_rejected_senses


_BODY_PATH = _paths.SPECS / "knight_body.json"
_REFERENCE_PATH = _paths.REHEARSAL / "knight" / "blender_knight.glb"
_MESH_RESOLUTION = 96
_IMAGE_SIZE = 192
_FLOW_TOLERANCE = 1.0e-10
_LEVERAGE_TARGET = 1.5
_TISSUE_KEYS = frozenset(("myotendinous_units", "integument_layers"))


@dataclass(frozen=True)
class _AcceptedCompilation:
    compiled: body.CompiledBody
    anatomy: anatomy_kernel.AcceptedAnatomy


@dataclass(frozen=True)
class _GraphEvaluation:
    intersection_over_union: float
    mesh_integrity: float


class AnatomyMeshVariant(StrEnum):
    CONTROL = "control"
    ANATOMY = "anatomy"
    REFERENCE = "reference"


@dataclass(frozen=True)
class AnatomySurfaceFormationObstruction:
    variant: AnatomyMeshVariant
    obstructions: tuple[SurfaceFormationObstruction, ...]


@dataclass(frozen=True)
class AnatomySensesObstruction:
    variant: AnatomyMeshVariant
    obstructions: tuple[SensesObstruction, ...]


@dataclass(frozen=True)
class AnatomyEvaluation(AcceptedEvaluation):
    control_iou: float
    anatomy_iou: float
    control_mesh_integrity: float
    anatomy_mesh_integrity: float
    rootedness: float
    interface_addressability: float
    authoring_leverage: float
    tissue_coverage: float
    anatomy_closure: float

    @property
    def reference_iou_gain(self) -> float:
        """Raw candidate-minus-control evidence; deliberately not normalized."""
        return self.anatomy_iou - self.control_iou

    @property
    def anatomy_guided_construction_score(self) -> float:
        structural_factor = prod(
            (
                self.anatomy_mesh_integrity,
                self.rootedness,
                self.interface_addressability,
                self.authoring_leverage,
                self.tissue_coverage,
                self.anatomy_closure,
            )
        ) ** (1.0 / 6.0)
        return self.anatomy_iou * structural_factor

    def evaluation_payload(self) -> dict[str, object]:
        return {
            "anatomy_closure": self.anatomy_closure,
            "anatomy_guided_construction_score": (
                self.anatomy_guided_construction_score
            ),
            "anatomy_iou": self.anatomy_iou,
            "anatomy_mesh_integrity": self.anatomy_mesh_integrity,
            "authoring_leverage": self.authoring_leverage,
            "control_iou": self.control_iou,
            "control_mesh_integrity": self.control_mesh_integrity,
            "interface_addressability": self.interface_addressability,
            "reference_iou_gain": self.reference_iou_gain,
            "rootedness": self.rootedness,
            "tissue_coverage": self.tissue_coverage,
        }


AnatomyEvaluationObstruction = EvaluationObstruction


type AnatomyEvaluationResult = (
    AnatomyEvaluation
    | AnatomyEvaluationObstruction
    | AnatomySensesObstruction
    | AnatomySurfaceFormationObstruction
)
type _CompilationResult = _AcceptedCompilation | AnatomyEvaluationObstruction
type _GraphEvaluationResult = (
    _GraphEvaluation | AnatomySurfaceFormationObstruction
)


def evaluate_anatomy() -> AnatomyEvaluationResult:
    """Evaluate the complete control/candidate cover or return one obstruction."""
    try:
        candidate_spec = json.loads(_BODY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as failure:
        return AnatomyEvaluationObstruction(str(_BODY_PATH), str(failure))
    control_spec = _without_authored_tissue(candidate_spec)
    if isinstance(control_spec, AnatomyEvaluationObstruction):
        return control_spec
    control = _compile_variant(control_spec, "control")
    if isinstance(control, AnatomyEvaluationObstruction):
        return control
    candidate = _compile_variant(candidate_spec, "anatomy")
    if isinstance(candidate, AnatomyEvaluationObstruction):
        return candidate
    reference_result = silhouette.load_geometry(_REFERENCE_PATH, _MESH_RESOLUTION)
    match reference_result:
        case silhouette.Geometry() as reference:
            pass
        case silhouette.GeometryLoadFailure(path, reason):
            return AnatomyEvaluationObstruction(str(path), reason)
        case silhouette.BodyGeometryLoadObstruction(path, obstructions):
            return AnatomyEvaluationObstruction(
                str(path),
                json.dumps(
                    body.project_rejected_body(body.RejectedBody(obstructions)),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        case silhouette.SurfaceFormationGeometryLoadObstruction(
            obstructions=obstructions
        ):
            return AnatomySurfaceFormationObstruction(
                AnatomyMeshVariant.REFERENCE,
                obstructions,
            )
        case _ as unreachable:
            assert_never(unreachable)
    control_graph = _score_graph(
        control.compiled.graph,
        reference,
        AnatomyMeshVariant.CONTROL,
    )
    if isinstance(control_graph, AnatomySurfaceFormationObstruction):
        return control_graph
    candidate_graph = _score_graph(
        candidate.compiled.graph,
        reference,
        AnatomyMeshVariant.ANATOMY,
    )
    if isinstance(candidate_graph, AnatomySurfaceFormationObstruction):
        return candidate_graph
    candidate_senses = proprio.build_senses(
        candidate.compiled.graph, candidate.compiled.graph["intent"]
    )
    match candidate_senses:
        case RejectedSenses(obstructions=obstructions):
            return AnatomySensesObstruction(
                AnatomyMeshVariant.ANATOMY,
                obstructions,
            )
        case Senses() as accepted_senses:
            pass
        case _ as unreachable:
            assert_never(unreachable)
    return AnatomyEvaluation(
        control_iou=control_graph.intersection_over_union,
        anatomy_iou=candidate_graph.intersection_over_union,
        control_mesh_integrity=control_graph.mesh_integrity,
        anatomy_mesh_integrity=candidate_graph.mesh_integrity,
        rootedness=_rootedness(candidate.anatomy, accepted_senses),
        interface_addressability=_interface_addressability(candidate.anatomy),
        authoring_leverage=min(
            1.0, candidate.anatomy.authoring_leverage / _LEVERAGE_TARGET
        ),
        tissue_coverage=_tissue_coverage(candidate.anatomy),
        anatomy_closure=1.0,
    )


def _without_authored_tissue(
    spec: object,
) -> dict[str, object] | AnatomyEvaluationObstruction:
    if not isinstance(spec, dict):
        return AnatomyEvaluationObstruction(str(_BODY_PATH), "expected an object")
    anatomy_payload = spec.get("anatomy")
    if not isinstance(anatomy_payload, dict):
        return AnatomyEvaluationObstruction(
            "/anatomy", "expected an anatomy object"
        )
    overall = anatomy_payload.get("overall")
    if not isinstance(overall, dict):
        return AnatomyEvaluationObstruction(
            "/anatomy/overall", "expected an overall anatomy object"
        )
    return {
        **spec,
        "anatomy": {
            **anatomy_payload,
            "overall": {
                key: value
                for key, value in overall.items()
                if key not in _TISSUE_KEYS
            },
        },
    }


def _compile_variant(spec: dict[str, object], variant: str) -> _CompilationResult:
    try:
        compile_result = body.Compiler(spec, spec_dir=_paths.SPECS).compile()
    except (OSError, ValueError, TypeError, KeyError) as failure:
        return AnatomyEvaluationObstruction(
            f"{_BODY_PATH}#{variant}", str(failure)
        )
    match compile_result:
        case body.CompiledBody() as compiled:
            pass
        case body.RejectedBody() as rejected:
            return AnatomyEvaluationObstruction(
                f"{_BODY_PATH}#{variant}",
                json.dumps(
                    body.project_rejected_body(rejected),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        case _ as unreachable:
            assert_never(unreachable)
    match compiled.anatomy:
        case anatomy_kernel.AcceptedAnatomy() as accepted:
            return _AcceptedCompilation(compiled, accepted)
        case anatomy_kernel.RejectedAnatomy() as rejected:
            return AnatomyEvaluationObstruction(
                f"/anatomy/{variant}",
                json.dumps(
                    anatomy_kernel.anatomy_to_dict(rejected)["obstructions"],
                    sort_keys=True,
                ),
            )
        case None:
            return AnatomyEvaluationObstruction(
                f"/anatomy/{variant}", "compiler emitted no anatomy result"
            )
        case _ as unreachable:
            assert_never(unreachable)


def _score_graph(
    graph: dict[str, object],
    reference: silhouette.Geometry,
    variant: AnatomyMeshVariant,
) -> _GraphEvaluationResult:
    evaluated = engine.evaluate(graph, res=_MESH_RESOLUTION)
    if isinstance(evaluated, engine.RejectedSurfaceFormation):
        return AnatomySurfaceFormationObstruction(
            variant,
            evaluated.obstructions,
        )
    coherence = engine.coherence_report(evaluated.vertices, evaluated.faces)
    score = silhouette.score_geometry(
        silhouette.Geometry(
            vertices=np.asarray(evaluated.vertices, dtype=np.float64),
            faces=np.asarray(evaluated.faces, dtype=np.int64),
        ),
        reference,
        brief_id=f"knight-anatomy-{variant}",
        image_size=_IMAGE_SIZE,
    )
    return _GraphEvaluation(
        intersection_over_union=score.mean_intersection_over_union,
        mesh_integrity=float(
            int(coherence["components"]) == 1
            and bool(coherence["watertight_main"])
        ),
    )


def _rootedness(
    accepted: anatomy_kernel.AcceptedAnatomy, senses: Senses
) -> float:
    maximum_free_imbalance, boundary_relative_error = accepted.conservation
    return float(
        maximum_free_imbalance <= _FLOW_TOLERANCE
        and boundary_relative_error <= _FLOW_TOLERANCE
        and senses.n_components == 1
    )


def _interface_addressability(
    accepted: anatomy_kernel.AcceptedAnatomy,
) -> float:
    segments = tuple(
        segment
        for circuit in accepted.circuits
        for segment in (
            *circuit.distributing_arteries,
            circuit.resistance_arteriole,
            circuit.collecting_venule,
            *circuit.returning_veins,
        )
    )
    return float(
        bool(segments)
        and bool(accepted.tissue_envelopes)
        and all(
            segment.interface_address.startswith("skeleton/")
            and segment.interface_address.endswith("/joint")
            for segment in segments
        )
        and all(
            envelope.shape_address.startswith("skeleton/")
            and "/flesh[" in envelope.shape_address
            for envelope in accepted.tissue_envelopes
        )
    )


def _tissue_coverage(accepted: anatomy_kernel.AcceptedAnatomy) -> float:
    muscles = accepted.overall.muscles
    anchor_bones = frozenset(
        anchor.bone_id
        for muscle in muscles
        for anchor in (muscle.origin, muscle.insertion)
    )
    envelope_bones = frozenset(
        envelope.bone_id for envelope in accepted.tissue_envelopes
    )
    muscle_regions = frozenset(muscle.region_id for muscle in muscles)
    skin_regions = frozenset(
        layer.region_id for layer in accepted.overall.integument
    )
    bone_coverage = (
        len(anchor_bones & envelope_bones) / len(anchor_bones)
        if anchor_bones
        else 0.0
    )
    skin_coverage = (
        len(muscle_regions & skin_regions) / len(muscle_regions)
        if muscle_regions
        else 0.0
    )
    return min(bone_coverage, skin_coverage)


def _render_surface_formation(
    obstruction: AnatomySurfaceFormationObstruction,
) -> RenderedEvaluation:
    return RenderedEvaluation(
        stdout="",
        stderr=json.dumps(
            {
                "address": f"mesh/{obstruction.variant}",
                "obstructions": tuple(
                    {
                        "obstruction": type(member).__name__,
                        **asdict(member),
                    }
                    for member in obstruction.obstructions
                ),
                "status": "surface_formation_obstructed",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        exit_code=2,
    )


def _render_senses_obstruction(
    obstruction: AnatomySensesObstruction,
) -> RenderedEvaluation:
    payload = {
        **project_rejected_senses(
            RejectedSenses(obstruction.obstructions)
        ),
        "address": f"senses/{obstruction.variant}",
    }
    return RenderedEvaluation(
        stdout="",
        stderr=json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        exit_code=2,
    )


def _render_anatomy_evaluation(
    result: AnatomyEvaluationResult,
) -> RenderedEvaluation:
    match result:
        case AnatomySurfaceFormationObstruction():
            return _render_surface_formation(result)
        case AnatomySensesObstruction():
            return _render_senses_obstruction(result)
        case AnatomyEvaluation() | EvaluationObstruction():
            return render_outcome(result)
        case _ as unreachable:
            assert_never(unreachable)


def render_evaluation(result: AnatomyEvaluationResult) -> str:
    rendered = _render_anatomy_evaluation(result)
    return (rendered.stdout or rendered.stderr).rstrip("\n")


if __name__ == "__main__":
    raise SystemExit(
        emit_rendered_evaluation(_render_anatomy_evaluation(evaluate_anatomy()))
    )
