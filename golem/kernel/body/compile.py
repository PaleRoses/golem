"""The one-shot body-plan compiler: spine state + phase orchestration."""

from __future__ import annotations

import difflib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from golem.kernel.engine import CompositionOperator

from golem.kernel.body.anatomy_bridge import (
    _anatomy_view,
    _derive_anatomy_result,
    _skin_formation_anatomy_obstructions,
    _skin_layer_view,
)
from golem.kernel.body.carves import compile_carves
from golem.kernel.body.webs import compile_webs
from golem.kernel.body.canonical import _r
from golem.kernel.body.eyes import compile_eyes
from golem.kernel.body.emit import _EmitMixin
from golem.kernel.body.goals import _GoalsMixin
from golem.kernel.body.intent import _IntentMixin
from golem.kernel.body.kinematics import _KinematicsMixin
from golem.kernel.body.myology import (
    AcceptedMyology,
    RejectedMyology,
    compile_myology,
)
from golem.kernel.body.relations import (
    FixedLocalAttachment,
    NamedSiteAttachment,
    RejectedRelationDecode,
    decode_attachment,
    decode_relations,
)
from golem.kernel.body.types import (
    BODY_SPEC_TOP_LEVEL_KEYS,
    BodyCompileResult,
    BodySpecDecodeObstruction,
    CompiledBody,
    EyeHostFrame,
    FleshCompositionRule,
    MalformedFleshCompositionObstruction,
    MalformedBodySpecSectionObstruction,
    MisplacedBodySpecFieldObstruction,
    MissingBodySpecSectionObstruction,
    RejectedBody,
    RejectedCarves,
    RejectedEyes,
    RejectedWebs,
    RelationalSolveReceipt,
    SolvedLocalPose,
    SkinFormationAnatomyObstruction,
    UnknownBodySpecKeyObstruction,
)
from golem.kernel.body.validate import _ValidateMixin


def _closest_body_spec_keys(authored_key: str) -> tuple[str, ...]:
    return (
        ("contract",)
        if authored_key == "intent"
        else tuple(
            difflib.get_close_matches(
                authored_key,
                sorted(BODY_SPEC_TOP_LEVEL_KEYS),
                n=3,
            )
        )
    )


def _unknown_body_spec_key_obstructions(
    spec: Mapping[str, object],
) -> tuple[UnknownBodySpecKeyObstruction, ...]:
    return tuple(
        UnknownBodySpecKeyObstruction(
            address=f"spec/{authored_key}",
            authored_key=authored_key,
            closest_valid_candidates=_closest_body_spec_keys(authored_key),
        )
        for authored_key in sorted(set(spec) - BODY_SPEC_TOP_LEVEL_KEYS)
    )


def _required_body_section_obstructions(
    spec: Mapping[str, object],
) -> tuple[MissingBodySpecSectionObstruction | MalformedBodySpecSectionObstruction, ...]:
    if "skeleton" not in spec:
        return (
            MissingBodySpecSectionObstruction(
                "skeleton", "skeleton", "top-level body"
            ),
        )
    skeleton = spec["skeleton"]
    if not isinstance(skeleton, Mapping):
        return (
            MalformedBodySpecSectionObstruction(
                "skeleton", "skeleton", "object", type(skeleton).__name__
            ),
        )
    missing_sections = tuple(
        MissingBodySpecSectionObstruction(
            f"skeleton/{section}", section, "skeleton"
        )
        for section in ("root", "bones")
        if section not in skeleton
    )
    root = skeleton.get("root")
    malformed_root = (
        (
            MalformedBodySpecSectionObstruction(
                "skeleton/root", "root", "object", type(root).__name__
            ),
        )
        if "root" in skeleton and not isinstance(root, Mapping)
        else ()
    )
    bones = skeleton.get("bones")
    malformed_bones = (
        (
            MalformedBodySpecSectionObstruction(
                "skeleton/bones",
                "bones",
                "array of bone objects",
                type(bones).__name__,
            ),
        )
        if "bones" in skeleton
        and (
            not isinstance(bones, Sequence)
            or isinstance(bones, (str, bytes))
        )
        else ()
    )
    malformed_bone_records = (
        tuple(
            MalformedBodySpecSectionObstruction(
                f"skeleton/bones/{index}",
                "bone",
                "object",
                type(bone).__name__,
            )
            for index, bone in enumerate(bones)
            if not isinstance(bone, Mapping)
        )
        if isinstance(bones, Sequence) and not isinstance(bones, (str, bytes))
        else ()
    )
    return (
        *missing_sections,
        *malformed_root,
        *malformed_bones,
        *malformed_bone_records,
    )


def _optional_body_section_obstructions(
    spec: Mapping[str, object],
) -> tuple[MalformedBodySpecSectionObstruction, ...]:
    pose = spec.get("pose")
    return (
        (
            MalformedBodySpecSectionObstruction(
                "pose", "pose", "object", type(pose).__name__
            ),
        )
        if "pose" in spec and pose is not None and not isinstance(pose, Mapping)
        else ()
    )


def _misplaced_body_spec_field_obstructions(
    spec: Mapping[str, object],
) -> tuple[MisplacedBodySpecFieldObstruction, ...]:
    skeleton = spec.get("skeleton")
    bones = skeleton.get("bones") if isinstance(skeleton, Mapping) else None
    return (
        tuple(
            MisplacedBodySpecFieldObstruction(
                address=(
                    f"skeleton/{bone['id']}/pose/relations"
                    if isinstance(bone.get("id"), str) and bone["id"]
                    else f"skeleton/bones/{index}/pose/relations"
                ),
                field="relations",
                authored_parent="bone pose",
                legal_parent="top-level pose",
                legal_address="pose/relations",
            )
            for index, bone in enumerate(bones)
            if isinstance(bone, Mapping)
            and isinstance(bone.get("pose"), Mapping)
            and "relations" in bone["pose"]
        )
        if isinstance(bones, Sequence) and not isinstance(bones, (str, bytes))
        else ()
    )


_COMPOSITION_OPERATORS = tuple(operator.value for operator in CompositionOperator)


def _finite_non_negative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _flesh_composition_record_obstructions(
    address: str, flesh: Mapping[str, object]
) -> tuple[MalformedFleshCompositionObstruction, ...]:
    blend = flesh.get("blend")
    operator = flesh.get("operator")
    return (
        (
            MalformedFleshCompositionObstruction(
                f"{address}/blend",
                FleshCompositionRule.FINITE_NON_NEGATIVE_BLEND,
                blend,
                {"finite": True, "minimum": 0.0},
            ),
        )
        if "blend" in flesh and not _finite_non_negative_number(blend)
        else ()
    ) + (
        (
            MalformedFleshCompositionObstruction(
                f"{address}/operator",
                FleshCompositionRule.OPERATOR,
                operator,
                _COMPOSITION_OPERATORS,
            ),
        )
        if "operator" in flesh and operator not in _COMPOSITION_OPERATORS
        else ()
    )


def _flesh_records(bone: Mapping[str, object]) -> Sequence[object]:
    flesh = bone.get("flesh", ())
    return (
        flesh
        if isinstance(flesh, Sequence) and not isinstance(flesh, (str, bytes))
        else ()
    )


def _flesh_composition_obstructions(
    spec: Mapping[str, object],
) -> tuple[MalformedFleshCompositionObstruction, ...]:
    skeleton = spec.get("skeleton")
    if not isinstance(skeleton, Mapping):
        return ()
    root = skeleton.get("root")
    bones = skeleton.get("bones")
    records = (
        ((f"skeleton/{root.get('id', 'root')}", root),)
        if isinstance(root, Mapping)
        else ()
    ) + (
        tuple(
            (
                (
                    f"skeleton/{bone['id']}"
                    if isinstance(bone.get("id"), str) and bone["id"]
                    else f"skeleton/bones/{index}"
                ),
                bone,
            )
            for index, bone in enumerate(bones)
            if isinstance(bone, Mapping)
        )
        if isinstance(bones, Sequence) and not isinstance(bones, (str, bytes))
        else ()
    )
    return tuple(
        obstruction
        for bone_address, bone in records
        for index, flesh in enumerate(_flesh_records(bone))
        if isinstance(flesh, Mapping)
        for obstruction in _flesh_composition_record_obstructions(
            f"{bone_address}/flesh[{index}]", flesh
        )
    )


def _body_spec_decode_obstructions(
    spec: Mapping[str, object],
) -> tuple[BodySpecDecodeObstruction, ...]:
    return (
        *_unknown_body_spec_key_obstructions(spec),
        *_required_body_section_obstructions(spec),
        *_optional_body_section_obstructions(spec),
        *_misplaced_body_spec_field_obstructions(spec),
        *_flesh_composition_obstructions(spec),
    )


class Compiler(
    _ValidateMixin,
    _KinematicsMixin,
    _GoalsMixin,
    _EmitMixin,
    _IntentMixin,
):
    """One-shot body-plan compiler. Construct then call :meth:`compile`."""

    def __init__(
        self,
        spec: dict,
        spec_dir: Path | None = None,
    ):
        self.spec = spec
        self.spec_dir = spec_dir or Path.cwd()
        self.target = spec.get("emit_target", "v03")
        self.ground_y = float(spec.get("ground_y", 0.02))
        self.blend = float(spec.get("blend", 0.05))
        self.violations: list[dict] = []
        self.bones: dict[str, dict] = {}       # id -> bone record (frame/head/...)
        self.order: list[str] = []             # DFS bone order
        self.landmarks: dict[str, list[float]] = {}
        self.parts: list[dict] = []            # emitted flat parts
        self.provenance: dict[str, str] = {}   # emitted id -> skeleton address
        self.ground_lift = 0.0
        self.solved_goals: list[dict] = []
        self.limit_violations: list[dict] = []
        self.clamps: list[dict] = []
        self.contacts = spec.get("contacts", [])
        self._solved_params: dict[str, SolvedLocalPose] = {}
        self._relational_receipt: RelationalSolveReceipt | None = None
        self._web_anchors: tuple[tuple[str, tuple[str, ...]], ...] = ()
        self._web_non_carrier: tuple[str, ...] = ()

    # -- violation helper ------------------------------------------------- #
    def _viol(self, rule: str, address: str, detail: str, **extra) -> None:
        rec = {"rule": rule, "address": address, "detail": detail}
        rec.update(extra)
        self.violations.append(rec)

    def _flesh_id(self, bid: str, fl: dict, i: int) -> str:
        return fl.get("name", f"{bid}.flesh{i}")

    # -- contact-group helpers -------------------------------------------- #
    def _contact_group_parts(self, group: str) -> list[str]:
        out: list[str] = []
        for c in self.contacts:
            if c.get("group") == group:
                out.extend(c.get("parts", []))
        return out

    # -- relational gate -------------------------------------------------- #
    def _is_relational(self) -> bool:
        pose = self.spec.get("pose", {}) or {}
        if pose.get("relations"):
            return True
        return any(
            isinstance(bone.get("attach"), dict) and "at" in bone["attach"]
            for bone in self.spec.get("skeleton", {}).get("bones", [])
        )

    def _decode_placement(self) -> tuple[dict, tuple, tuple]:
        attachments: dict = {}
        obstructions: tuple = ()
        for bone in self.spec.get("skeleton", {}).get("bones", []):
            decoded = decode_attachment(
                bone.get("attach"), f"skeleton/{bone['id']}/attach"
            )
            if isinstance(decoded, (FixedLocalAttachment, NamedSiteAttachment)):
                attachments[bone["id"]] = decoded
            else:
                obstructions = obstructions + (decoded,)
        relations = (self.spec.get("pose", {}) or {}).get("relations")
        decoded_relations = decode_relations(relations)
        if isinstance(decoded_relations, RejectedRelationDecode):
            return attachments, (), obstructions + decoded_relations.obstructions
        return attachments, decoded_relations.declarations, obstructions

    # -- top-level orchestration ------------------------------------------ #
    def compile(self) -> BodyCompileResult:
        decode_obstructions = _body_spec_decode_obstructions(self.spec)
        if decode_obstructions:
            return RejectedBody(decode_obstructions)
        self.validate()
        if not self._is_relational():
            return self._emit_body()
        attachments, explicit, obstructions = self._decode_placement()
        obstructions = obstructions + self._relation_validation(explicit, attachments)
        if obstructions:
            return RejectedBody(obstructions)
        solved, receipt, solve_obstructions = self.solve_relations(explicit, attachments)
        if solve_obstructions:
            return RejectedBody(solve_obstructions)
        self._solved_params = solved
        self._relational_receipt = receipt
        return self._emit_body()

    def _emit_body(self) -> BodyCompileResult:
        self.forward_kinematics()
        self.ground_lift_body()
        self.place_props()
        self.resolve_goals()
        anatomy = (
            _derive_anatomy_result(self.bones, self.order, self.spec["anatomy"])
            if "anatomy" in self.spec
            else None
        )
        skin_anatomy_obstructions = _skin_formation_anatomy_obstructions(
            anatomy,
            self.spec.get("anatomy"),
        )
        if skin_anatomy_obstructions:
            return RejectedBody(
                (
                    SkinFormationAnatomyObstruction(
                        skin_anatomy_obstructions
                    ),
                )
            )
        self.emit_parts(anatomy)
        myology = (
            compile_myology(
                self.spec["muscles"],
                self.spec,
                self.bones,
                self.landmarks,
                tuple(self.parts),
            )
            if "muscles" in self.spec
            else None
        )
        if isinstance(myology, RejectedMyology):
            return RejectedBody(myology.obstructions)
        body_parts = (
            [*self.parts, *myology.parts]
            if isinstance(myology, AcceptedMyology) and myology.parts
            else self.parts
        )
        body_provenance = (
            {**self.provenance, **dict(myology.provenance)}
            if isinstance(myology, AcceptedMyology) and myology.provenance
            else self.provenance
        )
        carve_result = compile_carves(
            self.bones,
            self.order,
            frozenset(str(part["id"]) for part in body_parts),
        )
        if isinstance(carve_result, RejectedCarves):
            return RejectedBody(carve_result.obstructions)
        eye_result = compile_eyes(
            self.spec.get("eyes"),
            {
                bone_id: EyeHostFrame(
                    origin=tuple(map(float, record["head"])),
                    rotation=tuple(
                        tuple(map(float, row)) for row in record["R"]
                    ),
                    length=float(record["length"]),
                )
                for bone_id, record in self.bones.items()
            },
            frozenset(str(part["id"]) for part in body_parts)
            | frozenset(str(carve["id"]) for carve in carve_result.carves),
        )
        if isinstance(eye_result, RejectedEyes):
            return RejectedBody(eye_result.obstructions)
        parts = (
            body_parts
            if not eye_result.brow_parts
            else [*body_parts, *eye_result.brow_parts]
        )
        carves = (*carve_result.carves, *eye_result.carves)
        web_result = compile_webs(
            self.spec.get("webs"),
            self.bones,
            frozenset(str(part["id"]) for part in parts)
            | frozenset(str(carve["id"]) for carve in carves)
            | frozenset(str(globe.eye_id) for globe in eye_result.globes),
        )
        if isinstance(web_result, RejectedWebs):
            return RejectedBody(web_result.obstructions)
        self._web_anchors = web_result.anchors
        self._web_non_carrier = web_result.non_carrier
        provenance = {
            **body_provenance,
            **dict(carve_result.provenance),
            **dict(eye_result.provenance),
            **dict(web_result.provenance),
        }
        anatomy_view = _anatomy_view(anatomy)
        skin = _skin_layer_view(anatomy)
        intent = self.build_intent(
            anatomy_view,
            parts=parts,
            provenance=provenance,
        )
        graph = {
            "name": self.spec.get("name", "?"),
            "blend": _r(self.blend),
            "parts": parts,
            **({"carves": list(carves)} if carves else {}),
            **({"webs": list(web_result.webs)} if web_result.webs else {}),
            **({"skin": skin} if skin is not None else {}),
            "intent": intent,
        }
        base_receipt = self.build_receipt(
            anatomy_view,
            parts=parts,
            provenance=provenance,
        )
        receipt = (
            {**base_receipt, "myology": myology.receipt}
            if isinstance(myology, AcceptedMyology) and myology.receipt
            else base_receipt
        )
        return CompiledBody(
            graph=graph,
            receipt=receipt,
            anatomy=anatomy,
            eye_globes=eye_result.globes,
        )


def compile_file(spec_path: str) -> BodyCompileResult:
    p = Path(spec_path)
    spec = json.loads(p.read_text())
    comp = Compiler(spec, spec_dir=p.resolve().parent)
    return comp.compile()
