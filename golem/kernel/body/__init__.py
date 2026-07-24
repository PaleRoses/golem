"""GOLEM v0.3 body compiler (M3) -- the minimal stance compiler.

A skeleton spec (``body/0.3``) compiles, in ONE deterministic pass, down to a
flat v0.2/v0.3 part-graph the frozen engine consumes, plus an embedded
**intent block** for the M2 sensory stack (proprio/asserts) and a **compile
receipt** sidecar. Skeletons are pure scaffolding: bones carry no geometry;
``flesh`` records attach to bones in bone-local coordinates, so a single
field edit (``forearm.length *= 0.88``) recomputes every downstream thing.
Gencyl flesh may supply a strictly increasing ``stations`` array matching its
``radii`` anchors; omitted stations retain uniform sampling over ``span``.

FRAME CONVENTION (R-7, written down once)
=========================================
Every bone has a right-handed local frame; its columns are the local +x, +y,
+z axes expressed in world coordinates (an SO(3) matrix ``R`` with
``R @ local = world_direction``).

  * local +z = the bone's proximal->distal axis (head -> tail).
  * local +y ("back") = the parent's +y projected orthogonal to +z, then
    rotated by ``twist_deg`` about +z. If the parent's +y lies within 1 degree
    of the bone axis, fall back to the parent's +x (a deterministic tie-break).
  * local +x = cross(+y, +z) (completes the right-handed frame).

``rest_dir`` is authored in the PARENT frame and normalized by the compiler;
the bone's world axis is ``R_parent @ normalize(rest_dir)``. The ROOT frame is
the world axes at ``root.world``. A PORT frame is the bone frame sampled at
parameter ``t`` (origin ``head + R @ [0,0,t*length]``) plus a twist about +z --
chosen so the hand sub-grammar's convention (fingers +z, back of hand +y,
palm -y) drops in unchanged, exactly as ``pilots/mount_hands.wrist_frame``.

FK: ``tail = head + R @ [0,0,length]``; ``child.head = parent.head +
R_parent @ (attach.offset + [0,0, attach.t * parent.length])``. Explicit pose:
``R_bone = R_parent @ R_rest @ R_pose`` with ``R_pose = Ry(yaw) Rx(pitch)
Rz(roll)`` (degrees) in the joint's rest-local frame.

ABSOLUTE FENCES (ExecPlan Decision Log -- these OVERRIDE fable-a)
================================================================
1. EXACTLY ONE analytic two-bone reach solve: law of cosines + author-declared
   pole vector. Closed form, no iteration, no seeds, one goal per chain.
   Unreachable target -> clamp at full extension aimed at the target and emit a
   ``goal_unreachable`` violation carrying the gap and per-knob single-dimension
   gradient. Solved angles recorded in the receipt. NO other IK.
2. NO masses in the compiler: the receipt has NO masses/volume/centroid/support
   block. Balance lives in proprio.py, computed from the emitted parts.
3. Joint limits are RECORDED and violations REPORTED, never enforced. This also
   covers IK-solved angles: a solved angle outside a declared limit is applied
   as solved and reported with a residual. The ONLY clamp anywhere is the
   full-extension clamp for an unreachable reach goal (fence 1).
4. Joint typology is exactly dof in {"fixed","hinge","ball"}. Posing a ``fixed``
   joint is a violation. No twist-limit machinery, no pose libraries/registry.
5. No pose libraries, no retargeting, no animation constructs.

CLI
===
    uv run --frozen python -m golem.kernel.body specs/knight_body.json \
        -o /tmp/knight_compiled.json [--receipt /tmp/receipt.json]

Exit code is 0 regardless of violations (informative, veto-poor).
"""

from __future__ import annotations

from golem.kernel.body.canonical import _canonical_json
from golem.kernel.body.carves import compile_carves
from golem.kernel.body.webs import compile_webs
from golem.kernel.body.cli import main
from golem.kernel.body.compile import Compiler, compile_file
from golem.kernel.body.eyes import compile_eyes, eye_globe_graph
from golem.kernel.body.linalg import mat_to_quat
from golem.kernel.body.myology import (
    AcceptedMyology,
    MyologyResult,
    RejectedMyology,
    compile_myology,
)
from golem.kernel.body.project import (
    obstruction_text,
    project_obstruction,
    project_rejected_body,
    rejected_body_text,
)
from golem.kernel.body.relations import (
    RelationKind,
    RelationSolvePolicy,
    decode_attachment,
    decode_relations,
)
from golem.kernel.body.report import render_digest, run_asserts, run_typed_asserts
from golem.kernel.body.types import (
    AcceptedCarves,
    AcceptedWebs,
    CARVE_KINDS,
    AcceptedEyes,
    BodyCompileResult,
    BodyObstruction,
    CarveCompileResult,
    CarveKind,
    CarveObstruction,
    CarvePartCollisionObstruction,
    CarveRule,
    BodySpecDecodeObstruction,
    BrowRidge,
    CyclicMuscleMirrorObstruction,
    CompiledBody,
    DegenerateMuscleSpanObstruction,
    DIALECT,
    DuplicateMuscleIdObstruction,
    DuplicateCarveIdObstruction,
    DuplicateEyeIdObstruction,
    DuplicateWebIdObstruction,
    EyeCompileResult,
    EyeGlobe,
    EyeHostFrame,
    EyeObstruction,
    EyePartCollisionObstruction,
    EyeRule,
    EyeSocket,
    EyeSpec,
    InvalidMuscleProfileObstruction,
    MalformedBodySpecSectionObstruction,
    MalformedCarveObstruction,
    MalformedEyeObstruction,
    MalformedWebObstruction,
    MalformedMuscleBulkObstruction,
    MalformedMuscleDeclarationObstruction,
    MalformedMuscleSectionsObstruction,
    MisplacedBodySpecFieldObstruction,
    MissingBodySpecSectionObstruction,
    MuscleMirrorAnchorMismatchObstruction,
    MuscleFormationObstruction,
    MuscleObstruction,
    MusclePartCollisionObstruction,
    NonGlossyEyeMaterialObstruction,
    RejectedBody,
    RejectedCarves,
    RejectedEyes,
    RejectedWebs,
    RelationalSolveReceipt,
    SkinFormationAnatomyObstruction,
    SolvedLocalPose,
    UnresolvableMuscleAnchorObstruction,
    UnresolvableMuscleBulkBasisObstruction,
    UnresolvedMuscleMirrorObstruction,
    UnsupportedMuscleAnchorObstruction,
    UnknownBodySpecKeyObstruction,
    UnknownEyeHostObstruction,
    UnknownEyeMaterialObstruction,
    UnknownWebAnchorObstruction,
    WebCompileResult,
    WebObstruction,
    WebPartCollisionObstruction,
    WebRule,
)

__all__ = [
    "BodyCompileResult",
    "AcceptedCarves",
    "AcceptedWebs",
    "CARVE_KINDS",
    "BodyObstruction",
    "BodySpecDecodeObstruction",
    "CarveCompileResult",
    "CarveKind",
    "CarveObstruction",
    "CarvePartCollisionObstruction",
    "CarveRule",
    "AcceptedMyology",
    "AcceptedEyes",
    "BrowRidge",
    "Compiler",
    "CompiledBody",
    "CyclicMuscleMirrorObstruction",
    "DegenerateMuscleSpanObstruction",
    "DIALECT",
    "DuplicateCarveIdObstruction",
    "DuplicateEyeIdObstruction",
    "DuplicateMuscleIdObstruction",
    "DuplicateWebIdObstruction",
    "EyeCompileResult",
    "EyeGlobe",
    "EyeHostFrame",
    "EyeObstruction",
    "EyePartCollisionObstruction",
    "EyeRule",
    "EyeSocket",
    "EyeSpec",
    "InvalidMuscleProfileObstruction",
    "MalformedBodySpecSectionObstruction",
    "MalformedCarveObstruction",
    "MalformedEyeObstruction",
    "MalformedWebObstruction",
    "MalformedMuscleBulkObstruction",
    "MalformedMuscleDeclarationObstruction",
    "MalformedMuscleSectionsObstruction",
    "MisplacedBodySpecFieldObstruction",
    "MissingBodySpecSectionObstruction",
    "MuscleMirrorAnchorMismatchObstruction",
    "MuscleFormationObstruction",
    "MuscleObstruction",
    "MusclePartCollisionObstruction",
    "MyologyResult",
    "NonGlossyEyeMaterialObstruction",
    "RejectedBody",
    "RejectedCarves",
    "RejectedEyes",
    "RejectedWebs",
    "RejectedMyology",
    "RelationKind",
    "RelationSolvePolicy",
    "RelationalSolveReceipt",
    "SkinFormationAnatomyObstruction",
    "SolvedLocalPose",
    "_canonical_json",
    "compile_file",
    "compile_carves",
    "compile_eyes",
    "decode_attachment",
    "decode_relations",
    "eye_globe_graph",
    "main",
    "mat_to_quat",
    "obstruction_text",
    "project_obstruction",
    "project_rejected_body",
    "rejected_body_text",
    "render_digest",
    "run_asserts",
    "run_typed_asserts",
    "UnknownBodySpecKeyObstruction",
    "UnknownEyeHostObstruction",
    "UnknownEyeMaterialObstruction",
    "UnknownWebAnchorObstruction",
    "WebCompileResult",
    "WebObstruction",
    "WebPartCollisionObstruction",
    "WebRule",
    "UnresolvableMuscleAnchorObstruction",
    "UnresolvableMuscleBulkBasisObstruction",
    "UnresolvedMuscleMirrorObstruction",
    "UnsupportedMuscleAnchorObstruction",
    "compile_myology",
    "compile_webs",
]
