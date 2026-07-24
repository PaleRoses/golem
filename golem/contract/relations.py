from __future__ import annotations

from collections.abc import Mapping

from golem.contract.model import ContractSection
from golem.kernel.body.relations import (
    RELATION_ALIASES,
    RELATION_CONVENTIONS,
    SUPPORTED_RELATION_SCOPES,
    BranchCycleObstruction,
    BranchSearchBudgetObstruction,
    DuplicateRelationIdObstruction,
    FixedPlacementConflictObstruction,
    GoalRelationAuthorityObstruction,
    MalformedRelationObstruction,
    RelationKind,
    RelationSolvePolicy,
    RelationalSolveExhaustedObstruction,
    RelationalSolverBudgetObstruction,
    UnderconstrainedRelationObstruction,
    UnknownRelationKindObstruction,
    UnreferenceableDerivedGeometryObstruction,
    UnresolvedRelationSelectorObstruction,
    UnsupportedRelationSelectorObstruction,
    _SUBJECT_SCOPE_OVERRIDES,
)
from golem.kernel.body.relation_solve import DeepeningStage, RelationalSolveConfig


_RESIDUAL_MEANING: Mapping[RelationKind, str] = {
    RelationKind.ATTACH_AT_NAMED_SITE: "child head coincides with the named site; required 0, observed Euclidean distance",
    RelationKind.ABOVE: "minimum subject y at least maximum reference y plus distance; required distance, observed subject_min_y minus reference_max_y",
    RelationKind.BELOW: "maximum subject y at most minimum reference y minus distance; required distance, observed reference_min_y minus subject_max_y",
    RelationKind.COINCIDENT: "selected points coincide; required 0, observed Euclidean separation",
    RelationKind.ALIGNED: "selected local +z axes parallel under the chosen convention; required 0 degrees, observed axis angle",
    RelationKind.PERPENDICULAR_TO: "selected local +z axes orthogonal; required 90 degrees, observed unsigned axis angle",
    RelationKind.MIRROR_OF: "subject pose is the sagittal reflection of reference across world x=0; required 0, observed maximum positional/angular deviation",
    RelationKind.BETWEEN: "subject bone head binds to the first reference site and tail to the second (two references, plural key); required 0, observed maximum endpoint gap",
}


def _aliases_of(kind: RelationKind) -> tuple[str, ...]:
    return tuple(
        alias for alias, canonical in RELATION_ALIASES.items() if canonical is kind
    )


def _pose_relation_kinds() -> tuple[RelationKind, ...]:
    return tuple(
        kind
        for kind in RelationKind
        if kind is not RelationKind.ATTACH_AT_NAMED_SITE
    )


def _kind_line(kind: RelationKind) -> str:
    meaning = _RESIDUAL_MEANING.get(kind, kind.value)
    aliases = _aliases_of(kind)
    alias_text = f" [aliases: {', '.join(aliases)}]" if aliases else ""
    return f"- {kind.value}: {meaning}{alias_text}"


def _capability_line(kind: RelationKind) -> str:
    references = ", ".join(
        scope.value for scope in SUPPORTED_RELATION_SCOPES[kind]
    )
    override = _SUBJECT_SCOPE_OVERRIDES.get(kind)
    if override is not None:
        subjects = ", ".join(scope.value for scope in override)
        return f"- {kind.value}: subject {subjects}; references {references}"
    return f"- {kind.value}: {references}"


def _convention_line(kind: RelationKind) -> str:
    names = ", ".join(convention.value for convention in RELATION_CONVENTIONS[kind])
    return f"- {kind.value}: {names}"


def render() -> str:
    config = RelationalSolveConfig()
    between_example = (
        '{"id": "span", "kind": "between", "subject": "bone:crossbar", '
        '"references": ["landmark:pillar_a/tail", "landmark:pillar_b/tail"], '
        '"solve": "subject"}'
    )
    attach_aliases = ", ".join(_aliases_of(RelationKind.ATTACH_AT_NAMED_SITE))
    pose_kinds = "\n".join(_kind_line(kind) for kind in _pose_relation_kinds())
    capability = "\n".join(
        _capability_line(kind) for kind in _pose_relation_kinds()
    )
    policies = ", ".join(policy.value for policy in RelationSolvePolicy)
    conventions = "\n".join(
        _convention_line(kind) for kind in RELATION_CONVENTIONS
    )
    stages = " then ".join(stage.value for stage in DeepeningStage)
    return f"""RELATIONAL PLACEMENT

Author placement as consequences over shared sites, not as coordinate arithmetic. A bone states where it belongs by naming a site every neighbour already names, or by declaring an explicit relation to another structure. The compiler defers equation construction until the whole skeleton and its constructed sites are final, then derives absolute poses by a deterministic hierarchical least-squares descent before any emission. This is the canonical surface. Fixed t/offset is the escape hatch for the rare case where you genuinely mean a coordinate.

ATTACHMENT
A bone's `attach` field takes exactly one of two disjoint forms.
- Named site (canonical): `{{"attach": {{"at": "landmark:upper_arm/wrist"}}}}`. The child head is movable and is driven to coincide with the selected constructed landmark. The authored `rest_dir` is the initial orientation. Two children whose `at` values normalize to the same selector share that site and receive identical coincidence equations; coincidence is implicit in the shared reference, never recomputed coordinates. Aliases for this form: {attach_aliases}.
- Fixed local (escape hatch): `{{"attach": {{"t": number?, "offset": [x, y, z]?}}}}`. Defaults are t=1 and offset=[0,0,0]. The resulting local translation is immutable through every deepening stage. Use it only when the coordinate is the intent.
A record mixing `at` with `t` or `offset` is rejected: one record admits exactly one placement owner. There is no top-level attach relation; the `attach` field is already the owner.

POSE RELATIONS
Explicit relations live under `pose.relations` as a list of records, each with a unique non-empty `id`, a `kind`, a `subject` selector, one `reference` selector (or, for between, a two-element `references` list), a `solve` policy, and (for above/below only) a finite non-negative `distance` in world units. The closed kind vocabulary, with accepted lexical aliases:
{pose_kinds}
An exact alias is accepted without warning; the receipt records both the authored kind and its canonical id. `distance` is lawful only on above and below. Named-site attachment is authored on the bone `attach` field above, never as a pose relation.

PLACEMENT VERSUS ASSERTION
Placement has one owner: the `attach` field. A named-site attachment's synthesized coincidence pins all three translation coordinates of its bone; a fixed attachment sets them immutably. above and below therefore never move a bone. They are verified assertions: the solver checks the vertical relation and reports quantitative required/observed evidence, satisfiable only as slack or reported as a conflict ({FixedPlacementConflictObstruction.__name__} for a fixed bone, {RelationalSolveExhaustedObstruction.__name__} for a named-site bone in a genuine tug-of-war). Express vertical placement freedom by site choice, and orientation freedom with aligned, perpendicular_to, or mirror_of, which yield swing and twist without touching the attach-owned translation. Do not reach for above/below to position a bone; author them to state the intent you want the solver to prove.

TWO-REFERENCE RELATIONS
between is the sole two-reference kind: it binds one subject bone across a pair of sites, authored with a plural `references` list of exactly two selectors, never a singular `reference`. The subject must be a Bone and both references must be Landmark sites. The subject's head is driven onto the first reference and its tail onto the second; the residual is those two endpoint coincidences and nothing else. Endpoint assignment is a frozen convention chosen by nearest initial satisfaction: between_forward binds head to the first site and tail to the second, between_reversed swaps them. `solve` accepts only subject or negotiate, because the two references are peers with no subject-versus-reference asymmetry to resolve; a reference policy is a {MalformedRelationObstruction.__name__}. distance is not lawful on between. Authoring a singular `reference` on between, or a plural `references` on any single-reference kind, is a {MalformedRelationObstruction.__name__} that names the exact key. Example:
  {between_example}
The subject's length is fixed authored data, never a solve variable, so between rotates and translates the bone but never stretches it. When the two references' separation differs from the subject's authored length, no eligible variable closes both endpoint gaps at once; the solve deepens through every stage and returns {RelationalSolveExhaustedObstruction.__name__} carrying the worst gap. That exhaustion is honest evidence, not a solver defect: the repair is to shorten or lengthen the subject bone, or to choose reference sites whose separation matches it.

SELECTOR CAPABILITY
Selectors refine existing Scope values by what each relation can constrain. Only pre-boolean constructed geometry is referenceable: an authored Bone, a named Part flesh primitive before SDF union, or a skeletal Landmark. Never the compiled boolean surface, a mesh face, an assembly solid, a plate, a conduit, or a vascular result. Whole, World, Chain, Contact, Element, Port, Mount, and Region scopes are rejected. Supported scope kinds per relation:
{capability}

SOLVE POLICY
`solve` is closed: {policies}. `subject` yields only the subject side, `reference` only the reference side, `negotiate` either side subject to fixed data, goal ownership, and deepening. between has no reference side to yield, so it admits only subject and negotiate; a `reference` policy on between is malformed. No policy unlocks a fixed attachment, the root world placement, or a goal-owned chain: absolute authoring stays fixed because the author chose it.

CONVENTIONS
A relation with more than one orientation reading exposes a finite convention set. The solver evaluates every convention against the provisional geometry, normalizes by the acceptance tolerances, selects the least residual, breaks a tie by canonical convention order, and freezes the choice for the whole solve. Conventions never switch mid-solve. above and below are unambiguous: they always mean world +y and world -y. mirror_of reflects across world x=0 and projects the reflected frame back to a proper right-handed SO(3) frame before comparison; the reflected frames match within tolerance only when the subtree's back-axis lies in the sagittal plane, so a spine authored off that plane reports residual angular deviation. Convention sets:
{conventions}

HIERARCHY AND DEEPENING
Each relation is owned by the lowest common ancestor of its subject and reference. The solve is a post-order fold: solve each child section, treat it as constant, solve the current node's geometry and its owned relations, glue if accepted. A rejected local problem deepens its eligible variable frontier by stage order: {stages}. LOCAL frees the current structure's eligible translation, swing, and twist. TRANSLATION adds descendant translations, moving a solved child rigidly. PARAMETER adds descendant swing and twist and reopens the affected constraints. Fixed attachments, the root, and goal chains never enter a frontier. Displacement from the seed is a tie-break, never a residual that hides missing intent.

ACTIVE SETS
Bone and Part extents come from finite constructed geometry, so above and below carry min/max branches. The solver evaluates the branches at the initial geometry, compiles only the active sites (ties broken by canonical selector order), solves, then re-evaluates. A changed active site recompiles and re-solves from the previous result. Acceptance requires a stable active set and the satisfied original expression. A repeating unsatisfied set is a cycle; reaching the sealed budget is a distinct budget outcome. Neither is mislabeled as geometric inconsistency.

GOAL-CHAIN NON-OVERLAP
The exact two-bone reach solver in pose.goals stays the sole owner of its chains. A relation whose movable closure overlaps a goal chain is rejected before either solver commits; this cut does not order two authorities over one bone.

OBSTRUCTIONS
Every failure is typed, addressed, and quantitative. Decode failures:
- {MalformedRelationObstruction.__name__}: structural defect; the reason names the exact field at the address.
- {DuplicateRelationIdObstruction.__name__}: two relations share an id; rename one.
- {UnknownRelationKindObstruction.__name__}: kind is neither canonical nor alias; use one of the suggested canonical ids.
- {UnresolvedRelationSelectorObstruction.__name__}: selector did not parse to a Scope; nearest constructed ids are suggested.
- {UnsupportedRelationSelectorObstruction.__name__}: selector parsed but its scope kind is not allowed for this relation; use a supported kind.
- {UnreferenceableDerivedGeometryObstruction.__name__}: selector names post-boolean or derived geometry; reference pre-boolean Bone/Part/Landmark only.
Solve failures:
- {FixedPlacementConflictObstruction.__name__}: a relation demanded a fixed datum move; required and observed quantify the contradiction. Relax the fixed attachment or the relation.
- {GoalRelationAuthorityObstruction.__name__}: the relation's movable closure overlaps a pose.goals chain; remove the overlap.
- {UnderconstrainedRelationObstruction.__name__}: the accepted Jacobian is rank-deficient and its null space moves the pose; free_dimension_count and parameter_addresses name the slack. Add a constraint or fix a parameter.
- {RelationalSolveExhaustedObstruction.__name__}: every deepening stage finished unsatisfied; the deepest frontier and worst required/observed are reported. This is not a proof of infeasibility.
- {RelationalSolverBudgetObstruction.__name__}: the evaluation budget was reached before convergence; the best partial iterate is not emitted.
- {BranchCycleObstruction.__name__}: a min/max active set cycled without satisfaction; the active_sets history shows the loop.
- {BranchSearchBudgetObstruction.__name__}: the active-set budget was reached, distinct from a cycle.

SOLVER CONSTANTS
The single numeric interpreter is scipy.optimize.least_squares (method trf, linear loss, no random restarts), imported only when at least one relation is present. Sealed RelationalSolveConfig values: position tolerance {config.position_tolerance}, angle tolerance {config.angle_tolerance_deg} degrees, singular-value tolerance {config.singular_value_tolerance}, maximum evaluations {config.maximum_evaluations}, maximum active-set re-solves {config.maximum_active_set_resolves}. Acceptance checks the original unnormalized measurements; residuals are normalized by the position and angle tolerances only inside the solver.
"""


SECTION = ContractSection("relations", render)
