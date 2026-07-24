from __future__ import annotations

from golem.contract.model import ContractSection
from golem.contract.receipts import CIRCULATION_WARNING
from golem.kernel.anatomy.graph import MissingProvenanceObstruction
from golem.kernel.anatomy.vocabulary import (
    IncompatibleIntegumentFormationObstruction,
    MissingTerminalRegionObstruction,
    NonterminalExchangeRegionObstruction,
)
from golem.kernel.body.types import (
    InvalidMuscleProfileObstruction,
    LoftSectionObstruction,
    MalformedEyeObstruction,
    MalformedMuscleBulkObstruction,
    MalformedMuscleDeclarationObstruction,
    NonGlossyEyeMaterialObstruction,
    SkinFormationAnatomyObstruction,
    UnknownEyeHostObstruction,
    UnresolvableMuscleAnchorObstruction,
    UnresolvableMuscleBulkBasisObstruction,
)
from golem.kernel.engine.types import (
    CanalEnvelopeObstruction,
    GeometryObstruction,
    IntegralRadiusUnsatisfied,
    MuscleFormationObstruction,
    SurfaceDetailObstruction,
)
from golem.materials.surface import (
    AppearancePaletteObstruction,
    SurfaceColorObstruction,
)


def render() -> str:
    return f"""GOLEM AUTHORING PRIMER

THESIS
Write consequences, not implementation choreography. A body/0.3 document is a deterministic body plan. `check` confirms the analytic consequence without a mesh; `compile` performs the expensive surface and assembly rite.

SKELETON
`skeleton.root` establishes the world origin and root flesh. `skeleton.bones` is an ordered attachment tree. A bone's `rest_dir` is expressed in its parent's frame; `length`, `attach`, `twist_deg`, `joint`, and optional pose determine its world frame. Bones are scaffolding and emit no surface by themselves.

FRAME COMPOSITION
Every `rest_dir` composes in the PARENT's frame, never in world: the child's world direction is `parent_R @ rest_dir`, and the child's own frame is derived, not authored (its +z is that composed direction; +y is the parent's +y projected orthogonal to it; +x is their cross). That derived frame orients the bone's own flesh axes and every one of its children's rest_dirs, so one girdle edit silently re-aims a whole limb chain. The axial trap: a chain continuing straight along its parent's axis needs local ≈ +z at EVERY generation — authoring the world direction into each child folds the chain, because each -z local inverts a generation (the root child is correct either way, so the fold only shows at generation 2+). To aim a bone at a desired world direction, author `parent_R^T @ desired_world`; dump actual world frames with the kernel's fk table before debugging geometry by hand.

PLACEMENT
State where a bone belongs as a consequence, not as coordinate arithmetic. The canonical `attach` form names a shared constructed site (`{{"attach": {{"at": "landmark:upper_arm/wrist"}}}}`); explicit `pose.relations` declare above, below, coincident, aligned, perpendicular_to, and mirror_of constraints over the selector vocabulary. The compiler defers equation construction until the whole tree and its sites are final, then derives absolute poses by a deterministic hierarchical solve before any emission. Fixed `{{"attach": {{"t": ..., "offset": ...}}}}` is the escape hatch for the rare coordinate you genuinely mean; a fixed coordinate is immutable through every solve stage. The RELATIONAL PLACEMENT section is the full authoring surface.

PARTS
`flesh` turns a bone frame into geometry. Start with the primitive matching the intended mass: `gencyl` for a shaft, `blob` for a rounded mass, `box` for a rounded prism, and `loft` only when the cross-section must be art-directed along the host bone. Every authored dimension is a half-extent in the bone frame: blob and box `size` are radii triples [hx, hy, hz] (size [0.1, 0.2, 0.3] emits a 0.20 x 0.40 x 0.60 extent), gencyl `radii` are radii, and loft `width`/`depth` are half-extents as the LOFT section states.

LOFT
The shortest loft is `{{"kind":"loft","name":"chest","sections":[{{"station":0,"width":0.3,"depth":0.2,"exponent":3,"roll":0}},{{"station":1,"width":0.2,"depth":0.14,"exponent":3,"roll":0}}]}}`. Stations are normalized host-bone coordinates; width and depth are half-extents, exponent is in [2,12], and roll is degrees. Add intermediate sections only to refine silhouette. Missing fields, nonpositive dimensions, bad exponents, or unordered stations produce `{LoftSectionObstruction.__name__}` and appear in `check` as `bad_loft_sections`; do not compensate with guessed extra stations.

OPERATORS
The closed flesh operator vocabulary is `blend | chamfer | crease | local_blend`. Omit `operator` for the byte-exact legacy smooth `blend`; author `"operator":"chamfer"` for a bevel-like join, `"operator":"crease"` for a hard union, or `"operator":"local_blend"` for a certified overlap-local gradient blend whose support is exact clean union outside the witnessed junction. Optional non-negative `blend` on a flesh record overrides the document-level composition radius for that part; blend, chamfer, and local_blend use the selected radius, while crease ignores it. Composition folds in declaration order, so the later incoming part owns the operator and radius at its junction. local_blend rejects uncalibrated fields, absent declared-iso overlap, escaped support, degenerate gradients, and topology changes instead of widening its reach. An unknown token produces `{GeometryObstruction.__name__}` with `R-composition-operator` for ordinary parts, or `{MalformedMuscleDeclarationObstruction.__name__}` for a muscle. Operators describe an incoming union edge, never a material or finish.

MYOLOGY
Author the intent first. The minimal base form is `{{"kind":"muscle","id":"quadriceps","origin":"part:thigh_mass","insertion":"landmark:shin/center","bulk":{{"relative":{{"to":"origin","width":1.0,"depth":0.7}}}},"definition":0.7}}`. The compiler places a five-section fusiform profile on the solved anchor span: tendinous ends, shoulders, and a belly at mid-span. `definition` is in [0,1]; higher values tighten the tendons and sharpen the belly. No stations or world-space widths are required.

`bulk.relative.to` selects the already-authored scale: `origin` or `insertion` uses part girth for a `part:` anchor and host-bone length for a `bone:` or `landmark:` anchor; `span` uses origin-to-insertion distance. `width` and `depth` are positive fractions of that scale. If a physical half-extent is truly intended, use `"bulk":{{"absolute":{{"width":0.14,"depth":0.09}}}}`. Never author `relative` and `absolute` together: that produces `{MalformedMuscleBulkObstruction.__name__}` with `relative_xor_absolute`. A zero or unavailable selected scale produces `{UnresolvableMuscleBulkBasisObstruction.__name__}`.

Detailed profile is opt-in: replace `bulk` and `definition` with legacy `sections` spanning exactly 0 to 1 and containing an interior belly wider and deeper than both ends. `sections`, `bulk`, and `mirror_of` are disjoint declaration forms; mixing them produces `{MalformedMuscleDeclarationObstruction.__name__}`. A non-fusiform explicit profile produces `{InvalidMuscleProfileObstruction.__name__}`. Bad selectors produce `{UnresolvableMuscleAnchorObstruction.__name__}`. A mirrored partner supplies its own reflected anchors plus `mirror_of` and inherits the exact reflected derived or explicit profile.

Radius-faithful formation is a separate opt-in on that same declaration: add `"formation":"skeleton_integral"` to derive the analytic rest-pose field, local frames, contact envelope, and solved receipt from the existing anchor span and sections. The admitted first family is circular, collinear, branch-free, and piecewise linear in radius; it does not reinterpret an anisotropic superellipse as a scalar radius. An unsatisfied circular radius returns `{MuscleFormationObstruction.__name__}` containing `{IntegralRadiusUnsatisfied.__name__}`; a singular radius slope returns `{CanalEnvelopeObstruction.__name__}` with the exact interval and bound. Branches remain separate muscle sections and join through the certified `local_blend` operator. Omit `formation` for the byte-exact legacy profiled gencyl.

MOUNTS
Body `mounts` attach a referenced hand subgrammar to a named skeletal port and move with that port. Assembly element mounts are different: they rigidly place an already-authored body or equipment graph. Equipment remains a separate solid; mounting never welds it into body flesh.

MIRRORS
`mirror: true` on a bone reflects the emitted subtree across world x=0. Author one canonical side and expect exact paired geometry. Receipts name the pair with a fixed vocabulary: parts gain `_m` mates, and vascular node ids mark the authored lane `:positive:` and its reflected mate `:negative:` (an unmirrored midline circuit is `:center:`). An explicit asymmetric pose on a mirrored subtree is reported because one declaration cannot predict two different sides. Muscles use explicit `mirror_of` declarations because each mass retains an independently addressable part id and union edge.

APPENDAGES
Top-level `appendages` are seeded growth declarations anchored to emitted part stations. They are expanded deterministically before meshing and then glued into the host field. The declaration remains the authoring source; generated branches are derived.

CONDUITS
Top-level `conduits` are authored surface loops or face lines. `band` emission creates a named surface stratum; `groove` displaces the host surface within the declared depth. `check` can preserve and inspect the declarations but cannot prove mesh occupancy. `compile` reports empty bands or emission failures as typed obstructions.

EYES
Start with `{{"id":"left_eye","host_bone_id":"skull","offset":[0.12,0.04,0.18],"radius":0.04,"socket":{{"radius":0.055}},"mirror":true}}`. The offset is bone-local, optional `t` moves along the host bone, and `mirror:true` derives the sagittal mate. The socket radius must exceed the globe radius; optional depth must remain below socket radius. Add `brow_ridge` only after the globe/socket read correctly. Bad shape or socket fields produce `{MalformedEyeObstruction.__name__}`, a missing host produces `{UnknownEyeHostObstruction.__name__}`, and a selected material rougher than the eye limit produces `{NonGlossyEyeMaterialObstruction.__name__}`.

PLATE POLICY
`plate_policy` belongs to the assembly element around the body or raw graph. The single-spec compile command projects a top-level policy onto that element. Cells, relief, grooves, and seam geometry are derived from the compiled surface; they are not authored replacement geometry.

PALETTE
For one local tint, author `"surface_color":{{"tint_linear_rgb":[0.08,0.06,0.1]}}`. `AppearanceMaterialId` is a CLOSED 10-id vocabulary: `obsidian_warden`, `obsidian_deep`, `steel_violet`, `emissive_seam`, `eye_gloss`, `vascular_supply_gold`, `vascular_return_violet`, `vascular_exchange_cyan`, `gambeson_dark`, and `neutral_gray`. An `appearance_palette` re-tints those ids; it never mints a new material. For a complete material-indexed appearance, author `"appearance_palette":{{"obsidian_warden":{{"tint_linear_rgb":[0.08,0.06,0.1]}}}}`; each key restricts onto every matching solid, eye, conduit, plate seam, or vascular record without replacing roughness, metallic, or emission. Both GLB/mesh export and `look` orthographic rendering honor palette tints. `tint_linear_rgb` remains linear through derivation and is converted exactly once to display sRGB at the shared face-color projection seam. Unknown material keys produce `{AppearancePaletteObstruction.__name__}`; invalid RGB or noise parameters produce `{SurfaceColorObstruction.__name__}`. A local `surface_color` and a palette entry for the solid's own material overlap and produce `{AppearancePaletteObstruction.__name__}` rather than precedence. Add `triplanar_value_noise` only when variation is intended.

FINISH
Color is not geometry. Optional `surface_detail` is the compile-time finish knob: `{{"kind":"curvature","amplitude":0.002,"scale":0.08}}` displaces the extracted creature surface using curvature. Omit it for the canonical surface. Non-finite or out-of-range fields produce `{SurfaceDetailObstruction.__name__}`; detail larger than `0.9*pitch`, larger than the mesh diagonal, or applied to equipment is rejected as an element surface-detail obstruction during `compile`. `check` is mesh-free and therefore cannot certify finish displacement.

ANATOMY
Top-level `anatomy` is anatomy/0.1 intent over the skeleton cover. Regions name hosted body areas; circulation names a pump region and exchange beds; optional myotendinous and integument declarations constrain tissue envelopes. The compiler derives the internal network. Do not author raw vessel meshes or confuse the conduit network with exterior body architecture.

An integument layer may opt into its own solved exterior with `"formation":"static_implicit_relaxation"`. Every anatomy region must own one formed integument layer at a shared thickness; incomplete cover, mixed legacy/formed cover, or unequal offsets produce `{IncompatibleIntegumentFormationObstruction.__name__}`. The compiler then withholds that thickness from flesh and derives a static projected, tangentially relaxed skin layer. If opted-in anatomy is rejected, `{SkinFormationAnatomyObstruction.__name__}` rejects the body instead of falling back to legacy flesh. Omit `formation` to preserve the legacy tissue envelope exactly.

Roles are explicit, opt-in contract vocabulary. A bone with `"role":"handle"` keeps normal frame composition but is contracted out of the perfused forest: it neither demands an exchange bed as a leaf nor revokes its parent's terminality. Flesh with `"role":"non_carrier"` keeps its authored dimensions and contributes no carrier sampling or provenance; a route left with only non-carrier flesh is rejected with `{MissingProvenanceObstruction.__name__}`, never silently inflated. Omitting either role preserves the default perfused-bone/carrier-flesh law.

The skeleton cover enforces four circulation invariants:
- Every terminal (leaf) bone must host an exchange bed; omission raises `{MissingTerminalRegionObstruction.__name__}`.
- Exchange beds may host only terminal bones; a nonterminal host raises `{NonterminalExchangeRegionObstruction.__name__}`.
- Every bone on every pump-to-bed circulation route must own emitted carrier-flesh provenance; omission raises `{MissingProvenanceObstruction.__name__}` during vascular realization.
- Exchange-bed paths follow their own region/route flesh topology and may leave the bone centerline. Every capsule still needs flesh cover from those same route/region bones; unrelated flesh, even nearby pump or thorax flesh, cannot cover it. See FEASIBLE ENVELOPE for the full aerial-cover law.

For each incoming part, body emission selects its per-part `blend` when present and otherwise the document-level value. Proprio predicts `k = selected_blend * ||padded_upper - padded_lower||_2`: the authored value is dimensionless and `k` is a world-space length in the body's authored distance units. Blend smooths with `k`, chamfer bevels with `k`, and crease is a hard union with zero smoothing reach.

{CIRCULATION_WARNING}

INTENT AND FEEDBACK
`contract`, `contacts`, `schematic`, pose goals, and anatomy become analytic intent carried by the emitted graph. Read `check` as the immediate mental-model confirmation. Read `compile` only when surface integrity, plate/conduit realization, assembly fit, or GLB exchange matters.

`look` follows the same compile dialect and assembly verdict before deriving front, side, top, or all orthographic PNG views. `eval` selects the stable evaluation reports registered for the supplied document kind.
"""


SECTION = ContractSection("primer", render)
