from __future__ import annotations

from golem.contract.model import ContractSection
from golem.kernel.anatomy.graph import (
    InfeasibleBifurcationObstruction,
    VascularSearchBudgetObstruction,
)


CIRCULATION_WARNING = f"""WARNING
`check`'s `CIRCULATION` line proves topological closure of anatomy intent only; it does not prove vascular feasibility.
`{InfeasibleBifurcationObstruction.__name__}` and the other realization obstructions are evaluated by the separate `VASCULAR feasibility` block and by `compile`.
Bifurcation solvability is globally coupled to the whole skeleton tree, pose, and exchange-bed count; the pump needs at least two exchange beds; and feasibility is not scale-invariant.
A green `CIRCULATION` line is not evidence that vascular realization will compile; require `VASCULAR feasibility: ACCEPTED` or a successful compile.
The FEASIBLE ENVELOPE section states the corridor, territory, and bifurcation constraints that decide realization."""


def render() -> str:
    return f"""READING RECEIPTS

CHECK
`PROPRIO ... | no mesh` proves the immediate tier stayed analytic. `GLOBAL` reports bounds, proportions, centroid, blend scale, and predicted connected components. `ASSERT` reports authored contract verdicts; failures name measured consequences and corrective knobs. `ANOMALY` reports undeclared fusion, missing fusion, deep burial, cross-plane fusion, blend ambiguity, or floating support. Its default count is the unexpected set: observations whose witness carries `"expected":true` are authored/derived intent and do not count as anomalies. `check --all` removes every presentation cap: all failed/unmeasurable assertions, all unexpected anomalies, and all suggested carrier-scale rows are rendered; it does not reclassify expected observations. `CIRCULATION` reports derived anatomy when anatomy intent exists. Its `N suggested carrier scales` rows (`suggested_scale`) are consumable advisory suggestions only — they do NOT mutate authored geometry. `OBSTRUCTIONS` contains typed body or geometry validation records. A successful check means the document decoded, senses were produced, every assert passed, no obstruction was rendered, and vascular feasibility was accepted when applicable. Anomaly records remain advisory.

SESSION
The CLI session verdict uses the same `witness.expected` discriminator. By default its `diagnostics` array omits entries whose witness has `"expected":true` and adds `diagnostics_summary` with total, expected, unexpected, and filtering state; `session --all` emits the raw diagnostic set. Acceptance requires zero unexpected diagnostics: expected authored-intent observations ride, while every undeclared observation blocks regardless of severity. Status is computed from the unfiltered evidence, so rendering never changes acceptance.

MIRROR IDS
Receipts map mirrored authored ids deterministically. A `mirror:true` subtree retains the authored part id and emits an `_m` reflected mate. Vascular ids name the authored circuit lane `:positive:`, the reflected mate `:negative:`, and an unmirrored circuit `:center:`. These are receipt vocabulary, not additional authored ids.

RELATIONS
A relational document's solved receipt carries a `relations` block: authored and canonical relation ids, resolved subject and `reference` selector addresses, a `reference_b` selector holding between's second reference (null for every single-reference kind), selected conventions, the deepest translation/parameter frontier reached, per-relation required/observed/normalized-residual measurements, Jacobian rank and eligible-variable counts, and the stable min/max active sites. Acceptance is decided on the original unnormalized measurements. The full solved frame table is not duplicated here; the landmark table is the derived coordinate view and the solved placement carrier is the authority. A legacy document with no relations emits no `relations` key and stays byte-identical.

REJECTION
A relational decode or solve failure is blocking: there is no authoritative pose to emit, so the compiler returns a typed `RejectedBody` rather than a partial graph. Every obstruction carries an authoring address and, where a geometric obligation exists, a quantitative required/observed pair. `check`, the body CLI, assembly ingestion, the silhouette loader, and the evaluation adapters all project the same typed evidence; none flatten it into a traceback. An under-constrained solve reports variable count, rank, and free dimensions; an exhausted solve reports the deepest frontier and worst residual and is not a proof of infeasibility.

{CIRCULATION_WARNING}
A vascular budget cutoff is reported as `{VascularSearchBudgetObstruction.__name__}` with its evaluated-state count and remaining queue, never as `{InfeasibleBifurcationObstruction.__name__}`: a search that ran out of budget has not proven geometry infeasible. The FEASIBLE ENVELOPE section states the staged-search trichotomy.

COMPILE
Each `element`, `conduit`, `plate seam`, or `vascular_*` line is a derived assembly stratum. Surface integrity reports components, watertightness, dust, faces, and violation count. Fit, physical, hydraulic, thermal, mechanics, and refinement lines appear only when that evidence was requested. The final `assembly verdict` is authoritative. A `RejectedAssembly` prints every typed obstruction and emits no GLB.

RESOLUTION
`--pitch` asks the compiler to choose a bounded resolution for a world-space pitch; vocabulary violations under that policy can reject integrity. `--res` is `PinnedResolution`: the author chose the grid. Feature-size and sharp-edge records are therefore advisories attached to that choice, not evidence that the compiler selected a bad grid. Change `--res` or the authored feature if the advisory matters.

GLB
`scene -> path` appears only after an accepted assembly was exported through the exchange boundary. The GLB and its material sidecar are derived views; the assembly receipt remains the authority.

`look` prints a PNG path only after the shared assembly descent and requested view render both accept. `eval` prints each applicable stable evaluation report; unavailable or obstructed evaluations remain typed rejections rather than fabricated scores.
"""


SECTION = ContractSection("receipts", render)
