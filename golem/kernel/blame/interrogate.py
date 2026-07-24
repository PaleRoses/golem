from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from golem.assembly.obstructions import (
    DuplicateElementIdObstruction,
    EmptySurfaceConduitObstruction,
    MissingMechanicalSupportObstruction,
    MissingPhysicalVasculatureObstruction,
    MissingRigidPayloadInterfaceObstruction,
    MissingSolidMaterialAssignmentObstruction,
    MissingVascularWallMaterialObstruction,
)
from golem.kernel.anatomy import (
    DemandDeliveryMismatchObstruction,
    DisconnectedAnatomyRegionObstruction,
    DisconnectedVascularGraphObstruction,
    DuplicateAnatomyRegionObstruction,
    DuplicateExchangeBedObstruction,
    DuplicateMyotendinousUnitObstruction,
    DuplicateRegionHostObstruction,
    EmptyPerfusionTerritoryObstruction,
    InsufficientTerminalSitesObstruction,
    InvalidMyotendinousPathObstruction,
    MissingIntegumentLayerObstruction,
    MissingProvenanceObstruction,
    MissingSkeletalCrossSectionObstruction,
    MissingTerminalRegionObstruction,
    SymmetryMismatchObstruction,
    VascularGluingObstruction,
    project_vascular_obstruction,
    VascularIntersectionObstruction,
    VascularSearchBudgetObstruction,
    VascularSolverResidualObstruction,
)
from golem.kernel.blame.core import (
    DecomposedDisposition,
    DeclinedDisposition,
    IllPosedTaxon,
    LocalizedDisposition,
    ObstructionSignature,
    WellPosednessDisposition,
)
from golem.kernel.blame.extract import DEFAULT_BUDGET, BlameResult, extract_blame
from golem.kernel.blame.shrink import VerdictOracle
from golem.kernel.blame.verdict import check_verdict, obstruction_signature
from golem.kernel.body.relations import (
    BranchCycleObstruction,
    BranchSearchBudgetObstruction,
    FixedPlacementConflictObstruction,
    GoalRelationAuthorityObstruction,
    RelationalSolveExhaustedObstruction,
    RelationalSolverBudgetObstruction,
    UnderconstrainedRelationObstruction,
)
from golem.kernel.body.types import (
    DuplicateEyeIdObstruction,
    DuplicateMuscleIdObstruction,
    MissingBodySpecSectionObstruction,
)


@dataclass(frozen=True)
class BlameBudgetAccounting:
    spent: int
    limit: int
    exhausted: bool

    def to_json(self) -> dict[str, object]:
        return {
            "spent": self.spent,
            "limit": self.limit,
            "exhausted": self.exhausted,
        }


@dataclass(frozen=True)
class BlameEntry:
    blamed_addresses: tuple[str, ...]
    budget: BlameBudgetAccounting
    disposition: WellPosednessDisposition
    witness: object | None = None

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "blamed_addresses": self.blamed_addresses,
            "budget": self.budget.to_json(),
            "disposition": self.disposition.render(),
        }
        return (
            payload
            if self.witness is None
            else {**payload, "witness": self.witness}
        )


@dataclass(frozen=True)
class BlameInterrogation:
    entries: tuple[tuple[ObstructionSignature, BlameEntry], ...]

    def to_json(self) -> dict[str, object]:
        return {
            signature.render(): entry.to_json()
            for signature, entry in self.entries
        }


def well_posedness_disposition(obstruction: object) -> WellPosednessDisposition:
    match obstruction:
        case VascularGluingObstruction(
            gluing_interrogation=interrogation
        ) if interrogation is not None:
            return DecomposedDisposition(IllPosedTaxon.GLOBAL_SOLVE)
        case (
            MissingBodySpecSectionObstruction()
            | MissingTerminalRegionObstruction()
            | MissingSkeletalCrossSectionObstruction()
            | MissingIntegumentLayerObstruction()
            | MissingProvenanceObstruction()
            | EmptyPerfusionTerritoryObstruction()
            | InsufficientTerminalSitesObstruction()
            | EmptySurfaceConduitObstruction()
            | MissingPhysicalVasculatureObstruction()
            | MissingSolidMaterialAssignmentObstruction()
            | MissingVascularWallMaterialObstruction()
            | MissingMechanicalSupportObstruction()
            | MissingRigidPayloadInterfaceObstruction()
        ):
            return DeclinedDisposition(IllPosedTaxon.ABSENCE_DEGENERATE)
        case (
            DuplicateAnatomyRegionObstruction()
            | DuplicateRegionHostObstruction()
            | DuplicateExchangeBedObstruction()
            | DuplicateMyotendinousUnitObstruction()
            | DuplicateEyeIdObstruction()
            | DuplicateMuscleIdObstruction()
            | DuplicateElementIdObstruction()
            | InvalidMyotendinousPathObstruction()
            | DisconnectedAnatomyRegionObstruction()
            | SymmetryMismatchObstruction()
            | VascularIntersectionObstruction()
            | FixedPlacementConflictObstruction()
            | GoalRelationAuthorityObstruction()
        ):
            return DeclinedDisposition(IllPosedTaxon.COUPLED_UNIT)
        case (
            UnderconstrainedRelationObstruction()
            | RelationalSolveExhaustedObstruction()
            | RelationalSolverBudgetObstruction()
            | BranchCycleObstruction()
            | BranchSearchBudgetObstruction()
            | VascularSearchBudgetObstruction()
            | VascularGluingObstruction()
            | DisconnectedVascularGraphObstruction()
            | VascularSolverResidualObstruction()
            | DemandDeliveryMismatchObstruction()
        ):
            return DeclinedDisposition(IllPosedTaxon.GLOBAL_SOLVE)
        case _:
            return LocalizedDisposition()


def _distinct_obstructions(
    obstructions: Sequence[object],
) -> tuple[tuple[ObstructionSignature, object], ...]:
    return tuple(
        sorted(
            {
                obstruction_signature(obstruction): obstruction
                for obstruction in reversed(obstructions)
            }.items(),
            key=lambda item: item[0],
        )
    )


def _budget_accounting(
    result: BlameResult | None,
    budget_limit: int,
) -> BlameBudgetAccounting:
    return BlameBudgetAccounting(
        spent=result.check_invocations if result is not None else 0,
        limit=result.budget_limit if result is not None else max(budget_limit, 0),
        exhausted=result.exhausted_budget if result is not None else False,
    )


def _decomposed_gluing(
    obstruction: object,
) -> tuple[tuple[str, ...], object] | None:
    if not isinstance(obstruction, VascularGluingObstruction):
        return None
    projection = project_vascular_obstruction(obstruction)
    interrogation = projection.get("interrogation")
    if not isinstance(interrogation, Mapping):
        return None
    raw_addresses = interrogation.get("suspect_addresses", ())
    addresses = (
        tuple(map(str, raw_addresses))
        if isinstance(raw_addresses, Sequence)
        and not isinstance(raw_addresses, (str, bytes))
        else ()
    )
    return addresses, interrogation


def interrogate_blame(
    spec: Mapping[str, object],
    obstructions: Sequence[object],
    *,
    spec_dir: Path | str | None = None,
    budget: int = DEFAULT_BUDGET,
    probe_timeout: float | None = None,
    oracle: VerdictOracle = check_verdict,
) -> BlameInterrogation:
    distinct = _distinct_obstructions(obstructions)
    localized = tuple(
        obstruction
        for _signature, obstruction in distinct
        if isinstance(well_posedness_disposition(obstruction), LocalizedDisposition)
    )
    result = (
        extract_blame(
            spec,
            localized,
            spec_dir=spec_dir,
            budget=budget,
            probe_timeout=probe_timeout,
            oracle=oracle,
        )
        if localized
        else None
    )
    subsets = (
        {subset.obstruction: subset for subset in result.subsets}
        if result is not None
        else {}
    )
    accounting = _budget_accounting(result, budget)
    return BlameInterrogation(
        tuple(
            (
                signature,
                BlameEntry(
                    blamed_addresses=(
                        decomposition[0]
                        if decomposition is not None
                        else subset.blamed_addresses
                        if isinstance(disposition, LocalizedDisposition)
                        and subset is not None
                        and not subset.pathological_fork
                        else ()
                    ),
                    budget=accounting,
                    disposition=(
                        DeclinedDisposition(IllPosedTaxon.PATHOLOGICAL_FORK)
                        if isinstance(disposition, LocalizedDisposition)
                        and (subset is None or subset.pathological_fork)
                        else disposition
                    ),
                    witness=(
                        decomposition[1]
                        if decomposition is not None
                        else None
                    ),
                ),
            )
            for signature, obstruction in distinct
            for disposition in (well_posedness_disposition(obstruction),)
            for subset in (subsets.get(signature),)
            for decomposition in (_decomposed_gluing(obstruction),)
        )
    )
