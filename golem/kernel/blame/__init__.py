"""Minimal-blame (WHY) layer: approximate MUS over authored declarations."""

from __future__ import annotations

from golem.kernel.blame.aggregate import NearMissDistribution, near_miss_distributions
from golem.kernel.blame.core import (
    AuthoredAddress,
    AuthoredUnit,
    AuthoredUnitKind,
    CheckVerdict,
    DecomposedDisposition,
    DeclinedDisposition,
    DeclarationFamily,
    IllPosedTaxon,
    LocalizedDisposition,
    ObstructionSignature,
    WellPosednessDisposition,
)
from golem.kernel.blame.extract import (
    DEFAULT_BUDGET,
    BlameResult,
    BlameSubset,
    extract_blame,
)
from golem.kernel.blame.interrogate import (
    BlameBudgetAccounting,
    BlameEntry,
    BlameInterrogation,
    interrogate_blame,
    well_posedness_disposition,
)
from golem.kernel.blame.shrink import ShrinkBudget, guarded_verdict, shrink_target
from golem.kernel.blame.verdict import check_verdict, obstruction_signature

__all__ = (
    "AuthoredAddress",
    "AuthoredUnit",
    "AuthoredUnitKind",
    "BlameBudgetAccounting",
    "BlameEntry",
    "BlameInterrogation",
    "BlameResult",
    "BlameSubset",
    "CheckVerdict",
    "DEFAULT_BUDGET",
    "DecomposedDisposition",
    "DeclinedDisposition",
    "DeclarationFamily",
    "IllPosedTaxon",
    "LocalizedDisposition",
    "NearMissDistribution",
    "ObstructionSignature",
    "ShrinkBudget",
    "WellPosednessDisposition",
    "check_verdict",
    "extract_blame",
    "guarded_verdict",
    "interrogate_blame",
    "near_miss_distributions",
    "obstruction_signature",
    "shrink_target",
    "well_posedness_disposition",
)
