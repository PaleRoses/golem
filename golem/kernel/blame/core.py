"""The closed vocabulary of the minimal-blame layer: addresses and verdicts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DeclarationFamily(StrEnum):
    """The addressable families of authored declarations a spec may carry."""

    FLESH = "flesh"
    MUSCLE = "muscle"
    EYE = "eye"
    CLAUSE = "clause"
    ATTACH = "attach"
    BONE = "bone"
    REGION = "region"
    EXCHANGE_BED = "bed"


class AuthoredUnitKind(StrEnum):
    LIMB = "limb"
    REGION = "region"
    EXCHANGE_BED = "bed"


class IllPosedTaxon(StrEnum):
    COUPLED_UNIT = "coupled-unit"
    ABSENCE_DEGENERATE = "absence-degenerate"
    GLOBAL_SOLVE = "global-solve"
    PATHOLOGICAL_FORK = "pathological-fork"


# Leaf families are exonerable one at a time; structural families anchor the
# skeleton/circulation referential graph and are therefore tried last, so the
# greedy scan removes the separable declarations before the load-bearing spine.
_FAMILY_SHRINK_RANK: dict[DeclarationFamily, int] = {
    DeclarationFamily.FLESH: 0,
    DeclarationFamily.MUSCLE: 1,
    DeclarationFamily.EYE: 2,
    DeclarationFamily.CLAUSE: 3,
    DeclarationFamily.ATTACH: 4,
    DeclarationFamily.BONE: 5,
    DeclarationFamily.REGION: 6,
    DeclarationFamily.EXCHANGE_BED: 7,
}

_FAMILY_DISPLAY_RANK: dict[DeclarationFamily, int] = {
    DeclarationFamily.BONE: 0,
    DeclarationFamily.FLESH: 1,
    DeclarationFamily.MUSCLE: 2,
    DeclarationFamily.EYE: 3,
    DeclarationFamily.REGION: 4,
    DeclarationFamily.EXCHANGE_BED: 5,
    DeclarationFamily.CLAUSE: 6,
    DeclarationFamily.ATTACH: 7,
}


@dataclass(frozen=True, order=True)
class AuthoredAddress:
    """A single removable authored declaration, keyed by its structural locus.

    ``parent_index`` names the owning bone for flesh entries and is ``None`` for
    every top-level family. ``index`` is the position within the owning list of
    the *original* spec, so a set of these addresses always refers to the
    pristine document rather than a partially shrunk one.
    """

    family: DeclarationFamily
    parent_index: int | None
    index: int
    address: str

    @property
    def shrink_key(self) -> tuple[int, int, int]:
        return (_FAMILY_SHRINK_RANK[self.family], self.parent_index or 0, self.index)

    @property
    def display_key(self) -> tuple[int, int, int]:
        return (_FAMILY_DISPLAY_RANK[self.family], self.parent_index or 0, self.index)


@dataclass(frozen=True)
class AuthoredUnit:
    identifier: str
    kinds: frozenset[AuthoredUnitKind]
    addresses: tuple[AuthoredAddress, ...]


@dataclass(frozen=True)
class LocalizedDisposition:
    def render(self) -> str:
        return "localized"


@dataclass(frozen=True)
class DecomposedDisposition:
    """A global solve that could not admit declaration-level MUS shrinking,
    but whose own algebra produced a typed, addressed decomposition."""

    taxon: IllPosedTaxon

    def render(self) -> str:
        return f"decomposed: {self.taxon.value}"


@dataclass(frozen=True)
class DeclinedDisposition:
    taxon: IllPosedTaxon

    def render(self) -> str:
        return f"declined: {self.taxon.value}"


type WellPosednessDisposition = (
    LocalizedDisposition | DecomposedDisposition | DeclinedDisposition
)


@dataclass(frozen=True, order=True)
class ObstructionSignature:
    """The presence-identity of an obstruction: its class and locating fields.

    Quantitative fields (required, observed, margins, candidate counts) are
    deliberately excluded — two runs whose solver drifted by a fraction still
    denote the *same* rejection, so matching is on locus, never on magnitude.
    """

    kind: str
    identity: tuple[tuple[str, str], ...]

    def render(self) -> str:
        locus = " ".join(f"{name}={value}" for name, value in self.identity)
        return f"{self.kind}({locus})" if locus else self.kind


@dataclass(frozen=True)
class CheckVerdict:
    """The outcome of one deterministic check probe over a (possibly forked) spec.

    ``resolved`` is ``False`` only when the pipeline raised — a forked spec that
    no longer decodes — which the shrinker reads as "this removal is not a clean
    reproduction of the target", never as acceptance.
    """

    stage: str
    rejected: bool
    resolved: bool
    obstructions: tuple[ObstructionSignature, ...]
    raw_obstructions: tuple[object, ...] = ()

    def reproduces(self, target: ObstructionSignature) -> bool:
        return self.rejected and target in self.obstructions
