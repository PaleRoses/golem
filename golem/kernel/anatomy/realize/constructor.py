"""Swappable circuit-constructor seam over the shared judgment layer.

A constructor proposes a candidate vascular network for one circulation
circuit's admissible domain.  The whole-body judgment (geometry clearance and
the hydraulic solver) verifies every constructor's proposal identically; a
constructor differs only in how it reaches the candidate, never in what makes
one acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from golem.kernel.anatomy.graph import RejectedVasculature

if TYPE_CHECKING:
    from golem.kernel.anatomy.realize.carriers import (
        _CircuitConstructionDomain,
        _CircuitGeometry,
        _StructuralCostIndex,
    )
    from golem.kernel.anatomy.vocabulary import (
        AcceptedAnatomy,
        CirculationCircuit,
        SealedVascularConfig,
    )


class CircuitConstructor(Protocol):
    """Propose a candidate circuit geometry for the judgment layer to verify."""

    def construct(
        self,
        circuit: CirculationCircuit,
        domain: _CircuitConstructionDomain,
        config: SealedVascularConfig,
        structural_cost_index: _StructuralCostIndex | None,
    ) -> _CircuitGeometry | RejectedVasculature: ...


@dataclass(frozen=True)
class GreedyCCOConstructor:
    """Default: constrained constructive optimization with staged relaxation."""

    def construct(
        self,
        circuit: CirculationCircuit,
        domain: _CircuitConstructionDomain,
        config: SealedVascularConfig,
        structural_cost_index: _StructuralCostIndex | None,
    ) -> _CircuitGeometry | RejectedVasculature:
        from golem.kernel.anatomy.realize.parity import _greedy_circuit_geometry

        return _greedy_circuit_geometry(
            circuit, domain, config, structural_cost_index
        )


DEFAULT_CIRCUIT_CONSTRUCTOR: CircuitConstructor = GreedyCCOConstructor()


def realize_circuit(
    constructor: CircuitConstructor,
    circuit: CirculationCircuit,
    terminal_pair_count: int,
    accepted: AcceptedAnatomy,
    part_by_id: dict[str, dict],
    provenance: dict[str, str],
    landmarks: dict[str, object],
    bone_use_count: dict[str, int],
    shared_pump_interfaces: tuple[tuple[float, float, float], ...],
    config: SealedVascularConfig,
    structural_cost_index: _StructuralCostIndex | None,
) -> _CircuitGeometry | RejectedVasculature:
    from golem.kernel.anatomy.realize.parity import _circuit_construction_domain

    domain = _circuit_construction_domain(
        circuit,
        terminal_pair_count,
        accepted,
        part_by_id,
        provenance,
        landmarks,
        bone_use_count,
        shared_pump_interfaces,
        config,
    )
    return (
        domain
        if isinstance(domain, RejectedVasculature)
        else constructor.construct(
            circuit, domain, config, structural_cost_index
        )
    )
