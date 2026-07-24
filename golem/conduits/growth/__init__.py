"""Typed appendage growth and compilation into host-fusing parts."""

from golem.conduits.growth.appendage import expand_appendages, grow_appendage
from golem.conduits.growth.grow import (
    appendage_envelope_graph,
    consolidate_appendage_chains,
    grow_appendage_network,
)
from golem.conduits.growth.types import (
    ContinueGrowth,
    GrowthNetwork,
    GrowthParameters,
    GrowthSection,
    GrowthState,
    GrowthTermination,
    TerminatedGrowth,
)

__all__ = [
    "ContinueGrowth",
    "GrowthNetwork",
    "GrowthParameters",
    "GrowthSection",
    "GrowthState",
    "GrowthTermination",
    "TerminatedGrowth",
    "appendage_envelope_graph",
    "consolidate_appendage_chains",
    "expand_appendages",
    "grow_appendage",
    "grow_appendage_network",
]
