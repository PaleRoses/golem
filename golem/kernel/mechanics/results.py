"""Frozen mechanics result products and the accepted/rejected verdict union."""

from __future__ import annotations

from dataclasses import dataclass

from .model import (
    CellIndex,
    DomainSource,
    MaterialFractionResolution,
    NodeIndex,
    StructuredHexDomain,
    UnevaluatedPhysics,
    Vector3,
)
from .obstructions import MechanicsObstruction


@dataclass(frozen=True)
class SymmetricTensor3:
    xx: float
    yy: float
    zz: float
    xy: float
    yz: float
    xz: float


@dataclass(frozen=True)
class NodeMechanics:
    node: NodeIndex
    position: Vector3
    displacement: Vector3
    applied_force: Vector3
    thermal_equivalent_force: Vector3
    reaction: Vector3


@dataclass(frozen=True)
class CellMechanics:
    cell: CellIndex
    center: Vector3
    volume: float
    solid_fraction: float
    fraction_resolution: MaterialFractionResolution
    total_strain: SymmetricTensor3
    thermal_strain: SymmetricTensor3
    elastic_strain: SymmetricTensor3
    stress: SymmetricTensor3
    von_mises_stress: float
    yield_margin: float
    yield_safety_factor: float | None


@dataclass(frozen=True)
class BucklingNotRequested:
    """The load case requested only the accepted static thermoelastic solve."""


@dataclass(frozen=True)
class LinearizedBucklingReceipt:
    eigenproblem: str
    prestress_model: str
    lowest_positive_load_factor: float
    reciprocal_eigenvalue: float
    normalized_eigen_residual: float
    geometric_stiffness_frobenius_norm: float
    follower_pressure_stiffness_frobenius_norm: float
    stability_matrix_relative_asymmetry: float
    eigensolver_kind: str
    free_degree_count: int


@dataclass(frozen=True)
class AcceptedLinearizedBuckling:
    mode_displacements: tuple[Vector3, ...]
    receipt: LinearizedBucklingReceipt


type BucklingResult = BucklingNotRequested | AcceptedLinearizedBuckling


@dataclass(frozen=True)
class MechanicsReceipt:
    constitutive_model: str
    discretization: str
    domain_source: DomainSource
    node_count: int
    cell_count: int
    minimum_solid_fraction: float
    fractional_cell_count: int
    material_fraction_model: str
    unconstrained_degree_count: int
    constrained_degree_count: int
    maximum_displacement: float
    maximum_absolute_principal_strain: float
    maximum_von_mises_stress: float
    minimum_yield_margin: float
    minimum_yield_safety_factor: float | None
    normalized_linear_solver_residual: float
    total_body_force: Vector3
    total_nodal_force: Vector3
    total_internal_pressure_force: Vector3
    total_external_force: Vector3
    total_reaction_force: Vector3
    force_balance: Vector3
    relative_force_imbalance: float
    moment_balance_about_origin: Vector3
    relative_moment_imbalance: float
    lowest_positive_buckling_load_factor: float | None
    normalized_buckling_eigen_residual: float | None
    not_evaluated: tuple[UnevaluatedPhysics, ...]


@dataclass(frozen=True)
class AcceptedMechanics:
    domain: StructuredHexDomain
    nodes: tuple[NodeMechanics, ...]
    cells: tuple[CellMechanics, ...]
    buckling: BucklingResult
    receipt: MechanicsReceipt

    def node_result(self, node: NodeIndex) -> NodeMechanics | None:
        return next((result for result in self.nodes if result.node == node), None)

    def cell_result(self, cell: CellIndex) -> CellMechanics | None:
        return next((result for result in self.cells if result.cell == cell), None)


@dataclass(frozen=True)
class RejectedMechanics:
    obstructions: tuple[MechanicsObstruction, ...]


type MechanicsResult = AcceptedMechanics | RejectedMechanics
