"""Structured-domain small-strain isotropic linear thermoelasticity.

This module is the D25/M7 mechanics owner.  It consumes an already-derived
structured solid domain and already-resolved physical properties; it owns no
body, mesh-authoring language, material catalogue, or vascular graph.  The
authoritative constitutive claim is deliberately narrow:

``small-strain isotropic linear thermoelasticity on a structured volumetric/
SDF-derived domain`` using conforming trilinear eight-node hexahedra and
2x2x2 Gauss integration.

Sub-grid channels are not wished into existence.  ``solid_cells`` is the
explicit active mask.  Every active cell additionally carries a frozen solid
fraction whose resolution is either ``FULL_SOLID`` or
``HOMOGENIZED_EFFECTIVE_PROPERTIES``.  A fractional cell is accepted only in
the latter case, where the caller supplies already-homogenized effective
properties; this solver does not scale bulk properties by a guessed mixture
law.  ``UNRESOLVED`` fractions are typed rejections.  Even an accepted
homogenized field does not resolve lumen-wall stress concentrations.

All coordinates are metres, Young's modulus and yield strength are pascals,
density is kg/m^3, forces are newtons, pressure is pascals, acceleration is
m/s^2, and ``NodalTemperatureChange`` is a temperature difference in kelvin.
An internal-pressure face is an exposed solid face adjacent to a lumen/void;
its traction on the solid is ``-pressure * solid_outward_normal``.  Its
linearized buckling contribution is a physical follower load: the consistent
surface load-stiffness is evaluated on the accepted displaced Q1 face rather
than laundering pressure into a fixed dead traction.

The pure entry point :func:`solve_mechanics` returns
``AcceptedMechanics | RejectedMechanics``.  Expected incompatibilities are
typed obstructions.  When ``require_linearized_buckling`` is true, the
accepted static prestress assembles the initial-stress geometric stiffness and
the follower-pressure tangent, then solves
``K phi = lambda (-K_G + K_pressure) phi``.  The smallest positive load factor
and mode are returned with an eigen-residual receipt; absent, singular, or
unresolved spectra reject rather than acquiring a ceremonial factor.
"""

from .embedded_channel import (
    AcceptedEmbeddedChannelMechanics,
    EmbeddedChannelConstitutiveResponse,
    EmbeddedChannelCriteria,
    EmbeddedChannelMechanicsReceipt,
    EmbeddedChannelMechanicsResult,
    EmbeddedChannelObstruction,
    EmbeddedChannelScaleSeparationObstruction,
    EmbeddedChannelSectionMechanics,
    EmbeddedChannelVolumeFractionObstruction,
    EmbeddedChannelVolumeKind,
    FullSolidEmbeddedChannelResponse,
    HomogenizedEmbeddedChannelResponse,
    InsufficientEmbeddedChannelBurstMarginObstruction,
    InvalidEmbeddedChannelInputObstruction,
    RejectedEmbeddedChannelMechanics,
    UnsupportedEmbeddedChannelExternalPressureObstruction,
    UnresolvedEmbeddedChannelConstitutiveObstruction,
    evaluate_embedded_channel_mechanics,
)
from .model import (
    DISCRETIZATION_NAME,
    MODEL_NAME,
    BoundaryFace,
    CellConstitutiveAssignment,
    CellIndex,
    DomainSource,
    InternalPressureFace,
    IsotropicConstitutiveProperties,
    MaterialFractionResolution,
    MechanicsCriteria,
    MechanicsProblem,
    NodalForce,
    NodalSupport,
    NodalTemperatureChange,
    NodeIndex,
    StructuredHexDomain,
    UnevaluatedPhysics,
    Vector3,
)
from .obstructions import (
    AbsentBucklingSpectrumObstruction,
    ConflictingSupportObstruction,
    DuplicateCellObstruction,
    DuplicateCellPropertiesObstruction,
    ExcessiveDisplacementObstruction,
    ForceBalanceObstruction,
    InsufficientBucklingMarginObstruction,
    InsufficientYieldMarginObstruction,
    InvalidAcceptanceCriteriaObstruction,
    InvalidBodyForceObstruction,
    InvalidBucklingEigenpairObstruction,
    InvalidConstitutivePropertiesObstruction,
    InvalidMaterialFractionObstruction,
    InvalidNodalLoadObstruction,
    InvalidPressureBoundaryObstruction,
    InvalidSupportObstruction,
    InvalidTemperatureFieldObstruction,
    LinearSolverFailureObstruction,
    MalformedDomainObstruction,
    MaterialFractionRule,
    MechanicsObstruction,
    MissingCellPropertiesObstruction,
    NonFiniteMechanicsResultObstruction,
    RigidBodyModeObstruction,
    SingularBucklingSpectrumObstruction,
    SmallStrainLimitObstruction,
    SolverResidualObstruction,
    UnknownCellPropertiesObstruction,
    UnresolvedBucklingSpectrumObstruction,
    UnresolvedMaterialFractionObstruction,
)
from .results import (
    AcceptedLinearizedBuckling,
    AcceptedMechanics,
    BucklingNotRequested,
    BucklingResult,
    CellMechanics,
    LinearizedBucklingReceipt,
    MechanicsReceipt,
    MechanicsResult,
    NodeMechanics,
    RejectedMechanics,
    SymmetricTensor3,
)
from .solve import solve_mechanics

__all__ = (
    "MODEL_NAME",
    "DISCRETIZATION_NAME",
    "Vector3",
    "DomainSource",
    "BoundaryFace",
    "UnevaluatedPhysics",
    "MaterialFractionResolution",
    "EmbeddedChannelVolumeKind",
    "NodeIndex",
    "CellIndex",
    "StructuredHexDomain",
    "IsotropicConstitutiveProperties",
    "CellConstitutiveAssignment",
    "NodalSupport",
    "NodalForce",
    "NodalTemperatureChange",
    "InternalPressureFace",
    "MechanicsCriteria",
    "MechanicsProblem",
    "EmbeddedChannelCriteria",
    "EmbeddedChannelSectionMechanics",
    "FullSolidEmbeddedChannelResponse",
    "HomogenizedEmbeddedChannelResponse",
    "UnresolvedEmbeddedChannelConstitutiveObstruction",
    "EmbeddedChannelConstitutiveResponse",
    "EmbeddedChannelMechanicsReceipt",
    "AcceptedEmbeddedChannelMechanics",
    "InvalidEmbeddedChannelInputObstruction",
    "EmbeddedChannelScaleSeparationObstruction",
    "EmbeddedChannelVolumeFractionObstruction",
    "UnsupportedEmbeddedChannelExternalPressureObstruction",
    "InsufficientEmbeddedChannelBurstMarginObstruction",
    "EmbeddedChannelObstruction",
    "RejectedEmbeddedChannelMechanics",
    "EmbeddedChannelMechanicsResult",
    "MalformedDomainObstruction",
    "DuplicateCellObstruction",
    "MissingCellPropertiesObstruction",
    "UnknownCellPropertiesObstruction",
    "DuplicateCellPropertiesObstruction",
    "InvalidConstitutivePropertiesObstruction",
    "InvalidMaterialFractionObstruction",
    "MaterialFractionRule",
    "UnresolvedMaterialFractionObstruction",
    "InvalidSupportObstruction",
    "ConflictingSupportObstruction",
    "InvalidNodalLoadObstruction",
    "InvalidBodyForceObstruction",
    "InvalidBucklingEigenpairObstruction",
    "InvalidTemperatureFieldObstruction",
    "InvalidPressureBoundaryObstruction",
    "InvalidAcceptanceCriteriaObstruction",
    "RigidBodyModeObstruction",
    "LinearSolverFailureObstruction",
    "NonFiniteMechanicsResultObstruction",
    "SolverResidualObstruction",
    "ForceBalanceObstruction",
    "SmallStrainLimitObstruction",
    "ExcessiveDisplacementObstruction",
    "InsufficientYieldMarginObstruction",
    "AbsentBucklingSpectrumObstruction",
    "SingularBucklingSpectrumObstruction",
    "UnresolvedBucklingSpectrumObstruction",
    "InsufficientBucklingMarginObstruction",
    "MechanicsObstruction",
    "SymmetricTensor3",
    "NodeMechanics",
    "CellMechanics",
    "BucklingNotRequested",
    "LinearizedBucklingReceipt",
    "AcceptedLinearizedBuckling",
    "BucklingResult",
    "MechanicsReceipt",
    "AcceptedMechanics",
    "RejectedMechanics",
    "MechanicsResult",
    "solve_mechanics",
    "evaluate_embedded_channel_mechanics",
)
