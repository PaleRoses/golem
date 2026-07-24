"""Mechanics assembly role carriers."""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass

from golem.assembly.carriers import RigidPayloadInterfaceReceipt
from golem.kernel.mechanics import NodalForce


@dataclass(frozen=True)
class _RigidPayloadTransfer:
    receipt: RigidPayloadInterfaceReceipt
    nodal_forces: tuple[NodalForce, ...]


@dataclass(frozen=True, eq=False)
class _PayloadTransferGeometry:
    scale: float
    mass_kilograms: float
    center_of_mass_metres: np.ndarray
    attachment_metres: np.ndarray
    gap: float

