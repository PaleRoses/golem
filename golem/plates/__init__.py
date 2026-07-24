"""Derived plated-surface cover for GOLEM morphology elements.

The morphology remains authoritative.  A :class:`PlatePolicy` restricts that
surface into local cells; seams and relief are derived views, never a second
mesh-authoring language.
"""

from golem.plates.core import (
    AcceptedPlatePolicy,
    AcceptedPlateSurface,
    PlateAdjacency,
    PlateLayout,
    PlatePolicy,
    PlatePolicyObstruction,
    PlatePolicyRule,
    PlateSurface,
    PlateSurfaceObstruction,
    PlateSurfaceRule,
    RejectedPlatePolicy,
    RejectedPlateSurface,
)
from golem.plates.decode import decode_plate_policy
from golem.plates.derive import derive_plate_surface
from golem.plates.geometry import minimum_point_segment_distances
from golem.plates.projection import plate_surface_report, seam_geometry

__all__ = [
    "AcceptedPlatePolicy",
    "AcceptedPlateSurface",
    "PlateAdjacency",
    "PlateLayout",
    "PlatePolicy",
    "PlatePolicyObstruction",
    "PlatePolicyRule",
    "PlateSurface",
    "PlateSurfaceObstruction",
    "PlateSurfaceRule",
    "RejectedPlatePolicy",
    "RejectedPlateSurface",
    "decode_plate_policy",
    "derive_plate_surface",
    "minimum_point_segment_distances",
    "plate_surface_report",
    "seam_geometry",
]
