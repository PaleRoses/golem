"""Legacy import paths for surface and vascular emission."""

from golem.conduits.surface.emit import (
    DEFAULT_BAND_LIFT,
    band_faces,
    groove,
)
from golem.conduits.vascular.emit import vascular_stratum_mesh

__all__ = [
    "DEFAULT_BAND_LIFT",
    "band_faces",
    "groove",
    "vascular_stratum_mesh",
]
