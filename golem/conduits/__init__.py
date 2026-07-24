"""GOLEM's surface-conduit, appendage-growth, and vascular-view subsystem.

``surface`` owns authored trim and rune embeddings, groove displacement, and
band strata. ``growth`` owns seeded appendage centerlines and their compiled
host-fusing part sequences. ``vascular`` derives diagnostic tube geometry from
anatomy-owned accepted vascular graphs. ``operator`` owns the typed command
algebra and the effectful terminal harness. Legacy modules remain import paths
for existing consumers; the subpackages are the semantic owners.

Surface emission is bounded by ``0.9 * pitch``. Violations return
``EmissionPitchObstruction`` rather than silently producing incomparable mesh
displacement. Appendage growth has no circulatory semantics. Vascular intent,
topology, radii, flow, and physical lumen ownership remain in
``golem.kernel.anatomy``; this package consumes accepted results only.
"""

from golem.conduits.surface import EmissionPitchObstruction, apply_conduits

__all__ = ["EmissionPitchObstruction", "apply_conduits"]
