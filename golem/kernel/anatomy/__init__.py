"""Typed whole-body anatomy descent for the GOLEM v0.3 body compiler.

The body skeleton is the attachment-tree cover.  An ``anatomy/0.1`` overall declares
typed body regions, one pump organ, and one capillary exchange bed for every
terminal region of that cover.  Optional myotendinous paths restrict bone-local
cross-sections through perfused muscle and integument into continuous tissue
envelopes.  Compilation derives both products before surface lowering.  Vessel
stages are real anatomical roles; circulation services tissue but never dictates
the exterior.
"""

from golem.kernel.anatomy.vocabulary import *
from golem.kernel.anatomy.graph import *
from golem.kernel.anatomy.hydraulics.types import *
from golem.kernel.anatomy.material.types import *
from golem.kernel.anatomy.geometry import *
from golem.kernel.anatomy.envelope import *
from golem.kernel.anatomy.lineage import *
from golem.kernel.anatomy.balance import *
from golem.kernel.anatomy.hydraulics.solve import *
from golem.kernel.anatomy.decode import *
from golem.kernel.anatomy.project import *
from golem.kernel.anatomy.descent import *
from golem.kernel.anatomy.material.carve import *
from golem.kernel.anatomy.realize import (
    allocate_terminal_pairs,
    realize_vasculature,
)

from golem.kernel.anatomy.geometry import (
    _certified_segment_capsule_margin,
    _segment_capsule_margin,
)
from golem.kernel.anatomy.realize.carriers import (
    _LocalVascularTree,
    _StructuralCostField,
    _index_structural_cost_field,
    _segment_structural_cost_multiplier,
)
from golem.kernel.anatomy.realize.cco import (
    _candidates_by_minimum_total_travel,
)
from golem.kernel.anatomy.realize.clearance import (
    _edge_clearance,
    _possible_vascular_edge_pairs,
    _vascular_edges_are_incident,
)
from golem.kernel.anatomy.realize.corridor import (
    _extend_corridor,
    _paired_corridor_section,
)
