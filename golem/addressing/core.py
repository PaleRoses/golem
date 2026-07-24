"""Compatibility surface for the distinct addressing grammars."""

from golem.addressing.anchor import (
    Anchor,
    AnchorObstruction,
    AnchorResult,
    RejectedAnchor,
)
from golem.addressing.anchor_grammar import parse_anchor, render_anchor
from golem.addressing.cell import CellId, GridDims, flatten, unflatten
from golem.addressing.scope import (
    Bone,
    Chain,
    Contact,
    Element,
    Landmark,
    Mount,
    Part,
    Port,
    Region,
    RejectedScope,
    Scope,
    ScopeKind,
    ScopeObstruction,
    ScopeResult,
    Whole,
    World,
)
from golem.addressing.scope_grammar import parse_scope, render_scope, scope_kind
from golem.addressing.session import (
    Address,
    AddressObstruction,
    AddressResult,
    FieldIndex,
    FieldName,
    FieldToken,
    RejectedAddress,
)
from golem.addressing.session_grammar import (
    field_tokens,
    parse_address,
    render_address,
)
