"""Tests for relation/attachment decoding, the closed vocabulary, and aliasing."""

from __future__ import annotations

import pytest

from golem.addressing.scope import Bone, Landmark, Part
from golem.kernel.body.relations import (
    AcceptedRelationDecode,
    DuplicateRelationIdObstruction,
    FixedLocalAttachment,
    MalformedRelationObstruction,
    NamedSiteAttachment,
    RejectedRelationDecode,
    RelationKind,
    UnknownRelationKindObstruction,
    UnresolvedRelationSelectorObstruction,
    UnsupportedRelationSelectorObstruction,
    decode_attachment,
    decode_relations,
)


def _relation(**overrides) -> dict:
    base = {
        "id": "r",
        "kind": "above",
        "subject": "part:head",
        "reference": "part:chest",
        "distance": 0.04,
        "solve": "subject",
    }
    return {**base, **overrides}


# -- attachment forms ----------------------------------------------------- #
def test_fixed_local_attachment_decodes_with_defaults():
    decoded = decode_attachment({"t": 0.5, "offset": [0.1, 0.0, 0.0]}, "a")
    assert decoded == FixedLocalAttachment(t=0.5, offset=(0.1, 0.0, 0.0))
    assert decode_attachment(None, "a") == FixedLocalAttachment()


def test_named_site_attachment_decodes_to_landmark():
    decoded = decode_attachment({"at": "landmark:upper/tail"}, "a")
    assert isinstance(decoded, NamedSiteAttachment)
    assert decoded.site == Landmark("upper/tail")
    assert decoded.selector == "landmark:upper/tail"


def test_mixed_attachment_form_rejects():
    decoded = decode_attachment({"at": "landmark:upper/tail", "t": 0.5}, "skeleton/x/attach")
    assert isinstance(decoded, MalformedRelationObstruction)
    assert decoded.address == "skeleton/x/attach"


# -- vocabulary + aliasing ------------------------------------------------ #
@pytest.mark.parametrize(
    "authored,expected",
    [
        ("above", RelationKind.ABOVE),
        ("over", RelationKind.ABOVE),
        ("under", RelationKind.BELOW),
        ("perpendicular", RelationKind.PERPENDICULAR_TO),
        ("orthogonal_to", RelationKind.PERPENDICULAR_TO),
        ("parallel_to", RelationKind.ALIGNED),
        ("reflection_of", RelationKind.MIRROR_OF),
        ("coincides_with", RelationKind.COINCIDENT),
    ],
)
def test_canonical_and_alias_kinds_resolve_identically(authored, expected):
    subject, reference = ("bone:a", "bone:b") if expected in {
        RelationKind.MIRROR_OF, RelationKind.ALIGNED, RelationKind.PERPENDICULAR_TO,
    } else ("part:a", "part:b")
    kwargs = {"kind": authored, "subject": subject, "reference": reference}
    if expected not in {RelationKind.ABOVE, RelationKind.BELOW}:
        kwargs["distance"] = None
    result = decode_relations([_relation(**kwargs)])
    assert isinstance(result, AcceptedRelationDecode)
    (declaration,) = result.declarations
    assert declaration.kind is expected
    assert declaration.authored_kind == authored


def test_misspelled_kind_suggests_deduplicated_canonical_ids():
    result = decode_relations([_relation(kind="abov")])
    assert isinstance(result, RejectedRelationDecode)
    (obstruction,) = result.obstructions
    assert isinstance(obstruction, UnknownRelationKindObstruction)
    assert obstruction.authored_kind == "abov"
    assert "above" in obstruction.closest_valid_candidates
    assert len(obstruction.closest_valid_candidates) == len(set(obstruction.closest_valid_candidates))


def test_duplicate_relation_ids_reject():
    result = decode_relations([_relation(id="dup"), _relation(id="dup", subject="part:x")])
    assert isinstance(result, RejectedRelationDecode)
    assert any(
        isinstance(o, DuplicateRelationIdObstruction) and o.relation_id == "dup"
        for o in result.obstructions
    )


# -- selector capability -------------------------------------------------- #
def test_unknown_selector_retains_address_and_selector():
    result = decode_relations([_relation(subject="part:head", reference="not a scope")])
    assert isinstance(result, RejectedRelationDecode)
    (obstruction,) = result.obstructions
    assert isinstance(obstruction, UnresolvedRelationSelectorObstruction)
    assert obstruction.selector == "not a scope"
    assert obstruction.address == "pose/relations/r/reference"


def test_unsupported_assembly_scope_returns_typed_obstruction():
    # `above` admits only Bone/Part; a Landmark subject is out of capability.
    result = decode_relations([_relation(kind="above", subject="landmark:head/tail")])
    assert isinstance(result, RejectedRelationDecode)
    (obstruction,) = result.obstructions
    assert isinstance(obstruction, UnsupportedRelationSelectorObstruction)
    assert obstruction.selector == "landmark:head/tail"
    assert obstruction.supported_scope_kinds


def test_attach_at_named_site_is_rejected_as_a_pose_relation():
    result = decode_relations([_relation(kind="attach_at", subject="landmark:a/head",
                                         reference="landmark:b/tail", distance=None)])
    assert isinstance(result, RejectedRelationDecode)
    (obstruction,) = result.obstructions
    assert isinstance(obstruction, MalformedRelationObstruction)


def test_distance_on_non_metric_kind_rejects():
    result = decode_relations([_relation(kind="aligned", subject="bone:a",
                                         reference="bone:b", distance=0.1)])
    assert isinstance(result, RejectedRelationDecode)
    assert all(isinstance(o, MalformedRelationObstruction) for o in result.obstructions)
