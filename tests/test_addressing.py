"""The unified addressing vocabulary round-trips and fails closed on every grammar."""

from __future__ import annotations

import pytest

from golem.addressing import (
    Address,
    Anchor,
    Bone,
    CellId,
    Chain,
    Contact,
    Element,
    FieldIndex,
    FieldName,
    GridDims,
    Landmark,
    Mount,
    Part,
    Port,
    Region,
    RejectedAddress,
    RejectedAnchor,
    RejectedScope,
    ScopeKind,
    Whole,
    World,
    field_tokens,
    flatten,
    parse_address,
    parse_anchor,
    parse_scope,
    render_address,
    render_anchor,
    render_scope,
    scope_kind,
    unflatten,
)


# --------------------------------------------------------------------------- #
# Round-trip corpora: real strings drawn from each existing grammar.           #
# --------------------------------------------------------------------------- #
_ADDRESS_CORPUS: tuple[tuple[str, Address], ...] = (
    ("meta", Address(("meta",), ())),
    ("meta@name", Address(("meta",), (FieldName("name"),))),
    ("skeleton/root", Address(("skeleton", "root"), ())),
    (
        "skeleton/forearm@length",
        Address(("skeleton", "forearm"), (FieldName("length"),)),
    ),
    (
        "skeleton/root@world",
        Address(("skeleton", "root"), (FieldName("world"),)),
    ),
    (
        "skeleton/spine@rest_dir[2]",
        Address(("skeleton", "spine"), (FieldName("rest_dir"), FieldIndex(2))),
    ),
    (
        "skeleton/forearm@joint.limits.pitch",
        Address(
            ("skeleton", "forearm"),
            (FieldName("joint"), FieldName("limits"), FieldName("pitch")),
        ),
    ),
    (
        "skeleton/arm@flesh[2].size",
        Address(
            ("skeleton", "arm"),
            (FieldName("flesh"), FieldIndex(2), FieldName("size")),
        ),
    ),
    ("pose/forearm@pitch", Address(("pose", "forearm"), (FieldName("pitch"),))),
    (
        "pose/goals/reach@target",
        Address(("pose", "goals", "reach"), (FieldName("target"),)),
    ),
    ("props/sword@anchor", Address(("props", "sword"), (FieldName("anchor"),))),
    (
        "props/sword/blade@radii[0]",
        Address(("props", "sword", "blade"), (FieldName("radii"), FieldIndex(0))),
    ),
    ("mounts/hand@port", Address(("mounts", "hand"), (FieldName("port"),))),
    ("contacts/feet@solve", Address(("contacts", "feet"), (FieldName("solve"),))),
    ("foo@[0]", Address(("foo",), (FieldIndex(0),))),
)


_SCOPE_CORPUS: tuple[tuple[str, object], ...] = (
    ("whole", Whole()),
    ("world", World()),
    ("part:head", Part("head")),
    ("bone:forearm", Bone("forearm", None)),
    ("bone:knight/forearm", Bone("forearm", "knight")),
    ("landmark:pommel", Landmark("pommel", None)),
    ("landmark:forearm/tail", Landmark("forearm/tail", None)),
    ("landmark:forearm/tail@front", Landmark("forearm/tail", "front")),
    ("chain:hip..ankle", Chain("hip", "ankle")),
    ("contact:support", Contact("support", None)),
    ("contact:knight/hand", Contact("hand", "knight")),
    ("element:knight", Element("knight")),
    ("port:knight/grip", Port("knight", "grip")),
    ("mount:knight", Mount("knight")),
    ("region:knight/torso", Region("knight", "torso")),
)


_ANCHOR_CORPUS: tuple[tuple[str, Anchor], ...] = (
    ("forearm:0.5", Anchor("forearm", 0.5)),
    ("root:1.0", Anchor("root", 1.0)),
    ("spine:0.0", Anchor("spine", 0.0)),
    ("hand:0.25", Anchor("hand", 0.25)),
    ("elbow:-0.5", Anchor("elbow", -0.5)),
    ("wrist:1.5", Anchor("wrist", 1.5)),
)


# --------------------------------------------------------------------------- #
# Address.                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("source", "expected"), _ADDRESS_CORPUS)
def test_address_parses_to_the_expected_arm(source: str, expected: Address) -> None:
    assert parse_address(source) == expected


@pytest.mark.parametrize(("source", "expected"), _ADDRESS_CORPUS)
def test_address_round_trips(source: str, expected: Address) -> None:
    parsed = parse_address(source)
    assert isinstance(parsed, Address)
    assert render_address(parsed) == source


def test_address_segments_percent_encode_reserved_identity_characters() -> None:
    source = "skeleton/child%2Fleaf%40bud%25"
    expected = Address(
        ("skeleton", "child/leaf@bud%"),
        (),
        frozenset({1}),
    )

    assert parse_address(source) == expected
    assert render_address(expected) == source


def test_encoded_literal_and_unencoded_index_selector_remain_distinct() -> None:
    selector = parse_address("skeleton/root/flesh[0]")
    literal = parse_address("skeleton/root/flesh%5B0%5D")

    assert isinstance(selector, Address)
    assert isinstance(literal, Address)
    assert selector.encoded_segments == frozenset()
    assert literal.encoded_segments == frozenset({2})
    assert render_address(selector) == "skeleton/root/flesh[0]"
    assert render_address(literal) == "skeleton/root/flesh%5B0%5D"


@pytest.mark.parametrize(
    ("source", "canonical"),
    (
        ("skeleton/a#b", "skeleton/a%23b"),
        ("skeleton/a b", "skeleton/a%20b"),
        ("skeleton/a?b", "skeleton/a%3Fb"),
        ("skeleton/a:b", "skeleton/a%3Ab"),
    ),
)
def test_address_renderer_canonicalizes_non_unreserved_segments(
    source: str,
    canonical: str,
) -> None:
    parsed = parse_address(source)

    assert isinstance(parsed, Address)
    assert render_address(parsed) == canonical


def test_address_field_tokens_project_to_raw_str_int() -> None:
    parsed = parse_address("skeleton/arm@flesh[2].size")
    assert isinstance(parsed, Address)
    assert field_tokens(parsed) == ("flesh", 2, "size")


@pytest.mark.parametrize(
    "source",
    [
        "",
        "/skeleton",
        "skeleton/",
        "meta@1bad",
        "meta@[bad]",
        "meta@a!b",
        "skeleton@joint.",
        "skeleton/child%leaf",
        "skeleton/child%FF",
    ],
)
def test_malformed_address_fails_closed(source: str) -> None:
    assert isinstance(parse_address(source), RejectedAddress)


# --------------------------------------------------------------------------- #
# Scope.                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("source", "expected"), _SCOPE_CORPUS)
def test_scope_parses_to_the_expected_arm(source: str, expected: object) -> None:
    assert parse_scope(source) == expected


@pytest.mark.parametrize(("source", "expected"), _SCOPE_CORPUS)
def test_scope_round_trips(source: str, expected: object) -> None:
    parsed = parse_scope(source)
    assert not isinstance(parsed, RejectedScope)
    assert render_scope(parsed) == source


@pytest.mark.parametrize(
    "source",
    [
        "",
        "forearm",
        "part:",
        "region:knight",
        "port:knight",
        "element:knight/torso",
        "mount:knight/x",
        "chain:hip",
        "bogus:thing",
        "bone:knight/",
        "bone:/forearm",
        "landmark:@front",
    ],
)
def test_malformed_scope_fails_closed(source: str) -> None:
    assert isinstance(parse_scope(source), RejectedScope)


# --------------------------------------------------------------------------- #
# Anchor.                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("source", "expected"), _ANCHOR_CORPUS)
def test_anchor_parses_to_the_expected_arm(source: str, expected: Anchor) -> None:
    assert parse_anchor(source) == expected


@pytest.mark.parametrize(("source", "expected"), _ANCHOR_CORPUS)
def test_anchor_round_trips(source: str, expected: Anchor) -> None:
    parsed = parse_anchor(source)
    assert isinstance(parsed, Anchor)
    assert render_anchor(parsed) == source


def test_bare_bone_anchor_defaults_to_the_tail_parameter() -> None:
    assert parse_anchor("forearm") == Anchor("forearm", 1.0)


@pytest.mark.parametrize(
    "source",
    ["", ":0.5", "forearm:abc", "forearm:2.0", "forearm:-1.0", "forearm:nan"],
)
def test_malformed_anchor_fails_closed(source: str) -> None:
    assert isinstance(parse_anchor(source), RejectedAnchor)


def test_anchor_is_distinct_from_scope_on_the_shared_colon() -> None:
    assert parse_anchor("forearm:0.5") == Anchor("forearm", 0.5)
    assert parse_scope("bone:forearm") == Bone("forearm", None)


# --------------------------------------------------------------------------- #
# Exhaustiveness smoke: every arm constructs, renders, and re-parses.          #
# --------------------------------------------------------------------------- #
def test_every_scope_arm_constructs_renders_and_reparses() -> None:
    arms = (
        Whole(),
        World(),
        Part("head"),
        Bone("forearm", None),
        Bone("forearm", "knight"),
        Landmark("pommel", None),
        Landmark("forearm/tail", "front"),
        Chain("hip", "ankle"),
        Contact("support", None),
        Contact("hand", "knight"),
        Element("knight"),
        Port("knight", "grip"),
        Mount("knight"),
        Region("knight", "torso"),
    )
    for arm in arms:
        rendered = render_scope(arm)
        assert parse_scope(rendered) == arm
        assert isinstance(scope_kind(arm), ScopeKind)


def test_every_scope_kind_is_reachable_from_an_arm() -> None:
    reached = {
        scope_kind(arm)
        for arm in (
            Whole(),
            World(),
            Part("p"),
            Bone("b"),
            Landmark("l"),
            Chain("a", "z"),
            Contact("c"),
            Element("e"),
            Port("e", "p"),
            Mount("e"),
            Region("e", "r"),
        )
    }
    assert reached == set(ScopeKind)


def test_address_and_anchor_arms_construct_and_render() -> None:
    address = Address(("skeleton", "arm"), (FieldName("flesh"), FieldIndex(0)))
    assert render_address(address) == "skeleton/arm@flesh[0]"
    assert render_anchor(Anchor("elbow", 0.75)) == "elbow:0.75"


# --------------------------------------------------------------------------- #
# CellId: the row-major flatten/unflatten bridge is a bijection on the grid.   #
# --------------------------------------------------------------------------- #
def test_flatten_unflatten_round_trip_over_a_grid() -> None:
    dims = GridDims(3, 4, 5)
    triples = tuple(
        (x, y, z)
        for x in range(dims.nx)
        for y in range(dims.ny)
        for z in range(dims.nz)
    )
    for x, y, z in triples:
        cell = flatten(x, y, z, dims)
        assert unflatten(cell, dims) == (x, y, z)


def test_flatten_covers_the_dense_id_range_exactly_once() -> None:
    dims = GridDims(2, 3, 4)
    ids = sorted(
        int(flatten(x, y, z, dims))
        for x in range(dims.nx)
        for y in range(dims.ny)
        for z in range(dims.nz)
    )
    assert ids == list(range(dims.cell_count))


def test_unflatten_inverts_flatten_on_raw_ids() -> None:
    dims = GridDims(4, 4, 4)
    for cell_id in range(dims.cell_count):
        x, y, z = unflatten(CellId(cell_id), dims)
        assert flatten(x, y, z, dims) == cell_id


def test_grid_dims_bounds_predicate() -> None:
    dims = GridDims(2, 2, 2)
    assert dims.contains(0, 0, 0)
    assert dims.contains(1, 1, 1)
    assert not dims.contains(2, 0, 0)
    assert not dims.contains(0, -1, 0)
