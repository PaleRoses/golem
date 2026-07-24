"""Type-surface tests for the frozen senses model (golem/senses/model.py)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import assert_never

import numpy as np
import pytest

from golem.senses.model import (
    Balance,
    Band,
    FusionBand,
    FusionRow,
    GlobalDims,
    InstanceDatum,
    NoSupport,
    PartDatum,
    Senses,
    Supported,
)


def _supported() -> Supported:
    return Supported(
        hull_x=(-0.5, 0.5),
        hull_z=(-0.4, 0.6),
        centroid_xz=(0.0, 0.1),
        margin_x=0.5,
        margin_z=0.4,
        margin_z_fwd=0.5,
        margin_z_aft=0.3,
        inside=True,
    )


def _part_datum() -> PartDatum:
    return PartDatum(
        bbox_lo=np.array([-1.0, -1.0, -1.0]),
        bbox_hi=np.array([1.0, 1.0, 1.0]),
        mirror=True,
        instances=["torso.L", "torso.R"],
        min_r=0.25,
    )


def _instance_datum() -> InstanceDatum:
    return InstanceDatum(
        part={"id": "torso", "type": "blob", "center": [0, 0, 0], "size": [1, 1, 1]},
        mirrored=False,
        samples=np.zeros((4, 3)),
        lo=np.array([-1.0, -1.0, -1.0]),
        hi=np.array([1.0, 1.0, 1.0]),
        blend_r=0.05,
        min_r=0.25,
        pid="torso",
    )


def _senses(balance: Balance) -> Senses:
    return Senses(
        name="knight",
        txn=1,
        n_parts=1,
        n_mirror=1,
        n_instances=2,
        global_dims=GlobalDims(
            W=2.0, H=3.0, D=1.5, HW=1.5, centroid=np.array([0.0, 1.0, -0.2]), k=0.1
        ),
        raw_lo=np.array([-1.0, -1.0, -1.0]),
        raw_hi=np.array([1.0, 1.0, 1.0]),
        parts={"torso": _part_datum()},
        inst_data={"torso.L": _instance_datum()},
        inst_gaps={("torso.L", "torso.R"): -0.02},
        declared_pair_gap={("torso", "arm"): 0.01},
        fusion_rows={"torso": [FusionRow("arm", -0.02, FusionBand.FUSED)]},
        cross_plane={"torso": -0.02},
        n_components=1,
        fused_adj={"torso": {"arm"}},
        ground={"foot": 0.0},
        near_ground={"shin": 0.03},
        ground_decl=["foot"],
        ground_tol=0.02,
        balance=balance,
        bands=[Band(ylo=0.0, yhi=0.3, width=1.2, depth=0.8, zmid=-0.1)],
        landmarks={"crown": [0.0, 3.0, 0.0]},
        schematic={"front": {"crown": [0.0, 1.0]}},
        attach=[("torso", "arm")],
        provenance={"torso": "skel.torso"},
        anatomy=None,
        graph={"name": "knight", "parts": []},
    )


def test_senses_exposes_typed_fields_through_the_record_tree():
    s = _senses(_supported())
    assert s.name == "knight" and s.txn == 1
    assert isinstance(s.raw_lo, np.ndarray) and s.raw_lo[0] == -1.0
    assert isinstance(s.global_dims, GlobalDims)
    assert s.global_dims.centroid[1] == 1.0
    assert s.parts["torso"].instances == ("torso.L", "torso.R")
    assert s.inst_data["torso.L"].pid == "torso"
    assert s.inst_gaps[("torso.L", "torso.R")] == -0.02
    assert s.fusion_rows["torso"][0].band is FusionBand.FUSED
    assert s.bands[0].zmid == -0.1
    assert s.fused_adj["torso"] == {"arm"}
    assert isinstance(s.balance, Supported) and s.balance.inside is True
    assert s.anatomy is None


def test_fusion_band_vocabulary_is_closed():
    assert {band.value for band in FusionBand} == {"FUSED", "BLEND"}


@pytest.mark.parametrize(
    "record",
    [
        GlobalDims(2.0, 3.0, 1.5, 1.5, np.array([0.0, 1.0, -0.2]), 0.1),
        _part_datum(),
        _instance_datum(),
        Band(0.0, 0.3, 1.2, 0.8, -0.1),
        FusionRow("arm", -0.02, FusionBand.FUSED),
        _supported(),
        NoSupport(),
        _senses(NoSupport()),
    ],
)
def test_every_record_is_frozen(record):
    with pytest.raises(FrozenInstanceError):
        record.frozen_probe = object()


def _summarize(balance: Balance) -> str:
    match balance:
        case Supported() as supported:
            return f"supported inside={supported.inside}"
        case NoSupport():
            return "unsupported"
        case _:
            assert_never(balance)


def test_balance_dispatch_hits_both_arms():
    assert _summarize(_supported()) == "supported inside=True"
    assert _summarize(NoSupport()) == "unsupported"


def test_balance_dispatch_guard_rejects_foreign_value():
    with pytest.raises(AssertionError):
        _summarize(object())  # type: ignore[arg-type]
