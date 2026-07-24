"""Enumerate authored declarations and fork a spec with a subset removed."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

from golem.kernel.blame.core import (
    AuthoredAddress,
    AuthoredUnit,
    AuthoredUnitKind,
    DeclarationFamily,
)


def _sequence(value: object) -> Sequence:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _overall(spec: Mapping[str, object]) -> Mapping[str, object]:
    anatomy = spec.get("anatomy")
    overall = anatomy.get("overall") if isinstance(anatomy, Mapping) else None
    return overall if isinstance(overall, Mapping) else {}


def _skeleton_bones(spec: Mapping[str, object]) -> Sequence:
    skeleton = spec.get("skeleton")
    return _sequence(skeleton.get("bones")) if isinstance(skeleton, Mapping) else ()


def authored_addresses(spec: Mapping[str, object]) -> tuple[AuthoredAddress, ...]:
    """Every removable authored declaration, in canonical display order."""
    bones = _skeleton_bones(spec)
    bone_addresses = tuple(
        AuthoredAddress(DeclarationFamily.BONE, None, index, f"skeleton/bones/{index}")
        for index, _bone in enumerate(bones)
    )
    flesh_addresses = tuple(
        AuthoredAddress(
            DeclarationFamily.FLESH,
            bone_index,
            flesh_index,
            f"skeleton/bones/{bone_index}/flesh/{flesh_index}",
        )
        for bone_index, bone in enumerate(bones)
        if isinstance(bone, Mapping)
        for flesh_index, _flesh in enumerate(_sequence(bone.get("flesh")))
    )
    muscle_addresses = tuple(
        AuthoredAddress(DeclarationFamily.MUSCLE, None, index, f"muscles/{index}")
        for index, _muscle in enumerate(_sequence(spec.get("muscles")))
    )
    eye_addresses = tuple(
        AuthoredAddress(DeclarationFamily.EYE, None, index, f"eyes/{index}")
        for index, _eye in enumerate(_sequence(spec.get("eyes")))
    )
    overall = _overall(spec)
    region_addresses = tuple(
        AuthoredAddress(
            DeclarationFamily.REGION, None, index, f"anatomy/overall/regions/{index}"
        )
        for index, _region in enumerate(_sequence(overall.get("regions")))
    )
    circulation = overall.get("circulation")
    beds = _sequence(circulation.get("exchange_beds")) if isinstance(circulation, Mapping) else ()
    bed_addresses = tuple(
        AuthoredAddress(
            DeclarationFamily.EXCHANGE_BED,
            None,
            index,
            f"anatomy/overall/circulation/exchange_beds/{index}",
        )
        for index, _bed in enumerate(beds)
    )
    contract = spec.get("contract")
    clause_addresses = (
        tuple(
            AuthoredAddress(DeclarationFamily.CLAUSE, None, index, f"contract/clauses/{index}")
            for index, _clause in enumerate(_sequence(contract.get("clauses")))
        )
        if isinstance(contract, Mapping)
        else ()
    )
    attach_addresses = (
        tuple(
            AuthoredAddress(DeclarationFamily.ATTACH, None, index, f"contract/attach/{index}")
            for index, _pair in enumerate(_sequence(contract.get("attach")))
        )
        if isinstance(contract, Mapping)
        else ()
    )
    return tuple(
        sorted(
            (
                *bone_addresses,
                *flesh_addresses,
                *muscle_addresses,
                *eye_addresses,
                *region_addresses,
                *bed_addresses,
                *clause_addresses,
                *attach_addresses,
            ),
            key=lambda address: address.display_key,
        )
    )


def _limb_root_index(
    bone_index: int,
    bones: Sequence,
    root_id: str | None,
    index_by_id: Mapping[str, int],
    visited: frozenset[int] = frozenset(),
) -> int:
    if bone_index in visited or not 0 <= bone_index < len(bones):
        return bone_index
    bone = bones[bone_index]
    if not isinstance(bone, Mapping):
        return bone_index
    parent = bone.get("parent")
    parent_index = index_by_id.get(parent) if isinstance(parent, str) else None
    return (
        bone_index
        if parent == root_id or parent_index is None
        else _limb_root_index(
            parent_index,
            bones,
            root_id,
            index_by_id,
            visited | {bone_index},
        )
    )


def authored_units(
    spec: Mapping[str, object],
    addresses: tuple[AuthoredAddress, ...] | None = None,
) -> tuple[AuthoredUnit, ...]:
    authored = authored_addresses(spec) if addresses is None else addresses
    bones = _skeleton_bones(spec)
    skeleton = spec.get("skeleton")
    root = skeleton.get("root") if isinstance(skeleton, Mapping) else None
    root_id = root.get("id") if isinstance(root, Mapping) and isinstance(root.get("id"), str) else None
    index_by_id = {
        str(bone["id"]): index
        for index, bone in enumerate(bones)
        if isinstance(bone, Mapping) and isinstance(bone.get("id"), str)
    }
    limb_root_by_bone = {
        index: _limb_root_index(index, bones, root_id, index_by_id)
        for index, _bone in enumerate(bones)
    }
    skeletal = tuple(
        address
        for address in authored
        if address.family in (DeclarationFamily.BONE, DeclarationFamily.FLESH)
    )
    skeletal_by_root = {
        root_index: tuple(
            address
            for address in skeletal
            if limb_root_by_bone.get(
                address.index
                if address.family is DeclarationFamily.BONE
                else address.parent_index if address.parent_index is not None else -1,
                -1,
            )
            == root_index
        )
        for root_index in frozenset(limb_root_by_bone.values())
    }
    overall = _overall(spec)
    region_records = _sequence(overall.get("regions"))
    circulation = overall.get("circulation")
    bed_records = (
        _sequence(circulation.get("exchange_beds"))
        if isinstance(circulation, Mapping)
        else ()
    )
    region_addresses = {
        address.index: address
        for address in authored
        if address.family is DeclarationFamily.REGION
    }
    bed_addresses = {
        address.index: address
        for address in authored
        if address.family is DeclarationFamily.EXCHANGE_BED
    }
    region_units = tuple(
        AuthoredUnit(
            str(region.get("region_id", region_index)),
            frozenset(
                (
                    AuthoredUnitKind.REGION,
                    *(
                        (AuthoredUnitKind.LIMB,)
                        if limb_root is not None
                        else ()
                    ),
                    *(
                        (AuthoredUnitKind.EXCHANGE_BED,)
                        if matching_beds
                        else ()
                    ),
                )
            ),
            tuple(
                sorted(
                    (
                        *skeletal_by_root.get(limb_root, ()),
                        *(region_addresses.get(region_index),),
                        *(bed_addresses[index] for index in matching_beds),
                    ),
                    key=lambda address: address.display_key,
                )
            ),
        )
        for region_index, region in enumerate(region_records)
        if isinstance(region, Mapping)
        for host_bone_id in (
            region.get("host_bone_id")
            if isinstance(region.get("host_bone_id"), str)
            else None,
        )
        for host_bone_index in (
            index_by_id.get(host_bone_id) if host_bone_id is not None else None,
        )
        for limb_root in (
            limb_root_by_bone.get(host_bone_index)
            if host_bone_index is not None
            else None,
        )
        for matching_beds in (
            tuple(
                index
                for index, bed in enumerate(bed_records)
                if isinstance(bed, Mapping)
                and bed.get("region_id") == region.get("region_id")
            ),
        )
        if region_addresses.get(region_index) is not None
    )
    covered_limb_roots = frozenset(
        limb_root_by_bone[host_index]
        for region in region_records
        if isinstance(region, Mapping)
        and isinstance(region.get("host_bone_id"), str)
        for host_index in (index_by_id.get(region["host_bone_id"]),)
        if host_index is not None
    )
    covered_beds = frozenset(
        address.index
        for unit in region_units
        for address in unit.addresses
        if address.family is DeclarationFamily.EXCHANGE_BED
    )
    orphan_limbs = tuple(
        AuthoredUnit(
            str(
                bones[root_index].get("id", root_index)
                if 0 <= root_index < len(bones) and isinstance(bones[root_index], Mapping)
                else root_index
            ),
            frozenset((AuthoredUnitKind.LIMB,)),
            skeletal_by_root[root_index],
        )
        for root_index in sorted(skeletal_by_root)
        if root_index not in covered_limb_roots
    )
    orphan_beds = tuple(
        AuthoredUnit(
            str(
                bed_records[index].get("region_id", index)
                if 0 <= index < len(bed_records) and isinstance(bed_records[index], Mapping)
                else index
            ),
            frozenset((AuthoredUnitKind.EXCHANGE_BED,)),
            (address,),
        )
        for index, address in sorted(bed_addresses.items())
        if index not in covered_beds
    )
    return (*region_units, *orphan_limbs, *orphan_beds)


def _filter_list(values: Sequence, family: DeclarationFamily, removed) -> list:
    return [
        value
        for index, value in enumerate(values)
        if AuthoredAddress(family, None, index, "") not in _by_family(removed, family)
    ]


def _by_family(removed, family: DeclarationFamily) -> frozenset[AuthoredAddress]:
    return frozenset(
        AuthoredAddress(address.family, None, address.index, "")
        for address in removed
        if address.family is family and address.parent_index is None
    )


def restrict(spec: Mapping[str, object], removed: frozenset[AuthoredAddress]) -> dict:
    """A deep copy of ``spec`` with the removed declarations deleted by original index.

    Removal is computed against the pristine indices carried by each address, so
    the result is independent of how many other declarations were dropped — the
    fork always starts from the original document.
    """
    forked = copy.deepcopy(dict(spec))
    dropped_flesh = {
        (address.parent_index, address.index)
        for address in removed
        if address.family is DeclarationFamily.FLESH
    }
    dropped_bones = _by_family(removed, DeclarationFamily.BONE)
    skeleton = forked.get("skeleton")
    if isinstance(skeleton, Mapping) and isinstance(skeleton.get("bones"), Sequence):
        rebuilt_bones = []
        for bone_index, bone in enumerate(skeleton["bones"]):
            if AuthoredAddress(DeclarationFamily.BONE, None, bone_index, "") in dropped_bones:
                continue
            if isinstance(bone, Mapping) and isinstance(bone.get("flesh"), Sequence):
                bone = {
                    **bone,
                    "flesh": [
                        flesh
                        for flesh_index, flesh in enumerate(bone["flesh"])
                        if (bone_index, flesh_index) not in dropped_flesh
                    ],
                }
            rebuilt_bones.append(bone)
        forked["skeleton"] = {**skeleton, "bones": rebuilt_bones}
    for key, family in (("muscles", DeclarationFamily.MUSCLE), ("eyes", DeclarationFamily.EYE)):
        if isinstance(forked.get(key), Sequence) and not isinstance(forked[key], (str, bytes)):
            forked[key] = _filter_list(forked[key], family, removed)
    anatomy = forked.get("anatomy")
    if isinstance(anatomy, Mapping) and isinstance(anatomy.get("overall"), Mapping):
        overall = dict(anatomy["overall"])
        if isinstance(overall.get("regions"), Sequence):
            overall["regions"] = _filter_list(
                overall["regions"], DeclarationFamily.REGION, removed
            )
        circulation = overall.get("circulation")
        if isinstance(circulation, Mapping) and isinstance(
            circulation.get("exchange_beds"), Sequence
        ):
            overall["circulation"] = {
                **circulation,
                "exchange_beds": _filter_list(
                    circulation["exchange_beds"], DeclarationFamily.EXCHANGE_BED, removed
                ),
            }
        forked["anatomy"] = {**anatomy, "overall": overall}
    contract = forked.get("contract")
    if isinstance(contract, Mapping):
        rebuilt = dict(contract)
        if isinstance(contract.get("clauses"), Sequence):
            rebuilt["clauses"] = _filter_list(
                contract["clauses"], DeclarationFamily.CLAUSE, removed
            )
        if isinstance(contract.get("attach"), Sequence):
            rebuilt["attach"] = _filter_list(
                contract["attach"], DeclarationFamily.ATTACH, removed
            )
        forked["contract"] = rebuilt
    return forked
