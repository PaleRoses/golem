"""Load-case section decoders -- loads, supports, joints, heat sources."""

from __future__ import annotations

from golem.assembly.service.address import SemanticAddress
from golem.assembly.service.model import (
    AppliedLoad,
    ForceLoad,
    HeatSource,
    JointInterface,
    JointLaw,
    MomentLoad,
    ServiceCase,
    Support,
    SupportLaw,
)
from golem.assembly.service.outcome import (
    ContradictorySupportObstruction,
    DuplicateServiceCaseObstruction,
    MalformedServiceIntentObstruction,
    ServiceIntentObstruction,
)
from golem.assembly.service.primitives import (
    _DOMAIN_ADDRESS_KINDS,
    _DOMAIN_ADDRESS_TYPES,
    _INTERFACE_ADDRESS_KINDS,
    _INTERFACE_ADDRESS_TYPES,
    _Decoded,
    _DecodeResult,
    _closed_value,
    _decode_bound_address,
    _duplicate_strings,
    _nonempty_string,
    _positive_budget,
    _record,
    _result_obstructions,
    _traverse_array,
    _vector3,
)


def _decode_cases(
    value: object, known_addresses: frozenset[SemanticAddress]
) -> _DecodeResult[tuple[ServiceCase, ...]]:
    decoded = _traverse_array(
        value,
        "/service/cases",
        lambda item, index: _decode_case(item, index, known_addresses),
        require_nonempty=True,
    )
    if not isinstance(decoded, _Decoded):
        return decoded
    cases = decoded.value
    duplicate_ids = tuple(
        DuplicateServiceCaseObstruction(case_id)
        for case_id in _duplicate_strings(tuple(case.case_id for case in cases))
    )
    return duplicate_ids if duplicate_ids else _Decoded(cases)


def _decode_case(
    value: object,
    index: int,
    known_addresses: frozenset[SemanticAddress],
) -> _DecodeResult[ServiceCase]:
    path = f"/service/cases/{index}"
    keys = frozenset(
        (
            "id",
            "gravity_m_s2",
            "inertial_acceleration_m_s2",
            "loads",
            "supports",
            "joints",
            "heat_sources",
        )
    )
    obj = _record(value, path, keys)
    if not isinstance(obj, _Decoded):
        return obj
    case_id = _nonempty_string(obj.value["id"], f"{path}/id")
    gravity = _vector3(obj.value["gravity_m_s2"], f"{path}/gravity_m_s2")
    inertia = _vector3(
        obj.value["inertial_acceleration_m_s2"],
        f"{path}/inertial_acceleration_m_s2",
    )
    loads = _traverse_array(
        obj.value["loads"],
        f"{path}/loads",
        lambda item, item_index: _decode_load(
            item, f"{path}/loads/{item_index}", known_addresses
        ),
    )
    supports = _traverse_array(
        obj.value["supports"],
        f"{path}/supports",
        lambda item, item_index: _decode_support(
            item, f"{path}/supports/{item_index}", known_addresses
        ),
    )
    joints = _traverse_array(
        obj.value["joints"],
        f"{path}/joints",
        lambda item, item_index: _decode_joint(
            item, f"{path}/joints/{item_index}", known_addresses
        ),
    )
    heat_sources = _traverse_array(
        obj.value["heat_sources"],
        f"{path}/heat_sources",
        lambda item, item_index: _decode_heat_source(
            item, f"{path}/heat_sources/{item_index}", known_addresses
        ),
    )
    results = (case_id, gravity, inertia, loads, supports, joints, heat_sources)
    obstructions = _result_obstructions(results)
    if obstructions:
        return obstructions
    contradictions = _support_contradictions(case_id.value, supports.value)
    return (
        contradictions
        if contradictions
        else _Decoded(
            ServiceCase(
                case_id=case_id.value,
                gravity_metres_per_second_squared=gravity.value,
                inertial_acceleration_metres_per_second_squared=inertia.value,
                loads=loads.value,
                supports=supports.value,
                joints=joints.value,
                heat_sources=heat_sources.value,
            )
        )
    )


def _decode_load(
    value: object,
    path: str,
    known_addresses: frozenset[SemanticAddress],
) -> _DecodeResult[AppliedLoad]:
    if not isinstance(value, dict):
        return (MalformedServiceIntentObstruction(path, "expected an object"),)
    kind = value.get("kind")
    vector_key = "vector_newtons" if kind == "force" else "vector_newton_metres"
    expected_keys = frozenset(("kind", "address", vector_key))
    obj = _record(value, path, expected_keys)
    if kind not in ("force", "moment"):
        return (
            MalformedServiceIntentObstruction(
                f"{path}/kind", "expected 'force' or 'moment'"
            ),
        )
    if not isinstance(obj, _Decoded):
        return obj
    address = _decode_bound_address(
        obj.value["address"],
        f"{path}/address",
        known_addresses,
        _INTERFACE_ADDRESS_TYPES,
        _INTERFACE_ADDRESS_KINDS,
    )
    vector = _vector3(obj.value[vector_key], f"{path}/{vector_key}")
    obstructions = _result_obstructions((address, vector))
    return (
        obstructions
        if obstructions
        else _Decoded(
            ForceLoad(address.value, vector.value)
            if kind == "force"
            else MomentLoad(address.value, vector.value)
        )
    )


def _decode_support(
    value: object,
    path: str,
    known_addresses: frozenset[SemanticAddress],
) -> _DecodeResult[Support]:
    obj = _record(value, path, frozenset(("address", "law")))
    if not isinstance(obj, _Decoded):
        return obj
    address = _decode_bound_address(
        obj.value["address"],
        f"{path}/address",
        known_addresses,
        _INTERFACE_ADDRESS_TYPES,
        _INTERFACE_ADDRESS_KINDS,
    )
    law = _closed_value(obj.value["law"], SupportLaw, f"{path}/law")
    obstructions = _result_obstructions((address, law))
    return obstructions if obstructions else _Decoded(Support(address.value, law.value))


def _decode_joint(
    value: object,
    path: str,
    known_addresses: frozenset[SemanticAddress],
) -> _DecodeResult[JointInterface]:
    obj = _record(value, path, frozenset(("address", "law")))
    if not isinstance(obj, _Decoded):
        return obj
    address = _decode_bound_address(
        obj.value["address"],
        f"{path}/address",
        known_addresses,
        _INTERFACE_ADDRESS_TYPES,
        _INTERFACE_ADDRESS_KINDS,
    )
    law = _closed_value(obj.value["law"], JointLaw, f"{path}/law")
    obstructions = _result_obstructions((address, law))
    return (
        obstructions
        if obstructions
        else _Decoded(JointInterface(address.value, law.value))
    )


def _decode_heat_source(
    value: object,
    path: str,
    known_addresses: frozenset[SemanticAddress],
) -> _DecodeResult[HeatSource]:
    obj = _record(value, path, frozenset(("address", "watts")))
    if not isinstance(obj, _Decoded):
        return obj
    address = _decode_bound_address(
        obj.value["address"],
        f"{path}/address",
        known_addresses,
        _DOMAIN_ADDRESS_TYPES,
        _DOMAIN_ADDRESS_KINDS,
    )
    watts = _positive_budget(obj.value["watts"], f"{path}/watts")
    obstructions = _result_obstructions((address, watts))
    return obstructions if obstructions else _Decoded(HeatSource(address.value, watts.value))


def _support_contradictions(
    case_id: str, supports: tuple[Support, ...]
) -> tuple[ServiceIntentObstruction, ...]:
    addresses = tuple(dict.fromkeys(map(lambda support: support.address, supports)))
    return tuple(
        ContradictorySupportObstruction(
            case_id,
            address,
            tuple(
                dict.fromkeys(
                    support.law for support in supports if support.address == address
                )
            ),
        )
        for address in addresses
        if len(
            frozenset(
                support.law for support in supports if support.address == address
            )
        )
        > 1
    )
