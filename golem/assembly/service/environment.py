"""Ambient and coolant environment section decoders."""

from __future__ import annotations

from golem.materials import (
    decode_compressible_gas,
    decode_incompressible_fluid,
)
from golem.kernel.sheaf import (
    ChurchillChuIsothermalVerticalPlate,
    HeatTransferCorrelationId,
    HeatTransferCorrelationKind,
    NaturalConvectionSurfaceOrientation,
)

from golem.assembly.service.model import (
    AmbientCondition,
    AmbientHeatTransferPolicy,
    CoolantService,
    PowerPumpBudget,
    PressurePumpBudget,
    PumpBudget,
)
from golem.assembly.service.outcome import (
    MalformedServiceIntentObstruction,
    MaterialValidityObstruction,
    ServiceIntentObstruction,
)
from golem.assembly.service.primitives import (
    _Decoded,
    _DecodeResult,
    _decode_material,
    _expected_implemented,
    _positive_budget,
    _record,
    _result_obstructions,
)


def _decode_ambient(value: object) -> _DecodeResult[AmbientCondition | None]:
    if value is None:
        return _Decoded(None)
    path = "/service/ambient"
    obj = _record(
        value,
        path,
        frozenset(
            (
                "temperature_k",
                "heat_transfer_policy",
                "gas_material",
                "pressure_pa",
                "natural_convection",
            )
        ),
    )
    if not isinstance(obj, _Decoded):
        return obj
    temperature = _positive_budget(
        obj.value["temperature_k"], f"{path}/temperature_k"
    )
    policy = _expected_implemented(
        obj.value["heat_transfer_policy"],
        AmbientHeatTransferPolicy.STILL_AIR_NATURAL_CONVECTION,
        f"{path}/heat_transfer_policy",
        "policy",
    )
    gas = _decode_material(
        decode_compressible_gas,
        obj.value["gas_material"],
        f"{path}/gas_material",
    )
    pressure = _positive_budget(obj.value["pressure_pa"], f"{path}/pressure_pa")
    correlation = _decode_natural_convection_correlation(
        obj.value["natural_convection"], f"{path}/natural_convection"
    )
    local_obstructions = _result_obstructions(
        (temperature, policy, gas, pressure, correlation)
    )
    if local_obstructions:
        return local_obstructions
    ambient_gas = gas.value
    validity_obstructions: tuple[ServiceIntentObstruction, ...] = (
        (
            MaterialValidityObstruction(
                f"{path}/temperature_k",
                ambient_gas.material_id.value,
                temperature.value,
                ambient_gas.operating_temperature.minimum_kelvin,
                ambient_gas.operating_temperature.maximum_kelvin,
            ),
        )
        if not ambient_gas.operating_temperature.contains(temperature.value)
        else ()
    ) + (
        (
            MaterialValidityObstruction(
                f"{path}/pressure_pa",
                ambient_gas.material_id.value,
                pressure.value,
                ambient_gas.allowable_pressure.minimum_pascal,
                ambient_gas.allowable_pressure.maximum_pascal,
            ),
        )
        if not ambient_gas.allowable_pressure.contains(pressure.value)
        else ()
    )
    return (
        validity_obstructions
        if validity_obstructions
        else _Decoded(
            AmbientCondition(
                temperature.value,
                policy.value,
                ambient_gas,
                pressure.value,
                correlation.value,
            )
        )
    )


def _decode_natural_convection_correlation(
    value: object, path: str
) -> _DecodeResult[ChurchillChuIsothermalVerticalPlate]:
    obj = _record(
        value,
        path,
        frozenset(
            ("correlation", "surface_orientation", "characteristic_length_m")
        ),
    )
    if not isinstance(obj, _Decoded):
        return obj
    implemented_kind = (
        HeatTransferCorrelationKind.CHURCHILL_CHU_ISOTHERMAL_VERTICAL_PLATE
    )
    correlation_kind = _expected_implemented(
        obj.value["correlation"],
        implemented_kind,
        f"{path}/correlation",
        "correlation",
    )
    orientation = _expected_implemented(
        obj.value["surface_orientation"],
        NaturalConvectionSurfaceOrientation.VERTICAL_PLATE,
        f"{path}/surface_orientation",
        "orientation",
    )
    characteristic_length = _positive_budget(
        obj.value["characteristic_length_m"],
        f"{path}/characteristic_length_m",
    )
    obstructions = _result_obstructions(
        (correlation_kind, orientation, characteristic_length)
    )
    return (
        obstructions
        if obstructions
        else _Decoded(
            ChurchillChuIsothermalVerticalPlate(
                HeatTransferCorrelationId("service/ambient/natural-convection"),
                characteristic_length.value,
                orientation.value,
            )
        )
    )


def _decode_coolant(value: object) -> _DecodeResult[CoolantService | None]:
    if value is None:
        return _Decoded(None)
    path = "/service/coolant"
    obj = _record(
        value, path, frozenset(("material", "inlet_temperature_k", "pump"))
    )
    if not isinstance(obj, _Decoded):
        return obj
    material = _decode_material(
        decode_incompressible_fluid, obj.value["material"], f"{path}/material"
    )
    inlet = _positive_budget(
        obj.value["inlet_temperature_k"], f"{path}/inlet_temperature_k"
    )
    pump = _decode_pump(obj.value["pump"])
    obstructions = _result_obstructions((material, inlet, pump))
    if obstructions:
        return obstructions
    fluid = material.value
    validity_obstructions: tuple[ServiceIntentObstruction, ...] = (
        (
            MaterialValidityObstruction(
                f"{path}/inlet_temperature_k",
                fluid.material_id.value,
                inlet.value,
                fluid.operating_temperature.minimum_kelvin,
                fluid.operating_temperature.maximum_kelvin,
            ),
        )
        if not fluid.operating_temperature.contains(inlet.value)
        else ()
    ) + (
        (
            MaterialValidityObstruction(
                f"{path}/pump/maximum_pressure_pa",
                fluid.material_id.value,
                pump.value.maximum_pressure_pascal,
                0.0,
                fluid.allowable_pressure.maximum_pascal,
            ),
        )
        if isinstance(pump.value, PressurePumpBudget)
        and pump.value.maximum_pressure_pascal
        > fluid.allowable_pressure.maximum_pascal
        else ()
    )
    return (
        validity_obstructions
        if validity_obstructions
        else _Decoded(CoolantService(fluid, inlet.value, pump.value))
    )


def _decode_pump(value: object) -> _DecodeResult[PumpBudget]:
    path = "/service/coolant/pump"
    if not isinstance(value, dict):
        return (MalformedServiceIntentObstruction(path, "expected an object"),)
    kind = value.get("kind")
    budget_key = (
        "maximum_pressure_pa" if kind == "pressure_budget" else "maximum_power_w"
    )
    obj = _record(value, path, frozenset(("kind", budget_key)))
    if kind not in ("pressure_budget", "power_budget"):
        return (
            MalformedServiceIntentObstruction(
                f"{path}/kind", "expected 'pressure_budget' or 'power_budget'"
            ),
        )
    if not isinstance(obj, _Decoded):
        return obj
    budget = _positive_budget(obj.value[budget_key], f"{path}/{budget_key}")
    return (
        budget
        if not isinstance(budget, _Decoded)
        else _Decoded(
            PressurePumpBudget(budget.value)
            if kind == "pressure_budget"
            else PowerPumpBudget(budget.value)
        )
    )
