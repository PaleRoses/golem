"""Typed capability evaluation of the authoritative elemental-flow view."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from golem import paths as _paths
from golem.evals.harness import (
    AcceptedEvaluation,
    EvaluationObstruction,
    ProcessReceipt,
    emit_rendered_evaluation,
    render_outcome,
    run_process,
)


class BoundaryRole(StrEnum):
    SOURCE = "source"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class SeamFlow:
    left_cell: int
    right_cell: int
    conductance: float
    left_to_right_flow: float


@dataclass(frozen=True)
class CellBalance:
    cell_id: int
    is_boundary: bool
    net_outflow: float
    normalized_imbalance: float


@dataclass(frozen=True)
class Conservation:
    free_cell_count: int
    maximum_normalized_free_cell_imbalance: float


@dataclass(frozen=True)
class BoundaryFlow:
    boundary_id: str
    cell_id: int
    role: BoundaryRole
    net_outflow: float


@dataclass(frozen=True)
class ConduitDelivery:
    source_boundary_id: str
    target_boundary_id: str
    is_complete: bool
    path_cell_count: int
    delivered_flow: float
    minimum_path_flow: float


@dataclass(frozen=True)
class CellChannel:
    cell_id: int
    potential: float
    normalized_potential: float
    normalized_flow: float
    is_conduit: bool


@dataclass(frozen=True)
class ElementalFlowDocument:
    seam_flows: tuple[SeamFlow, ...]
    cell_balances: tuple[CellBalance, ...]
    conservation: Conservation
    boundary_flows: tuple[BoundaryFlow, ...]
    relative_boundary_balance_error: float
    conduit_deliveries: tuple[ConduitDelivery, ...]
    cell_channels: tuple[CellChannel, ...]


@dataclass(frozen=True)
class ElementalFlowCase:
    cell_count: int
    seam_count: int
    free_cell_count: int
    source_boundary_id: str
    terminal_boundary_ids: frozenset[str]
    conduit_cell_count: int
    tolerance: float

    @property
    def boundary_ids(self) -> frozenset[str]:
        return self.terminal_boundary_ids | {self.source_boundary_id}


ELEMENTAL_FLOW_CASE = ElementalFlowCase(
    cell_count=110,
    seam_count=324,
    free_cell_count=104,
    source_boundary_id="heart-rune",
    terminal_boundary_ids=frozenset(
        {
            "crown-rune",
            "left-gauntlet-rune",
            "right-gauntlet-rune",
            "left-foot-rune",
            "right-foot-rune",
        }
    ),
    conduit_cell_count=23,
    tolerance=1.0e-10,
)


@dataclass(frozen=True)
class ElementalFlowCapabilities:
    seam_flow: bool
    local_conservation: bool
    boundary_balance: bool
    conduit_delivery: bool
    cell_channels: bool

    @property
    def score(self) -> float:
        return float(
            sum(
                map(
                    int,
                    (
                        self.seam_flow,
                        self.local_conservation,
                        self.boundary_balance,
                        self.conduit_delivery,
                        self.cell_channels,
                    ),
                )
            )
        )


@dataclass(frozen=True)
class ElementalFlowEvaluation(AcceptedEvaluation):
    capabilities: ElementalFlowCapabilities

    def evaluation_payload(self) -> dict[str, object]:
        return {
            "boundary_balance": self.capabilities.boundary_balance,
            "cell_channels": self.capabilities.cell_channels,
            "conduit_delivery": self.capabilities.conduit_delivery,
            "elemental_flow_capability_score": self.capabilities.score,
            "local_conservation": self.capabilities.local_conservation,
            "seam_flow": self.capabilities.seam_flow,
        }


type ElementalFlowDecodeResult = ElementalFlowDocument | EvaluationObstruction
type ElementalFlowResult = ElementalFlowEvaluation | EvaluationObstruction


def decode_elemental_flow_document(
    receipt: ProcessReceipt,
) -> ElementalFlowDecodeResult:
    if receipt.return_code != 0:
        return EvaluationObstruction(
            "elemental-flow/process",
            receipt.stderr.strip() or f"exit code {receipt.return_code}",
        )
    lines = tuple(filter(None, map(str.strip, receipt.stdout.splitlines())))
    if not lines:
        return EvaluationObstruction(
            "elemental-flow/stdout", "expected a JSON document"
        )
    try:
        payload = json.loads(lines[-1])
    except (TypeError, ValueError, json.JSONDecodeError) as failure:
        return EvaluationObstruction("elemental-flow/stdout", str(failure))
    document = _record(payload)
    if document is None:
        return EvaluationObstruction("elemental-flow/stdout", "expected an object")
    seam_flows = _decode_records(document, "seam_flows", _decode_seam_flow)
    if isinstance(seam_flows, EvaluationObstruction):
        return seam_flows
    cell_balances = _decode_records(
        document, "cell_balances", _decode_cell_balance
    )
    if isinstance(cell_balances, EvaluationObstruction):
        return cell_balances
    conservation = _decode_conservation(document.get("conservation"))
    if isinstance(conservation, EvaluationObstruction):
        return conservation
    boundary_flows = _decode_records(
        document, "boundary_flows", _decode_boundary_flow
    )
    if isinstance(boundary_flows, EvaluationObstruction):
        return boundary_flows
    relative_error = _number(document.get("relative_boundary_balance_error"))
    if relative_error is None:
        return EvaluationObstruction(
            "/relative_boundary_balance_error", "expected a finite number"
        )
    deliveries = _decode_records(
        document, "conduit_deliveries", _decode_conduit_delivery
    )
    if isinstance(deliveries, EvaluationObstruction):
        return deliveries
    channels = _decode_records(document, "cell_channels", _decode_cell_channel)
    if isinstance(channels, EvaluationObstruction):
        return channels
    return ElementalFlowDocument(
        seam_flows=seam_flows,
        cell_balances=cell_balances,
        conservation=conservation,
        boundary_flows=boundary_flows,
        relative_boundary_balance_error=relative_error,
        conduit_deliveries=deliveries,
        cell_channels=channels,
    )


def measure_capabilities(
    document: ElementalFlowDocument,
    case: ElementalFlowCase = ELEMENTAL_FLOW_CASE,
) -> ElementalFlowCapabilities:
    expected_cells = frozenset(range(case.cell_count))
    seam_flow = (
        len(document.seam_flows) == case.seam_count
        and all(
            0 <= flow.left_cell < flow.right_cell < case.cell_count
            and flow.conductance > 0.0
            for flow in document.seam_flows
        )
    )
    local_conservation = (
        len(document.cell_balances) == case.cell_count
        and frozenset(balance.cell_id for balance in document.cell_balances)
        == expected_cells
        and all(
            0 <= balance.cell_id < case.cell_count
            and balance.normalized_imbalance >= 0.0
            for balance in document.cell_balances
        )
        and document.conservation.free_cell_count == case.free_cell_count
        and document.conservation.maximum_normalized_free_cell_imbalance
        <= case.tolerance
    )
    boundary_balance = (
        len(document.boundary_flows) == len(case.boundary_ids)
        and frozenset(flow.boundary_id for flow in document.boundary_flows)
        == case.boundary_ids
        and sum(
            flow.role is BoundaryRole.SOURCE for flow in document.boundary_flows
        )
        == 1
        and sum(
            flow.role is BoundaryRole.TERMINAL
            for flow in document.boundary_flows
        )
        == len(case.terminal_boundary_ids)
        and all(
            0 <= flow.cell_id < case.cell_count
            and (
                flow.role is BoundaryRole.SOURCE and flow.net_outflow > 0.0
                or flow.role is BoundaryRole.TERMINAL and flow.net_outflow < 0.0
            )
            for flow in document.boundary_flows
        )
        and document.relative_boundary_balance_error <= case.tolerance
    )
    conduit_delivery = (
        len(document.conduit_deliveries) == len(case.terminal_boundary_ids)
        and frozenset(
            delivery.target_boundary_id
            for delivery in document.conduit_deliveries
        )
        == case.terminal_boundary_ids
        and all(
            delivery.source_boundary_id == case.source_boundary_id
            and delivery.is_complete
            and delivery.path_cell_count >= 2
            and delivery.delivered_flow > 0.0
            and delivery.minimum_path_flow > 0.0
            for delivery in document.conduit_deliveries
        )
    )
    potentials = tuple(channel.normalized_potential for channel in document.cell_channels)
    flows = tuple(channel.normalized_flow for channel in document.cell_channels)
    cell_channels = (
        len(document.cell_channels) == case.cell_count
        and frozenset(channel.cell_id for channel in document.cell_channels)
        == expected_cells
        and all(
            0 <= channel.cell_id < case.cell_count
            and 0.0 <= channel.normalized_potential <= 1.0
            and 0.0 <= channel.normalized_flow <= 1.0
            for channel in document.cell_channels
        )
        and min(potentials, default=-1.0) == 0.0
        and max(potentials, default=-1.0) == 1.0
        and max(flows, default=-1.0) == 1.0
        and sum(channel.is_conduit for channel in document.cell_channels)
        == case.conduit_cell_count
    )
    return ElementalFlowCapabilities(
        seam_flow=seam_flow,
        local_conservation=local_conservation,
        boundary_balance=boundary_balance,
        conduit_delivery=conduit_delivery,
        cell_channels=cell_channels,
    )


def evaluate_elemental_flow() -> ElementalFlowResult:
    receipt = run_process(
        (
            sys.executable,
            "-m",
            "golem.kernel.sheaf",
            str(_paths.OUTPUTS / "plate_complex.json"),
            str(_paths.SPECS / "elemental_field.json"),
            "--elemental-flow-json",
        )
    )
    decoded = decode_elemental_flow_document(receipt)
    return (
        decoded
        if isinstance(decoded, EvaluationObstruction)
        else ElementalFlowEvaluation(measure_capabilities(decoded))
    )


def elemental_flow_capability_score() -> float:
    result = evaluate_elemental_flow()
    return (
        result.capabilities.score
        if isinstance(result, ElementalFlowEvaluation)
        else 0.0
    )


def _decode_records[record](
    document: Mapping[str, object],
    key: str,
    decoder: Callable[[object, str], record | EvaluationObstruction],
) -> tuple[record, ...] | EvaluationObstruction:
    payload = document.get(key)
    if not isinstance(payload, list):
        return EvaluationObstruction(f"/{key}", "expected an array")
    decoded = tuple(
        decoder(value, f"/{key}/{index}")
        for index, value in enumerate(payload)
    )
    obstruction = next(
        (
            value
            for value in decoded
            if isinstance(value, EvaluationObstruction)
        ),
        None,
    )
    return (
        obstruction
        if obstruction is not None
        else tuple(
            value
            for value in decoded
            if not isinstance(value, EvaluationObstruction)
        )
    )


def _decode_seam_flow(value: object, address: str) -> SeamFlow | EvaluationObstruction:
    record = _record(
        value,
        ("left_cell", "right_cell", "conductance", "left_to_right_flow"),
    )
    if record is None:
        return EvaluationObstruction(address, "invalid seam-flow record")
    left = _integer(record.get("left_cell"))
    right = _integer(record.get("right_cell"))
    conductance = _number(record.get("conductance"))
    flow = _number(record.get("left_to_right_flow"))
    return (
        SeamFlow(left, right, conductance, flow)
        if None not in (left, right, conductance, flow)
        else EvaluationObstruction(address, "invalid seam-flow fields")
    )


def _decode_cell_balance(
    value: object, address: str
) -> CellBalance | EvaluationObstruction:
    record = _record(
        value,
        ("cell_id", "is_boundary", "net_outflow", "normalized_imbalance"),
    )
    if record is None:
        return EvaluationObstruction(address, "invalid cell-balance record")
    cell_id = _integer(record.get("cell_id"))
    is_boundary = record.get("is_boundary")
    net_outflow = _number(record.get("net_outflow"))
    imbalance = _number(record.get("normalized_imbalance"))
    return (
        CellBalance(cell_id, is_boundary, net_outflow, imbalance)
        if cell_id is not None
        and isinstance(is_boundary, bool)
        and net_outflow is not None
        and imbalance is not None
        else EvaluationObstruction(address, "invalid cell-balance fields")
    )


def _decode_conservation(value: object) -> Conservation | EvaluationObstruction:
    record = _record(
        value,
        ("free_cell_count", "maximum_normalized_free_cell_imbalance"),
    )
    if record is None:
        return EvaluationObstruction("/conservation", "invalid conservation record")
    free_count = _integer(record.get("free_cell_count"))
    imbalance = _number(record.get("maximum_normalized_free_cell_imbalance"))
    return (
        Conservation(free_count, imbalance)
        if free_count is not None and imbalance is not None
        else EvaluationObstruction("/conservation", "invalid conservation fields")
    )


def _decode_boundary_flow(
    value: object, address: str
) -> BoundaryFlow | EvaluationObstruction:
    record = _record(
        value,
        ("boundary_id", "cell_id", "role", "net_outflow"),
    )
    if record is None:
        return EvaluationObstruction(address, "invalid boundary-flow record")
    boundary_id = record.get("boundary_id")
    cell_id = _integer(record.get("cell_id"))
    role = next(
        (candidate for candidate in BoundaryRole if candidate.value == record.get("role")),
        None,
    )
    net_outflow = _number(record.get("net_outflow"))
    return (
        BoundaryFlow(boundary_id, cell_id, role, net_outflow)
        if isinstance(boundary_id, str)
        and cell_id is not None
        and role is not None
        and net_outflow is not None
        else EvaluationObstruction(address, "invalid boundary-flow fields")
    )


def _decode_conduit_delivery(
    value: object, address: str
) -> ConduitDelivery | EvaluationObstruction:
    record = _record(
        value,
        (
            "source_boundary_id",
            "target_boundary_id",
            "is_complete",
            "path_cell_count",
            "delivered_flow",
            "minimum_path_flow",
        ),
    )
    if record is None:
        return EvaluationObstruction(address, "invalid conduit-delivery record")
    source = record.get("source_boundary_id")
    target = record.get("target_boundary_id")
    complete = record.get("is_complete")
    path_count = _integer(record.get("path_cell_count"))
    delivered = _number(record.get("delivered_flow"))
    minimum = _number(record.get("minimum_path_flow"))
    return (
        ConduitDelivery(source, target, complete, path_count, delivered, minimum)
        if isinstance(source, str)
        and isinstance(target, str)
        and isinstance(complete, bool)
        and path_count is not None
        and delivered is not None
        and minimum is not None
        else EvaluationObstruction(address, "invalid conduit-delivery fields")
    )


def _decode_cell_channel(
    value: object, address: str
) -> CellChannel | EvaluationObstruction:
    record = _record(
        value,
        (
            "cell_id",
            "potential",
            "normalized_potential",
            "normalized_flow",
            "is_conduit",
        ),
    )
    if record is None:
        return EvaluationObstruction(address, "invalid cell-channel record")
    cell_id = _integer(record.get("cell_id"))
    potential = _number(record.get("potential"))
    normalized_potential = _number(record.get("normalized_potential"))
    normalized_flow = _number(record.get("normalized_flow"))
    is_conduit = record.get("is_conduit")
    return (
        CellChannel(
            cell_id,
            potential,
            normalized_potential,
            normalized_flow,
            is_conduit,
        )
        if cell_id is not None
        and potential is not None
        and normalized_potential is not None
        and normalized_flow is not None
        and isinstance(is_conduit, bool)
        else EvaluationObstruction(address, "invalid cell-channel fields")
    )


def _number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if isfinite(number) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _record(
    value: object,
    fields: tuple[str, ...] | None = None,
) -> Mapping[str, object] | None:
    return (
        value
        if isinstance(value, dict)
        and all(map(lambda key: isinstance(key, str), value.keys()))
        and (fields is None or frozenset(value) == frozenset(fields))
        else None
    )


def main() -> int:
    return emit_rendered_evaluation(render_outcome(evaluate_elemental_flow()))


if __name__ == "__main__":
    raise SystemExit(main())
