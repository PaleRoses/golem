from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ContractSection:
    name: str
    render: Callable[[], str]
