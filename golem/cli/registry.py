from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, cast

from golem.cli.model import CommandDescriptor

_COMMAND_MODULES = (
    ("check", "golem.cli.check"),
    ("compile", "golem.cli.compile"),
    ("contract", "golem.cli.contract"),
    ("eval", "golem.cli.eval"),
    ("forge", "golem.cli.forge"),
    ("look", "golem.cli.look"),
    ("pose-prior", "golem.cli.pose_prior"),
    ("session", "golem.cli.session"),
)

if TYPE_CHECKING:
    COMMANDS: tuple[CommandDescriptor, ...]


@cache
def command_by_name(name: str) -> CommandDescriptor | None:
    module_name = next(
        (
            module
            for command_name, module in _COMMAND_MODULES
            if command_name == name
        ),
        None,
    )
    return (
        cast(
            CommandDescriptor,
            import_module(module_name).COMMAND,
        )
        if module_name is not None
        else None
    )


@cache
def all_commands() -> tuple[CommandDescriptor, ...]:
    return tuple(
        command
        for name, _module in _COMMAND_MODULES
        for command in (command_by_name(name),)
        if command is not None
    )


def __getattr__(name: str) -> object:
    if name == "COMMANDS":
        return all_commands()
    raise AttributeError(name)
