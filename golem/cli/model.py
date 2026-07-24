from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PositionalArgument:
    name: str
    help_line: str
    value_parser: Callable[[str], object] = str
    nargs: str | None = None
    choices: tuple[object, ...] = ()


@dataclass(frozen=True)
class OptionArgument:
    flags: tuple[str, ...]
    destination: str
    help_line: str
    value_parser: Callable[[str], object]
    metavar: str | None = None
    default: object = None
    required: bool = False


@dataclass(frozen=True)
class SwitchArgument:
    flags: tuple[str, ...]
    destination: str
    help_line: str
    default: bool = False


@dataclass(frozen=True)
class ExclusiveOptions:
    options: tuple[OptionArgument, ...]
    required: bool = False


type ParserArgument = (
    PositionalArgument | OptionArgument | SwitchArgument | ExclusiveOptions
)


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class CommandRunner:
    arguments: tuple[ParserArgument, ...]
    evaluate: Callable[[Namespace], CommandResult]

    def __call__(self, namespace: Namespace) -> CommandResult:
        return self.evaluate(namespace)


@dataclass(frozen=True)
class CommandDescriptor:
    name: str
    help_line: str
    runner: CommandRunner
