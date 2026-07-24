from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from functools import reduce
from typing import Protocol, assert_never

from golem.cli.model import (
    CommandDescriptor,
    CommandResult,
    ExclusiveOptions,
    OptionArgument,
    ParserArgument,
    PositionalArgument,
    SwitchArgument,
)
from golem.cli.registry import all_commands, command_by_name


class ArgumentSink(Protocol):
    def add_argument(self, *name_or_flags: str, **kwargs: object) -> object: ...


class SubparserSink(Protocol):
    def add_parser(self, name: str, **kwargs: object) -> argparse.ArgumentParser: ...


def _positional_keywords(argument: PositionalArgument) -> dict[str, object]:
    return {
        "help": argument.help_line,
        "type": argument.value_parser,
        **({"nargs": argument.nargs} if argument.nargs is not None else {}),
        **({"choices": argument.choices} if argument.choices else {}),
    }


def _option_keywords(argument: OptionArgument) -> dict[str, object]:
    return {
        "dest": argument.destination,
        "help": argument.help_line,
        "type": argument.value_parser,
        "default": argument.default,
        "required": argument.required,
        **({"metavar": argument.metavar} if argument.metavar is not None else {}),
    }


def _add_option(sink: ArgumentSink, argument: OptionArgument) -> ArgumentSink:
    sink.add_argument(*argument.flags, **_option_keywords(argument))
    return sink


def _switch_keywords(argument: SwitchArgument) -> dict[str, object]:
    return {
        "dest": argument.destination,
        "help": argument.help_line,
        "action": "store_true",
        "default": argument.default,
    }


def _add_argument(
    parser: argparse.ArgumentParser,
    argument: ParserArgument,
) -> argparse.ArgumentParser:
    match argument:
        case PositionalArgument():
            parser.add_argument(argument.name, **_positional_keywords(argument))
        case OptionArgument():
            _add_option(parser, argument)
        case SwitchArgument():
            parser.add_argument(*argument.flags, **_switch_keywords(argument))
        case ExclusiveOptions():
            group = parser.add_mutually_exclusive_group(required=argument.required)
            reduce(_add_option, argument.options, group)
        case _ as unreachable:
            assert_never(unreachable)
    return parser


def _add_command(
    subparsers: SubparserSink,
    descriptor: CommandDescriptor,
) -> SubparserSink:
    parser = subparsers.add_parser(
        descriptor.name,
        help=descriptor.help_line,
        description=descriptor.help_line,
    )
    reduce(_add_argument, descriptor.runner.arguments, parser)
    parser.set_defaults(command_runner=descriptor.runner)
    return subparsers


def _build_parser(
    commands: tuple[CommandDescriptor, ...],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m golem",
        description="GOLEM authoring, checking, and assembly",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    reduce(_add_command, commands, subparsers)
    return parser


def build_parser() -> argparse.ArgumentParser:
    return _build_parser(all_commands())


def _write_result(result: CommandResult) -> None:
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    selected = (
        None
        if not arguments or arguments[0] in ("-h", "--help")
        else command_by_name(arguments[0])
    )
    parser = _build_parser(
        all_commands() if selected is None else (selected,)
    )
    if not arguments:
        parser.print_help()
        return 0
    namespace = parser.parse_args(arguments)
    result = namespace.command_runner(namespace)
    _write_result(result)
    return result.exit_code
