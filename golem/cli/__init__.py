from golem.cli.parser import build_parser, main
from golem.cli.registry import all_commands, command_by_name

__all__ = ["COMMANDS", "build_parser", "command_by_name", "main"]


def __getattr__(name: str) -> object:
    if name == "COMMANDS":
        return all_commands()
    raise AttributeError(name)
