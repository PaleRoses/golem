from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

from golem.cli.model import (
    CommandDescriptor,
    CommandResult,
    CommandRunner,
    PositionalArgument,
)
from golem.contract import render_section, section_by_name, section_names


@dataclass(frozen=True)
class ContractArguments:
    section: str | None


def _contract_arguments(namespace: Namespace) -> ContractArguments:
    return ContractArguments(section=namespace.section)


def run(namespace: Namespace) -> CommandResult:
    arguments = _contract_arguments(namespace)
    if arguments.section is None:
        return CommandResult(
            0,
            stdout="\n".join(("AUTHORING CONTRACT SECTIONS", *section_names(), "")),
        )
    descriptor = section_by_name(arguments.section)
    return (
        CommandResult(0, stdout=render_section(descriptor.name))
        if descriptor is not None
        else CommandResult(
            2,
            stderr=f"REJECTED [UnknownContractSection] {arguments.section}\n",
        )
    )


COMMAND = CommandDescriptor(
    name="contract",
    help_line="list or render the live GOLEM authoring contract",
    runner=CommandRunner(
        arguments=(
            PositionalArgument(
                "section",
                "contract section to render",
                str,
                "?",
                section_names(),
            ),
        ),
        evaluate=run,
    ),
)
