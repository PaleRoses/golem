from __future__ import annotations

from golem.contract.envelope import SECTION as ENVELOPE_SECTION
from golem.contract.exemplar import SECTION as EXEMPLAR_SECTION
from golem.contract.model import ContractSection
from golem.contract.primer import SECTION as PRIMER_SECTION
from golem.contract.receipts import SECTION as RECEIPTS_SECTION
from golem.contract.relations import SECTION as RELATIONS_SECTION
from golem.contract.schema import SECTION as SCHEMA_SECTION
from golem.contract.vocabulary import SECTION as VOCABULARY_SECTION


SECTIONS: tuple[ContractSection, ...] = (
    PRIMER_SECTION,
    RELATIONS_SECTION,
    VOCABULARY_SECTION,
    EXEMPLAR_SECTION,
    RECEIPTS_SECTION,
    ENVELOPE_SECTION,
    SCHEMA_SECTION,
)


def section_by_name(name: str) -> ContractSection | None:
    return next((section for section in SECTIONS if section.name == name), None)


def section_names() -> tuple[str, ...]:
    return tuple(section.name for section in SECTIONS)


def render_section(name: str) -> str:
    section = section_by_name(name)
    if section is None:
        raise KeyError(name)
    rendered = section.render()
    return rendered if rendered.endswith("\n") else rendered + "\n"
