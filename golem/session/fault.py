"""Typed engine-fault projection at oracle boundaries (wave-5 R11).

Every engine exception crossing an oracle boundary (session, check, look,
compile) projects to exactly one typed obstruction: an :class:`EngineFault`
naming the seam, the exception kind, and a stable message digest. No oracle
may surface a raw exception or an untyped catch-all; the retired
``unexpected_compile_exception`` classifier is replaced by this projection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

ENGINE_FAULT_CODE = "engine.fault"

BODY_COMPILE_SEAM = "body.compile"
ASSEMBLY_COMPILE_SEAM = "assembly.compile"
SESSION_EVIDENCE_SEAM = "session.evidence"
SESSION_SUBMIT_SEAM = "session.submit"
CHECK_DECODE_SEAM = "check.decode"
CHECK_SENSES_SEAM = "check.senses"
ORTHOGRAPHIC_RENDER_SEAM = "orthographic.render"
ORTHOGRAPHIC_WRITE_SEAM = "orthographic.write"


@dataclass(frozen=True)
class EngineFault:
    """One typed obstruction for an engine exception at an oracle boundary."""

    seam: str
    exception_kind: str
    message: str

    @property
    def message_digest(self) -> str:
        payload = f"{self.exception_kind}\n{self.message}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def render(self) -> str:
        return f"{self.exception_kind} @ {self.seam}: {self.message}"

    def to_json(self) -> dict[str, str]:
        return {
            "code": ENGINE_FAULT_CODE,
            "seam": self.seam,
            "exception_kind": self.exception_kind,
            "message_digest": self.message_digest,
            "message": self.message,
        }

    def render_refusal(self) -> str:
        return (
            f"REJECTED [{ENGINE_FAULT_CODE}] "
            f"{json.dumps(self.to_json(), sort_keys=True)}\n"
        )


def project_engine_fault(seam: str, exception: Exception) -> EngineFault:
    return EngineFault(seam, type(exception).__name__, str(exception))


__all__ = [
    "ASSEMBLY_COMPILE_SEAM",
    "BODY_COMPILE_SEAM",
    "CHECK_DECODE_SEAM",
    "CHECK_SENSES_SEAM",
    "ENGINE_FAULT_CODE",
    "EngineFault",
    "ORTHOGRAPHIC_RENDER_SEAM",
    "ORTHOGRAPHIC_WRITE_SEAM",
    "SESSION_EVIDENCE_SEAM",
    "SESSION_SUBMIT_SEAM",
    "project_engine_fault",
]
