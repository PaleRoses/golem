#!/usr/bin/env python3
"""Effect boundary for compiling and rendering accepted coupled evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from golem.assembly.core import compile_assembly
from rehearsal.coupled.render_acceptance import (
    RejectedCoupledRender,
    render_coupled_acceptance,
    render_obstruction,
)


def main(argv: tuple[str, ...]) -> int:
    if len(argv) not in (2, 3):
        print("usage: render_cli.py ASSEMBLY.json OUTPUT_DIRECTORY [IMAGE_SIZE]")
        return 2
    spec_path = Path(argv[0])
    output_directory = Path(argv[1])
    try:
        image_size = int(argv[2]) if len(argv) == 3 else 560
        assembly_spec = json.loads(spec_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as failure:
        print(f"REJECTED [RenderInput] {failure}")
        return 1
    result = render_coupled_acceptance(
        compile_assembly(assembly_spec, spec_path.parent),
        output_directory,
        image_size,
    )
    if isinstance(result, RejectedCoupledRender):
        print(
            "\n".join(
                render_obstruction(obstruction)
                for obstruction in result.obstructions
            )
        )
        return 1
    print("\n".join(map(str, result.paths.all_paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
