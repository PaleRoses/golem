"""Typed accumulation and gluing for local assembly verdicts."""

from __future__ import annotations

from collections.abc import Callable

from golem.assembly.carriers import AssemblyObstruction, RejectedAssembly


def traverse_results[T, R](
    results: tuple[T | RejectedAssembly, ...],
    glue: Callable[[tuple[T, ...]], R],
) -> R | RejectedAssembly:
    obstructions = tuple(
        obstruction
        for result in results
        if isinstance(result, RejectedAssembly)
        for obstruction in result.obstructions
    )
    return (
        RejectedAssembly(obstructions)
        if obstructions
        else glue(
            tuple(
                result
                for result in results
                if not isinstance(result, RejectedAssembly)
            )
        )
    )


def collect_results[T](
    results: tuple[T | RejectedAssembly, ...],
) -> tuple[T, ...] | RejectedAssembly:
    return traverse_results(results, lambda accepted: accepted)


def descend_local_results[T](
    results: tuple[T | AssemblyObstruction, ...],
    accepted_type: type[T],
) -> tuple[T, ...] | RejectedAssembly:
    return collect_results(
        tuple(
            result
            if isinstance(result, accepted_type)
            else RejectedAssembly((result,))
            for result in results
        )
    )
