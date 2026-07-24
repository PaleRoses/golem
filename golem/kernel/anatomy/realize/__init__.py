"""Vasculature realization: CCO growth, corridors, materialization, clearance."""

__all__ = ["allocate_terminal_pairs", "realize_vasculature"]


def __getattr__(name: str):
    if name in __all__:
        from golem.kernel.anatomy.realize import allocation

        return getattr(allocation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
