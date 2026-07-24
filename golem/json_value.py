from __future__ import annotations

import math


def is_json_value(value: object) -> bool:
    return (
        value is None
        or isinstance(value, (bool, int, str))
        or isinstance(value, float)
        and math.isfinite(value)
        or isinstance(value, list)
        and all(map(is_json_value, value))
        or isinstance(value, dict)
        and all(
            isinstance(key, str) and is_json_value(item)
            for key, item in value.items()
        )
    )


__all__ = ("is_json_value",)
