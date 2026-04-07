from __future__ import annotations

from collections.abc import Callable


ADAPTERS: dict[str, type] = {}
STAGES: dict[str, type] = {}


def register_adapter(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        ADAPTERS[name] = cls
        return cls

    return decorator


def register_stage(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        STAGES[name] = cls
        return cls

    return decorator


def get_adapter(name: str) -> type:
    if name not in ADAPTERS:
        raise KeyError(f"Unknown adapter kind: {name}")
    return ADAPTERS[name]


def get_stage(name: str) -> type:
    if name not in STAGES:
        raise KeyError(f"Unknown stage kind: {name}")
    return STAGES[name]