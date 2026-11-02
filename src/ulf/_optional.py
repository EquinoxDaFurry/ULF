
"""Checks and messages for features supplied by optional dependencies."""

from __future__ import annotations

import importlib.util
import sys

INSTALL_ALL = 'python -m pip install "unicode-lossful-format[all]"'


def has_optional_dependencies(*modules: str) -> bool:
    """Return True when all requested optional modules are available."""
    return all(importlib.util.find_spec(module) is not None for module in modules)


def warn_optional(feature: str) -> None:
    """Tell users how to enable an optional enhancement."""
    print(
        f"WARNING: {feature} will not work without optional dependencies. "
        f"Install every optional feature with: {INSTALL_ALL}",
        file=sys.stderr,
    )
