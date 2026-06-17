#!/usr/bin/env python3
"""Print the active Python environment and key package versions."""

from __future__ import annotations

import importlib
import sys


PACKAGES = ("torch", "transformers", "PIL")


def version_for(package: str) -> str:
    try:
        module = importlib.import_module(package)
    except Exception as exc:
        return f"not importable ({exc.__class__.__name__}: {exc})"
    return getattr(module, "__version__", "installed, version unknown")


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    for package in PACKAGES:
        print(f"{package}: {version_for(package)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
