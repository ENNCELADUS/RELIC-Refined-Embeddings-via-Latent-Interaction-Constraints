"""Unit tests for lazy package imports in ``src.optimize``."""

from __future__ import annotations

import importlib
import sys


def test_importing_optimize_package_does_not_import_run_submodule() -> None:
    sys.modules.pop("src.optimize.run", None)
    sys.modules.pop("src.optimize", None)

    optimize_package = importlib.import_module("src.optimize")

    assert "src.optimize.run" not in sys.modules
    assert callable(optimize_package.run_optimization)
    assert "src.optimize.run" in sys.modules
