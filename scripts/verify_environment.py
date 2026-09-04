#!/usr/bin/env python3
"""Run the maintained official-environment contract verification."""

from __future__ import annotations

import json
import platform
import sys
import unittest
from importlib.metadata import version
from pathlib import Path


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(
        str(repository_root / "tests"), pattern="test_environment_contract.py"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {
        "environment": "kaggriculture",
        "kaggle_environments": version("kaggle-environments"),
        "python": platform.python_version(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
