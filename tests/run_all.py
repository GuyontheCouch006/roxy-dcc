# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Test discovery runner — auto-detects all *_tests.py modules,
#              collects test_ functions, and prints a pass/fail summary.
# ============================================

import importlib
import inspect
from pathlib import Path
from tests.utils import run_tests

test_files = sorted(Path(__file__).parent.glob("*_tests.py"))

total_passed, total_failed = 0, 0

for path in test_files:
    module_name = f"tests.{path.stem}"
    module = importlib.import_module(module_name)
    tests = [fn for _, fn in inspect.getmembers(module, inspect.isfunction)
             if fn.__name__.startswith("test_")]
    if tests:
        print(f"\n{path.stem}")
        passed, failed = run_tests(tests)
        total_passed += passed
        total_failed += failed

total = total_passed + total_failed
print(f"\n{'─' * 40}")
print(f"  {total_passed}/{total} passed", end="")
print(f"  |  {total_failed} failed" if total_failed else "")

