# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Shared test helpers — numeric comparison, Vec3 comparison,
#              and the run_tests runner used across all test modules.
# ============================================

def approx_eq(a, b, eps=1e-6):
    return abs(a - b) < eps

def vec3_approx_eq(a, b):
    return approx_eq(a.x, b.x) and approx_eq(a.y, b.y) and approx_eq(a.z, b.z)

def run_tests(tests):
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
    return passed, failed