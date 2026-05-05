# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Math utility functions — safe division, clamping, interpolation,
#              angle conversion, and random unit vector sampling.
# ============================================

import math
import numpy as np
from core.vectors import Vec3


# ─── Math Helpers ─────────────────────────────────────────────────────────────

def safe_div(a, b, eps=1e-10):
    """Divide a by b, returning ±inf instead of raising on near-zero b."""
    if abs(b) < eps:
        return float('inf') if a >= 0 else float('-inf')
    return a / b

def clamp(value, lo, hi):
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))

def lerp(a, b, t):
    """Linear interpolation between a and b at parameter t."""
    return a + (b - a) * t

def degrees_to_radians(degrees):
    return degrees * math.pi / 180

def radians_to_degrees(radians):
    return radians * 180 / math.pi


# ─── Sampling ─────────────────────────────────────────────────────────────────

def random_unit_vector():
    v = np.random.normal(0, 1, 3)
    v = v / np.linalg.norm(v)
    return Vec3(float(v[0]), float(v[1]), float(v[2]))