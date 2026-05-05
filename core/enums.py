# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Shared enumerations used across the ray tracer (e.g. RotationOrder).
# ============================================

from enum import Enum


class RotationOrder(Enum):
    """Euler rotation order applied when building a rotation matrix."""

    XYZ = 'xyz'
    ZYX = 'zyx'
    XZY = 'xzy'
    YXZ = 'yxz'
    YZX = 'yzx'
    ZXY = 'zxy'
