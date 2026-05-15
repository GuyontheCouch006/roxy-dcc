import math


_R2_X = 0.7548776662466927
_R2_Y = 0.5698402909980532


def _fract(value):
    return value - math.floor(value)


def _hash_float(x, y, salt=0):
    value = math.sin(x * 127.1 + y * 311.7 + salt * 74.7) * 43758.5453123
    return _fract(value)


def pixel_sample_offset(x, y, frame):
    """Low-discrepancy pixel offset in [0, 1) with a per-pixel phase shift."""
    i = frame + 1
    return (
        _fract(_hash_float(x, y, 0) + i * _R2_X),
        _fract(_hash_float(x, y, 1) + i * _R2_Y),
    )


def clamp_color_sample(color, sample_clamp):
    if sample_clamp is None or sample_clamp <= 0:
        return color
    return color.__class__(
        min(max(color[0], 0.0), sample_clamp),
        min(max(color[1], 0.0), sample_clamp),
        min(max(color[2], 0.0), sample_clamp),
    )


def mis_power_weight(sample_pdf, other_pdf, beta=2.0):
    """Return the power-heuristic MIS weight for one sampled strategy."""
    sample_pdf = max(0.0, sample_pdf)
    other_pdf = max(0.0, other_pdf)
    sample_power = sample_pdf ** beta
    other_power = other_pdf ** beta
    total = sample_power + other_power
    if total <= 0.0:
        return 0.0
    return sample_power / total
