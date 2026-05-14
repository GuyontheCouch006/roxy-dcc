import numpy as np


def edge_aware_denoise(linear_pixels, radius=1, sigma_color=0.08, amount=1.0):
    """Denoise linear RGB pixels with a small color-aware neighborhood filter."""
    if radius <= 0 or amount <= 0.0:
        return linear_pixels

    src = np.asarray(linear_pixels, dtype=np.float32)
    radius = int(radius)
    sigma2 = max(float(sigma_color) ** 2, 1e-8)
    amount = min(max(float(amount), 0.0), 1.0)

    padded = np.pad(src, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    center = padded[radius:radius + src.shape[0], radius:radius + src.shape[1]]
    accum = np.zeros_like(src)
    weight_sum = np.zeros(src.shape[:2], dtype=np.float32)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[
                radius + dy:radius + dy + src.shape[0],
                radius + dx:radius + dx + src.shape[1],
            ]
            color_delta = neighbor - center
            color_dist2 = np.sum(color_delta * color_delta, axis=2)
            spatial_weight = 1.0 / (1.0 + dx * dx + dy * dy)
            weight = spatial_weight * np.exp(-color_dist2 / (2.0 * sigma2)).astype(np.float32)
            accum += neighbor * weight[:, :, None]
            weight_sum += weight

    filtered = accum / np.maximum(weight_sum[:, :, None], 1e-8)
    return src * (1.0 - amount) + filtered * amount


def linear_to_display(linear_pixels):
    return np.sqrt(np.minimum(np.maximum(linear_pixels, 0.0), 1.0)).astype(np.float32)
