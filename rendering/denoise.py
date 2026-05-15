import numpy as np


def _guide_buffer(name, guide, image_shape, channels=None):
    if guide is None:
        return None

    arr = np.asarray(guide, dtype=np.float32)
    if arr.shape[:2] != image_shape:
        raise ValueError(f"{name} guide must match pixel height and width")

    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim != 3:
        raise ValueError(f"{name} guide must be a 2D or 3D array")

    if channels is not None and arr.shape[2] != channels:
        raise ValueError(f"{name} guide must have {channels} channel(s)")

    return arr


def edge_aware_denoise(
    linear_pixels,
    radius=1,
    sigma_color=0.08,
    amount=1.0,
    *,
    normal=None,
    albedo=None,
    depth=None,
    sigma_normal=0.35,
    sigma_albedo=0.20,
    sigma_depth=1.0,
):
    """Denoise linear RGB pixels with a small guided bilateral filter.

    Passing no guide buffers preserves the original color-aware behavior.
    Optional normal/albedo guides must be HxWx3; depth may be HxW or HxWx1.
    """
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
    guide_weights = []

    for guide, sigma, name, channels in (
        (normal, sigma_normal, "normal", 3),
        (albedo, sigma_albedo, "albedo", 3),
        (depth, sigma_depth, "depth", 1),
    ):
        guide = _guide_buffer(name, guide, src.shape[:2], channels)
        if guide is None:
            continue

        guide_padded = np.pad(
            guide,
            ((radius, radius), (radius, radius), (0, 0)),
            mode="reflect",
        )
        guide_center = guide_padded[
            radius:radius + src.shape[0],
            radius:radius + src.shape[1],
        ]
        guide_sigma2 = max(float(sigma) ** 2, 1e-8)
        guide_weights.append((guide_padded, guide_center, guide_sigma2))

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[
                radius + dy:radius + dy + src.shape[0],
                radius + dx:radius + dx + src.shape[1],
            ]
            color_delta = neighbor - center
            color_dist2 = np.sum(color_delta * color_delta, axis=2)
            spatial_weight = 1.0 / (1.0 + dx * dx + dy * dy)
            weight = spatial_weight * np.exp(
                -color_dist2 / (2.0 * sigma2)
            ).astype(np.float32)

            for guide_padded, guide_center, guide_sigma2 in guide_weights:
                guide_neighbor = guide_padded[
                    radius + dy:radius + dy + src.shape[0],
                    radius + dx:radius + dx + src.shape[1],
                ]
                guide_delta = guide_neighbor - guide_center
                guide_dist2 = np.sum(guide_delta * guide_delta, axis=2)
                weight *= np.exp(
                    -guide_dist2 / (2.0 * guide_sigma2)
                ).astype(np.float32)

            accum += neighbor * weight[:, :, None]
            weight_sum += weight

    filtered = accum / np.maximum(weight_sum[:, :, None], 1e-8)
    return src * (1.0 - amount) + filtered * amount


def linear_to_display(linear_pixels):
    return np.sqrt(np.minimum(np.maximum(linear_pixels, 0.0), 1.0)).astype(np.float32)
