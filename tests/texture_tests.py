import numpy as np

from core import Vec2
from scene.textures import ImageTexture
from tests.utils import run_tests, approx_eq


def test_image_texture_samples_uv_corners():
    pixels = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    ], dtype=np.float32)
    tex = ImageTexture.from_array(pixels, flip_v=False)

    c = tex.sample(Vec2(0.0, 0.0))
    assert approx_eq(c.r, 1.0)
    assert approx_eq(c.g, 0.0)
    assert approx_eq(c.b, 0.0)


def test_image_texture_repeats_uvs():
    pixels = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    ], dtype=np.float32)
    tex = ImageTexture.from_array(pixels, flip_v=False)

    wrapped = tex.sample(Vec2(1.0, 1.0))
    origin = tex.sample(Vec2(0.0, 0.0))
    assert approx_eq(wrapped.r, origin.r)
    assert approx_eq(wrapped.g, origin.g)
    assert approx_eq(wrapped.b, origin.b)


def test_image_texture_bilinear_center():
    pixels = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    ], dtype=np.float32)
    tex = ImageTexture.from_array(pixels, flip_v=False)

    c = tex.sample(Vec2(0.5, 0.5))
    assert approx_eq(c.r, 0.5)
    assert approx_eq(c.g, 0.5)
    assert approx_eq(c.b, 0.5)


def test_image_texture_sample_many_matches_scalar_samples():
    pixels = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    ], dtype=np.float32)
    tex = ImageTexture.from_array(pixels, flip_v=False)
    uvs = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [0.25, 0.75],
    ], dtype=np.float32)

    batch = tex.sample_many(uvs)

    for i, (u, v) in enumerate(uvs):
        scalar = tex.sample(Vec2(float(u), float(v)))
        assert approx_eq(batch[i, 0], scalar.r)
        assert approx_eq(batch[i, 1], scalar.g)
        assert approx_eq(batch[i, 2], scalar.b)


def test_image_texture_array_exports_uint8_pixels():
    pixels = np.array([
        [[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]],
    ], dtype=np.float32)
    tex = ImageTexture.from_array(pixels, flip_v=False)

    uploaded = tex.load_pixels_u8()
    assert uploaded.dtype == np.uint8
    assert uploaded.shape == (1, 2, 3)
    assert uploaded[0, 0, 0] == 255
    assert uploaded[0, 1, 1] in (127, 128)


if __name__ == "__main__":
    run_tests([
        test_image_texture_samples_uv_corners,
        test_image_texture_repeats_uvs,
        test_image_texture_bilinear_center,
        test_image_texture_sample_many_matches_scalar_samples,
        test_image_texture_array_exports_uint8_pixels,
    ])
