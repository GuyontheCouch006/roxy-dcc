from pathlib import Path
from tempfile import TemporaryDirectory

from scene.io.mtl_loader import MTLLoader
from tests.utils import run_tests


def test_map_kd_creates_image_texture_with_resolved_path():
    with TemporaryDirectory() as tmp:
        mtl_path = Path(tmp) / "material.mtl"
        mtl_path.write_text(
            "\n".join([
                "newmtl painted",
                "Kd 0.8 0.8 0.8",
                "map_Kd -bm 0.7 textures/albedo.jpg",
            ]),
            encoding="utf-8",
        )

        mat = MTLLoader.load(str(mtl_path))["painted"].to_material()
        assert mat._albedo_texture is not None
        assert mat._albedo_texture.path == str(Path(tmp) / "textures/albedo.jpg")


if __name__ == "__main__":
    run_tests([
        test_map_kd_creates_image_texture_with_resolved_path,
    ])
