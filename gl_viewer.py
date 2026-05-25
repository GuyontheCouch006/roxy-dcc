import argparse

from rendering import GLViewport
from main import build_bicycle, build_gallery, build_glossy_test


SCENE_BUILDERS = {
    "glossy": build_glossy_test,
    "bicycle": build_bicycle,
    "gallery": build_gallery,
}


def main():
    args = _parse_args()
    width, height = _parse_resolution(args.resolution)
    world = SCENE_BUILDERS[args.scene](width, height)

    viewport = GLViewport(
        width,
        height,
        f"Roxy GL Viewport - {args.scene}",
        world=world,
        sync_camera=True,
    )
    try:
        while not viewport.should_close and not viewport.consume_render_requested():
            viewport.poll_events()
            viewport.draw_scene()
    finally:
        viewport.close()


def _parse_args():
    parser = argparse.ArgumentParser(description="Open an interactive Roxy GL scene viewport.")
    parser.add_argument(
        "--scene",
        choices=sorted(SCENE_BUILDERS.keys()),
        default="glossy",
        help="Scene to load into the viewport.",
    )
    parser.add_argument(
        "--resolution",
        default="1280x720",
        help="Viewport size as WIDTHxHEIGHT.",
    )
    return parser.parse_args()


def _parse_resolution(value):
    try:
        width, height = value.lower().split("x", 1)
        return int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must look like 1280x720") from exc


if __name__ == "__main__":
    main()
