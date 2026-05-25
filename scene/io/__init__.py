import json
from pathlib import Path

from scene.world import World
from scene.io.assimp_importer import AssimpImporter

def load_scene(path) -> World:
    """Load a World from a JSON or Roxy ASCII scene file."""
    if Path(path).suffix.lower() == ".rxa":
        from scene.io.roxy_ascii import load_rxa
        return load_rxa(path)
    with open(path, "r") as f:
        return World.from_dict(json.load(f))


def save_scene(world: World, path) -> None:
    """Write a World to a JSON or Roxy ASCII scene file."""
    if Path(path).suffix.lower() == ".rxa":
        from scene.io.roxy_ascii import save_rxa
        save_rxa(world, path)
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(world.to_dict(), f, indent=2)
