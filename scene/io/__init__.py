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


def save_obj_as_roxy(obj_path, rxa_path, rxb_path=None, name=None, indexed=True):
    """Convert an OBJ into paired RXA/RXB scene files."""
    from scene.io.roxy_ascii import save_obj_as_roxy as _save_obj_as_roxy
    return _save_obj_as_roxy(
        obj_path,
        rxa_path,
        rxb_path=rxb_path,
        name=name,
        indexed=indexed,
    )
