# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: Scene serialization and deserialization to/from JSON.
# ============================================

import json
from scene.world import World


def save_scene(world, filepath):
    """Save a World to a JSON file."""
    data = world.to_dict()
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_scene(filepath):
    """Load a World from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return World.from_dict(data)