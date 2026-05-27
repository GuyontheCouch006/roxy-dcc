from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GeometrySourceNode:
    """Lightweight source-history node synthesized for a shape's geometry."""

    shape: object
    name: str
    type_name: str
    attrs: dict = field(default_factory=dict)

    @property
    def raw(self):
        return self
