from dataclasses import dataclass
from typing import Optional


@dataclass
class RenderStats:
    width: int
    height: int
    samples_requested: int
    samples_rendered: int
    max_depth: int
    rays_cast: int
    elapsed_seconds: float
    primitive_count: int = 0
    bvh_nodes: int = 0
    bvh_triangles: int = 0
    bvh_materials: int = 0
    bvh_leaf_size: int = 0
    adaptive_sampling: bool = False
    adaptive_stopped: bool = False
    adaptive_threshold: float = 0.0
    adaptive_error: Optional[float] = None
    adaptive_min_samples: int = 0

    @property
    def pixel_count(self):
        return self.width * self.height

    @property
    def rays_per_pixel(self):
        if self.pixel_count == 0:
            return 0.0
        return self.rays_cast / self.pixel_count

    @property
    def rays_per_second(self):
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.rays_cast / self.elapsed_seconds

    def format_report(self):
        lines = [
            "Render report",
            f"  resolution:      {self.width} x {self.height}",
            f"  samples:         {self.samples_rendered} / {self.samples_requested}",
            f"  max depth:       {self.max_depth}",
            f"  rays cast:       {self.rays_cast:,}",
            f"  rays / pixel:    {self.rays_per_pixel:,.2f}",
            f"  elapsed:         {self.elapsed_seconds:.3f}s",
            f"  rays / second:   {self.rays_per_second:,.0f}",
            f"  primitives:      {self.primitive_count:,}",
            f"  BVH nodes:       {self.bvh_nodes:,}",
            f"  BVH triangles:   {self.bvh_triangles:,}",
            f"  BVH materials:   {self.bvh_materials:,}",
            f"  BVH leaf size:   {self.bvh_leaf_size:,}",
        ]
        if self.adaptive_sampling:
            error = "n/a" if self.adaptive_error is None else f"{self.adaptive_error:.6f}"
            lines.extend([
                f"  adaptive stop:   {'yes' if self.adaptive_stopped else 'no'}",
                f"  adaptive error:  {error}",
                f"  adaptive target: {self.adaptive_threshold:.6f}",
            ])
        return "\n".join(lines)
