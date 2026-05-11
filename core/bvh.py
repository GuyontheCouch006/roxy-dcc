class BVHNode:
    def __init__(self, triangles, depth=0, max_leaf_size=4):
        self.bounds = self._compute_bounds(triangles)
        
        if len(triangles) <= max_leaf_size or depth > 20:
            self.triangles = triangles  # leaf
            self.left = self.right = None
        else:
            axis = self._longest_axis(self.bounds)
            triangles.sort(key=lambda t: t.centroid()[axis])
            mid = len(triangles) // 2
            self.left  = BVHNode(triangles[:mid], depth+1)
            self.right = BVHNode(triangles[mid:], depth+1)
            self.triangles = None  # interior node

    def intersect(self, ray):
        if not self.bounds.intersect(ray):
            return None
        if self.triangles is not None:  # leaf
            closest_t   = float('inf')
            closest_hit = None
            for tri in self.triangles:
                hit = tri.intersect(ray)
                if hit and hit[0] < closest_t:
                    closest_t   = hit[0]
                    closest_hit = hit
            return closest_hit
        # interior — test both children
        left  = self.left.intersect(ray)  if self.left  else None
        right = self.right.intersect(ray) if self.right else None
        if left and right:
            return left if left[0] < right[0] else right
        return left or right
    
    def _compute_bounds(self, triangles):
        bounds = triangles[0].local_bounds()
        for tri in triangles[1:]:
            bounds = bounds.union(tri.local_bounds())
        return bounds
    
    def _longest_axis(self, bounds):
        extents = bounds.max - bounds.min
        if extents.x >= extents.y and extents.x >= extents.z:
            return 0
        elif extents.y >= extents.z:
            return 1
        else:
            return 2
        
    def __repr__(self):
        if self.triangles is not None:
            return f"BVHNode(leaf, num_triangles={len(self.triangles)})"
        else:
            return f"BVHNode(interior, left={self.left}, right={self.right})"
        