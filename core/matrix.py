# ============================================
# Author: Seth
# Date: May 2026
# Version: 1.0
# Description: 4x4 matrix class supporting TRS construction, inversion, transposition,
#              point/vector/normal transforms, and a look-at camera matrix.
# ============================================

import math

from core.enums import RotationOrder
from core.vectors import Vec3


class Mat4x4:
    """Row-major 4x4 matrix for 3D homogeneous transforms."""

    def __init__(self, rows=None):
        self.rows = rows if rows is not None else [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

    @classmethod
    def identity(cls):
        return cls([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @classmethod
    def from_trs(cls, translation, rotation, scale,
                 shear=None, pivot=None, rotation_order=RotationOrder.XYZ):
        """Build a world matrix from TRS components.

        Composition order: T * (T_pivot * R * Sh * S * T_pivot_inv)
        Rotation axes are applied in the order specified by rotation_order.
        """
        Sh = cls.shear(shear.x, shear.y, shear.z) if shear else cls.identity()
        T = cls.translation(translation.x, translation.y, translation.z)
        S = cls.scale(scale.x, scale.y, scale.z)
        Rx = cls.rotation_x(rotation.x)
        Ry = cls.rotation_y(rotation.y)
        Rz = cls.rotation_z(rotation.z)

        if rotation_order == RotationOrder.XYZ:
            R = Rx * Ry * Rz
        elif rotation_order == RotationOrder.ZYX:
            R = Rz * Ry * Rx
        elif rotation_order == RotationOrder.XZY:
            R = Rx * Rz * Ry
        elif rotation_order == RotationOrder.YXZ:
            R = Ry * Rx * Rz
        elif rotation_order == RotationOrder.YZX:
            R = Ry * Rz * Rx
        elif rotation_order == RotationOrder.ZXY:
            R = Rz * Rx * Ry
        else:
            raise ValueError(f"Unknown rotation order: {rotation_order}")

        if pivot:
            T_pivot = Mat4x4.translation(pivot.x, pivot.y, pivot.z)
            T_pivot_inv = Mat4x4.translation(-pivot.x, -pivot.y, -pivot.z)
            return T * T_pivot * R * Sh * S * T_pivot_inv
        else:
            return T * R * Sh * S

    @classmethod
    def inverse_trs(cls, translation, rotation, scale,
                    shear=None, pivot=None, rotation_order=RotationOrder.XYZ):
        """Compute the inverse of a TRS matrix.

        Uses the analytic shortcut (S_inv * R_T * T_inv) when there is no shear
        or pivot, falling back to Gauss-Jordan for the general case.
        """
        has_shear = shear and (shear.x != 0 or shear.y != 0 or shear.z != 0)
        has_pivot = pivot and (pivot.x != 0 or pivot.y != 0 or pivot.z != 0)

        if has_shear or has_pivot:
            return cls.from_trs(
                translation, rotation, scale, shear, pivot, rotation_order
            ).inverse()

        S_inv = Mat4x4.scale(1 / scale.x, 1 / scale.y, 1 / scale.z)

        Rx = Mat4x4.rotation_x(rotation.x)
        Ry = Mat4x4.rotation_y(rotation.y)
        Rz = Mat4x4.rotation_z(rotation.z)

        if rotation_order == RotationOrder.XYZ:
            R = Rx * Ry * Rz
        elif rotation_order == RotationOrder.ZYX:
            R = Rz * Ry * Rx
        elif rotation_order == RotationOrder.XZY:
            R = Rx * Rz * Ry
        elif rotation_order == RotationOrder.YXZ:
            R = Ry * Rx * Rz
        elif rotation_order == RotationOrder.YZX:
            R = Ry * Rz * Rx
        elif rotation_order == RotationOrder.ZXY:
            R = Rz * Rx * Ry
        else:
            raise ValueError("Invalid rotation order")

        R_inv = R.transpose()  # For pure rotation, inverse equals transpose.
        T_inv = Mat4x4.translation(-translation.x, -translation.y, -translation.z)
        return S_inv * R_inv * T_inv

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = Mat4x4()
            for i in range(4):
                for j in range(4):
                    result.rows[i][j] = self.rows[i][j] * other
            return result
        elif isinstance(other, Mat4x4):
            result = Mat4x4()
            for i in range(4):
                for j in range(4):
                    result.rows[i][j] = sum(
                        self.rows[i][k] * other.rows[k][j] for k in range(4)
                    )
            return result
        else:
            raise TypeError("Unsupported multiplication")

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, Mat4x4):
            result = Mat4x4()
            for i in range(4):
                for j in range(4):
                    result.rows[i][j] = self.rows[i][j] + other.rows[i][j]
            return result
        raise TypeError("Unsupported addition")

    def __sub__(self, other):
        if isinstance(other, Mat4x4):
            result = Mat4x4()
            for i in range(4):
                for j in range(4):
                    result.rows[i][j] = self.rows[i][j] - other.rows[i][j]
            return result
        raise TypeError("Unsupported subtraction")

    def __eq__(self, other):
        if not isinstance(other, Mat4x4):
            return False
        return all(
            abs(self.rows[i][j] - other.rows[i][j]) < 1e-9
            for i in range(4) for j in range(4)
        )

    def __repr__(self):
        return f"Mat4x4({self.rows})"

    def transform_point(self, point):
        """Apply the matrix to a point (w=1), performing perspective divide."""
        px, py, pz = point
        r = self.rows
        x = px * r[0][0] + py * r[0][1] + pz * r[0][2] + r[0][3]
        y = px * r[1][0] + py * r[1][1] + pz * r[1][2] + r[1][3]
        z = px * r[2][0] + py * r[2][1] + pz * r[2][2] + r[2][3]
        w = px * r[3][0] + py * r[3][1] + pz * r[3][2] + r[3][3]
        if w != 0:
            return Vec3(x / w, y / w, z / w)
        return Vec3(x, y, z)

    def transform_vector(self, vector):
        """Apply the matrix to a direction vector (w=0), ignoring translation."""
        vx, vy, vz = vector
        r = self.rows
        x = vx * r[0][0] + vy * r[0][1] + vz * r[0][2]
        y = vx * r[1][0] + vy * r[1][1] + vz * r[1][2]
        z = vx * r[2][0] + vy * r[2][1] + vz * r[2][2]
        return Vec3(x, y, z)

    def transform_normal(self, normal):
        """Transform a surface normal using the inverse-transpose of this matrix."""
        inv_T = self.inverse().transpose()
        return inv_T.transform_vector(normal)

    def transpose(self):
        result = Mat4x4()
        for i in range(4):
            for j in range(4):
                result.rows[i][j] = self.rows[j][i]
        return result

    def inverse(self):
        """Invert the matrix using Gauss-Jordan elimination with partial pivoting."""
        # Build 4x8 augmented matrix [M | I]
        aug = [
            self.rows[i][:] + [1 if i == j else 0 for j in range(4)]
            for i in range(4)
        ]

        for col in range(4):
            # 1. Find pivot — row with the largest absolute value in this column.
            pivot = max(range(col, 4), key=lambda r: abs(aug[r][col]))

            # 2. Swap pivot row to the diagonal position.
            aug[col], aug[pivot] = aug[pivot], aug[col]

            # 3. Singular check.
            if abs(aug[col][col]) < 1e-10:
                raise ValueError("Matrix is not invertible")

            # 4. Scale pivot row so the diagonal becomes 1.
            scale = aug[col][col]
            aug[col] = [v / scale for v in aug[col]]

            # 5. Eliminate this column from every other row.
            for row in range(4):
                if row == col:
                    continue
                factor = aug[row][col]
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(8)]

        # Extract the right half — that's the inverse.
        return Mat4x4([aug[i][4:] for i in range(4)])

    @classmethod
    def translation(cls, tx, ty, tz):
        return cls([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0,  1],
        ])

    @classmethod
    def rotation_x(cls, angle):
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        return cls([
            [1,  0, 0, 0],
            [0,  c, -s, 0],
            [0,  s,  c, 0],
            [0,  0,  0, 1],
        ])

    @classmethod
    def rotation_y(cls, angle):
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        return cls([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1],
        ])

    @classmethod
    def rotation_z(cls, angle):
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        return cls([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])

    @classmethod
    def scale(cls, sx, sy, sz):
        return cls([
            [sx,  0,  0, 0],
            [ 0, sy,  0, 0],
            [ 0,  0, sz, 0],
            [ 0,  0,  0, 1],
        ])

    @classmethod
    def shear(cls, shx, shy, shz):
        return cls([
            [1, shx, shy, 0],
            [0,   1, shz, 0],
            [0,   0,   1, 0],
            [0,   0,   0, 1],
        ])

    @classmethod
    def look_at(cls, eye, target, up):
        """Build a view matrix that orients the camera from eye toward target."""
        forward = (target - eye).normalize()
        right = forward.cross(up).normalize()
        true_up = right.cross(forward)
        return cls([
            [ right.x,   right.y,   right.z,  -right.dot(eye)],
            [true_up.x, true_up.y, true_up.z, -true_up.dot(eye)],
            [-forward.x, -forward.y, -forward.z, forward.dot(eye)],
            [0,          0,          0,          1],
        ])
