import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from core import Color, Mat4x4, Point3, Vec3
from scene.camera import Camera
from scene.materials import Diffuse, Dielectric, Emissive, Glossy, Metal
from scene.mesh import IndexedMesh
from scene.primitives import Cube, Plane, Sphere
from scene.scene_object import SceneObject
from scene.shape import Shape
from scene.textures import ImageTexture
from scene.world import World


@dataclass
class RXAAttribute:
    type_name: str | None
    value: object


@dataclass
class RXANode:
    type_name: str
    name: str
    parent: str | None = None
    attrs: dict[str, RXAAttribute] = field(default_factory=dict)

    def attr(self, name, default=None):
        value = self.attrs.get(name)
        return default if value is None else value.value

    def attr_type(self, name):
        value = self.attrs.get(name)
        return None if value is None else value.type_name


@dataclass
class RXAConnection:
    source: str
    destination: str


@dataclass
class RXAScene:
    nodes: dict[str, RXANode] = field(default_factory=dict)
    connections: list[RXAConnection] = field(default_factory=list)

    def create_node(self, type_name, name, parent=None):
        if name in self.nodes:
            raise ValueError(f"Duplicate RXA node: {name}")
        node = RXANode(type_name=type_name, name=name, parent=parent)
        self.nodes[name] = node
        return node

    def set_attr(self, path, value, type_name=None):
        node_name, attr_name = split_attr_path(path)
        if node_name not in self.nodes:
            raise ValueError(f"setAttr references unknown RXA node: {node_name}")
        self.nodes[node_name].attrs[attr_name] = RXAAttribute(type_name, value)

    def connect_attr(self, source, destination):
        self.connections.append(RXAConnection(source, destination))


def load_rxa(path):
    path = Path(path)
    with open(path, "r") as f:
        return rxa_scene_to_world(parse_rxa(f.read()), base_dir=path.parent)


def save_rxa(world, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(dumps_rxa(world_to_rxa_scene(world)))


def rxa_scene_for_obj(path, name=None, indexed=True):
    path = Path(path)
    scene = RXAScene()
    root_name = name or path.stem
    root = scene.create_node("transform", root_name)
    root.attrs["matrix"] = RXAAttribute("matrix", Mat4x4.identity().rows)
    root.attrs["visible"] = RXAAttribute("bool", True)
    root.attrs["renderable"] = RXAAttribute("bool", True)
    root.attrs["selectable"] = RXAAttribute("bool", True)

    shape = scene.create_node("meshShape", f"{root_name}Shape", parent=root_name)
    shape.attrs["source"] = RXAAttribute("file", str(path))
    shape.attrs["sourceType"] = RXAAttribute("string", "obj")
    shape.attrs["indexed"] = RXAAttribute("bool", bool(indexed))
    return scene


def save_obj_as_roxy(obj_path, rxa_path, rxb_path=None, name=None, indexed=True):
    """Convert an OBJ into paired RXA/RXB files.

    RXA stores the scene graph, transforms, shape nodes, and shader connections.
    RXB stores the heavy mesh arrays referenced by each meshShape.geometry attr.
    """
    from scene.io.obj_reader import OBJReader
    from scene.io.roxy_binary import save_rxb_meshes

    if not indexed:
        raise ValueError("save_obj_as_roxy currently requires indexed=True for RXB mesh payloads")

    obj_root = OBJReader.load(obj_path, name=name, indexed=indexed)
    world = World(use_sky=True)
    world.add_object(obj_root)

    rxa_path = Path(rxa_path)
    rxb_path = Path(rxb_path) if rxb_path is not None else rxa_path.with_suffix(".rxb")
    rxb_ref_path = str(Path(_relative_path(rxb_path, rxa_path.parent)))
    rxb_meshes = {}

    scene = world_to_rxa_scene(
        world,
        rxb_ref_path=rxb_ref_path,
        rxb_meshes=rxb_meshes,
    )
    if not rxb_meshes:
        raise ValueError("OBJ export produced no IndexedMesh payloads for RXB")
    save_rxb_meshes(rxb_path, rxb_meshes)
    rxa_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rxa_path, "w") as f:
        f.write(dumps_rxa(scene))
    return scene


def parse_rxa(text):
    scene = RXAScene()
    for statement in _split_statements(text):
        tokens = _tokenize(statement)
        if not tokens:
            continue

        command = tokens[0]
        if command == "createNode":
            _parse_create_node(scene, tokens)
        elif command == "setAttr":
            _parse_set_attr(scene, tokens)
        elif command == "connectAttr":
            _parse_connect_attr(scene, tokens)
        elif command == "parent":
            _parse_parent(scene, tokens)
        else:
            raise ValueError(f"Unknown RXA command: {command}")
    return scene


def dumps_rxa(scene):
    lines = ["# Roxy ASCII 1.0"]
    for node in scene.nodes.values():
        line = f"createNode {node.type_name} -n {_quote(node.name)}"
        if node.parent:
            line += f" -p {_quote(node.parent)}"
        lines.append(line + ";")

        for attr_name, attr in node.attrs.items():
            attr_path = f"{node.name}.{attr_name}"
            type_part = f" -type {attr.type_name}" if attr.type_name else ""
            lines.append(f"setAttr {_quote(attr_path)}{type_part} {_format_value(attr.value, attr.type_name)};")

    for connection in scene.connections:
        lines.append(
            f"connectAttr {_quote(connection.source)} {_quote(connection.destination)};"
        )
    return "\n".join(lines) + "\n"


def world_to_rxa_scene(world, rxb_ref_path=None, rxb_meshes=None):
    scene = RXAScene()
    used_names = set()
    material_nodes = {}

    world_node = scene.create_node("roxyWorld", _unique_name("world", used_names))
    world_node.attrs["useSky"] = RXAAttribute("bool", bool(world.use_sky))
    world_node.attrs["backgroundColor"] = RXAAttribute(
        "color3", _vec_to_list(world.background_color)
    )

    def export_material(material, owner_name, group_name):
        if material is None:
            return None
        key = id(material)
        if key in material_nodes:
            return material_nodes[key]

        node_name = _unique_name(f"{owner_name}_{group_name}_shader", used_names)
        node = scene.create_node(_material_node_type(material), node_name)
        node.attrs["albedo"] = RXAAttribute("color3", _vec_to_list(material._albedo))

        if isinstance(material, (Metal, Glossy)):
            node.attrs["roughness"] = RXAAttribute("float", material._roughness)
        elif isinstance(material, Dielectric):
            node.attrs["ior"] = RXAAttribute("float", material._ior)
        elif isinstance(material, Emissive):
            node.attrs["intensity"] = RXAAttribute("float", material._intensity)

        texture = getattr(material, "_albedo_texture", None)
        if isinstance(texture, ImageTexture) and texture.path is not None:
            texture_name = _unique_name(f"{node_name}_albedo_texture", used_names)
            texture_node = scene.create_node("imageTexture", texture_name)
            texture_node.attrs["file"] = RXAAttribute("file", texture.path)
            texture_node.attrs["flipV"] = RXAAttribute("bool", bool(texture.flip_v))
            scene.connect_attr(f"{texture_name}.outColor", f"{node_name}.albedo")

        material_nodes[key] = node_name
        return node_name

    def export_object(obj, parent_name=None):
        node_name = _unique_name(obj.name or "transform", used_names)
        node = scene.create_node("transform", node_name, parent=parent_name)
        node.attrs["matrix"] = RXAAttribute("matrix", _matrix_rows(obj.local_matrix))
        node.attrs["visible"] = RXAAttribute("bool", bool(obj.visible))
        node.attrs["renderable"] = RXAAttribute("bool", bool(obj.renderable))
        node.attrs["selectable"] = RXAAttribute("bool", bool(obj.selectable))

        for index, shape in enumerate(obj.shapes):
            shape_name = _unique_name(shape.name or f"{node_name}Shape{index + 1}", used_names)
            shape_node = scene.create_node(_shape_node_type(shape.geometry), shape_name, parent=node_name)
            _write_geometry_attrs(
                shape_node,
                shape.geometry,
                rxb_ref_path=rxb_ref_path,
                rxb_meshes=rxb_meshes,
                mesh_name=shape_name,
            )
            for group_name, material in shape.material_groups.items():
                shader_name = export_material(material, shape_name, group_name)
                if shader_name:
                    attr = (
                        "surfaceShader"
                        if group_name == "default"
                        else f"materialGroups.{group_name}"
                    )
                    scene.connect_attr(f"{shader_name}.outSurface", f"{shape_name}.{attr}")

        for child in obj.children:
            export_object(child, node_name)

    for obj in world.objects:
        export_object(obj)

    for i, camera in enumerate(world.cameras, 1):
        camera_name = _unique_name(camera.name or f"camera{i}", used_names)
        node = scene.create_node("camera", camera_name)
        node.attrs["position"] = RXAAttribute("vec3", _vec_to_list(camera.position))
        node.attrs["forward"] = RXAAttribute("vec3", _vec_to_list(camera.forward))
        node.attrs["up"] = RXAAttribute("vec3", _vec_to_list(camera.up))
        node.attrs["fov"] = RXAAttribute("float", camera.fov)
        node.attrs["aspectRatio"] = RXAAttribute("float", camera.aspect_ratio)
        node.attrs["active"] = RXAAttribute("bool", camera is world.active_camera)

    return scene


def rxa_scene_to_world(scene, base_dir=None):
    world_node = _first_node_of_type(scene, "roxyWorld")
    world = World(
        background_color=Color(*world_node.attr("backgroundColor", [0, 0, 0])) if world_node else Color(0, 0, 0),
        use_sky=bool(world_node.attr("useSky", True)) if world_node else True,
    )

    objects = {}
    for node in scene.nodes.values():
        if node.type_name != "transform":
            continue
        obj = SceneObject(
            name=node.name,
            matrix=Mat4x4(node.attr("matrix", Mat4x4.identity().rows)),
            visible=bool(node.attr("visible", True)),
            renderable=bool(node.attr("renderable", True)),
            selectable=bool(node.attr("selectable", True)),
        )
        objects[node.name] = obj

    for name, obj in objects.items():
        parent_name = scene.nodes[name].parent
        if parent_name and parent_name in objects:
            objects[parent_name].add_child(obj)
        else:
            world.add_object(obj)

    materials = _build_materials(scene)
    shape_materials = _shape_material_connections(scene, materials)
    rxb_cache = {}
    for node in scene.nodes.values():
        if not _is_shape_node(node):
            continue
        if node.parent not in objects:
            raise ValueError(f"Shape node {node.name} has no transform parent")
        if node.type_name == "meshShape" and node.attr("source") is not None:
            _attach_external_mesh(node, objects[node.parent], base_dir=base_dir)
            continue
        geometry = _geometry_from_node(node, base_dir=base_dir, rxb_cache=rxb_cache)
        objects[node.parent].shapes.append(
            Shape(geometry, shape_materials.get(node.name, {}), name=node.name)
        )

    active_camera = None
    for node in scene.nodes.values():
        if node.type_name != "camera":
            continue
        camera = Camera(
            position=Point3(*node.attr("position", [0, 0, 0])),
            forward=Vec3(*node.attr("forward", [0, 0, -1])),
            up=Vec3(*node.attr("up", [0, 1, 0])),
            fov=node.attr("fov", 90),
            width=node.attr("aspectRatio", 1.0),
            height=1.0,
            name=node.name,
        )
        world.add_camera(camera)
        if node.attr("active", False):
            active_camera = camera
    if active_camera is not None:
        world.active_camera = active_camera

    return world


def split_attr_path(path):
    if "." not in path:
        raise ValueError(f"Attribute path must be node.attr: {path}")
    return path.split(".", 1)


def _parse_create_node(scene, tokens):
    if len(tokens) < 4:
        raise ValueError("createNode requires type and -n name")
    type_name = tokens[1]
    name = None
    parent = None
    i = 2
    while i < len(tokens):
        if tokens[i] == "-n":
            name = tokens[i + 1]
            i += 2
        elif tokens[i] == "-p":
            parent = tokens[i + 1]
            i += 2
        else:
            raise ValueError(f"Unknown createNode flag: {tokens[i]}")
    if not name:
        raise ValueError("createNode requires -n name")
    scene.create_node(type_name, name, parent=parent)


def _parse_set_attr(scene, tokens):
    if len(tokens) < 2:
        raise ValueError("setAttr requires an attribute path")
    path = tokens[1]
    type_name = None
    value_tokens = tokens[2:]
    if value_tokens[:1] == ["-type"]:
        if len(value_tokens) < 2:
            raise ValueError("setAttr -type requires a type name")
        type_name = value_tokens[1]
        value_tokens = value_tokens[2:]
    scene.set_attr(path, _parse_value(value_tokens, type_name), type_name=type_name)


def _parse_connect_attr(scene, tokens):
    if len(tokens) != 3:
        raise ValueError("connectAttr requires source and destination")
    scene.connect_attr(tokens[1], tokens[2])


def _parse_parent(scene, tokens):
    if len(tokens) != 3:
        raise ValueError("parent requires child and parent")
    child, parent = tokens[1], tokens[2]
    if child not in scene.nodes:
        raise ValueError(f"parent references unknown child node: {child}")
    scene.nodes[child].parent = parent


def _split_statements(text):
    statements = []
    current = []
    quote = None
    escaped = False
    for char in text:
        current.append(char)
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == ";":
            statements.append("".join(current[:-1]).strip())
            current = []
    tail = "".join(current).strip()
    if tail and not tail.startswith("#"):
        statements.append(tail)
    return statements


def _tokenize(statement):
    lexer = shlex.shlex(statement, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return list(lexer)


def _parse_value(tokens, type_name=None):
    if type_name == "matrix":
        values = [float(token) for token in tokens]
        if len(values) != 16:
            raise ValueError("matrix attributes require 16 values")
        return [values[i:i + 4] for i in range(0, 16, 4)]
    if type_name in ("vec3", "color3", "float3"):
        if len(tokens) != 3:
            raise ValueError(f"{type_name} attributes require 3 values")
        return [float(token) for token in tokens]
    if type_name == "bool":
        return _parse_bool(tokens[0])
    if type_name == "int":
        return int(tokens[0])
    if type_name == "float":
        return float(tokens[0])
    if type_name in ("string", "file", "rxbMesh", "message"):
        return tokens[0] if len(tokens) == 1 else " ".join(tokens)

    parsed = [_parse_untyped_token(token) for token in tokens]
    return parsed[0] if len(parsed) == 1 else parsed


def _parse_untyped_token(token):
    lowered = token.lower()
    if lowered in ("true", "false"):
        return _parse_bool(token)
    try:
        if re.match(r"^-?\d+$", token):
            return int(token)
        return float(token)
    except ValueError:
        return token


def _parse_bool(token):
    lowered = str(token).lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"Invalid bool value: {token}")


def _quote(value):
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_value(value, type_name=None):
    if type_name == "matrix":
        return " ".join(_format_number(v) for row in value for v in row)
    if type_name in ("vec3", "color3", "float3"):
        return " ".join(_format_number(v) for v in value)
    if type_name == "bool" or isinstance(value, bool):
        return "true" if value else "false"
    if type_name in ("string", "file", "rxbMesh", "message") or isinstance(value, str):
        return _quote(value)
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, (list, tuple)):
        return " ".join(_format_value(v) for v in value)
    return _quote(value)


def _format_number(value):
    return f"{float(value):.9g}"


def _unique_name(base, used):
    clean = re.sub(r"[^A-Za-z0-9_:]+", "_", base or "node").strip("_") or "node"
    if clean[0].isdigit():
        clean = f"n_{clean}"
    name = clean
    i = 1
    while name in used:
        i += 1
        name = f"{clean}{i}"
    used.add(name)
    return name


def _matrix_rows(matrix):
    return [[float(v) for v in row] for row in matrix.rows]


def _vec_to_list(value):
    return [float(value[0]), float(value[1]), float(value[2])]


def _shape_node_type(geometry):
    if isinstance(geometry, Sphere):
        return "sphereShape"
    if isinstance(geometry, Plane):
        return "planeShape"
    if isinstance(geometry, Cube):
        return "cubeShape"
    return "meshShape"


def _write_geometry_attrs(node, geometry, rxb_ref_path=None, rxb_meshes=None, mesh_name=None):
    if isinstance(geometry, Sphere):
        node.attrs["radius"] = RXAAttribute("float", geometry._radius)
    elif isinstance(geometry, Plane):
        node.attrs["normal"] = RXAAttribute("vec3", _vec_to_list(geometry._normal))
        node.attrs["distance"] = RXAAttribute("float", geometry._distance)
    elif isinstance(geometry, Cube):
        node.attrs["sideLength"] = RXAAttribute("float", geometry._side_length)
    elif isinstance(geometry, IndexedMesh) and rxb_ref_path is not None and rxb_meshes is not None:
        payload_name = mesh_name or geometry._name or node.name
        rxb_meshes[payload_name] = geometry
        node.attrs["geometry"] = RXAAttribute("rxbMesh", f"{rxb_ref_path}:meshes/{payload_name}")
        node.attrs["geometryStorage"] = RXAAttribute("string", "rxb")
        node.attrs["geometryName"] = RXAAttribute("string", payload_name)
    else:
        node.attrs["geometryStorage"] = RXAAttribute("string", "external-or-rxb-required")
        node.attrs["geometryType"] = RXAAttribute("string", geometry.to_dict().get("type", type(geometry).__name__))


def _is_shape_node(node):
    return node.type_name.endswith("Shape")


def _geometry_from_node(node, base_dir=None, rxb_cache=None):
    if node.type_name == "sphereShape":
        return Sphere(node.attr("radius", 1.0))
    if node.type_name == "planeShape":
        return Plane(Vec3(*node.attr("normal", [0, 1, 0])), node.attr("distance", 0.0))
    if node.type_name == "cubeShape":
        return Cube(node.attr("sideLength", 1.0))
    if node.type_name == "meshShape":
        geometry = node.attr("geometry")
        if geometry and (node.attr_type("geometry") == "rxbMesh" or ".rxb:" in geometry):
            from scene.io.roxy_binary import load_rxb_mesh_ref
            return load_rxb_mesh_ref(
                geometry,
                base_dir=base_dir,
                build_bvh=bool(node.attr("buildBvh", False)),
                cache=rxb_cache,
            )
        raise NotImplementedError("meshShape import needs external OBJ or RXB geometry payload support")
    raise ValueError(f"Unknown shape node type: {node.type_name}")


def _attach_external_mesh(node, target, base_dir=None):
    source_type = node.attr("sourceType", "obj")
    if source_type != "obj":
        raise NotImplementedError(f"Unsupported meshShape sourceType: {source_type}")

    source = Path(node.attr("source"))
    if not source.is_absolute() and base_dir is not None:
        source = Path(base_dir) / source

    from scene.io.obj_reader import OBJReader

    imported = OBJReader.load(
        source,
        name=node.name,
        indexed=bool(node.attr("indexed", True)),
    )
    for shape in imported.shapes:
        target.shapes.append(shape)
    for child in imported.children:
        target.add_child(child)


def _material_node_type(material):
    if isinstance(material, Diffuse):
        return "diffuse"
    if isinstance(material, Metal):
        return "metal"
    if isinstance(material, Dielectric):
        return "dielectric"
    if isinstance(material, Emissive):
        return "emissive"
    if isinstance(material, Glossy):
        return "glossy"
    return "shader"


def _build_materials(scene):
    materials = {}
    texture_connections = {
        split_attr_path(conn.destination)[0]: split_attr_path(conn.source)[0]
        for conn in scene.connections
        if split_attr_path(conn.destination)[1] == "albedo"
    }

    for node in scene.nodes.values():
        if node.type_name not in {"diffuse", "metal", "dielectric", "emissive", "glossy"}:
            continue
        albedo = Color(*node.attr("albedo", [1, 1, 1]))
        texture = None
        texture_node_name = texture_connections.get(node.name)
        if texture_node_name:
            texture_node = scene.nodes.get(texture_node_name)
            if texture_node and texture_node.type_name == "imageTexture":
                texture = ImageTexture(texture_node.attr("file"), flip_v=texture_node.attr("flipV", True))

        if node.type_name == "diffuse":
            materials[node.name] = Diffuse(albedo, albedo_texture=texture)
        elif node.type_name == "metal":
            materials[node.name] = Metal(albedo, node.attr("roughness", 0.0), albedo_texture=texture)
        elif node.type_name == "dielectric":
            materials[node.name] = Dielectric(albedo, node.attr("ior", 1.5), albedo_texture=texture)
        elif node.type_name == "emissive":
            materials[node.name] = Emissive(albedo, node.attr("intensity", 1.0), albedo_texture=texture)
        elif node.type_name == "glossy":
            materials[node.name] = Glossy(albedo, node.attr("roughness", 0.5), albedo_texture=texture)
    return materials


def _shape_material_connections(scene, materials):
    assignments = {}
    for connection in scene.connections:
        source_node, source_attr = split_attr_path(connection.source)
        destination_node, destination_attr = split_attr_path(connection.destination)
        if source_attr != "outSurface" or source_node not in materials:
            continue
        if destination_node not in scene.nodes or not _is_shape_node(scene.nodes[destination_node]):
            continue
        if destination_attr == "surfaceShader":
            group = "default"
        elif destination_attr.startswith("materialGroups."):
            group = destination_attr.split(".", 1)[1]
        else:
            continue
        assignments.setdefault(destination_node, {})[group] = materials[source_node]
    return assignments


def _first_node_of_type(scene, type_name):
    for node in scene.nodes.values():
        if node.type_name == type_name:
            return node
    return None


def _relative_path(path, start):
    try:
        return Path(path).resolve().relative_to(Path(start).resolve())
    except ValueError:
        import os
        return os.path.relpath(path, start)
