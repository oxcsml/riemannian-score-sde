import bpy
import bmesh
import math
from functools import partial
import numpy as np
import os
from mathutils import Vector
from mathutils import Euler

# directory = os.getcwd()
base_dir = os.path.expanduser(
    "~/Documents/projects/ExtrinsicGaugeEquivariantVectorGPs/"
)
scripts_dir = os.path.join(base_dir, "blender_scripts")
data_dir = os.path.join(base_dir, "blender")
texture_path = os.path.join(base_dir, "blender", "textures")
col_dir = os.path.join(base_dir, "blender", "col")

os.makedirs(os.path.join(data_dir, "blank_to_wrong", "renders"), exist_ok=True)

with open(os.path.join(scripts_dir, "render.py")) as file:
    exec(file.read())

reset_scene()
set_renderer_settings(num_samples=16 if bpy.app.background else 128)
(cam_axis, cam_obj) = setup_camera(
    distance=10,
    angle=(0, 0, 0),
    lens=85,
    height=1500,
)
setup_lighting(
    shifts=(-10, -10, 10),
    sizes=(9, 18, 15),
    energies=(1500, 150, 1125),
    horizontal_angles=(-np.pi / 6, np.pi / 3, np.pi / 3),
    vertical_angles=(-np.pi / 3, -np.pi / 6, np.pi / 4),
)
set_resolution(1080 / 2, aspect=(16, 9))

bd_obj = create_backdrop(location=(0, 0, -2), scale=(10, 5, 5))
arr_obj = create_vector_arrow(color=(1, 0, 0, 1))
# set_object_collections(backdrop=[bd_obj], instancing=[arr_obj])

bm = import_bmesh(os.path.join(data_dir, "unwrap_sphere", "frame_0.obj"))
# bm = import_bmesh(os.path.join(data_dir, "kernels", "S2.obj"))
# import_color(bm, name='white', color = (1,1,1,1))
import_color(
    bm,
    data_file=os.path.join(data_dir, "kernels", "s_wrong.csv"),
    palette_file=os.path.join(col_dir, "viridis.csv"),
)
earth_obj = add_mesh(bm, name="Earth")
earth_mat = add_vertex_colors(earth_obj)
add_texture(earth_mat, os.path.join(texture_path, "mercator_rot.png"))

# VECTOR FIELD
vf_bm = import_vector_field(os.path.join(data_dir, "unwrap_sphere", f"frame_0.csv"))
vf_obj = add_vector_field(vf_bm, arr_obj, scale=3, name="observations")

arr_obj = create_vector_arrow(color=(0.3, 0.3, 0.3, 1.0))  # (0.0, 0.75, 1.0, 1)
mean_bm = import_vector_field(
    os.path.join(data_dir, "kernels", f"mean_wrong_sphere.csv")
)
mean_obj = add_vector_field(mean_bm, arr_obj, scale=3, name="means")

bpy.ops.object.empty_add(
    type="PLAIN_AXES", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
)
empty = bpy.context.selected_objects[0]
earth_obj.parent = empty
vf_obj.parent = empty
mean_obj.parent = empty

bpy.context.scene.render.filepath = os.path.join(
    data_dir, "kernels", "renders", "wrong_front.png"
)
empty.rotation_euler = Euler((0, 0, math.radians(90)), "XYZ")
bpy.ops.render.render(use_viewport=True, write_still=True)

bpy.context.scene.render.filepath = os.path.join(
    data_dir, "kernels", "renders", "wrong_back.png"
)
empty.rotation_euler = Euler((0, 0, -math.radians(90)), "XYZ")
bpy.ops.render.render(use_viewport=True, write_still=True)

bpy.context.scene.render.filepath = os.path.join(
    data_dir, "kernels", "renders", "wrong_top.png"
)
empty.rotation_euler = Euler((0, -math.radians(90), math.radians(90)), "XYZ")
bpy.ops.render.render(use_viewport=True, write_still=True)

bpy.context.scene.render.filepath = os.path.join(
    data_dir, "kernels", "renders", "wrong_bottom.png"
)
empty.rotation_euler = Euler((0, math.radians(90), math.radians(90)), "XYZ")
bpy.ops.render.render(use_viewport=True, write_still=True)
