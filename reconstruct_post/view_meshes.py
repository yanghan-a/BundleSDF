#!/usr/bin/env python3
"""View mesh_cleaned.obj and textured_mesh.obj side-by-side in MuJoCo.

Renders both the world coordinate frame and each object's own body frame
(red=X, green=Y, blue=Z).
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "my_data" / "20260504_220811_erode_mask3" / "results"
DEFAULT_TEXTURED_NAME = "textured_mesh_calibrated_59mm_centered.obj"
DEFAULT_CALIBRATED_NAME = "textured_mesh_calibrated_59mm.obj"
DEFAULT_TEXTURE_NAME = "material_0.png"

DEFAULT_RESULTS_DIR2 = SCRIPT_DIR / "my_data" / "20260504_220811_erode_mask3_trunc2" / "results"
DEFAULT_TEXTURED_NAME2 = "textured_mesh.obj"
DEFAULT_TEXTURE_NAME2 = "material_0.png"


def make_axes_xml(prefix: str, length: float, radius: float) -> str:
    """Return XML snippet for an RGB triad at the local origin."""
    return f"""
      <geom name="{prefix}_axis_x" type="capsule" size="{radius}"
            fromto="0 0 0 {length} 0 0" rgba="1 0 0 1"
            contype="0" conaffinity="0"/>
      <geom name="{prefix}_axis_y" type="capsule" size="{radius}"
            fromto="0 0 0 0 {length} 0" rgba="0 1 0 1"
            contype="0" conaffinity="0"/>
      <geom name="{prefix}_axis_z" type="capsule" size="{radius}"
            fromto="0 0 0 0 0 {length}" rgba="0 0 1 1"
            contype="0" conaffinity="0"/>
    """


def build_xml(
    results_dir: Path,
    textured_obj: Path,
    calibrated_obj: Path,
    texture_png: Path,
    axis_length: float,
    axis_radius: float,
) -> str:
    world_axes = make_axes_xml("world", axis_length, axis_radius)
    # Object axes are slightly shorter so they remain distinguishable from world axes.
    body_axes_textured = make_axes_xml("textured", axis_length * 0.7, axis_radius)
    body_axes_calibrated = make_axes_xml("calibrated", axis_length * 0.7, axis_radius)

    return f"""
<mujoco model="mesh_viewer">
  <compiler angle="radian" meshdir="{results_dir}" texturedir="{results_dir}"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance=".2"/>

    <mesh name="textured_mesh" file="{textured_obj.name}"/>
    <mesh name="calibrated_mesh" file="{calibrated_obj.name}"/>

    <texture name="tex0" type="2d" file="{texture_png.name}"/>
    <material name="textured" texture="tex0" specular="0.1" shininess="0.3"/>
    <material name="cleaned"  rgba="0.7 0.75 0.85 1" specular="0.3" shininess="0.5"/>
  </asset>

  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="2 2 2" dir="-1 -1 -1" diffuse="0.5 0.5 0.5"/>

    <geom name="floor" type="plane" size="2 2 0.05" material="grid" pos="0 0 -0.5"/>

    {world_axes}

    <body name="textured_body" pos="0.25 0 0">
      <geom type="mesh" mesh="textured_mesh" material="textured"/>
      {body_axes_textured}
    </body>
    <body name="calibrated_body" pos="-0.25 0 0">
      <geom type="mesh" mesh="calibrated_mesh" material="textured"/>
      {body_axes_calibrated}
    </body>
  </worldbody>
</mujoco>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View reconstructed meshes in MuJoCo with world and object coordinate frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the mesh and texture files.",
    )
    parser.add_argument(
        "--textured-mesh",
        type=Path,
        default=None,
        help=f"Path to textured mesh .obj (default: <results_dir>/{DEFAULT_TEXTURED_NAME}).",
    )
    parser.add_argument(
        "--calibrated-mesh",
        type=Path,
        default=None,
        help=f"Path to calibrated mesh .obj (default: <results_dir>/{DEFAULT_CALIBRATED_NAME}).",
    )
    parser.add_argument(
        "--texture",
        type=Path,
        default=None,
        help=f"Path to texture image (default: <results_dir>/{DEFAULT_TEXTURE_NAME}).",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.1,
        help="Length of the world coordinate axes in meters.",
    )
    parser.add_argument(
        "--axis-radius",
        type=float,
        default=0.003,
        help="Radius of the coordinate axis capsules in meters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir: Path = args.results_dir.resolve()
    textured_obj = (args.textured_mesh or results_dir / DEFAULT_TEXTURED_NAME).resolve()
    calibrated_obj = (args.calibrated_mesh or results_dir / DEFAULT_CALIBRATED_NAME).resolve()
    texture_png = (args.texture or results_dir / DEFAULT_TEXTURE_NAME).resolve()

    for p in (textured_obj, calibrated_obj, texture_png):
        if not p.exists():
            raise FileNotFoundError(p)

    xml = build_xml(
        results_dir=results_dir,
        textured_obj=textured_obj,
        calibrated_obj=calibrated_obj,
        texture_png=texture_png,
        axis_length=args.axis_length,
        axis_radius=args.axis_radius,
    )

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
