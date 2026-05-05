#!/usr/bin/env python3
"""Recenter an OBJ mesh so its coordinate-frame origin lies at the geometric center.

Only `v ...` lines (vertex positions) are rewritten; `vt` (texture coords),
`vn` (normals), `f` (faces), `mtllib`, `usemtl`, `g`, `o`, comments, and any
other directives are copied verbatim. Textures and material references therefore
stay intact.

Usage:
    python recenter_mesh.py INPUT.obj [OUTPUT.obj] [--mode bbox|centroid]

If OUTPUT is omitted, writes <input_stem>_centered.obj next to the input.

--mode bbox      (default) shift by AABB center  -> 包围盒中心置于原点
--mode centroid  shift by mean of vertex positions -> 顶点平均位置置于原点

Example:
    python recenter_mesh.py \\
        /home/l/BundleSDF/my_data/20260504_220811_erode_mask3/results/textured_mesh_calibrated_59mm.obj
"""

import argparse
from pathlib import Path


def _is_vertex_line(line: str) -> bool:
    # `v ` excludes `vt`/`vn`/`vp` automatically.
    return line.startswith("v ")


def read_vertices(path: Path):
    verts = []
    with open(path) as f:
        for line in f:
            if _is_vertex_line(line):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return verts


def compute_center(verts, mode: str):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    if mode == "bbox":
        return (
            (min(xs) + max(xs)) / 2.0,
            (min(ys) + max(ys)) / 2.0,
            (min(zs) + max(zs)) / 2.0,
        )
    if mode == "centroid":
        n = len(verts)
        return (sum(xs) / n, sum(ys) / n, sum(zs) / n)
    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="input .obj file")
    ap.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="output .obj file (default: <input_stem>_centered.obj)",
    )
    ap.add_argument(
        "--mode",
        choices=["bbox", "centroid"],
        default="bbox",
        help="center definition (default: bbox = AABB center)",
    )
    args = ap.parse_args()

    src = args.input.expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"input not found: {src}")

    dst = (
        args.output.expanduser().resolve()
        if args.output
        else src.with_name(f"{src.stem}_centered{src.suffix}")
    )

    verts = read_vertices(src)
    if not verts:
        raise SystemExit(f"no vertices found in {src}")

    cx, cy, cz = compute_center(verts, args.mode)

    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    bbox_min = (min(xs), min(ys), min(zs))
    bbox_max = (max(xs), max(ys), max(zs))

    print(f"input    : {src}")
    print(f"vertices : {len(verts)}")
    print(f"old bbox : min={bbox_min}, max={bbox_max}")
    print(f"old size : ({bbox_max[0]-bbox_min[0]:.6f}, "
          f"{bbox_max[1]-bbox_min[1]:.6f}, {bbox_max[2]-bbox_min[2]:.6f})")
    print(f"mode     : {args.mode}")
    print(f"center   : ({cx:+.6f}, {cy:+.6f}, {cz:+.6f})")
    print(f"shift    : ({-cx:+.6f}, {-cy:+.6f}, {-cz:+.6f})")

    with open(src) as fin, open(dst, "w") as fout:
        fout.write(
            f"# recentered ({args.mode}); shifted by "
            f"({-cx:+.6f}, {-cy:+.6f}, {-cz:+.6f})\n"
        )
        for line in fin:
            if _is_vertex_line(line):
                parts = line.split()
                x = float(parts[1]) - cx
                y = float(parts[2]) - cy
                z = float(parts[3]) - cz
                tail = " ".join(parts[4:])  # vertex color or w-coord, if any
                if tail:
                    fout.write(f"v {x:.6f} {y:.6f} {z:.6f} {tail}\n")
                else:
                    fout.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            else:
                fout.write(line)

    # sanity check
    new_verts = read_vertices(dst)
    nxs = [v[0] for v in new_verts]; nys = [v[1] for v in new_verts]; nzs = [v[2] for v in new_verts]
    new_bbox_min = (min(nxs), min(nys), min(nzs))
    new_bbox_max = (max(nxs), max(nys), max(nzs))
    new_center = (
        (new_bbox_min[0] + new_bbox_max[0]) / 2.0,
        (new_bbox_min[1] + new_bbox_max[1]) / 2.0,
        (new_bbox_min[2] + new_bbox_max[2]) / 2.0,
    )
    print(f"output   : {dst}")
    print(f"new bbox : min={new_bbox_min}, max={new_bbox_max}")
    print(f"new bbox center: ({new_center[0]:+.6f}, "
          f"{new_center[1]:+.6f}, {new_center[2]:+.6f})  (应≈0,0,0 当 mode=bbox)")


if __name__ == "__main__":
    main()
