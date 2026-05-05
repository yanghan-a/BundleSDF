"""
把 BundleSDF 输出的逐帧位姿可视化到原始 RGB 上。
- 用 mesh 的 AABB(避开 trimesh.bounds.oriented_bounds 的 numpy/scipy 兼容问题)
- 同时画 3D bbox 和物体坐标系 XYZ 三轴
- 输出 pose_vis/*.png 和 pose_vis.mp4

usage:
  python ./reconstruct_post/draw_pose_video.py /home/l/BundleSDF/my_data/20260504_201434/results
"""
import os, sys, glob
import numpy as np
import cv2
import trimesh
import imageio


def project(K, ob_in_cam, pts3d):
    """pts3d: (N,3) in object frame -> (N,2) pixel coords, returns (uv, valid)."""
    pts_h = np.concatenate([pts3d, np.ones((len(pts3d), 1))], axis=1)
    pts_cam = (ob_in_cam @ pts_h.T).T[:, :3]
    valid = pts_cam[:, 2] > 1e-3
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
    return np.round(uv).astype(int), valid


def draw_bbox_3d(img, K, ob_in_cam, bbox_min, bbox_max, color=(255, 255, 0), thickness=2):
    xs = [bbox_min[0], bbox_max[0]]
    ys = [bbox_min[1], bbox_max[1]]
    zs = [bbox_min[2], bbox_max[2]]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])  # 8x3
    uv, valid = project(K, ob_in_cam, corners)
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    for a, b in edges:
        if valid[a] and valid[b]:
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, thickness, cv2.LINE_AA)
    return img


def draw_axes(img, K, ob_in_cam, length, thickness=3):
    origin = np.array([[0, 0, 0]])
    axes = np.array([[length, 0, 0], [0, length, 0], [0, 0, length]])
    pts = np.concatenate([origin, axes], axis=0)
    uv, valid = project(K, ob_in_cam, pts)
    if valid[0]:
        for i, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # X red, Y green, Z blue (RGB)
            if valid[i + 1]:
                cv2.arrowedLine(img, tuple(uv[0]), tuple(uv[i + 1]), c, thickness, cv2.LINE_AA, tipLength=0.15)
    return img


def main(out_folder):
    out_folder = out_folder.rstrip('/')
    K = np.loadtxt(f'{out_folder}/cam_K.txt').reshape(3, 3)
    color_files = sorted(glob.glob(f'{out_folder}/color/*.png'))
    if not color_files:
        raise SystemExit(f'no color frames at {out_folder}/color/')

    mesh = trimesh.load(f'{out_folder}/textured_mesh.obj', force='mesh')
    bbox_min, bbox_max = mesh.bounds            # (2,3) AABB in object frame
    axis_len = float(np.linalg.norm(bbox_max - bbox_min)) * 0.4
    print(f'mesh AABB(mm): {((bbox_max-bbox_min)*1000).round(2)}; axis_len={axis_len*1000:.1f}mm')

    out_dir = f'{out_folder}/pose_vis'
    os.makedirs(out_dir, exist_ok=True)

    sample = imageio.imread(color_files[0])
    H, W = sample.shape[:2]
    video_path = f'{out_folder}/pose_vis.mp4'
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))

    skipped = 0
    for color_file in color_files:
        id_str = os.path.basename(color_file).replace('.png', '')
        pose_file = f'{out_folder}/ob_in_cam/{id_str}.txt'
        if not os.path.exists(pose_file):
            skipped += 1
            continue
        pose = np.loadtxt(pose_file)            # ob_in_cam, 4x4
        if not np.all(np.isfinite(pose)):
            skipped += 1
            continue

        img = imageio.imread(color_file).copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = draw_bbox_3d(img, K, pose, bbox_min, bbox_max, color=(255, 255, 0), thickness=2)
        img = draw_axes(img, K, pose, axis_len, thickness=3)
        cv2.putText(img, id_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        imageio.imwrite(f'{out_dir}/{id_str}.png', img)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f'PNG dir : {out_dir}/  (skipped {skipped} frames without pose)')
    print(f'MP4     : {video_path}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
