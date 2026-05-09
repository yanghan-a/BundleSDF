"""
把 BundleSDF 输出的逐帧位姿可视化到原始 RGB 上。
- 用 mesh 的 AABB(避开 trimesh.bounds.oriented_bounds 的 numpy/scipy 兼容问题)
- 同时画 3D bbox 和物体坐标系 XYZ 三轴
- 输出 pose_vis/*.png 和 pose_vis.mp4

位姿来源 (--mode):
  tracking  [默认] 仅用 BundleTrack 输出 ob_in_cam/*.txt
            优点: 帧间连续平滑(相邻帧 <20° rot)
            适用: 对称物体(圆柱/球/正多面体), NeRF refine 在这类物体上会因
                  多解性逐帧抖动 80-120°
  refined   仅用 NeRF refine 后的 keyframe 位姿(非 keyframe 跳过)
            适用: 非对称物体 + 关心绝对精度的单帧
  corrected refined 关键帧 + 非关键帧用 delta 插值
            适用: 非对称物体 + 视频连续性

usage:
  python draw_pose_video.py /path/to/results            # 默认 tracking
  python draw_pose_video.py /path/to/results refined
  python draw_pose_video.py /path/to/results corrected
"""
import os, sys, glob
import numpy as np
import cv2
import trimesh
import imageio


def project(K, ob_in_cam, pts3d):
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
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
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
        for i, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            if valid[i + 1]:
                cv2.arrowedLine(img, tuple(uv[0]), tuple(uv[i + 1]), c, thickness, cv2.LINE_AA, tipLength=0.15)
    return img


def slerp_rotation(R0, R1, t):
    Rrel = R0.T @ R1
    rvec, _ = cv2.Rodrigues(Rrel)
    Rrel_t, _ = cv2.Rodrigues(rvec * t)
    return R0 @ Rrel_t


def interp_pose(P0, P1, t):
    P = np.eye(4)
    P[:3, :3] = slerp_rotation(P0[:3, :3], P1[:3, :3], t)
    P[:3, 3] = (1 - t) * P0[:3, 3] + t * P1[:3, 3]
    return P


def load_tracking_poses(out_folder, color_files):
    out = {}
    for cf in color_files:
        fid = os.path.basename(cf).replace('.png', '')
        pf = f'{out_folder}/ob_in_cam/{fid}.txt'
        if not os.path.exists(pf):
            continue
        P = np.loadtxt(pf)
        if not np.all(np.isfinite(P)):
            continue
        out[fid] = P
    return out


def load_refined_keyframe_poses(out_folder):
    """{id_str: ob_in_cam_4x4}, NeRF refine 后, OpenCV 约定"""
    stamp_dirs = [d for d in sorted(glob.glob(f'{out_folder}/[0-9]*'))
                  if os.path.isdir(d) and os.path.exists(f'{d}/poses_after_nerf.txt')]
    if not stamp_dirs:
        return {}
    last = stamp_dirs[-1]
    poses_file = f'{last}/poses_after_nerf.txt'
    frames_file = f'{last}/nerf_frames.txt'
    if not os.path.exists(frames_file):
        return {}
    cam_in_obs = np.loadtxt(poses_file).reshape(-1, 4, 4)
    with open(frames_file) as f:
        ids = [line.strip() for line in f if line.strip()]
    if len(ids) != len(cam_in_obs):
        print(f'[warn] nerf_frames count {len(ids)} != poses {len(cam_in_obs)}; refined disabled')
        return {}
    print(f'[info] loaded {len(ids)} refined keyframe poses from {last}')
    return {fid: np.linalg.inv(P) for fid, P in zip(ids, cam_in_obs)}


def build_corrected(tracking, refined):
    """非关键帧用前后 keyframe 的 delta 插值修正"""
    deltas = {fid: refined[fid] @ np.linalg.inv(tracking[fid])
              for fid in refined if fid in tracking}
    sorted_kfs = sorted(deltas.keys(), key=int)
    sorted_int = [int(k) for k in sorted_kfs]
    out = {}
    for fid, P_track in tracking.items():
        if fid in refined:
            out[fid] = (refined[fid], 'refined')
            continue
        i = int(fid)
        prev = next_ = None
        for k_id, k_int in zip(sorted_kfs, sorted_int):
            if k_int <= i:
                prev = (k_id, k_int)
            else:
                next_ = (k_id, k_int); break
        if prev is None and next_ is None:
            out[fid] = (P_track, 'tracking')
        elif prev is None:
            out[fid] = (deltas[next_[0]] @ P_track, 'corrected')
        elif next_ is None or prev[1] == next_[1]:
            out[fid] = (deltas[prev[0]] @ P_track, 'corrected')
        else:
            t = (i - prev[1]) / (next_[1] - prev[1])
            d = interp_pose(deltas[prev[0]], deltas[next_[0]], t)
            out[fid] = (d @ P_track, 'corrected')
    return out


def main(out_folder, mode='tracking'):
    out_folder = out_folder.rstrip('/')
    K = np.loadtxt(f'{out_folder}/cam_K.txt').reshape(3, 3)
    color_files = sorted(glob.glob(f'{out_folder}/color/*.png'))
    if not color_files:
        raise SystemExit(f'no color frames at {out_folder}/color/')

    mesh = trimesh.load(f'{out_folder}/textured_mesh.obj', force='mesh')
    bbox_min, bbox_max = mesh.bounds
    axis_len = float(np.linalg.norm(bbox_max - bbox_min)) * 0.4
    print(f'mode={mode}; mesh AABB(mm): {((bbox_max-bbox_min)*1000).round(2)}; axis_len={axis_len*1000:.1f}mm')

    tracking = load_tracking_poses(out_folder, color_files)

    if mode == 'tracking':
        pose_dict = {fid: (P, 'tracking') for fid, P in tracking.items()}
    elif mode == 'refined':
        refined = load_refined_keyframe_poses(out_folder)
        pose_dict = {fid: (P, 'refined') for fid, P in refined.items()}
    elif mode == 'corrected':
        refined = load_refined_keyframe_poses(out_folder)
        pose_dict = build_corrected(tracking, refined) if refined else \
                    {fid: (P, 'tracking') for fid, P in tracking.items()}
    else:
        raise SystemExit(f'unknown mode: {mode}')

    out_dir = f'{out_folder}/pose_vis'
    os.makedirs(out_dir, exist_ok=True)

    sample = imageio.imread(color_files[0])
    H, W = sample.shape[:2]
    video_path = f'{out_folder}/pose_vis.mp4'
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))

    src_color = {'refined': (0, 255, 0), 'corrected': (0, 200, 255), 'tracking': (255, 255, 255)}
    counter = {'refined': 0, 'corrected': 0, 'tracking': 0}
    skipped = 0
    for color_file in color_files:
        id_str = os.path.basename(color_file).replace('.png', '')
        if id_str not in pose_dict:
            skipped += 1
            continue
        pose, src = pose_dict[id_str]
        counter[src] += 1

        img = imageio.imread(color_file).copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = draw_bbox_3d(img, K, pose, bbox_min, bbox_max, color=(255, 255, 0), thickness=2)
        img = draw_axes(img, K, pose, axis_len, thickness=3)
        cv2.putText(img, f'{id_str} [{src}]', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    src_color[src], 2, cv2.LINE_AA)

        imageio.imwrite(f'{out_dir}/{id_str}.png', img)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f'PNG dir : {out_dir}/  (skipped {skipped} frames)')
    print(f'MP4     : {video_path}')
    print(f'pose source counts: {counter}')


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise SystemExit(__doc__)
    out_folder = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) == 3 else 'tracking'
    main(out_folder, mode)
