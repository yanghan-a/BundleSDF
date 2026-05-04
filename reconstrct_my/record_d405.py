"""
Record an RGBD sequence from Intel RealSense D405 for BundleSDF + SAM2 workflow.

Outputs (under --out_dir):
    rgb/{idx:06d}.png        BGR 8U color
    depth/{idx:06d}.png      uint16 depth in millimeters (aligned to color)
    cam_K.txt                3x3 color intrinsics (post-alignment)
    video.mp4                color video for SAM2 demo upload (<70MB target)

Controls (in preview window):
    r   start recording
    s   stop recording and quit (saves data)
    q   quit without saving

    # Defaults match FoundationPose track_single.py camera config:
    #   640x480 @ 90fps, color_exposure=5000us, depth_exposure=5000us,
    #   color_gain=80, depth_max=0.6m, auto WB
    python3 ./reconstrct_my/record_d405.py --max_seconds 30 --color_exposure 5000 --depth_exposure 5000 --color_gain 50 --depth_max 0.6
"""
import argparse
import os
import sys
import time
import numpy as np
import cv2
import pyrealsense2 as rs


def make_writer(path, fps, size):
    """Try H.264 (avc1) first, fall back to mp4v."""
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, fps, size)
        if writer.isOpened():
            print(f"[video] codec={fourcc_str}")
            return writer
    raise RuntimeError("No working video codec found in OpenCV")


def colorize_depth(depth_mm, max_mm=500):
    """Colormap for preview only (mm uint16 -> BGR uint8)."""
    d = np.clip(depth_mm.astype(np.float32) / max_mm, 0, 1)
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/home/l/BundleSDF/my_data",
                   help="Base directory; a timestamped subdirectory is created per run "
                        "to hold rgb/ depth/ cam_K.txt video.mp4")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--max_seconds", type=float, default=30.0,
                   help="Hard cap on recording duration (helps stay under 70MB)")
    p.add_argument("--depth_preview_max_mm", type=int, default=500,
                   help="Max depth (mm) used for depth preview colormap; D405 sweet spot ~70-300mm")
    # ---- Camera config (mirrors track_single.py defaults) ----
    p.add_argument("--color_exposure", type=int, default=5000,
                   help="color manual exposure (us); pass 0 / negative to enable auto-exposure")
    p.add_argument("--depth_exposure", type=int, default=5000,
                   help="depth manual exposure (us); pass 0 / negative to enable auto-exposure")
    p.add_argument("--color_gain", type=int, default=80,
                   help="color manual gain; pass negative to skip setting gain")
    p.add_argument("--white_balance", type=int, default=None,
                   help="color manual white-balance (K), e.g. 4000~6500. Default = auto WB")
    p.add_argument("--depth_max", type=float, default=0.6,
                   help="depths beyond this distance (meters) are zeroed before saving. "
                        "D405 sweet spot ~0.6m. Pass 0 / negative to disable clipping.")
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.realpath(args.out_dir)
    run_name = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, run_name)
    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    video_path = os.path.join(out_dir, "video.mp4")
    print(f"[out] saving run to {out_dir}")

    # ---- RealSense pipeline ----
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)

    # Locate color & depth sensors (D405 shares a single sensor for both)
    color_sensor = None
    depth_sensor = None
    for s in profile.get_device().query_sensors():
        for sp in s.get_stream_profiles():
            st = sp.stream_type()
            if st == rs.stream.color and color_sensor is None:
                color_sensor = s
            elif st == rs.stream.depth and depth_sensor is None:
                depth_sensor = s
        if color_sensor is not None and depth_sensor is not None:
            break
    if depth_sensor is None or color_sensor is None:
        raise RuntimeError("color/depth sensor not found on device")
    if color_sensor is depth_sensor:
        print("[d405] color & depth share one sensor (D405-style)")

    depth_scale = depth_sensor.as_depth_sensor().get_depth_scale()  # meters per unit
    print(f"[d405] depth_scale={depth_scale} m/unit")

    # ---- Apply manual exposure / gain / WB (matches track_single.py defaults) ----
    if args.color_exposure is not None and args.color_exposure > 0:
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, float(args.color_exposure))
        print(f"[d405] color: manual exposure {args.color_exposure}us")
    else:
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        print("[d405] color: AE on")
    if args.color_gain is not None and args.color_gain >= 0:
        color_sensor.set_option(rs.option.gain, float(args.color_gain))
        print(f"[d405] color: manual gain {args.color_gain}")
    if args.white_balance is None:
        try:
            color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
        except Exception:
            pass
    else:
        try:
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            color_sensor.set_option(rs.option.white_balance, float(args.white_balance))
            print(f"[d405] color: manual WB {args.white_balance}K")
        except Exception:
            pass
    if args.depth_exposure is not None and args.depth_exposure > 0:
        try:
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            depth_sensor.set_option(rs.option.exposure, float(args.depth_exposure))
            print(f"[d405] depth: manual exposure {args.depth_exposure}us")
        except Exception:
            pass
    else:
        try:
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            print("[d405] depth: AE on")
        except Exception:
            pass

    align = rs.align(rs.stream.color)
    depth_max_mm = int(args.depth_max * 1000) if args.depth_max and args.depth_max > 0 else 0

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    print(f"[d405] color intrinsics:\n{K}")

    # ---- Warm up ----
    for _ in range(10):
        pipeline.wait_for_frames()

    # ---- Preview + recording loop ----
    recording = False
    frame_idx = 0
    start_t = None
    writer = None
    max_frames = int(args.max_seconds * args.fps)
    print("\nPreview ready. Keys: [r]=start, [s]=stop+save, [q]=quit-no-save\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())            # (H,W,3) BGR
            depth_raw = np.asanyarray(depth_frame.get_data())        # (H,W) z16
            depth_mm = (depth_raw.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)
            if depth_max_mm > 0:
                depth_mm[depth_mm > depth_max_mm] = 0

            preview = np.hstack([color, colorize_depth(depth_mm, args.depth_preview_max_mm)])
            status = "REC" if recording else "IDLE"
            cv2.putText(preview, f"{status}  frame={frame_idx}  ({frame_idx/args.fps:.1f}s)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if recording else (0, 255, 0), 2)
            cv2.imshow("D405 (color | depth)  r=rec  s=stop  q=quit", preview)

            if recording:
                cv2.imwrite(os.path.join(rgb_dir, f"{frame_idx:06d}.png"), color)
                cv2.imwrite(os.path.join(depth_dir, f"{frame_idx:06d}.png"), depth_mm)
                writer.write(color)
                frame_idx += 1
                if frame_idx >= max_frames:
                    print(f"[stop] hit max_frames ({max_frames})")
                    break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') and not recording:
                writer = make_writer(video_path, args.fps, (args.width, args.height))
                recording = True
                start_t = time.time()
                print(f"[rec] started, max {args.max_seconds}s = {max_frames} frames")
            elif key == ord('s'):
                print("[stop] user pressed s")
                break
            elif key == ord('q'):
                print("[quit] no data saved")
                if writer is not None:
                    writer.release()
                # Clean up the partial files and the empty timestamped run dir
                for d in (rgb_dir, depth_dir):
                    for f in os.listdir(d):
                        os.remove(os.path.join(d, f))
                if os.path.exists(video_path):
                    os.remove(video_path)
                for d in (rgb_dir, depth_dir, out_dir):
                    if os.path.isdir(d):
                        try:
                            os.rmdir(d)
                        except OSError:
                            pass
                cv2.destroyAllWindows()
                pipeline.stop()
                return
    finally:
        cv2.destroyAllWindows()

    if writer is not None:
        writer.release()
    pipeline.stop()

    if frame_idx == 0:
        print("[warn] no frames recorded")
        return

    # ---- Write intrinsics ----
    np.savetxt(os.path.join(out_dir, "cam_K.txt"), K, fmt="%.6f")

    # ---- Summary ----
    duration = frame_idx / args.fps
    mp4_size = os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0
    print("\n========== Recording summary ==========")
    print(f" frames    : {frame_idx}")
    print(f" duration  : {duration:.2f} s")
    print(f" rgb dir   : {rgb_dir}")
    print(f" depth dir : {depth_dir}")
    print(f" K file    : {os.path.join(out_dir, 'cam_K.txt')}")
    print(f" video.mp4 : {mp4_size:.2f} MB  ({video_path})")
    if mp4_size > 70:
        print(f" [warn] mp4 is >70MB; SAM2 demo may reject. Re-record shorter or lower res.")
    elif mp4_size > 60:
        print(f" [note] mp4 is close to 70MB limit.")
    print("========================================")
    print("\nNext step:")
    print(f"  1) Upload {video_path} to https://sam2.metademolab.com/demo")
    print(f"  2) Annotate frame 0, export 'Object Cutout' video")
    print(f"  3) Save it as: {os.path.join(out_dir, 'sam2_cutout.mp4')}")
    print(f"  4) Run: python extract_masks_from_sam2.py")


if __name__ == "__main__":
    main()
