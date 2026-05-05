#!/usr/bin/env python3
"""Overlay per-frame masks on a video.

Reads frames from an input video, looks up a matching mask image for each
frame index in a mask directory (e.g. ``000000.png``, ``000001.png`` ...),
tints the masked region and draws its contour, then writes the result to a
new video file.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VIDEO = Path("/home/l/BundleSDF/my_data/20260505_150000_yida/video.mp4")
DEFAULT_MASKS = Path("/home/l/BundleSDF/my_data/20260505_150000_yida/masks")


def parse_color(s: str) -> np.ndarray:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3 or not all(0 <= v <= 255 for v in parts):
        raise argparse.ArgumentTypeError("color must be 'B,G,R' with values in 0-255")
    return np.array(parts, dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay per-frame masks on a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Input video file.",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=DEFAULT_MASKS,
        help="Directory containing per-frame mask PNGs (e.g. 000000.png).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: <video_stem>_overlay.mp4 next to input).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS (default: copy from input video).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Tint blend strength in [0, 1].",
    )
    parser.add_argument(
        "--color",
        type=parse_color,
        default=np.array([0, 0, 255], dtype=np.uint8),
        help="Tint color as 'B,G,R'.",
    )
    parser.add_argument(
        "--contour-color",
        type=parse_color,
        default=np.array([0, 255, 255], dtype=np.uint8),
        help="Contour color as 'B,G,R'.",
    )
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="{idx:06d}.png",
        help="Mask filename pattern; supports {idx} for the 0-based frame index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path: Path = args.video.resolve()
    masks_dir: Path = args.masks.resolve()

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not masks_dir.is_dir():
        raise NotADirectoryError(masks_dir)

    output_path: Path = (
        args.output.resolve()
        if args.output is not None
        else video_path.with_name(f"{video_path.stem}_overlay.mp4")
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = args.fps if args.fps is not None else cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open writer for: {output_path}")

    color = args.color
    contour_color = tuple(int(c) for c in args.contour_color)
    alpha = float(args.alpha)

    idx = 0
    matched = 0
    missing = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask_path = masks_dir / args.mask_pattern.format(idx=idx)
        if mask_path.exists():
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                if mask_img.shape[:2] != (height, width):
                    mask_img = cv2.resize(
                        mask_img, (width, height), interpolation=cv2.INTER_NEAREST
                    )
                mask = mask_img > 0
                if mask.any():
                    tint = np.zeros_like(frame)
                    tint[mask] = color
                    frame[mask] = (
                        frame[mask] * (1 - alpha) + tint[mask] * alpha
                    ).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(frame, contours, -1, contour_color, 1)
                matched += 1
            else:
                missing += 1
        else:
            missing += 1

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    print(
        f"wrote {output_path} ({idx} frames, {matched} with masks, "
        f"{missing} without)"
    )


if __name__ == "__main__":
    main()
