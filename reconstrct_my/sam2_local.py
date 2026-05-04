"""
Local SAM2 video segmentation for the D405 -> BundleSDF workflow.

Reads rgb/*.png, opens an OpenCV window on frame 0 for interactive prompting
(left-click = positive, right-click = negative, 'r' = reset, space/Enter =
confirm, q/ESC = abort), then runs SAM2 video predictor to propagate the
mask through all frames and writes masks/{idx:06d}.png matching rgb names.

Directory layout (matches record_d405.py / extract_masks_from_sam2.py):
    --out_dir / {YYYYMMDD_HHMMSS} / rgb/      <- created by record_d405.py
                                  / depth/
                                  / cam_K.txt
                                  / video.mp4
                                  / masks/    <- this script writes here

By default, the most recent timestamped run under --out_dir is used; pass
--run <name> to target a specific one.

Run inside the conda env created by setup_sam2.sh:
    conda activate sam2
    python sam2_local.py                    # auto-pick latest run
    python sam2_local.py --run 20260503_175423
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


def find_latest_run(base_dir: Path):
    """Return the most recent YYYYMMDD_HHMMSS subdir name inside base_dir, or None."""
    if not base_dir.is_dir():
        return None
    runs = [d.name for d in base_dir.iterdir()
            if d.is_dir() and TIMESTAMP_RE.match(d.name)]
    if not runs:
        return None
    runs.sort()  # lexical sort works because timestamp is fixed-width
    return runs[-1]


def parse_args():
    p = argparse.ArgumentParser()
    here = os.path.dirname(os.path.realpath(__file__))
    p.add_argument("--out_dir", default="/home/l/BundleSDF/my_data",
                   help="Base directory containing one timestamped subdir per recording "
                        "(matches record_d405.py)")
    p.add_argument("--run", default=None,
                   help="Specific timestamped subdir under --out_dir (e.g. 20260503_153800). "
                        "Default = most recent.")
    p.add_argument("--rgb_subdir", default="rgb")
    p.add_argument("--mask_subdir", default="masks")
    p.add_argument("--ckpt", default=os.path.join(here, "checkpoints", "sam2.1_hiera_base_plus.pt"),
                   help="SAM2 checkpoint (the weight stays with the scripts, not the data)")
    p.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml",
                   help="Hydra config relative to sam2 package")
    p.add_argument("--no_offload", action="store_true",
                   help="Disable CPU offload (faster but uses more GPU memory)")
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth rgb frame to limit RAM/GPU usage. "
                        "Masks are written only for the kept frames. You MUST pass "
                        "the same --stride to run_custom.py for the BundleSDF run.")
    p.add_argument("--vis", action="store_true",
                   help="After propagation, show overlay preview frame-by-frame")
    return p.parse_args()


# ----------------------------- Interactive prompt -----------------------------

class FirstFramePrompter:
    """Collect positive/negative clicks on frame 0 with live SAM2 preview."""

    def __init__(self, predictor, state, frame_bgr, obj_id=1):
        self.predictor = predictor
        self.state = state
        self.frame = frame_bgr.copy()
        self.obj_id = obj_id
        self.points = []   # list[(x, y)]
        self.labels = []   # list[int] 1=fg, 0=bg
        self.current_mask = None
        self.win = "SAM2 prompt | L=positive  R=negative  r=reset  space=confirm  q=abort"

    def _refresh(self):
        if self.points:
            try:
                _, _, mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.state,
                    frame_idx=0,
                    obj_id=self.obj_id,
                    points=np.array(self.points, dtype=np.float32),
                    labels=np.array(self.labels, dtype=np.int32),
                )
                # mask_logits: (num_obj, 1, H, W)
                m = (mask_logits[0, 0].detach().cpu().numpy() > 0).astype(np.uint8)
                self.current_mask = m
            except Exception as e:
                print(f"[warn] SAM2 prompt failed: {e}")
                self.current_mask = None
        else:
            self.predictor.reset_state(self.state)
            self.current_mask = None

    def _render(self):
        canvas = self.frame.copy()
        if self.current_mask is not None and self.current_mask.any():
            color = np.zeros_like(canvas)
            color[..., 1] = 255   # green
            alpha = (self.current_mask.astype(np.float32) * 0.45)[..., None]
            canvas = (canvas * (1 - alpha) + color * alpha).astype(np.uint8)
        for (x, y), lab in zip(self.points, self.labels):
            c = (0, 255, 0) if lab == 1 else (0, 0, 255)
            cv2.circle(canvas, (int(x), int(y)), 6, c, -1)
            cv2.circle(canvas, (int(x), int(y)), 7, (255, 255, 255), 1)
        n_pos = sum(1 for l in self.labels if l == 1)
        n_neg = sum(1 for l in self.labels if l == 0)
        cv2.putText(canvas, f"+{n_pos}  -{n_neg}   space=confirm  r=reset  q=abort",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return canvas

    def _on_mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.labels.append(1)
            self._refresh()
        elif ev == cv2.EVENT_RBUTTONDOWN:
            self.points.append((x, y))
            self.labels.append(0)
            self._refresh()

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self._on_mouse)
        print("\nClick the object (left=positive, right=negative). Space/Enter to confirm, q to abort.\n")
        while True:
            cv2.imshow(self.win, self._render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord(' '), 13):     # space or enter
                if not self.points:
                    print("[warn] add at least one point first")
                    continue
                cv2.destroyWindow(self.win)
                return True
            if key in (ord('q'), 27):     # q or ESC
                cv2.destroyWindow(self.win)
                return False
            if key == ord('r'):
                self.points.clear()
                self.labels.clear()
                self._refresh()


# ----------------------------- Main pipeline -----------------------------

def png_dir_to_jpg_dir(rgb_dir: Path, jpg_dir: Path, stride: int = 1):
    """SAM2 init_state requires a directory of .jpg files. Convert PNGs (subset by stride).

    Returns the list of kept original PNG paths (length = ceil(N / stride)).
    The output JPGs are renumbered 0.jpg, 1.jpg, ... to satisfy SAM2's int-sortable
    name requirement; the caller is responsible for mapping output masks back to
    the original PNG names.
    """
    jpg_dir.mkdir(parents=True, exist_ok=True)
    all_rgb = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() == ".png")
    if not all_rgb:
        raise FileNotFoundError(f"No PNGs in {rgb_dir}")
    rgb_files = all_rgb[::stride]
    print(f"[prep] converting {len(rgb_files)}/{len(all_rgb)} PNG -> JPG  (stride={stride})")
    for i, src in enumerate(tqdm(rgb_files)):
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        cv2.imwrite(str(jpg_dir / f"{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return rgb_files


def main():
    args = parse_args()
    base_dir = Path(args.out_dir).resolve()

    if args.run is not None:
        run_name = args.run
    else:
        run_name = find_latest_run(base_dir)
        if run_name is None:
            sys.exit(f"[err] no timestamped runs found under {base_dir}\n"
                     f"      run record_d405.py first, or pass --run/--out_dir explicitly")
        print(f"[info] auto-selected latest run: {run_name}")

    run_dir = base_dir / run_name
    if not run_dir.is_dir():
        sys.exit(f"[err] run dir does not exist: {run_dir}")

    rgb_dir = run_dir / args.rgb_subdir
    mask_dir = run_dir / args.mask_subdir

    if not rgb_dir.is_dir():
        sys.exit(f"[err] {rgb_dir} not found\n"
                 f"      did record_d405.py finish successfully for this run?")
    if not Path(args.ckpt).is_file():
        sys.exit(f"[err] checkpoint not found: {args.ckpt}\n      run setup_sam2.sh first")

    mask_dir.mkdir(exist_ok=True)

    # ---- Build predictor ----
    print("[init] loading SAM2")
    from sam2.build_sam import build_sam2_video_predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[warn] CUDA not available, running on CPU (very slow)")
    predictor = build_sam2_video_predictor(args.model_cfg, args.ckpt, device=device)

    # ---- Convert PNG->JPG into a temp dir for SAM2 (subsample by --stride) ----
    with tempfile.TemporaryDirectory(prefix="sam2_jpg_") as tmpdir:
        jpg_dir = Path(tmpdir)
        rgb_files = png_dir_to_jpg_dir(rgb_dir, jpg_dir, stride=args.stride)
        n = len(rgb_files)

        print("[init] init_state (this loads all frames)")
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            state = predictor.init_state(
                video_path=str(jpg_dir),
                offload_video_to_cpu=not args.no_offload,
                offload_state_to_cpu=not args.no_offload,
            )

            # ---- Interactive prompting on frame 0 ----
            frame0 = cv2.imread(str(rgb_files[0]))
            prompter = FirstFramePrompter(predictor, state, frame0, obj_id=1)
            ok = prompter.run()
            if not ok:
                print("[abort] user aborted")
                return

            # ---- Propagate ----
            print(f"[run] propagating across {n} frames")
            results = {}
            for frame_idx, obj_ids, mask_logits in tqdm(
                predictor.propagate_in_video(state), total=n
            ):
                # Single object (obj_id=1), index 0
                m = (mask_logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
                results[frame_idx] = m

        # ---- Save masks with original PNG filenames ----
        print(f"[save] writing {len(results)} masks to {mask_dir}")
        empty = 0
        for i, src in enumerate(rgb_files):
            m = results.get(i)
            if m is None:
                m = np.zeros((frame0.shape[0], frame0.shape[1]), dtype=np.uint8)
                empty += 1
            cv2.imwrite(str(mask_dir / src.name), m)
        if empty:
            print(f"[warn] {empty} frames had no SAM2 output (saved as black mask)")

        # ---- Optional viz ----
        if args.vis:
            print("[vis] press any key to advance, q to quit")
            for src in rgb_files:
                rgb = cv2.imread(str(src))
                m = cv2.imread(str(mask_dir / src.name), cv2.IMREAD_GRAYSCALE)
                overlay = rgb.copy()
                overlay[m > 0] = (0.4 * overlay[m > 0] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
                cv2.imshow("vis", np.hstack([rgb, overlay]))
                if (cv2.waitKey(0) & 0xFF) == ord('q'):
                    break
            cv2.destroyAllWindows()

    print("\n[done] masks saved. Next:")
    print("  conda deactivate")
    print(f"  cd /home/l/BundleSDF")
    stride_arg = f" --stride {args.stride}" if args.stride > 1 else ""
    print(f"  python3 run_custom.py --mode run_video --video_dir {run_dir} \\")
    print(f"      --out_folder {run_dir}/out --use_segmenter 0 --use_gui 0 --debug_level 2"
          f"{stride_arg}")
    if args.stride > 1:
        print(f"  # IMPORTANT: --stride {args.stride} must match what was used here, "
              "otherwise BundleSDF will fail on frames without masks.")


if __name__ == "__main__":
    main()
