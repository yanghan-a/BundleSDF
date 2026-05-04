# D405 → SAM2 (本地) → BundleSDF 工作流

完整流程: 用 RealSense **D405** 拍 RGBD 视频 → **本地 SAM2** 推理生成全部帧 mask → 直接喂 **BundleSDF** 重建。

> 之前用过的 SAM2 网页 demo 有 ~10s 时长上限,所以改成本地推理。如果想用网页 demo,`extract_masks_from_sam2.py` 仍然保留。

## 目录约定

```
reconstrct_my/
├── setup_sam2.sh               # ① 一次性: 创建 conda 环境 + 装 sam2 + 下权重
├── record_d405.py              # ② D405 录制
├── sam2_local.py               # ③ 交互式标注 + 全帧推理 (本地 SAM2)
├── extract_masks_from_sam2.py  # (备选) 从 SAM2 网页 cutout 提 mask
├── checkpoints/                # SAM2 权重
│   └── sam2.1_hiera_base_plus.pt
├── rgb/             *.png       BGR 8U
├── depth/           *.png       uint16 mm, 已对齐到 color
├── masks/           *.png       0/255 binary
├── cam_K.txt                    color 3x3 内参
└── video.mp4                    录制顺手产出的 mp4(本地流程不再需要,留作快速预览)
```

---

## ① 一次性环境准备

```bash
cd /home/l/BundleSDF/reconstrct_my
bash setup_sam2.sh
```

它会:
- 用 conda 创建 `sam2` 环境 (Python 3.10)
- 装 PyTorch 2.5.1 (cu124) + 官方 sam2 包 + opencv 等
- 下载 `sam2.1_hiera_base_plus.pt` (~324MB) 到 `checkpoints/`
- 跑一次 import 自检

可重复执行,已存在的环境/权重会跳过。

---

## ② D405 录制

物体放在相机正前方 10–30 cm,光线均匀,**慢慢**转动相机或物体让 BundleSDF 看到尽量多视角。

```bash
cd /home/l/BundleSDF/reconstrct_my
python3 record_d405.py --max_seconds 30
# (本地 SAM2 没 10s 上限,可以放心录长一点;但仍受 8GB 显存限制,30s 是安全值)
```

预览窗口按键:

| 键 | 作用 |
|---|---|
| `r` | 开始录制 |
| `s` | 停止并保存 |
| `q` | 退出不保存 |

输出: `rgb/000000.png ...`, `depth/000000.png ...`, `cam_K.txt`, `video.mp4`。

**录制要点**:
- 距离 7–30 cm 最佳 (D405 是短距相机)
- 相机移动 **要慢**,每帧旋转 < 5°,否则 BundleSDF 会判 FAIL
- 第 0 帧物体必须清晰、无遮挡,它将被设为世界坐标系原点

---

## ③ 本地 SAM2 出 mask

```bash
cd /home/l/BundleSDF/reconstrct_my
conda activate sam2
python sam2_local.py
```

弹出窗口显示第 0 帧:

| 鼠标/键 | 作用 |
|---|---|
| **左键** | 加正点 (绿色) — "这是物体" |
| **右键** | 加负点 (红色) — "这不是物体",用来修掉过分割 |
| `r` | 清空所有点重来 |
| `空格` / `Enter` | 确认开始全帧传播 |
| `q` / `ESC` | 中止退出 |

加点过程中实时显示 SAM2 的预测 mask (半透明绿叠加),不满意继续加点。点选满意后按空格,SAM2 会传播到所有帧并写到 `masks/`。

输出文件名与 `rgb/` 完全一致,直接对齐。

**可选**: 加 `--vis` 标志逐帧检查 mask 质量:
```bash
python sam2_local.py --vis
```

---

## ④ 跑 BundleSDF

BundleSDF 是在 docker 里编译运行的(看主目录 `docker/run_container.sh`),conda env 不影响它:

```bash
conda deactivate          # 离开 sam2 conda 环境
cd /home/l/BundleSDF

# 4.1 在线跟踪 + 重建
python3 run_custom.py --mode run_video \
    --video_dir /home/l/BundleSDF/reconstrct_my \
    --out_folder /home/l/BundleSDF/reconstrct_my/out \
    --use_segmenter 0 --use_gui 0 --debug_level 2

# 4.2 离线全局精修(更细 NeRF + 更高精度 mesh)
python3 run_custom.py --mode global_refine \
    --video_dir /home/l/BundleSDF/reconstrct_my \
    --out_folder /home/l/BundleSDF/reconstrct_my/out

python run_custom.py --mode global_refine --video_dir my_data/20260503_163946 --out_folder /home/l/BundleSDF/my_data/20260503_163946/results
```

输出:
- `out/textured_mesh.obj` — 带纹理的重建 mesh
- `out/ob_in_cam/` — 每帧物体位姿

`--use_segmenter 0` 表示不调用内建分割器,直接用 `masks/` 下的 mask。

---

## 故障排查

### D405 找不到
```bash
rs-enumerate-devices    # 看设备列表
# 如果系统能看到但 Python 看不到:
sudo cp /etc/udev/rules.d/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### SAM2 OOM (8GB 卡跑长视频)
默认开了 CPU offload,30s @ 640x480 没问题。如果还是 OOM:
- 用更小的模型: 把 `setup_sam2.sh` 里的 `base_plus` 换成 `small`,把 `sam2_local.py` 的 `--model_cfg` 改成 `configs/sam2.1/sam2.1_hiera_s.yaml`,`--ckpt` 改对应权重
- 或者降低分辨率重录: `python record_d405.py --width 480 --height 360`

### SAM2 mask 边缘抖
SAM2 base_plus 已经够稳。如果某些帧分割糊掉了,在 `sam2_local.py` 里改加几个负点,或换 large 模型(8GB 显存边缘)。

### BundleSDF 大量 FAIL
- 相机太快: `BundleTrack/config_ho3d.yml` 里 `ransac.max_rot_deg_neighbor` 默认 30°
- 物体太远 (>1m): 调大 `depth_processing.zfar`
- mask 漏了物体: `masks/` 下检查那一帧是否物体几乎全部为 0

---

## 备选: SAM2 网页 demo (≤10s)

`extract_masks_from_sam2.py` 仍然保留,如果你只录了 ≤10s 的短视频,可以走网页 demo:

```bash
# 录制 ≤10s
python3 record_d405.py --max_seconds 10
# 上 https://sam2.metademolab.com/demo,导出 Object Cutout
# 把下载视频另存为 sam2_cutout.mp4
python3 extract_masks_from_sam2.py
```
