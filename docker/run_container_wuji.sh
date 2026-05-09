#!/usr/bin/env bash
# 中国节点 wuji 服务器(8×RTX4090)专用 - 进入 BundleSDF 容器(交互式)
# 用 attempt #10 build 出来的 nvcr.io/nvidian/bundlesdf:bundlesdf 镜像
# 已注入所有跑训练所需 env:
#   - LD_LIBRARY_PATH    : my_cpp.so 的 C++ 依赖搜索路径
#   - BUNDLESDF_NERF_GPU : NeRF 子进程隔离到物理 cuda:1, 主进程占 cuda:0
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True : 减少 PyTorch 显存碎片

set -e

CONTAINER_NAME=bundlesdf
IMAGE_NAME=nvcr.io/nvidian/bundlesdf:bundlesdf
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
xhost + >/dev/null 2>&1 || true

docker run --gpus all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -it \
    --network=host \
    --name "${CONTAINER_NAME}" \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /home:/home \
    -v /tmp:/tmp \
    -v /mnt:/mnt \
    -v "${DIR}:${DIR}" \
    --ipc=host \
    -e LD_LIBRARY_PATH=/root/wuji_ws_0/BundleSDF/BundleTrack/build \
    -e BUNDLESDF_NERF_GPU=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e DISPLAY="${DISPLAY}" \
    -e GIT_INDEX_FILE \
    --entrypoint /bin/bash \
    "${IMAGE_NAME}" \
    -c "cd ${DIR} && exec bash"
