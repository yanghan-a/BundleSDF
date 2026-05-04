#!/usr/bin/env bash
# Launch the BundleSDF-on-FoundationPose container.
# Mirrors the original run_container.sh layout (volume bindings, network, X11)
# but uses the bundlesdf-fp:latest image and a different container name so it
# can coexist with the original BundleSDF and FoundationPose containers.

set -e

CONTAINER_NAME=bundlesdf-fp
IMAGE_NAME=bundlesdf-fp:latest
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
    -e DISPLAY="${DISPLAY}" \
    -e GIT_INDEX_FILE \
    "${IMAGE_NAME}" \
    bash -c "cd ${DIR} && bash"
