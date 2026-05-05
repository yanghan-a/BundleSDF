#!/usr/bin/env bash
# One-shot setup for local SAM2 inference on D405 captures.
#
# - Conda env "sam2" (Python 3.10) with PyTorch 2.5.1 (cu124).
# - Clones the official SAM2 repo to /home/l/sam2 and pip-installs it editable.
# - Downloads sam2.1_hiera_base_plus.pt to ./checkpoints.
# Idempotent: safe to re-run.

set -euo pipefail

ENV_NAME="sam2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM2_REPO_DIR="/home/l/sam2"
SAM2_GIT_URL="https://github.com/facebookresearch/sam2.git"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
CKPT_NAME="sam2.1_hiera_base_plus.pt"
CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/${CKPT_NAME}"

echo "==> Bootstrapping conda"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "    env '${ENV_NAME}' exists, reusing"
else
    echo "==> Creating conda env '${ENV_NAME}' (Python 3.10)"
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"

echo "==> Installing PyTorch 2.5.1 (CUDA 12.4 wheel)"
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

echo "==> Cloning / updating SAM2 repo at ${SAM2_REPO_DIR}"
if [[ -d "${SAM2_REPO_DIR}/.git" ]]; then
    echo "    (already cloned, fetching latest)"
    git -C "${SAM2_REPO_DIR}" fetch --quiet
    git -C "${SAM2_REPO_DIR}" pull --ff-only --quiet || true
else
    git clone "${SAM2_GIT_URL}" "${SAM2_REPO_DIR}"
fi

echo "==> Installing SAM2 (editable from ${SAM2_REPO_DIR})"
pip install -e "${SAM2_REPO_DIR}"

echo "==> Installing helper deps"
pip install opencv-python numpy tqdm pillow

echo "==> Downloading checkpoint"
mkdir -p "${CKPT_DIR}"
if [[ -f "${CKPT_DIR}/${CKPT_NAME}" ]]; then
    echo "    (already present, skipping)"
else
    if command -v wget >/dev/null 2>&1; then
        wget -O "${CKPT_DIR}/${CKPT_NAME}" "${CKPT_URL}"
    else
        curl -L -o "${CKPT_DIR}/${CKPT_NAME}" "${CKPT_URL}"
    fi
fi

echo "==> Sanity check"
python - <<'PY'
import sam2, torch
print("sam2:", sam2.__file__)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
PY

echo
echo "==> Done."
echo "    Activate env:    conda activate ${ENV_NAME}"
echo "    Repo location:   ${SAM2_REPO_DIR}"
echo "    Run inference:   python ${SCRIPT_DIR}/sam2_local.py"
