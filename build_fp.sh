#!/usr/bin/env bash
# Build BundleSDF native modules inside the FoundationPose conda env "my".
# Run this INSIDE the bundlesdf-fp container (or any container that has the same env).
#
# Usage:
#   bash build_fp.sh
#
# Differences from the original build.sh:
#   - activates conda env `my` (FoundationPose convention) instead of using system Python 3.10
#   - drops sm_90 from TORCH_CUDA_ARCH_LIST (PyTorch 2.0+cu118 wheels don't include it)
#   - passes the conda env's python to BundleTrack's cmake explicitly so pybind picks the
#     right interpreter when system python differs from conda python

set -e

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- activate conda env -------------------------------------------------------
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate my
else
    echo "[build_fp] WARNING: /opt/conda not found, assuming caller has activated env" >&2
fi

PY=$(which python)
PY_VER=$(python -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
echo "[build_fp] python = ${PY} (${PY_VER})"
python -c "import torch; print('[build_fp] torch =', torch.__version__, 'cuda =', torch.version.cuda, 'avail =', torch.cuda.is_available())"

# --- env vars BundleSDF native code needs ------------------------------------
TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"
export TORCH_LIBRARIES="${TORCH_LIB}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6}"
export FORCE_CUDA=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_fp"

echo "[build_fp] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "[build_fp] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# conda env's libffi.so.7 is a symlink to 8.1.2 which lacks LIBFFI_BASE_7.0 symbols
# expected by system libgobject-2.0; force system libffi.so.7 via LD_PRELOAD when it exists.
if [ -f /lib/x86_64-linux-gnu/libffi.so.7 ]; then
    export LD_PRELOAD="/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:${LD_PRELOAD}}"
    echo "[build_fp] LD_PRELOAD=${LD_PRELOAD}"
fi

# --- mycuda (PyTorch CUDA extension) -----------------------------------------
echo "[build_fp] === building mycuda ==="
cd "${ROOT}/mycuda"
rm -rf build *.egg-info *.so
python -m pip install --no-build-isolation -e .
# `from mycuda import common` needs (a) BundleSDF root on sys.path and (b) torch
# imported first so libc10.so is loaded into the process.
( cd "${ROOT}" && python -c "import torch; from mycuda import common; print('[build_fp] mycuda OK')" )

# --- BundleTrack (C++ pybind module) -----------------------------------------
echo "[build_fp] === building BundleTrack ==="
cd "${ROOT}/BundleTrack"
rm -rf build
mkdir -p build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="${PY}" \
    -DPython_EXECUTABLE="${PY}" \
    -DPython3_EXECUTABLE="${PY}"
make -j"$(nproc)"

# --- sanity check ------------------------------------------------------------
cd "${ROOT}"
python -c "
import sys, os
sys.path.insert(0, os.path.join('${ROOT}', 'BundleTrack', 'build'))
import my_cpp
print('[build_fp] my_cpp OK:', my_cpp.__file__)
"

echo "[build_fp] DONE"
