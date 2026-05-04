#!/usr/bin/env bash
# Build BundleSDF native modules directly inside the host conda env "foundationpose".
# No docker needed.
#
# Prereqs (run once on host, requires sudo):
#   sudo apt-get install -y libzmq3-dev libgflags-dev libgoogle-glog-dev
#
# These are the only system deps the host is missing — PCL 1.12, OpenCV 4.5,
# yaml-cpp 0.7, pybind11 2.9, Eigen 3.4, Boost 1.74, libflann, libhdf5, freeglut3
# are already installed.
#
# Usage:
#   conda activate foundationpose   # (the script will activate it again to be safe)
#   bash build_conda.sh

set -e

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME="${BUNDLESDF_ENV:-foundationpose}"

# --- activate the right conda env --------------------------------------------
if [ -z "${CONDA_DEFAULT_ENV}" ] || [ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
        source /opt/conda/etc/profile.d/conda.sh
    fi
    conda activate "${ENV_NAME}"
fi

PY=$(which python)
PY_VER=$(python -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
echo "[build_conda] env    = ${CONDA_DEFAULT_ENV}"
echo "[build_conda] python = ${PY} (${PY_VER})"
python -c "import torch; print('[build_conda] torch =', torch.__version__, 'cuda =', torch.version.cuda, 'avail =', torch.cuda.is_available())"

# --- system deps sanity ------------------------------------------------------
need_apt=()
for hdr in /usr/include/pcl-1.* ; do [ -d "$hdr" ] || need_apt+=("libpcl-dev"); done
for hdr in /usr/include/opencv4 ; do [ -d "$hdr" ] || need_apt+=("libopencv-dev"); done
for hdr in /usr/include/yaml-cpp ; do [ -d "$hdr" ] || need_apt+=("libyaml-cpp-dev"); done
[ -f /usr/include/zmq.h ]              || need_apt+=("libzmq3-dev")
[ -f /usr/include/gflags/gflags.h ]    || need_apt+=("libgflags-dev")
[ -f /usr/include/glog/logging.h ]     || need_apt+=("libgoogle-glog-dev")
[ -f /usr/include/GL/glut.h ]          || need_apt+=("freeglut3-dev")
[ -f /usr/lib/cmake/pybind11/pybind11Config.cmake ] || need_apt+=("pybind11-dev")

if [ ${#need_apt[@]} -gt 0 ]; then
    echo "[build_conda] ERROR: missing system packages:" "${need_apt[@]}"
    echo "[build_conda] run: sudo apt-get install -y ${need_apt[*]}"
    exit 1
fi

# --- env vars BundleSDF native code needs ------------------------------------
TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"
export TORCH_LIBRARIES="${TORCH_LIB}"
# RTX 4060 Ti is sm_89; PyTorch 2.0+cu118 wheel ships sm_86 + PTX which JITs at runtime.
# Including 8.9 explicitly avoids the JIT pause on first kernel launch.
# Drop sm_90 — cu118 wheels don't include it.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9}"
export FORCE_CUDA=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${ENV_NAME}"

# Force the build to use conda env's nvcc (CUDA 11.8) not the host one (which is 11.5).
# PyTorch was compiled against 11.8, so cpp_extension needs the matching toolkit.
CONDA_CUDA="${CONDA_PREFIX}"   # e.g. /home/l/miniconda3/envs/foundationpose
if [ -x "${CONDA_CUDA}/bin/nvcc" ]; then
    export CUDA_HOME="${CONDA_CUDA}"
    export PATH="${CONDA_CUDA}/bin:${PATH}"
    echo "[build_conda] using conda nvcc: $(${CONDA_CUDA}/bin/nvcc --version | tail -1)"
else
    echo "[build_conda] WARN: no nvcc in conda env; falling back to system $(which nvcc)"
fi

echo "[build_conda] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "[build_conda] CUDA_HOME=${CUDA_HOME}"

# --- 1) mycuda (PyTorch CUDA extension) --------------------------------------
echo "[build_conda] === building mycuda ==="
cd "${ROOT}/mycuda"
rm -rf build *.egg-info ./*.so
python -m pip install --no-build-isolation -e .
# `from mycuda import common` needs (a) BundleSDF root on sys.path so the mycuda package
# is found, and (b) torch imported first so libc10.so is dlopened into the process.
( cd "${ROOT}" && python -c "import torch; from mycuda import common; print('[build_conda] mycuda OK')" )

# --- 2) BundleTrack (C++ pybind module) --------------------------------------
echo "[build_conda] === building BundleTrack ==="
cd "${ROOT}/BundleTrack"
rm -rf build
mkdir -p build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="${PY}" \
    -DPython_EXECUTABLE="${PY}" \
    -DPython3_EXECUTABLE="${PY}" \
    -Dpybind11_DIR=/usr/lib/cmake/pybind11
make -j"$(nproc)"

# --- 3) sanity check ---------------------------------------------------------
cd "${ROOT}"
python -c "
import sys, os
sys.path.insert(0, os.path.join('${ROOT}', 'BundleTrack', 'build'))
import my_cpp
print('[build_conda] my_cpp OK:', my_cpp.__file__)
"

echo "[build_conda] DONE"
echo ""
echo "Next step: run BundleSDF, e.g."
echo "  cd ${ROOT}"
echo "  python run_custom.py --mode run_video --video_dir <your_data> --out_folder <out> --use_segmenter 0 --use_gui 0 --debug_level 2"
