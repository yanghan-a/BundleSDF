# BundleSDF on top of FoundationPose image.
# Goal: reuse FP's conda env "my" (Python 3.8/3.9 + PyTorch 2.0+cu118 + kaolin + nvdiffrast + pytorch3d)
# and only add the C++ system dependencies BundleTrack needs (PCL, OpenCV system libs)
# plus the LoFTR Python deps that FP doesn't already ship.
#
# Build:  docker build -f docker/dockerfile.fp -t bundlesdf-fp:latest .
# Run:    bash docker/run_container_fp.sh

ARG FP_BASE=foundationpose:latest
FROM ${FP_BASE}

ENV DEBIAN_FRONTEND=noninteractive

# ----- BundleTrack C++ system deps -----
# FP image already provides: pybind11 v2.10 (source build), Eigen 3.4 (source build),
# libboost-all-dev, libyaml-cpp-dev, libzmq3-dev, libgflags-dev, libgoogle-glog-dev,
# libflann-dev, freeglut3-dev, libhdf5-dev. PCL is missing.
# We do NOT use apt's libopencv-dev (4.2) because BundleTrack uses cv::cuda::DescriptorMatcher
# which lives in opencv_contrib/cudafeatures2d, only built when OpenCV is compiled from source
# WITH_CUDA=ON + opencv_contrib. So we mirror BundleSDF's original dockerfile here.
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        libpcl-dev \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# ----- OpenCV 4.5.5 with CUDA + opencv_contrib (built from source) -----
# Branch 4.5.5 is chosen because it cleanly compiles with CUDA 11.x + gcc 9 (FP base image).
# Drop arch 9.0 (cu118 wheels don't ship sm_90); host GPU sm_89 (RTX 4060 Ti) JITs from PTX.
ARG OPENCV_VERSION=4.5.5
RUN cd /tmp && \
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv && \
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib && \
    mkdir -p /tmp/opencv/build && cd /tmp/opencv/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_CXX_STANDARD=17 \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN="7.0 7.5 8.0 8.6" \
        -DCUDA_ARCH_PTX="8.6" \
        -DCUDA_FAST_MATH=ON \
        -DWITH_OPENMP=ON \
        -DWITH_OPENCL=OFF \
        -DBUILD_opencv_cudacodec=OFF \
        -DBUILD_opencv_wechat_qrcode=OFF \
        -DBUILD_opencv_xfeatures2d=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        -DENABLE_PRECOMPILED_HEADERS=OFF && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf /tmp/opencv /tmp/opencv_contrib

SHELL ["/bin/bash", "--login", "-c"]

# ----- Python deps that BundleSDF needs but FP env doesn't ship -----
# loguru / pytorch-lightning are imported via LoFTR's src tree.
# BundleSDF Python code also uses ruamel.yaml, transformations, dearpygui (gui), pymeshlab.
# Most of these FP already has; only fill the gaps. Use --no-deps where possible to avoid
# pulling in conflicting torch/numpy versions.
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate my && \
    pip install --no-cache-dir \
        loguru==0.7.2 \
        yacs==0.1.8 \
        ruamel.yaml \
        transformations \
        dearpygui \
        pymeshlab \
        sacred \
        pymongo \
        zmq && \
    pip install --no-cache-dir --no-deps pytorch-lightning==1.5.10

# ----- kaolin sanity: BundleSDF uses kaolin.ops.spc / kaolin.render.spc.
# These APIs are stable since 0.13. FP installs kaolin from source at /kaolin (FORCE_CUDA=1).
# Only force-install a known-good wheel if import fails (defensive).
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate my && \
    python -c "import kaolin; print('kaolin ok:', kaolin.__version__)" || \
    pip install --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# ----- Build env vars for BundleSDF native modules -----
# PyTorch 2.0+cu118 wheels do NOT ship sm_90 binaries, so drop 9.0 from BundleSDF's
# original "7.0;7.5;8.0;8.6;9.0" arch list. Add 5.2/6.0/6.1 to match PyTorch wheel coverage.
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
# Conda env "my" ships libffi 8.1.2 but symlinks libffi.so.7 -> libffi.so.8.1.2
# (lossy compat shim). System libgobject-2.0.so needs LIBFFI_BASE_7.0 symbols which
# 8.1.2 doesn't export, so import my_cpp (which pulls GLib via OpenCV) crashes.
# Force the real libffi.so.7 via LD_PRELOAD.
ENV LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7

# Make the conda env the default interactive shell
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate my" >> /root/.bashrc

CMD ["/bin/bash"]
