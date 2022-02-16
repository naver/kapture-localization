#FROM ubuntu:18.04
FROM nvidia/cudagl:10.0-devel-ubuntu18.04
MAINTAINER naverlabs "kapture@naverlabs.com"

# setup environment
ENV     LANG C.UTF-8
ENV     LC_ALL C.UTF-8
ENV     DEBIAN_FRONTEND noninteractive
# arguments
ARG     MAKE_OPTIONS="-j8"
ARG     SOURCE_PREFIX="/opt/src"

RUN mkdir -p ${SOURCE_PREFIX}

# Get dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
    wget curl \
    unzip openssh-client libssl-dev \
    python3.6 python3-pip python3-dev \
    pandoc asciidoctor \
    build-essential \
    libboost-all-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    freeglut3-dev \
    libxmu-dev \
    libxi-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libcgal-qt5-dev \
    libqt5opengl5-dev \
    qt5-default \
    x11-apps \
    mesa-utils \
  && rm -rf /var/lib/apt/lists/*


########################################################################################################################
# PYTHON-PIP ###########################################################################################################
# make sure pip 3 is >= 20.0 to enable use-feature=2020-resolver
RUN     python3 -m pip install --upgrade pip
RUN     python3 -m pip install --upgrade setuptools wheel twine

# force upgrade of cmake (more than apt get)
## CMAKE version 3.23.0-rc1
WORKDIR ${SOURCE_PREFIX}
RUN     wget https://github.com/Kitware/CMake/releases/download/v3.23.0-rc1/cmake-3.23.0-rc1.tar.gz && \
        tar -xf cmake-3.23.0-rc1.tar.gz
RUN     cd cmake-3.23.0-rc1 && ./bootstrap && make install

########################################################################################################################
# COLMAP ###############################################################################################################
# ├── eigen
# └── ceres

# Eigen 3.3.9
WORKDIR ${SOURCE_PREFIX}
RUN     git clone -b 3.3.9 https://gitlab.com/libeigen/eigen.git eigen
RUN     mkdir -p eigen/build
WORKDIR ${SOURCE_PREFIX}/eigen/build
RUN     cmake \
        -DCMAKE_BUILD_TYPE=Release \
         .. && \
        make ${MAKE_OPTIONS} && make install && make clean

# ceres 2.0.0
WORKDIR ${SOURCE_PREFIX}
RUN     git clone -b 2.0.0 https://github.com/ceres-solver/ceres-solver.git
RUN     mkdir -p ceres-solver/build
WORKDIR ${SOURCE_PREFIX}/ceres-solver/build
RUN     cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
        ../ && \
        make ${MAKE_OPTIONS} && make install && make clean

# colmap
WORKDIR ${SOURCE_PREFIX}
RUN     git clone -b 3.7 https://github.com/colmap/colmap.git
WORKDIR ${SOURCE_PREFIX}/colmap
RUN     mkdir -p build
WORKDIR ${SOURCE_PREFIX}/colmap/build
RUN     cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DTESTS_ENABLED=OFF \
        .. && \
        make ${MAKE_OPTIONS} && make install && make clean

######### POSELIB ##############################################################
WORKDIR ${SOURCE_PREFIX}
RUN     git clone --recursive -n https://github.com/vlarsson/PoseLib.git
WORKDIR ${SOURCE_PREFIX}/PoseLib
RUN     git checkout 67dc757c619a320ae3cf4a4ffd4e4b7fc5daa692 && \
        git submodule update --recursive # version required by PYRANSACLIB
RUN     mkdir -p ${SOURCE_PREFIX}/PoseLib/_build
RUN     cd ${SOURCE_PREFIX}/PoseLib/_build && \
        cmake -DCMAKE_INSTALL_PREFIX=../_install .. && \
        cmake --build . --target install -j 8 && \
        cmake --build . --target clean

######### PYCOLMAP #############################################################
WORKDIR ${SOURCE_PREFIX}
RUN     git clone --recursive -b v0.1.0 https://github.com/colmap/pycolmap.git
WORKDIR ${SOURCE_PREFIX}/pycolmap
RUN     python3 -m pip install ./

######### PYRANSACLIB ##########################################################
WORKDIR ${SOURCE_PREFIX}
RUN     git clone --recursive -n https://github.com/tsattler/RansacLib.git
WORKDIR ${SOURCE_PREFIX}/RansacLib
RUN     git checkout 8b5a8b062711ee9cc57bc73907fbe0ae769d5113 && \
        git submodule update --recursive
RUN     sed -i '4i set(CMAKE_CXX_STANDARD 17)' CMakeLists.txt
RUN     CMAKE_PREFIX_PATH=${SOURCE_PREFIX}/PoseLib/_install/lib/cmake/PoseLib  python3 -m pip install ./

#########################################################################################################################
# install kapture from pip.
RUN      python3 -m pip install kapture

# install kapture-localization
ADD      . ${SOURCE_PREFIX}/kapture-localization
WORKDIR  ${SOURCE_PREFIX}/kapture-localization
RUN      python3 -m pip install "torch==1.4.0" "torchvision==0.5.0" "scikit_learn==0.20.2"
RUN      python3 -m pip install -r requirements.txt
RUN      python3 setup.py install

### FINALIZE ###################################################################
# save space: purge apt-get
RUN     rm -rf /var/lib/apt/lists/*
USER  root
WORKDIR ${SOURCE_PREFIX}/
