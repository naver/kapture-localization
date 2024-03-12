FROM nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04
MAINTAINER naverlabs "kapture@naverlabs.com"

# setup environment
ENV     LANG C.UTF-8
ENV     LC_ALL C.UTF-8
ENV     DEBIAN_FRONTEND noninteractive
# arguments
ARG     MAKE_OPTIONS="-j8"
ARG     SOURCE_PREFIX="/opt/src"
ARG     CUDA_ARCHITECTURES=75

RUN mkdir -p ${SOURCE_PREFIX}

# base tools
RUN     apt-get update \
     && apt-get install -y \
        git \
        wget cmake \
        build-essential \
     && rm -rf /var/lib/apt/lists/*

# PYTHON & PIP ###########################################################################################################
RUN     apt-get update \
     && apt-get install -y python3 python3-pip \
     && python3 -m pip install --upgrade pip \
     && python3 -m pip install --upgrade setuptools wheel twine


# COLMAP ###############################################################################################################
RUN     apt-get update \
     && apt-get install -y --no-install-recommends --no-install-suggests \
            git \
            cmake \
            ninja-build \
            build-essential \
            libboost-program-options-dev \
            libboost-filesystem-dev \
            libboost-graph-dev \
            libboost-system-dev \
            libeigen3-dev \
            libflann-dev \
            libfreeimage-dev \
            libmetis-dev \
            libgoogle-glog-dev \
            libgtest-dev \
            libsqlite3-dev \
            libglew-dev \
            qtbase5-dev \
            libqt5opengl5-dev \
            libcgal-dev \
            libceres-dev
## colmap
WORKDIR ${SOURCE_PREFIX}
RUN     git clone -b 3.9 https://github.com/colmap/colmap.git
RUN     cd colmap \
     && mkdir build \
     && cd build \
     && cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
     && ninja install

######### POSELIB ##############################################################
WORKDIR ${SOURCE_PREFIX}
RUN     git clone --recursive -n https://github.com/vlarsson/PoseLib.git
WORKDIR ${SOURCE_PREFIX}/PoseLib
RUN     git checkout 67dc757c619a320ae3cf4a4ffd4e4b7fc5daa692 && \
        git submodule update --recursive # version required by PYRANSACLIB
RUN     mkdir -p ${SOURCE_PREFIX}/PoseLib/_build
RUN     cd ${SOURCE_PREFIX}/PoseLib/_build \
     && cmake -DCMAKE_INSTALL_PREFIX=../_install .. \
     && cmake --build . --target install -j 8 \
     && cmake --build . --target clean

######### PYCOLMAP #############################################################
RUN     python3 -m pip install pycolmap

######### PYRANSACLIB ##########################################################
WORKDIR ${SOURCE_PREFIX}
RUN     git clone --recursive -n https://github.com/tsattler/RansacLib.git
WORKDIR ${SOURCE_PREFIX}/RansacLib
RUN     git checkout 8b5a8b062711ee9cc57bc73907fbe0ae769d5113 \
     && git submodule update --recursive \
     && sed -i '4i set(CMAKE_CXX_STANDARD 14)' CMakeLists.txt
RUN     CMAKE_PREFIX_PATH=${SOURCE_PREFIX}/PoseLib/_install/lib/cmake/PoseLib  python3 -m pip install ./


########################################################################################################################
# install kapture from pip.
RUN      python3 -m pip install kapture==1.1.10

# install kapture-localization
ADD      . ${SOURCE_PREFIX}/kapture-localization
WORKDIR  ${SOURCE_PREFIX}/kapture-localization
RUN      python3 -m pip install "torch==2.2.1" "torchvision==0.17.1" "scikit_learn==1.3.2" \
      && python3 -m pip install -r requirements.txt \
      && python3 -m pip install .

### FINALIZE ###################################################################
# save space: purge apt-get
RUN     rm -rf /var/lib/apt/lists/*
USER  root
WORKDIR ${SOURCE_PREFIX}/
