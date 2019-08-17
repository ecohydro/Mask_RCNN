# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# tensorflow    latest (pip)
# ==================================================================

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \ 

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    add-apt-repository ppa:ubuntugis/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        # geospatial dependencies
        software-properties-common \
        libfreetype6-dev \
        libpng-dev \
        libzmq3-dev \
        libspatialindex-dev \
        gdal-bin \
        libgdal-dev \
        python3-gdal \
        libsm6 \
        nano \
        zip \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        matplotlib \
        PyQt5 \
        seaborn \
        xlrd \
        scipy \
        scikit-learn \
        scikit-image \
        xarray \
        dask \
        pandas==0.25.0 \
        cloudpickle \
        Cython \
        Pillow \
        opencv-python \
        IPython[all] \
        rasterstats \
        geopy \
        geos \
        pyproj \
        pyshp \
        six \
        cartopy \
        geopandas \
        rasterio==1.0.24 \
        contextily \
        pysal \
        pyproj \
        folium \
        rtree \
        urllib3 \
        networkx \
        geojson \
        affine \
        albumentations \
        rio-cogeo==1.0.0 \
        rioxarray \
        solaris \
        keras \
        && \

# ==================================================================
# tensorflow
# ------------------------------------------------------------------

    $PIP_INSTALL \
        tensorflow-gpu==1.14.0 \
        && \

# ==================================================================
# Azure ML
#

    $PIP_INSTALL \
        azureml-sdk[automl] --ignore-installed PyYAML \
        && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 6006
