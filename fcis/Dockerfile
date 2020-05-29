FROM mxnet/python:gpu_0.12.1

RUN rm -rf /mxnet

RUN apt-get update

RUN apt-get install -y --fix-missing git build-essential pkg-config libopencv-dev libglib2.0-0

RUN pip install pillow pandas matplotlib pyaml jupyterlab scipy scikit-image hickle easydict==1.6 opencv-python==3.2.0.6 Cython

# can't compile mxnet with versions of cuda that support faster convolution
#ENV CUDNN_VERSION=7.1.4.18

#LABEL com.nvidia.cudnn.version=7.1.4.18

#RUN apt-get update && apt-get install -y --no-install-recommends libcudnn7=$CUDNN_VERSION-1+cuda8.0 libcudnn7-dev=$CUDNN_VERSION-1+cuda8.0

RUN cd /home \
    && git clone https://github.com/msracver/FCIS.git \
    && cd /home/FCIS \
    && sh ./init.sh

RUN mkdir /home/code \
    && mkdir /home/data

RUN apt-get install -y --fix-missing python-tk libopenblas-dev

RUN cd /home && git clone --recursive https://github.com/dmlc/mxnet.git && cd mxnet \
  && git checkout 998378a \
  && git submodule init \
  && git submodule update

RUN cp -r /home/FCIS/fcis/operator_cxx/channel_operator* /home/mxnet/src/operator/contrib/

RUN cd /home/mxnet \
  && make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1

RUN cd /home/mxnet \
  && cd python \
  && python setup.py develop

RUN sed -i -e 's/allow_extra=allow_extra//g' /home/mxnet/python/mxnet/module/base_module.py
