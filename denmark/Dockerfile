# Pull in the AI for Earth Base Image, so we can extract necessary libraries.
FROM mxnet/python:gpu_0.11.0 as mxnet

RUN pip install Cython \
    && pip install opencv-python==3.2.0.6 \
    && pip install easydict==1.6 \
    && pip install hickle

RUN git clone https://github.com/msracver/FCIS.git

RUN apt-get install -y libopenblas-dev

RUN cd mxnet \
    && make clean

RUN cd FCIS \
    && sh ./init.sh

RUN cp -r FCIS/fcis/operator_cxx/channel_operator* mxnet/src/operator/contrib/

RUN  cd mxnet \
    && make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1

RUN cd mxnet/python \
    && python setup.py install

COPY fcis_coco-0000.params /FCIS/model/

RUN apt-get update

RUN apt-get install -y --fix-missing python-tk \
    && pip install pillow matplotlib pyaml

RUN pip install jupyterlab
