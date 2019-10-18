FROM ufoym/deepo:tensorflow-py36-cu100
# ==================================================================
# setting up conda
# ------------------------------------------------------------------
RUN chmod 775 /usr/local
RUN echo 'export PATH=/usr/local/miniconda/bin:$PATH' > /etc/profile.d/conda.sh
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -o /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -bufp /usr/local
RUN rm -f /tmp/miniconda.sh
COPY ./azureml_env.yml /root/environment.yml
COPY ./ /root/CropMask_RCNN
RUN export PATH=/usr/local/bin:$PATH \
    && conda install anaconda-client python==3.6.6 -n base \
    && conda install conda-build \
    && conda create -y --name azureml python=3.6 \
    && echo "source activate azureml" >> ~/.bashrc \
    && conda env update --file /root/environment.yml \
    && export CONDA_AUTO_UPDATE_CONDA=false \
    && export CONDA_DEFAULT_ENV=azureml \
    && export CONDA_PREFIX=/usr/local/envs/$CONDA_DEFAULT_ENV \
    && export PATH=$CONDA_PREFIX/bin:$PATH \
    && conda clean -ya
# ==================================================================
# Azure ML
#
RUN pip install --upgrade --ignore-installed PyYAML azureml-sdk[automl]
ENV PATH /envs/azureml/bin:$PATH
ENV PYTHONPATH /root/CropMask_RCNN/cropmask:$PYTHONPATH

