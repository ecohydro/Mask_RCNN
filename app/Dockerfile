# Pull in the AI for Earth Base Image, so we can extract necessary libraries.
FROM mcr.microsoft.com/aiforearth/base-py:1.8-cuda-9.0-runtime as ai4e_base

RUN conda update conda

RUN conda install -c conda-forge -n ai4e_py_api keras-gpu=2.2.4 tensorflow-gpu=1.10.0 \
    && /usr/local/envs/ai4e_py_api/bin/pip install imgaug scikit-image==0.16.1 PyQt5 Cython \
    && /usr/local/envs/ai4e_py_api/bin/pip install git+git://github.com/ecohydro/CropMask_RCNN@master

RUN apt-get update \
    && apt-get install software-properties-common -y \
    && add-apt-repository ppa:ubuntugis/ppa \
    && apt-get install gdal-bin -y \
    && apt-get install libgdal-dev -y

RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal \
    && export C_INCLUDE_PATH=/usr/include/gdal \
    && pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') --global-option=build_ext --global-option="-I/usr/include/gdal" \
    && pip install rasterio \
    && pip install geopandas

# Now add our custom image commands ---------------------------------------------------
# installing environement, replaced 3 lines in original repo
# RUN conda install -c conda-forge -n ai4e_py_api geopandas rasterio=1.0 poppler=0.61 \
#     && conda clean -t

EXPOSE 55678 50001

# Note: supervisor.conf reflects the location and name of your api code.
# If the default (./my_api/runserver.py) is renamed, you must change supervisor.conf
COPY ./supervisord.conf /etc/supervisord.conf
# startup.sh is a helper script
COPY ./startup.sh /
RUN chmod +x /startup.sh

# Copy your API code
COPY ./cropmask_api /app/cropmask_api/

ENV PYTHONPATH="${PYTHONPATH}:/app/"

# Application Insights keys and trace configuration
ENV APPINSIGHTS_INSTRUMENTATIONKEY= \
    TRACE_SAMPLING_RATE=1.0

# The following variables will allow you to filter logs in AppInsights
ENV SERVICE_OWNER=AI4E_cropmask_app \
    SERVICE_CLUSTER=Local\ Docker \
    SERVICE_MODEL_NAME=AI4E_cropmask_app \
    SERVICE_MODEL_FRAMEWORK=Python \
    SERVICE_MODEL_FRAMEOWRK_VERSION=3.6.6 \
    SERVICE_MODEL_VERSION=1.0

ENV API_PREFIX=/v1/keras_api

ENV STORAGE_ACCOUNT_NAME= \
    STORAGE_ACCOUNT_KEY=

# Expose the port that is to be used when calling your API
EXPOSE 80
HEALTHCHECK --interval=1m --timeout=3s --start-period=20s \
  CMD curl -f http://localhost/${API_PREFIX}/  || exit 1
ENTRYPOINT [ "/startup.sh" ]
# Use the following entrypoint to debug bridge issues.
#ENTRYPOINT ["uwsgi", "--chdir", "/app/cropmask_api/", "--virtualenv", "/usr/local/envs/ai4e_py_api", "--http", "0.0.0.0:80", "--callable", "app", "--enable-threads", "--wsgi-file", "/app/cropmask_api/runserver.py", "--wsgi-disable-file-wrapper"]
