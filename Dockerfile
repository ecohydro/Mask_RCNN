FROM ufoym/deepo:tensorflow-py36-cu100
# ==================================================================
# setting up conda
# ------------------------------------------------------------------
COPY ./bash-scripts/setup_azureml_env.sh ./setup_azureml_env.sh
RUN bash setup_azureml_env.sh
# ==================================================================
# setting up cropmask environment
# ------------------------------------------------------------------

COPY ./environment.yml /root/environment.yml

RUN conda env update --file /root/environment.yml

# ==================================================================
# Azure ML
#
RUN pip install --upgrade --ignore-installed PyYAML azureml-sdk[automl]