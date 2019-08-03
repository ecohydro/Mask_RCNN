
#%%
import yaml
from azureml.core import Workspace, Datastore
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

with open("/home/rave/azure_configs.yaml") as f:
    configs = yaml.safe_load(f)
# run if no workspace
# ws = Workspace.create(
#     name=configs["account"]["workspace_name"],
#     subscription_id=configs["account"]['subscription_id'],
#     location=configs["account"]["location"],
#     storage_account=configs["account"]['resource_id'] # get from properties of storage account
# )

#ws.write_config(path = "/home/rave/.azureml/config.json")
#%%

from azureml.core import Workspace
ws = Workspace.from_config(path="/home/rave/.azureml/config.json")

#%%
datastore = Datastore.register_azure_file_share(workspace=ws,
                                                datastore_name='azure_file_share_modeling',
                                                file_share_name=configs['azureml']['file_share_name']'azureml-filestore-896933ab-f4fd-42b2-a154-0abb35dfb0b0',
                                                account_name=azureml['azureml']['account_name'],
                                                account_key=configs['azureml']['account_key'],
                                                create_if_not_exists=False)


# datastore = Datastore.register_azure_blob_container(workspace=ws, 
#                                                     datastore_name='your datastore name', 
#                                                     container_name='your azure blob container name',
#                                                     account_name='your storage account name', 
#                                                     account_key='your storage account key',
#                                                     create_if_not_exists=True)

#define default datastore for current workspace
ws.set_default_datastore('azure_file_share_modeling')


#to mount the full contents in your storage to the compute target
mounted_data_reference= datastore.as_mount()

#to download the contents of the `./bar` directory in your storage to the compute target
# datastore.path('./bar').as_download()
#%%

# Choose a name for your CPU cluster
gpu_cluster_name = "gpucluster"

# Verify that cluster does not exist already
try:
    gpu_cluster = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC6s_v2',
                                                           max_nodes=1)
    gpu_cluster = ComputeTarget.create(ws, gpu_cluster_name, compute_config)

gpu_cluster.wait_for_completion(show_output=True)

#%%
# Choose a name for your CPU cluster
cpu_cluster_name = "cpucluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS4_V2',
                                                           max_nodes=1)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)

#%% 
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

# Create a new runconfig object
run_amlcompute = RunConfiguration()

# Use the cpu_cluster you created above. 
run_amlcompute.target = cpu_cluster

# Enable Docker
run_amlcompute.environment.docker.enabled = True

# Set Docker base image to the default CPU-based image
run_amlcompute.environment.docker.base_image = DEFAULT_CPU_IMAGE

# Use conda_dependencies.yml to create a conda environment in the Docker image for execution
run_amlcompute.environment.python.user_managed_dependencies = True # this was false, conda env not being used by compute

# Specify CondaDependencies obj, add necessary packages
run_amlcompute.environment.python.conda_dependencies = CondaDependencies("/home/rave/CropMask_RCNN/cropmask-env.yml").create()

#%%
from azureml.core import Experiment
experiment_name = 'my_experiment'

exp = Experiment(workspace=ws, name=experiment_name)

from azureml.core import ScriptRunConfig
import os 

script_folder = os.getcwd()
src = ScriptRunConfig(source_directory = script_folder, script = 'cropmask/run_cropmask.py', run_config = run_amlcompute)
run = exp.submit(src)
run.wait_for_completion(show_output = True)

#%%

from azureml.core.model import Model

model = Model.register(workspace=ws, model_path="churn-model.pkl", model_name="churn-model-test")


#%%
from azureml.core.model import Model
import os

model = Model(workspace=ws, name="churn-model-test")
model.download(target_dir=os.getcwd())

model.delete()

#%%

from azureml.train.automl import AutoMLConfig

automl_config = AutoMLConfig(task="classification",
                             X=your_training_features,
                             y=your_training_labels,
                             iterations=30,
                             iteration_timeout_minutes=5,
                             primary_metric="AUC_weighted",
                             n_cross_validations=5
                            )


from azureml.core.experiment import Experiment

experiment = Experiment(ws, "automl_test_experiment")
run = experiment.submit(config=automl_config, show_output=True)

best_model = run.get_output()
y_predict = best_model.predict(X_test)
