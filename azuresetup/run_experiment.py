
#%%
import yaml
from azureml.core import Workspace, Datastore
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

with open("/home/rave/azure_configs.yaml") as f:
    configs = yaml.safe_load(f)['azureml']

if Workspace.list(configs['subscription_id']) == {}:
    ws = Workspace.create(
        name=configs["workspace_name"],
        subscription_id=configs['subscription_id'],
        location=configs["location"],
        storage_account=configs['resource_id'],
        container_registry=configs['container_registry']
    )

    ws.write_config(file_name = "/home/rave/.azureml/config.json")

ws = Workspace.from_config(path="/home/rave/.azureml/config.json")

#%%
if ws.get_default_datastore().name != 'azure_file_share_modeling':
    datastore = Datastore.register_azure_file_share(workspace=ws,
                                                    datastore_name='azure_file_share_modeling',
                                                    file_share_name=configs['file_share_name'],
                                                    account_name=configs['account_name'],
                                                    account_key=configs['account_key'],
                                                    create_if_not_exists=False)

    #define default datastore for current workspace
    ws.set_default_datastore('azure_file_share_modeling')


#to mount the full contents in your storage to the compute target
# mounted_data_reference= datastore.as_mount()

#to download the contents of the `./bar` directory in your storage to the compute target
# datastore.path('./bar').as_download()


#%%
# Choose a name for your GPU cluster
gpu_cluster_name = "gpucluster"

# Verify that cluster does not exist already
try:
    gpu_cluster = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                           max_nodes=1)
    gpu_cluster = ComputeTarget.create(ws, gpu_cluster_name, compute_config)

gpu_cluster.wait_for_completion(show_output=True)

#%% 
from azureml.train.estimator import Estimator
import os

script_params = {
    "--dataset": ws.get_default_datastore().path("landsat-512").as_mount(),
    "--subset": "train",
    "--weights": "coco"
}

exec_environment = Estimator(source_directory=os.getcwd(),
                             entry_script='cropmask/run_cropmask.py',
                             script_params=script_params,              
                             compute_target=gpu_cluster,
                             custom_docker_image=os.path.basename(os.path.split(configs['container_registry'])[-2])+".azurecr.io/cropmask:1",
                             user_managed=True,
                             use_docker=True)

#%%
from azureml.core import Experiment
experiment_name = 'my_experiment'

exp = Experiment(workspace=ws, name=experiment_name)
run = exp.submit(config=exec_environment)
run.wait_for_completion(show_output = True)

#%%
# for registering trained models
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
