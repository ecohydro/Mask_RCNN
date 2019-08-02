
#%%
import yaml
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

with open("/home/rave/azure_configs.yaml") as f:
    configs = yaml.safe_load(f)

ws = Workspace.create(
    name=configs["account"]["workspace_name"],
    subscription_id=configs["account"]['subscription_id'],
    location=configs["account"]["location"],
    storage_account=configs["account"]['resource_id'] # get from properties of storage account
)

