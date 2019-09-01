
#%%
from azureml.core import Workspace
# see https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace
# for details on how the workspace was made and how to make config file
ws = Workspace.from_config()

#%%
