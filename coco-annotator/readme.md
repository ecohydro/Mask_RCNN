# Azure Setup Instructions 
(if you already followed this section in the terraform folder, you are good to go)

Portion of the following README and terraform scripts come from Andreas Offenhaeuser's excellent [Automated dev workflow for using Data Science VM on Azure](https://medium.com/@an0xff/automated-dev-workflow-for-using-data-science-vm-on-azure-13c1a5b56f91). Before following the instructions below to provision an Azure cluster, you'll need an Azure account and [Azure CLI tools](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) installed on your local machine. Then run

```sh
# have your account username and password handy
az login
```

Next, login to the portal and create a permanent storage account under it's own resource group, meaning a seperate resource group from what you choose for the resource group that is called by terraform (Otherwise you could end up deleting your storage with `terraform destroy`). 

To start the vm, run `terraform apply`. You can use `make stop-annotation` and other make commands to control the serve instance (see the Makefile).  Once you run terraform apply, docker compose should be installed and the server should be running, but the first time you will need to ssh in with `make ssh-annotation` and mount your storage using the connection script provided through the Azure portal. Create all datasets in an annotation folder in your mounted file storage so that it is not coupled to the vm and can't be accidentally deleted with terraform destroy.

after `terraform apply`

```
git clone https://github.com/ecohydro/coco-annotator
sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
newgrp docker
cd coco-annotator && docker-compose up", # starts server at port 5000
# get the mount script from azure portal by clicking the Connect to file share button
bash mnt.sh # for mounting permanently after stopping and starting. point to ~/coco-annotator/datasets

```

See the [COCO Annotator page and wiki](https://github.com/jsbroks/coco-annotator/wiki) for an overview of the tool. I'm currently testing it out but it looks excellent. I selected it because it's free, supports free hand drawing, magic wand, and .h5 models for assisted annotation.
