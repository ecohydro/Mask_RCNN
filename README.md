[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

CropMask_RCNN is a project to train and deploy instance segmentation models for mapping fallow and irrigated center pivot agriculture from multispectral satellite imagery. The primary goal is to develop a model or set of models that can map center pivot agriculture with high precision and recall across different dryland agriculture regions. The main strategy I'm using relies on transfer learning from COCO before finetuning on Landsat tiles from multiple cloud-free scenes acquired in Nebraska during the 2005 growing season (late May - early October). This time period coincides with the labeled dataset described below. It extends [matterport's module](https://github.com/matterport/Mask_RCNN) , which is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. CropMask_RCNN work with multispectral Landsat satellite imagery, contains infrastructure-as-code via terraform to build a GPU enabled Azure Data Science VM (from [Andreus Affenhaeuser's guide](https://medium.com/@an0xff/automated-dev-workflow-for-using-data-science-vm-on-azure-13c1a5b56f91)), and a REST API for submitting Landsat geotiff imagery for center pivot detection. 

See [matterport's mrcnn repo](https://github.com/matterport/Mask_RCNN) for an explanation of the Mask R-CNN architecture and a general guide to notebook tutorials and notebooks for inspecting model inputs and outputs.

For an overview of the project in poster form, see this poster I presented at the Fall 2018 Meeting on [Center Pivot Crop Water Use](assets/cropmask_agu2018.pdf). 



### See `terraform/` folder for instructions to isntall the package and set up an Azure workstation for remote development.
