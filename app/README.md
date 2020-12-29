# PyTorch example

This deploys a PyTorch model via an AI for Earth container. It serves a custom trained detectron2 Mask R-CNN model meant to run inference on 8-bit encoded, georeferenced Landsat tiffs.

## Download the model

The Mask R-CNN model and config yaml can be downloaded [here]().

Place these two files in the `pytorch_api` folder, which will be copied to the Docker container (see the `COPY` commands in `Dockerfile`). There are other ways of accessing a model, such as placing it in a Azure blob storage container (a unit of blob storage, do not confuse with Docker _containers_) and mount that blob container.

## Using the Service

nvidia-docker is required on the machine running the gpu enabled docker container. See this faq and repo for info: https://github.com/NVIDIA/nvidia-docker/wiki

The dev image runs a jupyter notebook and sets up an image with more python libraries for testing.

Build the docker image:
```
docker build -f Dockerfile-dev -t pytorchapp-dev .
```

Run dev image locally, mounting the host folder to the container to have access to notebooks, data downloads, models etc.:
```
cd /path/to/CropMask_RCNN
docker run -it -p 8888:8888 --runtime=nvidia --mount type=bind,source="$(pwd)",target=/app/test pytorchapp-dev
```

Run an instance of this image interactively and start bash to debug:
```
docker run -it --entrypoint /bin/bash --runtime=nvidia pytorchapp-dev
```

The production runs a webserver and includes minimal dependencies to do inference and convert predictions into vector format.

For the production image

Build the docker image:
```
docker build -f Dockerfile-prod -t pytorchapp-prod .
```

Run production image locally:
```
docker run -it --runtime=nvidia pytorchapp-prod
```

Run an instance of this image interactively and start bash to debug:
```
docker run -it --runtime=nvidia --entry-point /bin/bash pytorchapp-prod
```


## Testing and calling the service

Testing locally, the end point would be at

```
http://localhost:8081/v1/pytorch_api/classify
```

You can use a tool like Postman to test the end point:

![Calling the API](../screenshots/postman_pytorch_api.png)

