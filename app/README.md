# PyTorch example

This deploys a PyTorch model via an AI for Earth container. It serves a custom trained detectron2 Mask R-CNN model meant to run inference on 8-bit encoded, georeferenced Landsat tiffs.

## Download the model

The Mask R-CNN model and config yaml can be downloaded [here]().

Place these two files in the `pytorch_api` folder, which will be copied to the Docker container (see the `COPY` commands in `Dockerfile`). There are other ways of accessing a model, such as placing it in a Azure blob storage container (a unit of blob storage, do not confuse with Docker _containers_) and mount that blob container.

## Using the Service

nvidia-docker is required on the machine running the gpu enabled docker container. See this faq and repo for info: https://github.com/NVIDIA/nvidia-docker/wiki

The AI for Earth base-py folder needs to be checked out from this commit: https://github.com/microsoft/AIforEarth-API-Development/blob/2afd4df1e8ca8d1f2f975d067991e40411324fd5/Containers/base-py/Dockerfile

Since this properly sets up conda. Then, a more recent conda installer needs to be used to avoid conda import errors during build. Finally, a 10.1 cuda base image needs to be used to rebuild the AI for earth base py docker container, which is used as a base image for the pytorch app.

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

You can test an example request with `example_reguest.ipynb` and test the model and server side code interactively in `testpytorchapi.ipynb`. Outputs will currently be saved to the pytorch_api folder on the host and the production container.

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
docker run -it -p 8081:80 --runtime=nvidia pytorchapp-prod 
```

Run an instance of this image interactively for debugging. stop it with CNTRL+C and restarrt it to remount and changed server side code.:
```
docker run -it --mount type=bind,source="$(pwd)/pytorch_api",target=/app/pytorch_api -p 8081:80 --runtime=nvidia pytorchapp-prod

```


## Testing and calling the service

Testing locally, the end point would be at

```
http://localhost:8081/v1/pytorch_api/classify
```

You can use a tool like Postman to test the end point:

![Calling the API](../screenshots/postman_pytorch_api.png)

