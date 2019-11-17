    # run if no workspace, make sure to publish container to registry
    az ml workspace show -w CropMask_RCNN_Workspace -g cropmaskresource_grouphz --query containerRegistry
    az acr login --name cropmaskcontainers 
    az acr update -n cropmaskcontainers --admin-enabled true
    docker tag image_tag_on_local cropmaskcontainers.azurecr.io/remote_tag
    docker push cropmaskcontainers.azurecr.io/remote_tag