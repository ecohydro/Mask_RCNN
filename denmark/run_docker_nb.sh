docker run -p 8888:8888 --gpus all 497 jupyter lab --ip=0.0.0.0 --no-browser --allow-root
#requires nvidia-docker-toolkit and the bimage to be built
# port forwarding assumes that the Azure vm is forwarding on 8888 if it is used from an azure vm
