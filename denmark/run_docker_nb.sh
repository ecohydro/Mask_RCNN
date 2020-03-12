docker run -p 8888:8888 -v /home/ryan/InstanceSegmentation_Sentinel2:/home/code -v /datadrive/denmark-data:/home/data --gpus all fcis:latest jupyter lab --ip=0.0.0.0 --notebook-dir=/ --no-browser --allow-root
#requires nvidia-docker-toolkit and the bimage to be built
# port forwarding assumes that the Azure vm is forwarding on 8888 if it is used from an azure vm
