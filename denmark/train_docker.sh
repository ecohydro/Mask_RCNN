docker run -p 8888:8888 -v /home/ryan/InstanceSegmentation_Sentinel2:/home/code -v /datadrive/denmark-data:/home/data --gpus all fcis:latest bash /home/code/train.sh
#requires nvidia-docker-toolkit and the bimage to be built
# port forwarding assumes that the Azure vm is forwarding on 8888 if it is used from an azure vm
