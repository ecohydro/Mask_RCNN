start:
	docker run --name pytorchapp-prod-test -d -p 8081:80 --runtime=nvidia --mount type=bind,source="/home/rave/CropMask_RCNN/app/pytorch_api",target=/app/pytorch_api pytorchapp-prod
	# hot reloading doesn't seem to work, uwsgi can be touche dif folder is mounted but this doesn't trigger tocuh-reload in supervisord.conf
	# watchman-make -p '**/*.py' -s 1 --run 'touch /home/rave/CropMask_RCNN/app/pytorch_api/uwsgi-reload'
reload:
	docker stop pytorchapp-prod-test
	docker container rm pytorchapp-prod-test
	docker run --name pytorchapp-prod-test -d -p 8081:80 --runtime=nvidia --mount type=bind,source="/home/rave/CropMask_RCNN/app/pytorch_api",target=/app/pytorch_api pytorchapp-prod
stop:
	docker stop pytorchapp-prod-test
	docker container rm pytorchapp-prod-test
	# watchman watch-del-all
logs:
	docker container logs --follow pytorchapp-prod-test
interact:
	docker exec -it pytorchapp-prod-test /bin/bash