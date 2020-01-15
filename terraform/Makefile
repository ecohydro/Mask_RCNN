# Settings

PATH_SSH_KEY=~/.ssh/id_rsa
REMOTE_DIR=~/work
LOCAL_DIR=.
PATH_VM_ID=./terraform/.vm-id
PATH_VM_IP=./terraform/.vm-ip
PATH_VM_ID_GPU=./terraform/.vm-id-gpu
PATH_VM_IP_GPU=./terraform/.vm-ip-gpu
PATH_VM_ID_MGPU=./terraform/.vm-id-mgpu
PATH_VM_IP_MGPU=./terraform/.vm-ip-mgpu

##### You should not have to touch stuff below this line

VM_ID=$(shell cat ${PATH_VM_ID}) # Azure resource ID
VM_CONN=$(shell cat ${PATH_VM_IP}) # user@IP

mkfile_path=$(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR=$(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

# sync code disabled strict key checking for syncup for terraform build
syncup:
	rsync -e "ssh -i ${PATH_SSH_KEY} -o StrictHostKeyChecking=no" -avz --exclude=".git/" --exclude-from=.gitignore ${LOCAL_DIR} $(shell echo ${VM_CONN}:${REMOTE_DIR}/${CURRENT_DIR} | tr -d '[:space:]')
syncdown:
	rsync -e "ssh -i ${PATH_SSH_KEY}" -avz --exclude=".git/" --exclude-from=.gitignore $(shell echo ${VM_CONN}:${REMOTE_DIR}/${CURRENT_DIR}/ | tr -d '[:space:]') ${LOCAL_DIR}
# start/stop instance
stop:
	az vm deallocate --ids ${VM_ID}
start:
	az vm start --ids ${VM_ID}
status:
	az vm show -d --ids ${VM_ID} | grep "powerState" | cut -d\" -f4

# ssh into machine and forward jupyter port
ssh:
	ssh -i ${PATH_SSH_KEY} -L 8888:localhost:8888 ${VM_CONN}

VM_ID_GPU=$(shell cat ${PATH_VM_ID_GPU}) # Azure resource ID
VM_CONN_GPU=$(shell cat ${PATH_VM_IP_GPU}) # user@IP

# gpu commands
syncup-gpu:
		rsync -e "ssh -i ${PATH_SSH_KEY} -o StrictHostKeyChecking=no" -avz --exclude=".git/" --exclude-from=.gitignore ${LOCAL_DIR} $(shell echo ${VM_CONN_GPU}:${REMOTE_DIR}/${CURRENT_DIR} | tr -d '[:space:]')
syncdown-gpu:
		rsync -e "ssh -i ${PATH_SSH_KEY}" -avz --exclude=".git/" --exclude-from=.gitignore $(shell echo ${VM_CONN_GPU}:${REMOTE_DIR}/${CURRENT_DIR}/ | tr -d '[:space:]') ${LOCAL_DIR}
# start/stop instance
stop-gpu:
		az vm deallocate --ids ${VM_ID_GPU}
start-gpu:
		az vm start --ids ${VM_ID_GPU}
status-gpu:
		az vm show -d --ids ${VM_ID_GPU} | grep "powerState" | cut -d\" -f4

# ssh into machine and forward jupyter port
ssh-gpu:
		ssh -i ${PATH_SSH_KEY} -L 8888:localhost:8888 -L 6006:localhost:6006 -L 8787:localhost:8787 ${VM_CONN_GPU}

VM_ID_MGPU=$(shell cat ${PATH_VM_ID_MGPU}) # Azure resource ID
VM_CONN_MGPU=$(shell cat ${PATH_VM_IP_MGPU}) # user@IP

# gpu commands
syncup-mgpu:
		rsync -e "ssh -i ${PATH_SSH_KEY} -o StrictHostKeyChecking=no" -avz --exclude=".git/" --exclude-from=.gitignore ${LOCAL_DIR} $(shell echo ${VM_CONN_MGPU}:${REMOTE_DIR}/${CURRENT_DIR} | tr -d '[:space:]')

syncdown-mgpu:
		rsync -e "ssh -i ${PATH_SSH_KEY}" -avz --exclude=".git/" --exclude-from=.gitignore $(shell echo ${VM_CONN_MGPU}:${REMOTE_DIR}/${CURRENT_DIR}/ | tr -d '[:space:]') ${LOCAL_DIR}
# start/stop instance
stop-mgpu:
		az vm deallocate --ids ${VM_ID_MGPU}
start-mgpu:
		az vm start --ids ${VM_ID_MGPU}
status-mgpu:
		az vm show -d --ids ${VM_ID_MGPU} | grep "powerState" | cut -d\" -f4

# ssh into machine and forward jupyter port
ssh-mgpu:
		ssh -i ${PATH_SSH_KEY} -L 8888:localhost:8888 ${VM_CONN_MGPU}


