source ~/.bashrc
conda update -n base -c defaults conda -y
/data/anaconda/envs/py36/bin/conda env update -f ~/work/$1/cropmask-env.yml -y
cd ~/work/$1/
/data/anaconda/envs/py36/bin/python setup.py develop
conda install -c conda-forge jupyterlab=1.0.2 # updating to jupyterlab 1.x
