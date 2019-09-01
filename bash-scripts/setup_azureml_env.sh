chmod 775 /usr/local
echo 'export PATH=/usr/local/miniconda/bin:$PATH' > /etc/profile.d/conda.sh
curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -bufp /usr/local
rm -rf /tmp/miniconda.sh

export PATH=/usr/local/bin:$PATH
export CONDA_AUTO_UPDATE_CONDA=false
export CONDA_DEFAULT_ENV=cropmask
export CONDA_PREFIX=/usr/local/envs/$CONDA_DEFAULT_ENV
export PATH=$CONDA_PREFIX/bin:$PATH

# Create a Python 3.6 environment
conda install anaconda-client python==3.6.6 -n base # necessary to run conda update later
conda install conda-build
conda create -y --name cropmask python=3.7.3
conda clean -ya
echo "source activate cropmask" >> ~/.bashrc