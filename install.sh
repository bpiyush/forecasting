# Remove the old environment
export CUDA_HOME=/apps/software/software/CUDA/11.3.1/
conda deactivate
conda env remove -n ego4d_forecasting

# Create new environment
conda create -n ego4d_forecasting python=3.8
conda activate ego4d_forecasting

# Install PyTorch with CUDA 11.3 support first
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# Downgrade pip temporarily to install pytorch-lightning 1.5.6 (from your original requirements)
pip install pip==23.3.2

# Install pytorch-lightning 1.5.6 (your original version) with --no-deps to prevent torch upgrade
pip install pytorch-lightning==1.5.6 --no-deps

# Install pytorch-lightning dependencies manually (excluding torch/torchvision)
pip install tensorboard>=2.2.0
pip install fsspec[http]>=2021.05.0
pip install packaging>=17.0
pip install PyYAML>=5.4
pip install tqdm>=4.57.0

# Install other requirements
pip install pytorchvideo==0.1.5
pip install editdistance==0.6.0
pip install scikit-learn==0.24.2
pip install psutil==5.9.0
pip install opencv-python==4.5.3.56
pip install einops==0.3.0
pip install decord==0.6.0
pip install lmdb==1.2.1
pip install imutils==0.5.4
pip install submitit==1.3.3
pip install pandas==1.3.5

# Install detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Upgrade pip back to latest
pip install --upgrade pip
