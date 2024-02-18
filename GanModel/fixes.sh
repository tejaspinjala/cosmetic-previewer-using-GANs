echo "Cuda is incorrectly installed on aws ec2 (usr/local/cuda/bin folder doesnt exist). Reinstall using the link below to fix ninja issues"
echo "This issue also affects pip install inplace_abn"
echo "https://medium.com/@yulin_li/how-to-update-cuda-and-cudnn-on-ubuntu-18-04-4bfb762cf0b8"
echo "This only has to be done the first time torch is being installed into the conda env"

conda uninstall pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*"
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Fixes open cv issues on aws sagemaker instance
pip uninstall opencv-python
pip install opencv-python-headless

# Fixes gdown not downloadingg the model
pip install --upgrade --no-cache-dir gdown

# Fixes dlib issues
#conda install gxx_linux-64=12.1
#conda install -c conda-forge libstdcxx-ng
#pip install cmake
# pip install dlib