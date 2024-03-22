FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as deps

SHELL ["/bin/bash", "--login", "-c"]

# Step 1. Set up Ubuntu
RUN apt update && apt install --yes wget ssh git git-lfs vim build-essential gcc-12 ninja-build

# Step 1.5 Set up Cuda
RUN apt-get remove --purge --yes --allow-change-held-packages "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*"
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt install --yes cuda


# NOTE: libcuda.so.1 doesn't exist in NVIDIA's base image, link the stub file to work around
# RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
# Fixes mount error
RUN rm -rf umount /usr/lib/x86_64-linux-gnu/libcuda.so.1 && rm -rf /usr/lib/x86_64-linux-gnu/libcuda.so.1
RUN rm -rf umount /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 && rm -rf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
RUN rm -rf umount /usr/lib/x86_64-linux-gnu/libcudadebugger.so.1 && rm -rf /usr/lib/x86_64-linux-gnu/libcudadebugger.so.1

WORKDIR /root

# Step 2. Set up Conda
RUN wget -O miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
RUN bash miniconda.sh -b -p "/root/conda"
RUN rm -rf miniconda.sh

RUN echo "export PATH=/usr/local/cuda/bin/:/root/conda/bin:\$PATH" >> /root/.profile 
RUN echo "source /root/conda/etc/profile.d/conda.sh" >> /root/.profile
RUN conda init bash

# Step 3. Set up Python
RUN conda create --yes -n mlc python=3.12 && \
  echo "conda activate mlc" >> /root/.profile

FROM deps as compiler

WORKDIR /root

# See https://github.com/PanQiWei/AutoGPTQ/issues/194#issuecomment-1638480640
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX;8.9;9.0"

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install dlib cmake boto3 ftfy gdown imageio-ffmpeg matplotlib opencv-python-headless pandas scipy scikit-learn scikit-image sixdrepnet supervision timm tk ultralytics typing_extensions
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install ipython

RUN git clone https://github.com/tejaspinjala/cosmetic-previewer-using-GANs.git
WORKDIR /root/cosmetic-previewer-using-GANs/GanModel
RUN chmod +x ./bash.sh

CMD ./bash.sh