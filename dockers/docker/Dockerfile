ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX" \
        && export TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
        && export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# apt and apt-get update
RUN apt-get update && apt update && apt upgrade -y && apt-get clean
# apt and apt-get installations
RUN apt install -y vim git unzip \
	&& apt-get install -y libosmesa6-dev build-essential libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev

# Install BLD model
RUN echo "Installing BLD model..." \
	&& conda clean --all \
	&& git clone https://github.com/alex-pv01/binary-latent-diffusion.git \
	&& cd binary-latent-diffusion \
	&& conda env create -f environment.yml \
        && conda activate bld-env