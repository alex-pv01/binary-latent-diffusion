ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX" \
        && export TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
        && export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# INIT OF FIX FOR GPG KEY -- Hopefully they will fix this soon
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# END OF FIX FOR GPG KEY

# apt and apt-get update
RUN apt-get update && apt update && apt upgrade -y && apt-get clean 

# Install dependencies
RUN apt-get install -y git
RUN pip install matplotlib 

# Install BLD model
#RUN echo "Installing BLD model..." \
#	&& git clone https://github.com/alex-pv01/binary-latent-diffusion.git 