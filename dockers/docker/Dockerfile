ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX" \
        && export TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
        && export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/* \
        && pip install cython xtcocotools \
        && pip install mmcv-full==latest+torch${PYTORCH}+cu${CUDA} -f https://download.openmmlab.com/mmcv/dist/index.html


RUN echo "hello"