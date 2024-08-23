# FROM mambaorg/micromamba:jammy-cuda-12.1.0
# FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
# torch 2.4.0 torchaudio 2.4.0 torchtext 0.15.0
FROM bitnami/pytorch:latest 

LABEL maintainer="Sylvain Le Groux <sylvain.legroux@gmail.com>"
USER root

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    build-essential \
#     libboost-all-dev \
#     libeigen3-dev \
#     cmake \
#     zlib1g-dev libbz2-dev liblzma-dev \
#     festival \
    espeak-ng \
#     mbrola \
    git \
    git-lfs \
    curl \
    ca-certificates \     
    wget \
    pkg-config && \
    # Remove the effect of `apt-get update`
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/kpu/kenlm && \
#     cd kenlm; mkdir build; cd build && \
#     cmake ..; make -j 4; make install
# taking care of dependencies between torch/torchtext & lightning/numpy
RUN pip install jupyterlab nbdev torchvision torchaudio torchtext==0.15.2 torchdata numpy==1.26.4 lightning
RUN pip install spacy && python -m spacy download en_core_web_sm && nbdev_install_quarto
# RUN nbdev_install_quarto 
# #&& git clone https://github.com/slegroux/nimrod
# WORKDIR /workspaces/nimrod
COPY . .
RUN pip install -e .
# # RUN git config --global --add safe.directory /workspaces/nimrod

# EXPOSE 8888
# CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''","--NotebookApp.password=''"]
