# FROM mambaorg/micromamba:jammy-cuda-12.1.0
# FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
# because of deps with torchtext will downgrade to torch 2.0.1
FROM bitnami/pytorch:2.3.1

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
    # spacy takes forever to install (wheels not available)
    pip install spacy && \
    python -m spacy download en_core_web_sm && \
    # Remove the effect of `apt-get update`
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/kpu/kenlm && \
#     cd kenlm; mkdir build; cd build && \
#     cmake ..; make -j 4; make install

# taking care of dependencies between torch/torchtext & lightning/numpy
# RUN pip install torchtext torchvision torchaudio torchdata jupyterlab nbdev numpy lightning

# RUN python -m spacy download en_core_web_sm
# #&& git clone https://github.com/slegroux/nimrod
# WORKDIR /workspaces/nimrod
COPY . .
RUN pip install -e .
RUN nbdev_install_quarto && nbdev_install_hooks
# # RUN git config --global --add safe.directory /workspaces/nimrod

# EXPOSE 8888
# CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''","--NotebookApp.password=''"]
