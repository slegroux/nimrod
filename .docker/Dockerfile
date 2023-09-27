# FROM condaforge/mambaforge
FROM mambaorg/micromamba:jammy-cuda-12.1.0
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
     apt-get install -y --no-install-recommends \
     build-essential \
     git \
     curl \
     ca-certificates \     
     wget \
     pkg-config && \
     # Remove the effect of `apt-get update`
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

WORKDIR /workspace
COPY environment.yml .

RUN micromamba install --yes --name base -f environment.yml
RUN micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN nbdev_install_quarto
RUN git clone https://github.com/slegroux/nimrod