FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set bash as the default shell
ENV SHELL=/bin/bash

# Create a working directory
WORKDIR /app/

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    apt-utils \
    vim \
    git \
    cmake \
    libboost-all-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# kenlm
RUN git clone https://github.com/kpu/kenlm && \
    cd kenlm; mkdir build; cd build && \
    cmake ..; make -j"$(nproc)"; make install

# build with some basic python packages
RUN pip install \
    numpy \
    torch \
    jupyterlab \
    nbdev \
    slg-nimrod

RUN nbdev_install_quarto


EXPOSE 8888 6006
CMD jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True

# CMD /bin/bash