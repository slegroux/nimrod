FROM mambaorg/micromamba:jammy-cuda-12.1.0
LABEL maintainer="Sylvain Le Groux <sylvain.legroux@gmail.com>"
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
     apt-get install -y --no-install-recommends \
     build-essential \
     libboost-all-dev \
     libeigen3-dev \
     cmake \
     zlib1g-dev libbz2-dev liblzma-dev \
     festival \
     espeak-ng \
     mbrola \
     git \
     curl \
     ca-certificates \     
     wget \
     pkg-config && \
     # Remove the effect of `apt-get update`
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/kpu/kenlm && \
    cd kenlm; mkdir build; cd build && \
    cmake ..; make -j 4; make install

USER $MAMBA_USER

WORKDIR /home/$MAMBA_USER
COPY environment.yml .
COPY .devcontainer/start_jupyterlab.sh .

RUN micromamba install --yes --name base -f environment.yml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python -m spacy download en_core_web_sm
RUN nbdev_install_quarto && git clone https://github.com/slegroux/nimrod
WORKDIR /home/$MAMBA_USER/nimrod
RUN pip install -e .
EXPOSE 8888
CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''","--NotebookApp.password=''"]
