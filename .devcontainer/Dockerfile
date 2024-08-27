FROM docker.io/bitnami/pytorch:2.3.1
LABEL maintainer="Sylvain Le Groux <sylvain.legroux@gmail.com>"
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
    # build-essential \
    espeak-ng \
    git-lfs \
    libsndfile-dev \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install slg-nimrod==0.0.11
RUN python -m spacy download en_core_web_sm
RUN nbdev_install_hooks
# RUN nbdev_install_quarto && \
