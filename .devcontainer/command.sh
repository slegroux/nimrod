#!/usr/bin/env bash
# -u 0 gives root access to the container
# docker run -it -u 0 --rm bitnami/pytorch:2.3.1 /bin/bash

apt-get update
apt-get install -y --no-install-recommends espeak-ng \
    espeak-ng \
    git-lfs \
    libsndfile-dev \
    curl 
apt-get clean
rm -rf /var/lib/apt/lists/*

# install nimrod
git clone https://github.com/slegroux/nimrod
cd nimrod
pip install -e .
# alternative install from python package
# pip install slg-nimrod==0.0.11

# spacy requirements for en
python -m spacy download en_core_web_sm
# nbdev stuff
nbdev_install_hooks && nbdev_install_quarto
# prevent some cloud GPU providers to automatically start tmux on login
touch ~/.no_auto_tmux;