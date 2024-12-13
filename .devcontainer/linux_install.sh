#!/usr/bin/env
git clone https://github.com/slegroux/nimrod
cd nimrod
pip install spacy
python -m spacy download en_core_web_sm
pip install -e .
nbdev_install_hooks
apt-get install git-lfs
# git lfs install 
git lfs fetch --all
git lfs checkout
