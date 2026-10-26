#!/usr/bin/env
git clone https://github.com/slegroux/nimrod
cd nimrod
pip install -e .
nbdev_install_hooks
apt-get install git-lfs
# git lfs install 
git lfs fetch --all
git lfs checkout
