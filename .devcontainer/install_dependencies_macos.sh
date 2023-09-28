#!/usr/bin/env bash

# KENLM
brew install boost
cd /tmp
git clone https://github.com/kpu/kenlm
cd kenlm
mkdir -p build
cd build
cmake ..
make -j 8
sudo make install

# SPACY
python -m spacy download en_core_web_sm

# PHONEMIZER
brew install espeak
export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib