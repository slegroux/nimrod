#!/usr/bin/env bash
#
# Re-fetch datasets that used to be committed to git.
# Data is intentionally NOT tracked (see .gitignore); this script restores it.
#
# Usage:
#   scripts/download_data.sh            # text + MNIST + hub image datasets
#   scripts/download_data.sh --audio    # also fetch LibriSpeech + LJSpeech (large)
#   scripts/download_data.sh --all      # everything
#
# Idempotent: existing files are left untouched.
#
# PROVENANCE
# Every source below is pinned: raw files to an upstream commit SHA and verified
# by sha256, hub datasets to a dataset-repo commit SHA. Unpinned fetches are
# not reproducible -- upstream can move a branch under you and the data silently
# changes between runs. The hub revisions here are the same ones recorded in
# config/**/*.yaml as `revision:`; if you change one, change both.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"

# --- pinned upstream commits -------------------------------------------------
CHAR_RNN_SHA=6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e
MAKEMORE_SHA=988aa59e4d8fefa526d06f3b453ad116258398d4

# --- pinned hub dataset revisions (keep in sync with config/**/*.yaml) --------
REV_MNIST=77f3279092a1c1579b2250db8eafed0ad422088c
REV_CIFAR10=0b2714987fa478483af9968de7c934580d0bb9a2
REV_FASHION_MNIST=531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2
REV_BUTTERFLIES=3cdedf844922ab40393d46d4c7f81c596e1c6d45
REV_TINY_IMAGENET_CLEAN=4b908d89fab3eb36aa8ebcd41c1996b28da7d6f2
REV_TINY_IMAGENET_ZH=5a77092c28e51558c5586e9c5eb71a7e17a5e43f

WITH_AUDIO=0
case "${1:-}" in
  --audio|--all) WITH_AUDIO=1 ;;
  "" ) ;;
  * ) echo "unknown arg: $1" >&2; exit 2 ;;
esac

# fetch <url> <dest> [expected_sha256]
# With a checksum, a mismatch is fatal and the bad file is removed.
# Without one, the computed digest is printed so it can be pinned here later.
fetch() {
  local url="$1" dest="$2" want="${3:-}"
  if [[ -s "$dest" ]]; then
    echo "  exists, skip: ${dest#$ROOT/}"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  echo "  downloading: ${dest#$ROOT/}"
  curl -fsSL "$url" -o "$dest"
  local got
  got="$(shasum -a 256 "$dest" | cut -d' ' -f1)"
  if [[ -n "$want" ]]; then
    if [[ "$got" != "$want" ]]; then
      rm -f "$dest"
      echo "  CHECKSUM MISMATCH for ${dest#$ROOT/}" >&2
      echo "    expected $want" >&2
      echo "    got      $got" >&2
      exit 1
    fi
    echo "    sha256 ok"
  else
    echo "    sha256 $got  (unpinned -- add it to this script to enforce)"
  fi
}

echo "==> Text datasets (pinned commits, checksum-verified)"
fetch "https://raw.githubusercontent.com/karpathy/char-rnn/$CHAR_RNN_SHA/data/tinyshakespeare/input.txt" \
      "$DATA/text/tiny_shakespeare.txt" \
      86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
fetch "https://raw.githubusercontent.com/karpathy/makemore/$MAKEMORE_SHA/names.txt" \
      "$DATA/text/names.txt" \
      0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d

echo "==> MNIST (via torchvision, downloads to data/image + recipes/image/mnist/data)"
if python -c "import torchvision" 2>/dev/null; then
  python - <<'PY'
from torchvision.datasets import MNIST
for root in ("data/image", "recipes/image/mnist/data"):
    MNIST(root, train=True, download=True)
    MNIST(root, train=False, download=True)
print("  MNIST ready")
PY
else
  echo "  SKIP: torchvision not installed. Install project deps first (pip install -e .), then re-run."
fi

echo "==> Hub image datasets (pinned revisions)"
if python -c "import datasets" 2>/dev/null; then
  REV_MNIST="$REV_MNIST" \
  REV_CIFAR10="$REV_CIFAR10" \
  REV_FASHION_MNIST="$REV_FASHION_MNIST" \
  REV_BUTTERFLIES="$REV_BUTTERFLIES" \
  REV_TINY_IMAGENET_CLEAN="$REV_TINY_IMAGENET_CLEAN" \
  REV_TINY_IMAGENET_ZH="$REV_TINY_IMAGENET_ZH" \
  DATA_DIR="$DATA/image" python - <<'PY'
import os
from datasets import load_dataset

CACHE = os.environ["DATA_DIR"]
SETS = [
    ("mnist",                                 os.environ["REV_MNIST"]),
    ("cifar10",                               os.environ["REV_CIFAR10"]),
    ("fashion_mnist",                         os.environ["REV_FASHION_MNIST"]),
    ("huggan/smithsonian_butterflies_subset", os.environ["REV_BUTTERFLIES"]),
    ("slegroux/tiny-imagenet-200-clean",      os.environ["REV_TINY_IMAGENET_CLEAN"]),
    ("zh-plus/tiny-imagenet",                 os.environ["REV_TINY_IMAGENET_ZH"]),
]
for name, rev in SETS:
    print(f"  {name} @ {rev[:12]}")
    load_dataset(name, revision=rev, cache_dir=CACHE,
                 download_mode="reuse_dataset_if_exists")
print("  hub image datasets ready")
PY
else
  echo "  SKIP: 'datasets' not installed. Install project deps first (pip install -e .), then re-run."
fi

if [[ "$WITH_AUDIO" -eq 1 ]]; then
  # These two are large; their digests are printed on first fetch rather than
  # guessed here. Pin them by pasting the printed value into the fetch calls.
  echo "==> LibriSpeech dev-clean (~337 MB)"
  if [[ ! -d "$DATA/en/LibriSpeech/dev-clean" ]]; then
    tmp="$(mktemp -t librispeech.XXXX.tar.gz)"
    fetch "https://www.openslr.org/resources/12/dev-clean.tar.gz" "$tmp"
    mkdir -p "$DATA/en"
    tar -xzf "$tmp" -C "$DATA/en"
    rm -f "$tmp"
  else
    echo "  exists, skip"
  fi

  echo "==> LJSpeech-1.1 (~2.6 GB)"
  if [[ ! -d "$DATA/en/LJSpeech-1.1/wavs" ]]; then
    tmp="$(mktemp -t ljspeech.XXXX.tar.bz2)"
    fetch "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2" "$tmp"
    mkdir -p "$DATA/en"
    tar -xjf "$tmp" -C "$DATA/en"
    rm -f "$tmp"
  else
    echo "  exists, skip"
  fi
else
  echo "==> Skipping audio corpora (LibriSpeech, LJSpeech). Run with --audio to fetch."
fi

echo "Done. Data restored under $DATA"
