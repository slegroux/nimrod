#!/usr/bin/env bash
#
# Re-fetch datasets that used to be committed to git.
# Data is intentionally NOT tracked (see .gitignore); this script restores it.
#
# Usage:
#   scripts/download_data.sh            # fetch small datasets (text + MNIST)
#   scripts/download_data.sh --audio    # also fetch LibriSpeech + LJSpeech (large)
#   scripts/download_data.sh --all      # everything
#
# Idempotent: existing files are left untouched.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"

WITH_AUDIO=0
case "${1:-}" in
  --audio|--all) WITH_AUDIO=1 ;;
  "" ) ;;
  * ) echo "unknown arg: $1" >&2; exit 2 ;;
esac

fetch() {  # fetch <url> <dest>
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "  exists, skip: ${dest#$ROOT/}"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  echo "  downloading: ${dest#$ROOT/}"
  curl -fsSL "$url" -o "$dest"
}

echo "==> Text datasets"
fetch "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
      "$DATA/text/tiny_shakespeare.txt"
fetch "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt" \
      "$DATA/text/names.txt"

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

if [[ "$WITH_AUDIO" -eq 1 ]]; then
  echo "==> LibriSpeech dev-clean (~337 MB)"
  if [[ ! -d "$DATA/en/LibriSpeech/dev-clean" ]]; then
    tmp="$(mktemp -t librispeech.XXXX.tar.gz)"
    curl -fsSL "https://www.openslr.org/resources/12/dev-clean.tar.gz" -o "$tmp"
    mkdir -p "$DATA/en"
    tar -xzf "$tmp" -C "$DATA/en"
    rm -f "$tmp"
  else
    echo "  exists, skip"
  fi

  echo "==> LJSpeech-1.1 (~2.6 GB)"
  if [[ ! -d "$DATA/en/LJSpeech-1.1/wavs" ]]; then
    tmp="$(mktemp -t ljspeech.XXXX.tar.bz2)"
    curl -fsSL "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2" -o "$tmp"
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
