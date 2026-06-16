#!/usr/bin/env bash
# Optional native extras for nimrod's LM (kenlm) and speech (espeak / phonemizer)
# recipes. SKIP this unless you actually run those recipes — the core image and
# audio recipes do not need it. Safe to re-run. Requires Homebrew.
#
# Run with the `nimrod` env active so `pip` targets the right interpreter.
set -eu

if ! command -v brew >/dev/null 2>&1; then
  echo "error: Homebrew not found. Install from https://brew.sh first." >&2
  exit 1
fi

# --- espeak: required by `phonemizer` (TTS / text normalizers) ---------------
brew list espeak >/dev/null 2>&1 || brew install espeak
ESPEAK_PREFIX="$(brew --prefix espeak)"
ESPEAK_LIB="$(find "$ESPEAK_PREFIX/lib" -name 'libespeak*.dylib' 2>/dev/null | head -1 || true)"
if [ -n "$ESPEAK_LIB" ]; then
  echo
  echo "espeak library found: $ESPEAK_LIB"
  echo "Add this to your shell profile so phonemizer can find it:"
  echo "  export PHONEMIZER_ESPEAK_LIBRARY=\"$ESPEAK_LIB\""
else
  echo "warning: could not locate libespeak*.dylib under $ESPEAK_PREFIX/lib" >&2
fi

# --- kenlm: required by nimrod/models/ngram.py -------------------------------
# Python bindings only — enough to *query* an n-gram LM. Needs boost to build.
brew list boost >/dev/null 2>&1 || brew install boost
pip install "https://github.com/kpu/kenlm/archive/master.zip"

# To also build the kenlm CLI tools (lmplz / build_binary) for *training* n-grams
# (no sudo needed — binaries stay under the build dir, add it to PATH):
#   git clone https://github.com/kpu/kenlm /tmp/kenlm
#   cmake -S /tmp/kenlm -B /tmp/kenlm/build && cmake --build /tmp/kenlm/build -j
#   export PATH="/tmp/kenlm/build/bin:$PATH"

echo
echo "Done. Optional LM/speech native extras installed."
