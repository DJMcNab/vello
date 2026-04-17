#!/bin/sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

curl -L -o "$SCRIPT_DIR/SourceSerif4Variable-Roman.ttf" \
    "https://github.com/adobe-fonts/source-serif/raw/refs/heads/release/VAR/SourceSerif4Variable-Roman.ttf"

curl -L -o "$SCRIPT_DIR/Arimo.ttf" \
    "https://github.com/google/fonts/raw/refs/heads/main/apache/arimo/Arimo%5Bwght%5D.ttf"
