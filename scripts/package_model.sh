#!/usr/bin/env bash
# Collect the three files the web service needs into service/model_files/.
# Run scripts/evaluate.py first (it creates artifacts/model.int8.onnx).
set -euo pipefail
cd "$(dirname "$0")/.."
dest=service/model_files
mkdir -p "$dest"
cp artifacts/model.int8.onnx model/saved_model/tokenizer.json model/saved_model/metadata.json "$dest/"
ls -lh "$dest"
