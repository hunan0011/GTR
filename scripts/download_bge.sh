#!/bin/bash

set -e

MODEL_NAME="BAAI/bge-base-en-v1.5"
MODEL_DIR="./models/bge-base-en-v1.5"

echo "=========================================="
echo "Downloading dual-encoder model"
echo "Model: ${MODEL_NAME}"
echo "Save to: ${MODEL_DIR}"
echo "=========================================="

mkdir -p "${MODEL_DIR}"

if ! command -v huggingface-cli &> /dev/null
then
    echo "huggingface-cli is not found. Installing huggingface_hub..."
    pip install -U huggingface_hub
fi

huggingface-cli download "${MODEL_NAME}" \
  --local-dir "${MODEL_DIR}"

echo "=========================================="
echo "BGE encoder download completed."
echo "Model path: ${MODEL_DIR}"
echo "=========================================="