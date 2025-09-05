#!/bin/bash
set -e

# This script is intended to be run from the project root directory

# --- Activate Virtual Environment ---
echo "--- Activating virtual environment ---"
source ./.venv/bin/activate

# --- Change to script directory ---
cd scripts

# --- whisper.cpp setup ---
echo "--- Setting up whisper.cpp ---"
if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp.git
fi
cd whisper.cpp

echo "--- Building whisper.cpp with CUDA support ---"
# Remove previous build directory to ensure a clean build
rm -rf build
# Add -DGGML_CUDA=1 to enable Nvidia GPU support and -j for parallel build
cmake -B build -DGGML_CUDA=1
cmake --build build -j --config Release

MODEL_NAME="ggml-base.en.bin"
if [ ! -f "../../models/$MODEL_NAME" ]; then
    echo "--- Downloading Whisper model ($MODEL_NAME) ---"
    # The download script is now in the base directory
    ./models/download-ggml-model.sh base.en
    mv ./models/$MODEL_NAME ../../models/
else
    echo "Whisper model already exists. Skipping download."
fi
cd .. # back to scripts dir

# --- LLM setup ---
echo "--- Setting up LLM ---"
LLM_MODEL_NAME="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
if [ ! -f "../models/$LLM_MODEL_NAME" ]; then
    echo "--- Downloading LLM model ($LLM_MODEL_NAME) ---"
    LLM_URL="https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    wget -O "../models/$LLM_MODEL_NAME" "$LLM_URL"
else
    echo "LLM model already exists. Skipping download."
fi

# --- Python Dependencies ---
echo "--- Installing Python dependencies ---"
uv pip install -r ../requirements.txt

echo "--- Setup complete! ---"
