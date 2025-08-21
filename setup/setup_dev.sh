#!/bin/bash

set -e

# -- directory setup --
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."
echo "▶️  Changed working directory to project root: $(pwd)"

# -- environment setup --
CONDA_ENV_NAME="llm"
PYTORCH_CMD=(pip3 install -v torch torchvision --index-url https://download.pytorch.org/whl/cu129)
BUILD_GRAMMAR_CMD=(sh build_grammar.sh)

# check existance of `CONDA_ENV_NAME`
if conda info --envs | grep -wq "^$CONDA_ENV_NAME\s"; then
    echo "▶️  Conda environment '$CONDA_ENV_NAME' already exists."
else
    echo "▶️  Creating conda environment '$CONDA_ENV_NAME' with Python 3.13..."
    conda create -n $CONDA_ENV_NAME python=3.13 -y
fi

# -- run installation commands --
echo "▶️  Installing PyTorch dependencies inside '$CONDA_ENV_NAME'..."
conda run -v -n "$CONDA_ENV_NAME" "${PYTORCH_CMD[@]}"

echo "▶️  Installing project in editable mode inside '$CONDA_ENV_NAME'..."
conda run -v -n $CONDA_ENV_NAME pip install -e ".[dev]"

echo "▶️  Building the tree-sitter grammar..."
conda run -v -n $CONDA_ENV_NAME "${BUILD_GRAMMAR_CMD[@]}"

# -- log ending --
echo "✅  Development environment setup complete!"
echo "➡️  To activate, run: conda activate $CONDA_ENV_NAME"
