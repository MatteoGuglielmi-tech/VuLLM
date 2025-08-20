#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

PACKAGE_DIR="./libs/tree-sitter-extended-c"
GRAMMAR_DIR="${PACKAGE_DIR}/src/tree_sitter_extended_c/tree-sitter-c"

echo "🚀 Starting the build process...🚀"
echo " 🚧 Generating 'parser.c' in '${GRAMMAR_DIR}/src/tree_sitter_extended_c/src'...🚧"
(cd "$GRAMMAR_DIR" && tree-sitter generate)
echo " 🎉 'parser.c' generated 🎉"

echo " 🚧 Installing Python package from '${PACKAGE_DIR}'...🚧"
(pip install --no-cache-dir -e ${PACKAGE_DIR})

echo " 🎉 Python package installer and library generated 🎉"

echo "✅ Process completed successfully! ✅"
