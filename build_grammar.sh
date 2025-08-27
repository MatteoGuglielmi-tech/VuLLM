#!/bin/bash
set -e

PACKAGE_DIR="./libs/tree-sitter-extended-c"
GRAMMAR_DIR="${PACKAGE_DIR}/src/tree_sitter_extended_c/tree-sitter-c"

# --- Clean everything ---
echo "🧹 Starting clean build process..."
echo "--> Removing old build artifacts..."
rm -f "$GRAMMAR_DIR/src/parser.c"
rm -f "$GRAMMAR_DIR/src/tree_sitter/parser.h"
rm -rf "$GRAMMAR_DIR/build"
rm -rf "${PACKAGE_DIR}/build" "${PACKAGE_DIR}/dist" "${PACKAGE_DIR}"/*.egg-info
find . -type d -name "__pycache__" -exec rm -rf {} +

# --- Rebuild everything ---
echo "🚀 Starting the build process...🚀"

echo " --> 🚧 Generating 'parser.c' in '${GRAMMAR_DIR}/src/tree_sitter_extended_c/src'...🚧"
(cd "$GRAMMAR_DIR" && tree-sitter generate)
echo " 🎉 'parser.c' generated 🎉"

echo " --> 🚧 Installing Python package from '${PACKAGE_DIR}'...🚧"
(cd "$PACKAGE_DIR" && pip install --no-cache-dir -e .)
echo " 🎉 Python package installer and library generated 🎉"

echo "✅ Process completed successfully! ✅"
