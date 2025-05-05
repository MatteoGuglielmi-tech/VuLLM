#!/bin/bash
# this script is used to install all dependencies for the current project to run

looking_for_deps() {
	builtin echo "Checking for dependencies file"
	if [ ! -f deps.yml ]; then
		builtin echo "Error: deps.yml not found"
		return 1
	else
		builtin echo "deps.yml file found!"
		return 0
	fi
}

looking_for_conda() {
	builtin echo "Running checks for conda instance"
	if command -V conda >&2; then
		builtin echo "conda instance is ready"
		return 0
	else
		builtin echo "Error: command not found"
		return 1
	fi
}

pip_path="$CONDA_PREFIX/envs/prova/bin/pip3"
looking_for_pip() {
	builtin echo "Running checks for pip3 instance"
	if command -V "$pip_path" >&2; then
		builtin echo "pip3 setup for current env"
		return 0
	else
		builtin echo "Error: command not found"
		return 1
	fi
}

main() {
	set -e
	if (looking_for_conda && looking_for_deps); then
		builtin echo "$1 environment"
		if [ "$1" = "Creating" ]; then
			conda env create --file deps.yml
		else
			conda env update --file deps.yml --prune
		fi

		if (looking_for_pip); then
			builtin echo "Installing pip3 deps"
			__pip_install="$pip_path install tree-sitter-language-pack"
			eval "$__pip_install"
		fi
	fi
}

main "$1"
