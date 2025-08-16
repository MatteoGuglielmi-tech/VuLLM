from setuptools import setup, Extension
import os
from pathlib import Path


BASE_PATH: Path = Path("src") / "tree_sitter_extended_c" / "tree-sitter-c" / "src"

# define the C sources to be compiled from the submodule
sources = [str(BASE_PATH / "parser.c")]
include_dirs = [str(BASE_PATH)]


extended_c_parser = Extension(
    "tree_sitter_extended_c.parser",
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args=(
        ["-std=c11", "-Wno-unused-variable"]  # use GCC/Clang specific flags on non-Windows systems
        if os.name != "nt"
        else ["/std:c11"]  # Use MSVC specific flags on Windows
    ),
)

# minimal setup call that only provides the C extension
setup(ext_modules=[extended_c_parser])
