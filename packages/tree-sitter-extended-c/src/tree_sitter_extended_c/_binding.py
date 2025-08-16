import ctypes
from importlib.resources import files
from importlib.resources.abc import Traversable

# build parser path
lib_path: str = ""
package_files: Traversable = files("tree_sitter_extended_c")
for f in package_files.iterdir():
    if f.name.startswith("parser") and f.name.endswith(".so"):
        lib_path = str(f)
        break

if not lib_path: raise FileNotFoundError("Could not find compiled parser.so library.")

# load library and get language function symbol
lib: ctypes.CDLL = ctypes.cdll.LoadLibrary(lib_path)
language_symbol = lib.tree_sitter_c
language_symbol.restype = ctypes.c_void_p # set return type to C pointer

def language():
    """Calls the C function and returns the integer pointer to the language."""
    return language_symbol()
