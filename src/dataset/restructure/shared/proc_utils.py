import codecs
import json
import re
import shutil
import subprocess
from typing import Any

from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp

C_LANGUAGE = Language(tsc.language())
CPP_LANGUAGE = Language(tscpp.language())


def write2file(fp: str, content: str, encoding: str = "utf-8") -> None:
    """Utility to serialize string.

    Params:
        fp: str
            target filepath where conent will be written to.
        conent: str
            content to be serialized.
        encoding: Optional[str]
            encoding to be used during serialization.

    Returns:
        None
    """

    with open(file=fp, mode="w", encoding=encoding) as f:
        f.write(content)


def append2file(fp: str, content: str, encoding: str = "utf-8") -> None:
    with open(file=fp, mode="a", encoding=encoding) as f:
        f.write(content)


def read_file(fp: str, encoding: str = "utf-8", strip: bool = True) -> str:
    """Utility to read file content as string.

    Params:
        fp: str
            path of file to be read.
        encoding: Optional[str]
            encoding to be used for reading.
        strip: bool
            whether to strip conent.

    Returns:
        str:
            file content
    """

    with open(file=fp, mode="r", encoding=encoding) as f:
        return f.read().strip() if strip else f.read()


def read_lines(fp: str, encoding:str="utf-8") -> list[str]:
    """Utility to read file content as string.

    Params:
        fp: str
            path of file to be read.

    Returns:
        list[str]:
            file conent as list of lines
    """

    with open(file=fp, mode="r", encoding=encoding) as f:
        return f.readlines()


def load_config(fp:str) -> dict[str,Any]:
    with open(file=fp, mode="r") as f:
        return json.load(f)

def update_language_clang_format(fp: str, language_to_set: str) -> None:
    """Utility to update `Language` field of .clang-format file.

    Params:
        fp: str
            path of .clang-format config file.
        language_to_set: str
            target language

    Returns:
        None
    """

    # read current file content
    config_lines: list[str] = read_lines(fp=fp)
    with open(file=fp, mode="w") as f:
        for line in config_lines:
            if line.strip().startswith("Language:"):
                f.write(f"Language: {language_to_set.capitalize()}\n")
            else:
                f.write(line)


def decode_escaped_string(raw_string: str) -> str:
    """
    Decodes a string that contains C-style escape sequences from its
    raw, double-escaped form in the JSON file.

    Params:
        raw_string: str
            entry to be processed for escape removal

    Returns:
        str
            original string cleaned up from escape sequences
    """

    return codecs.decode(obj=raw_string, encoding="unicode_escape")


def is_cpp(code: str) -> bool:
    """Uses robust heuristics to detect if a code snippet is C++. It prioritizes
    fast regex checks for unambiguous C++ keywords and falls back to a more
    accurate tree-sitter parse for ambiguous syntax like references and
    default parameters.

    Params:
        code: str
            code to investigate
    Returns:
        bool:
            whether the function is written in C++.
    """
    # 1. Fast regex checks for unambiguous C++ keywords.
    cpp_patterns: list[re.Pattern] = [
        re.compile(r"\bnamespace\b"),
        re.compile(r"\b(public|protected|private)\s*:"),
        re.compile(r"\b\w+::\w+"),
        re.compile(r"\b(template)\s*<"),
        re.compile(r"\b(class)\s+\w+\s*\{"),
        re.compile(r"\b(try|catch|throw|new|delete)\b"),
    ]
    if any(pattern.search(code) for pattern in cpp_patterns):
        return True

    # 2. Slower, more accurate tree-sitter check for ambiguous syntax.
    try:
        parser = Parser(CPP_LANGUAGE)
        tree = parser.parse(bytes(code, encoding="utf-8"))

        if tree.root_node.has_error: return False

        # Query for C++ specific features within function parameters.
        # - `reference_declarator` is used for `&` and `&&` in types.
        # - `optional_parameter_declaration` is used for `type name = value`.
        query_string = """ [ 
          (reference_declarator)
          (optional_parameter_declaration)
        ] @cpp_feature
        """
        query = Query(CPP_LANGUAGE, query_string)
        captures = QueryCursor(query).captures(tree.root_node)
        if captures: return True

    except Exception:
        # If parsing fails for any reason, we cannot confirm it's C++.
        return False

    return False


def extract_function_signature(code: str, language_name: str) -> str | None:
    """Extracts the function signature from a C code snippet using a tree-sitter query.

    Params:
        code: A string containing the C function.
        parser: An initialized tree-sitter Parser object.
        language: The tree-sitter Language object for C.
    Returns:
        The function signature as a string (e.g., "void process_data(int p_id)"),
        or None if no function definition is found.
    """

    language : Language = C_LANGUAGE if language_name == "c" else CPP_LANGUAGE
    parser = Parser(language)

    tree = parser.parse(bytes(code, encoding="utf-8"))
    root_node = tree.root_node

    # 2. Define a query to capture the return type and the declarator
    #    The declarator includes the function name and parameters.
    query_string = """
        (function_definition
          (storage_class_specifier) @storage_class
          type: (_) @return_type
          declarator: (_) @declarator
        )"""
    query = Query(language, query_string)

    # 3. Execute the query
    captures = QueryCursor(query=query).captures(root_node)
    if not captures: return None

    # 4. Store captured nodes by name for easy access
    specifiers = captures.get("storage_class", [])
    return_type_node = captures.get("return_type", [])[0]
    declarator_node = captures.get("declarator", [])[0]

    # 5. Extract text and combine into the final signature
    if (
        return_type_node
        and declarator_node
        and return_type_node.text
        and declarator_node.text
    ):
        return_type = return_type_node.text.decode("utf-8")
        declarator = declarator_node.text.decode("utf-8")

        signature = f"{return_type} {declarator}"

        if specifiers:
            storage_class = " ".join([n.text.decode() for n in specifiers if n.text])
            signature = storage_class + " " + signature

        cleaned_signature = ' '.join(signature.split())
        return cleaned_signature

    return None



def spawn_clang_format(filepath: str, lang_name: str, clang_format_config_file: str) -> str:
    """Checks the .clang-format file, updates the language if necessary,
    and formats the source code in the given file using clang-format.

    Params:
        filepath: str
            Location of the temporary file to format.
        language_name: str
            The target language ("c" or "cpp").
        clang_format_config_file: str
            Path to the .clang-format configuration file.

    Returns:
        str: The refactored code read from the file.
    """

    current_language: str|None = ""

    # 1. Check current language in .clang-format and update if necessary.
    try:
        with open(file=clang_format_config_file, mode='r', encoding='utf-8') as f:
            current_config: str = f.read()

        m: re.Match[str]|None = re.search(pattern=r"^\s*Language:\s*(\w+)", string=current_config, flags=re.MULTILINE)
        current_language = m.group(1).lower() if m else None

        if current_language != lang_name.lower():
            update_language_clang_format(fp=clang_format_config_file, language_to_set=lang_name)
    except IOError as e:
        print(f"Warning: Could not read or update .clang-format file: {e}")

    # 2. Execute clang-format command.
    clang_format_exe = shutil.which("clang-format") # get command full path
    if not clang_format_exe:
        raise RuntimeError("`clang-format` not found. Please ensure it is in your PATH.")

    command: list[str] = [
        clang_format_exe,
        f"-style=file:{clang_format_config_file}",
        "-i",
        filepath,
    ]

    try:
        subprocess.run(
            args=command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return read_file(fp=filepath)
    except FileNotFoundError:
        raise RuntimeError(f"Executable not found at '{clang_format_exe}'.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"clang-format failed with exit code {e.returncode}:\n{e.stderr}")


def get_refactored_code(code: str, lang_name: str, fp: str, clang_format_file_path: str) -> str:
    write2file(fp=fp, content=code)
    return spawn_clang_format(fp, lang_name, clang_format_file_path)


def pause_exec(keyword: str ="ok"):
    """Pauses execution and waits for the user to type a specific keyword."""

    # Print a newline to move the cursor off the tqdm progress bar line.
    print()
    while True:
        if input(f"Type `{keyword}` when you're ready to proceed.  ").strip().lower() == keyword:
            break

