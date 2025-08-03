# import re
#
# class InterleavedBlockFixer:
#     def _rewrite_structure(self, match: re.Match, keyword: str) -> str:
#         """A helper called by re.sub. It validates the match before rewriting."""
#
#         preprocessor_block = match.group(1)
#         shared_body = match.group(2)
#
#         # VALIDATION STEP: Check if the preprocessor block actually contains
#         # the incomplete opener we're looking for (e.g., "if (...) {").
#         opener_pattern = rf"{keyword}\s*\((?:.|\n)*?\)\s*\{{"
#         if not re.search(opener_pattern, preprocessor_block):
#             return match.group(0)
#
#         stripped_shared_body = shared_body.strip()
#         replacement_block = f"{{\n    {stripped_shared_body}\n}}"
#         corrected_block = preprocessor_block.replace("{", replacement_block)
#
#         return corrected_block
#
#     def _fix_interleaved_block_general(self, c_code: str, keyword: str, is_complex_body: bool) -> str:
#         """Generic fixer that finds candidate blocks and passes them to the 
#         _rewrite_structure function for validation and correction.
#         """
#
#         body_quantifier = "*" if is_complex_body else "*?"
#
#         pattern = re.compile(
#             # Group 1: The entire preprocessor block (greedy for nesting).
#             rf"((?:#if|#ifdef)(?:.|\n)*#endif)\s*"
#             # Group 2: The shared body content.
#             rf"((?:.|\n){body_quantifier})\s*^\s*}}",
#             re.MULTILINE | re.DOTALL
#         )
#
#         return pattern.sub(lambda m: self._rewrite_structure(m, keyword), c_code)
#
#     def fix_interleaved_if(self, c_code: str) -> str:
#         # if bodies are usually simple (non-greedy body capture is safer).
#         return self._fix_interleaved_block_general(c_code, "if", is_complex_body=False)
#
#     def fix_interleaved_switch(self, c_code: str) -> str:
#         # switch bodies are also simple (non-greedy is safer).
#         return self._fix_interleaved_block_general(c_code, "switch", is_complex_body=False)
#
#     def fix_interleaved_while(self, c_code: str) -> str:
#         # while/for loops can have complex bodies with braces (greedy capture is needed).
#         return self._fix_interleaved_block_general(c_code, "while", is_complex_body=True)
#
#     def fix_interleaved_for(self, c_code: str) -> str:
#         # while/for loops can have complex bodies with braces (greedy capture is needed).
#         return self._fix_interleaved_block_general(c_code, "for", is_complex_body=True)
#
#     def _fix_interleaved_do_while(self, c_code: str) -> str:
#         pattern = re.compile(
#             r"do\s*\{((?:.|\n)*?)\s*"
#             r"(#if(?:.|\n)*\}\s*while(?:.|\n)*#endif)",
#             re.DOTALL
#         )
#
#         def _rewrite_structure(match: re.Match) -> str:
#             do_body_content = match.group(1).strip()
#             preprocessor_block = match.group(2)
#
#             replicated_do_block = f"do {{\n    {do_body_content}\n}}"
#
#             corrected_block = re.sub(
#                 r"\}\s*while",
#                 f"{replicated_do_block} while",
#                 preprocessor_block
#             )
#
#             return corrected_block
#
#         return pattern.sub(_rewrite_structure, c_code)
#
#     def full_structural_refactor(self, c_code: str) -> str:
#         """Applies all interleaved block fixers in a cascade to repair code."""
#
#         code = self._fix_interleaved_do_while(c_code)
#         code = self.fix_interleaved_if(code)
#         code = self.fix_interleaved_switch(code)
#         code = self.fix_interleaved_while(code)
#         code = self.fix_interleaved_for(code)
#
#         return code

import re

class InterleavedBlockFixer:
    """
    A robust C code restructurer using a state-based parser to correctly
    handle complex nested preprocessor blocks. This approach is more reliable
    than a pure regex solution for nested structures.
    """

    def _rewrite_block(self, block_lines: list[str]) -> str:
        """Takes a list of lines representing a complete malformed block (from
        the first #if to the final shared '}') and returns the corrected string.
        """

        shared_body_lines: list[str] = []

        block_lines.pop()
        while block_lines and not block_lines[-1].strip().startswith('#endif'):
            shared_body_lines.insert(0, block_lines.pop())

        preprocessor_block:str = "\n".join(block_lines)
        shared_body:str = "\n".join(shared_body_lines)

        opener_pattern: re.Pattern = re.compile(
            pattern=r"(\b(?:if|switch|while|for)\s*\((?:.|\n)*?\)\s*\{)\s*$",
            flags=re.MULTILINE | re.DOTALL
        )

        def replacer(opener_match: re.Match[str]) -> str:
            original_opener:str = opener_match.group(1)
            return f"{original_opener}\n{shared_body.strip()}\n}}"

        return opener_pattern.sub(repl=replacer, string=preprocessor_block)

    def _fix_interleaved_do_while(self, c_code: str) -> str:
        """
        Fixes the unique structure of interleaved do-while loops using a
        single-pass, greedy-pattern approach to handle nesting correctly.
        """

        # This greedy pattern finds the entire do-while construct, from the
        # 'do' to the final '#endif' of its block.
        # group 1: 'do {' + body
        # group 2: body only
        # group 3: The preprocessor block (greedy)
        pattern = re.compile(
            pattern=r"(do\s*\{((?:.|\n)*?))\s*"
            r"((?:#if|#ifdef)(?:.|\n)*#endif)",
            flags=re.DOTALL
        )

        match: re.Match[str]|None = pattern.search(string=c_code)
        if not match: return c_code

        do_block_prefix:str = match.group(1)
        preprocessor_block:str = match.group(3)


        if "} while" not in preprocessor_block:
            return c_code

        corrected_preprocessor = re.sub(
            # Capture the entire 'while' clause, including parentheses.
            pattern=r"\}\s*(while\s*\(.*\);?)",
            repl=lambda m: f"{do_block_prefix}}} {m.group(1)}",
            string=preprocessor_block
        )

        return c_code.replace(match.group(0), corrected_preprocessor)

    def _fix_interleaved_statement_content(self, c_code: str) -> str:
        """Fixes cases where a statement's body is split across directives.
        Example: for(...) { #if ... } #else ... } #endif
        This is fixed by duplicating the parent statement inside each directive.
        """

        # Group 1: The full statement opener, e.g., "for (...) {"
        # Group 3: The entire preprocessor block that splits the body.
        pattern: re.Pattern = re.compile(
            pattern=r"(\b(for|while|if|switch)\s*\((?:.|\n)*?\)\s*\{)\s*"
            r"((?:#if|#ifdef)(?:.|\n)*?#endif)",
            flags=re.DOTALL
        )

        def _rewrite(match: re.Match[str]) -> str:
            opener:str = match.group(1)
            preprocessor_block:str = match.group(3)

            # VALIDATION: This pattern is only valid if a brace is prematurely
            # closed before an #else or #elif.
            if not re.search(pattern=r"\}\s*#(?:else|elif)", string=preprocessor_block):
                return match.group(0)

            clean_opener:str = opener.strip()

            def insert_opener(directive_match: re.Match[str]) -> str:
                directive:str = directive_match.group(0)
                return f"{directive}\n{clean_opener}"

            # Insert the opener after each preprocessor directive.
            modified_block:str = re.sub(
                pattern=r"^\s*#(?:if|ifdef|elif|else).*$",
                repl=insert_opener,
                string=preprocessor_block,
                flags=re.MULTILINE
            )

            return modified_block

        while True: # for nested patterns
            c_code, num_subs = pattern.subn(repl=_rewrite, string=c_code, count=1)
            if num_subs == 0: break

        return c_code

    def full_structural_refactor(self, c_code: str) -> str:
        """
        Applies all interleaved block fixers in a cascade to repair code.
        This method uses a state-based parser to correctly handle nesting.
        """

        output_lines, buffer = [], [] # list[str]
        if_level, i = 0, 0 # int

        code: str = self._fix_interleaved_statement_content(c_code)
        code = self._fix_interleaved_do_while(code)

        lines: list[str] = code.split('\n')

        while i < len(lines):
            line:str = lines[i]
            stripped_line:str = line.strip()

            # --- State 1: Outside a preprocessor block ---
            if if_level == 0:
                if stripped_line.startswith(('#if', '#ifdef')):
                    buffer.append(line)
                    if_level = 1
                else:
                    output_lines.append(line)
                i += 1
                continue

            # --- State 2: Inside a preprocessor block ---
            if if_level > 0:
                buffer.append(line)
                if stripped_line.startswith(('#if', '#ifdef')):
                    if_level += 1
                elif stripped_line.startswith('#endif'):
                    if_level -= 1

                # Check if we have just closed the top-level block
                if if_level == 0:
                    # Look ahead for a shared body ending in '}'
                    lookahead_index:int = i + 1
                    shared_body_lines:list[str] = []
                    found_structure:bool = False
                    brace_level:int = 0

                    while lookahead_index < len(lines):
                        next_line:str = lines[lookahead_index]
                        brace_level += next_line.count('{')
                        brace_level -= next_line.count('}')
                        if brace_level < 0:
                            # The line with the final brace is part of the body
                            shared_body_lines.append(next_line)
                            found_structure = True
                            break

                        shared_body_lines.append(next_line)
                        lookahead_index += 1

                    buffered_block_str:str = "\n".join(buffer)
                    is_truly_malformed:bool = buffered_block_str.count('{') > buffered_block_str.count('}')


                    if found_structure and is_truly_malformed:
                        full_block_lines:list[str] = buffer + shared_body_lines# + ["}"]
                        fixed_block:str = self._rewrite_block(full_block_lines)
                        output_lines.append(fixed_block)

                        i = lookahead_index + 1
                        buffer = [] # Reset buffer
                        continue
                    else:
                        # Not a malformed block, flush buffer and continue
                        output_lines.extend(buffer)
                        if found_structure:
                            output_lines.extend(shared_body_lines)
                            i = lookahead_index + 1
                        else:
                            i += 1
                        buffer = []
            i += 1

        # Add any remaining lines from the buffer (for incomplete files)
        if buffer:
            output_lines.extend(buffer)

        return "\n".join(output_lines)



