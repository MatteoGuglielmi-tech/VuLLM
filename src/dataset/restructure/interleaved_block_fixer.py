import re


class InterleavedBlockFixer:
    """
    A robust C code restructurer using a single-pass state machine to
    correctly handle multiple types of complex, nested interleaved blocks.
    """

    # This pattern identifies statement openers like for(...){, if(...){, etc.
    OPENER_PATTERN = re.compile(pattern= r"^\s*(\b(for|while|if|switch)\s*\([^{}]*?\)\s*\{)\s*$")

    def _rewrite_split_statement(self, opener: str, preprocessor_block: str) -> str:
        """Rewrites a statement that is split by preprocessor directives.

        This function takes the opening part of a statement and a block of code
        containing preprocessor directives (e.g., #if, #elif, #else). It then
        duplicates the opener within each preprocessor branch to make the code
        syntactically valid within each branch.

        Parameters
        ----------
        opener : str
            The opening part of the statement to be duplicated.
        preprocessor_block : str
            The block of code with preprocessor directives.

        Returns
        -------
        str
            The modified block with the opener inserted into each directive.
        """

        def insert_opener(directive_match: re.Match[str]) -> str:
            return f"{directive_match.group(0)}\n{opener.strip()}"

        modified_block:str = re.sub(
            pattern=r"^\s*#(if|ifdef|elif|else).*$",
            repl=insert_opener,
            string=preprocessor_block,
            flags=re.MULTILINE
        )

        return modified_block

    def _rewrite_inject_shared_body(self, preprocessor_block: str, shared_body: str) -> str:
        """Injects a shared code body into preprocessor directive branches.

        This function uses a regular expression to find all lines within a
        preprocessor block that end with an opening brace '{'. For each match,
        it injects the provided shared code body and adds a corresponding
        closing brace '}' to complete the block.

        Parameters
        ----------
        preprocessor_block : str
            The input string containing preprocessor directives and the
            opening braces for each code branch.
        shared_body : str
            The common block of code to be injected into each branch.

        Returns
        -------
        str
            The modified preprocessor block with the shared body injected
            into each branch.
        """

        # find a line ending in an opening brace, capturing everything before it.
        pattern = re.compile(pattern=r"^(.*\s*\{)\s*$", flags=re.MULTILINE)
        # replacement adds the shared code and the closing brace.
        # \\1 is automatically substituted with the match of the first group
        replacement:str = f"\\1 {shared_body.strip()} }}" 

        return pattern.sub(repl=replacement, string=preprocessor_block)

    def _rewrite_block_with_shared_body(self, block_lines: list[str]) -> str:
        """Refactors a code block by injecting a shared final body into each branch.

        This function processes a list of code lines that form a block, such as
        an if-else structure, where each branch concludes with the same set
        of statements (a "shared body"). It isolates this shared body from the
        preceding control statements and preprocessor directives. It then
        programmatically injects the shared body into each distinct branch,
        ensuring each is a complete, valid code block.

        Parameters
        ----------
        block_lines : list[str]
            A list of strings, where each string is a line of code from the
            block to be refactored. The block is expected to end with a shared
            body of code followed by a closing brace.

        Returns
        -------
        str
            A single string representing the fully rewritten code, with the
            shared body correctly placed inside each control statement's branch.
        """

        shared_body_lines:list[str] = []

        block_lines.pop() # Remove the final '}'

        # --- accumulate the shared body ---
        while block_lines and not block_lines[-1].strip().startswith('#endif'):
            shared_body_lines.insert(0, block_lines.pop())

        preprocessor_block:str = "\n".join(block_lines)
        shared_body:str = "\n".join(shared_body_lines)

        # this pattern finds the { that opens the body inside each directive
        opener_pattern = re.compile(
            pattern=r"(\b(?:if|switch|while|for)\s*\((?:.|\n)*?\)\s*\{)\s*$",
            flags=re.MULTILINE | re.DOTALL
        )

        def replacer(opener_match: re.Match[str]) -> str:
            return f"{opener_match.group(1)}\n{shared_body.strip()}\n}}"

        return opener_pattern.sub(repl=replacer, string=preprocessor_block)
 
    def _fix_interleaved_do_while(self, c_code: str) -> str:
        """
        Corrects a `do-while` loop that is improperly split by a potentially
        nested preprocessor block.
        """
        # Find the 'do { body' part that is immediately followed by a preprocessor directive.
        # Group 1 captures the entire 'do { body' prefix.
        match = re.search(r"(do\s*\{(?:[^{}]|\{[^{}]*\})*?)\s*(#if|#ifdef)", c_code, re.DOTALL)
        if not match:
            return c_code

        do_prefix = match.group(1)
        # The character index where the preprocessor block starts.
        block_start_index = match.start(2)

        # --- Use a counter to find the matching #endif for the nested block ---
        if_level = 0
        block_end_index = -1
        # Create an iterator over all #if, #ifdef, #ifndef, and #endif directives
        directive_finder = re.finditer(r"#\s*(if|ifdef|ifndef|endif)\b", c_code[block_start_index:])

        for directive_match in directive_finder:
            directive = directive_match.group(1)
            if directive in ("if", "ifdef", "ifndef"):
                if_level += 1
            elif directive == "endif":
                if_level -= 1

            # When the level returns to 0, we've found the matching #endif
            if if_level == 0:
                block_end_index = block_start_index + directive_match.end()
                break
                
        if block_end_index == -1:
            # No matching #endif was found; this is not the pattern we're fixing.
            return c_code

        # Isolate the full preprocessor block
        preprocessor_block = c_code[block_start_index:block_end_index]
        
        # If the block doesn't contain the 'while' clause, it's not our pattern.
        if "} while" not in preprocessor_block:
            return c_code

        # --- Perform the Rewrite ---
        # Define a replacer function that injects the 'do { body }' before the 'while'
        def replacer(m: re.Match[str]) -> str:
            # m.group(1) is the 'while(...);' part.
            # We add the 'do' prefix and a closing brace to form a complete loop.
            return f"{do_prefix.strip()} }} {m.group(1)}"

        # Apply the replacement to all '} while' clauses within the block
        corrected_pp_block = re.sub(
            r"\}\s*(while\s*\(.*\);?)",
            replacer,
            preprocessor_block
        )

        # Reconstruct the original code with the fix
        original_full_block = c_code[match.start(1):block_end_index]
        return c_code.replace(original_full_block, corrected_pp_block)

    def full_structural_refactor(self, c_code: str) -> str:
        """Applies all structural fixes to C code using a state machine.

        This function serves as the primary engine for refactoring C code with
        interleaved preprocessor directives. It iterates through the code line by
        line, using a state machine to identify and buffer complete, top-level
        `#if...#endif` blocks.

        Once a complete block is isolated, the function determines which type of
        structural issue is present and applies the corresponding fix:
        1.  Split Statement: if a control statement opener (e.g., `for (...) {`)
            is immediately followed by the preprocessor block, it calls
            `_rewrite_split_statement` to duplicate the opener inside each branch.
        2.  Interleaved do-while: it checks for and corrects `do-while` loops
            that have been split by the preprocessor block.
        3.  Shared Body Injection: if the preprocessor block is missing closing
            braces and is followed by a shared block of code, it injects that
            shared code into each preprocessor branch.

        The function reconstructs the code with the fixed blocks, leaving
        correctly structured code untouched.

        Parameters
        ----------
        c_code : str
            The input C source code as a single string.

        Returns
        -------
        str
            The refactored C code with structural issues corrected.
        """

        output_lines, buffer = [], [] # list[str]
        if_level, i = 0, 0 #int
        statement_opener: str|None = None

        lines: list[str] = self._fix_interleaved_do_while(c_code=c_code).split(sep="\n")

        while i < len(lines):
            line: str = lines[i]
            stripped_line: str = line.strip()

            # --- outside a preprocessor block ---
            if if_level == 0:
                # Look for 'opener' followed immediately by '#if'
                opener_match: re.Match[str]|None = self.OPENER_PATTERN.match(line)
                is_last_line:bool = (i + 1) >= len(lines)

                # CASE 1: Opener followed by #if (e.g., `for { #if ...`)
                if opener_match and not is_last_line and lines[i + 1].strip().startswith(('#if', '#ifdef', '#ifndef')):
                    # PATTERN 1 DETECTED: capture the state and start buffering
                    statement_opener = opener_match.group(1)
                    buffer.append(line); buffer.append(lines[i + 1])
                    if_level = 1
                    i += 2
                    continue

                # CASE 2: A standalone #if directive that should start the buffering process.
                elif stripped_line.startswith(('#if', '#ifdef', '#ifndef')):
                    buffer.append(line)
                    if_level = 1
                    i += 1
                    continue

                # CASE 3: A normal line of code
                else:
                    output_lines.append(line)
                    i += 1
                    continue

            # --- inside a preprocessor block ---
            if if_level > 0:
                buffer.append(line)
                if stripped_line.startswith(('#if', '#ifdef', '#ifndef')):
                    if_level += 1
                elif stripped_line.startswith('#endif'):
                    if_level -= 1

                # --- DECISION POINT: A block has just been fully buffered ---
                if if_level == 0:
                    #  -- run fixes only on this isolated block. --
                    buffered_code_str:str = "\n".join(buffer)

                    # --- PATH A: We captured a statement opener earlier ---
                    if statement_opener:
                        preprocessor_block:str = "\n".join(buffer[1:])
                        fixed_block = self._rewrite_split_statement(statement_opener, preprocessor_block)
                        output_lines.append(fixed_block)

                        i += 1
                        buffer = []
                        statement_opener = None
                        continue 

                    # --- PATH B: No opener was captured, check for a shared body ---
                    else:
                        # --- lookahead logic ---
                        lookahead_index: int = i+1 #, brace_level = i + 1, 0
                        shared_body_lines, found_structure = [], False

                        while lookahead_index < len(lines):
                            next_line = lines[lookahead_index]
                            # brace_level += next_line.count('{') - next_line.count('}')
                            # if brace_level < 0:
                            #     shared_body_lines.append(next_line)
                            #     found_structure = True
                            #     break
                            if next_line.strip() == "}":
                                found_structure = True
                                break
                            shared_body_lines.append(next_line)
                            lookahead_index += 1

                        is_malformed: bool = buffered_code_str.count('{') > buffered_code_str.count('}')

                        if found_structure and is_malformed:
                            shared_body_str = "\n".join(shared_body_lines)
                            rewritten_block = self._rewrite_inject_shared_body(buffered_code_str, shared_body_str)
                            output_lines.append(rewritten_block)
                            i = lookahead_index + 1
                        else:
                            output_lines.extend(buffer)
                            i += 1

                    # Reset state for the next iteration
                    buffer = []
                    continue

            i += 1

        if buffer:
            output_lines.extend(buffer)

        return "\n".join(output_lines)

