import pytest
import os
import shutil

from dataset.restructure.proc_utils import write2file, decode_escaped_string, is_cpp, spawn_clang_format
from dataset.restructure.tree_sitter_parser import TreeSitterParser
from dataset.restructure.interleaved_block_fixer import InterleavedBlockFixer
from dataset.restructure.code_sanitizer import CodeSanitizer


# =================================================================================
@pytest.fixture(scope="module")
def interleaved_block_fixer_instance():
    yield InterleavedBlockFixer()


@pytest.fixture(scope="module")
def code_sanitizer_instance():
    """Sets up a CodeSanitizer instance."""
    yield CodeSanitizer()

    # Teardown the './misc' directory created by the pipeline.
    misc_dir = os.path.join(os.getcwd(), "misc")
    if os.path.exists(misc_dir):
        shutil.rmtree(misc_dir)


project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
clang_format_file_path: str = os.path.join(project_root, ".clang-format")
def _get_refactored_code(code: str, lang_name: str, fp: str):
    write2file(fp=fp, content=code)
    formatted_result = spawn_clang_format(fp, lang_name, clang_format_file_path)
    os.remove(fp)

    return formatted_result


# --- Unit Tests for Individual Methods ---
class TestPipelineUnit:
    """A test class to group unit tests for individual pipeline methods."""

    # ==============================================================================================================
    # `decode_escaped_string` TESTS
    # =============================================================================================================

    # ---
    SIMPLE_FUNC_INPUT = 'void func() {\\n\\tprintf(\\"Hello World!\\");\\n}'
    SIMPLE_FUNC_OUTPUT = 'void func() {\n\tprintf("Hello World!");\n}'
    # ---

    # ---
    REAL_LIKE_EXAMPLE_INPUT = r"""void yajl_string_decode(yajl_buf buf, const unsigned char *str, unsigned int len) {\n    unsigned int beg = 0;\n    unsigned int end = 0;\n\n    while (end < len) {\n        if (str[end] == '\\\\') {\n            char utf8Buf[5];\n            const char * unescaped = \"?\";\n            yajl_buf_append(buf, str + beg, end - beg);\n            switch (str[++end]) {\n                case 'r': unescaped = \"\\r\"; break;\n                case 'n': unescaped = \"\\n\"; break;\n                case '\\\\': unescaped = \"\\\\\"; break;\n                case '/': unescaped = \"/\"; break;\n                case '\"': unescaped = \"\\\"\"; break;\n                case 'f': unescaped = \"\\f\"; break;\n                case 'b': unescaped = \"\\b\"; break;\n                case 't': unescaped = \"\\t\"; break;\n            }\n            yajl_buf_append(buf, unescaped, (unsigned int)strlen(unescaped));\n            beg = ++end;\n        } else {\n            end++;\n        }\n    }\n    yajl_buf_append(buf, str + beg, end - beg);\n}"""
    REAL_LIKE_EXAMPLE_OUTPUT = r"""void yajl_string_decode(yajl_buf buf, const unsigned char *str, unsigned int len) {
        unsigned int beg = 0;
        unsigned int end = 0;

        while (end < len) {
            if (str[end] == '\\') {
                char utf8Buf[5];
                const char * unescaped = "?";
                yajl_buf_append(buf, str + beg, end - beg);
                switch (str[++end]) {
                    case 'r': unescaped = "\r"; break;
                    case 'n': unescaped = "\n"; break;
                    case '\\': unescaped = "\\"; break;
                    case '/': unescaped = "/"; break;
                    case '"': unescaped = "\""; break;
                    case 'f': unescaped = "\f"; break;
                    case 'b': unescaped = "\b"; break;
                    case 't': unescaped = "\t"; break;
                }
                yajl_buf_append(buf, unescaped, (unsigned int)strlen(unescaped));
                beg = ++end;
            } else {
                end++;
            }
        }
        yajl_buf_append(buf, str + beg, end - beg);
    }"""
    # ---
  
    # ---
    DATASET_EXAMPLE_INPUT = 'static int snd_mem_proc_read(char *page, char **start, off_t off, int count, int *eof, void *data)\n{\n\tint len = 0;\n\tlong pages = snd_allocated_pages >> (PAGE_SHIFT-12);\n\tstruct snd_mem_list *mem;\n\tint devno;\n\tstatic char *types[] = { "UNKNOWN", "CONT", "DEV", "DEV-SG", "SBUS" };\n\n\tmutex_lock(&list_mutex);\n\tlen += snprintf(page + len, count - len,\n\t\t\t"pages  : %li bytes (%li pages per %likB)\\n",\n\t\t\tpages * PAGE_SIZE, pages, PAGE_SIZE / 1024);\n\tdevno = 0;\n\tlist_for_each_entry(mem, &mem_list_head, list) {\n\t\tdevno++;\n\t\tlen += snprintf(page + len, count - len,\n\t\t\t\t"buffer %d : ID %08x : type %s\\n",\n\t\t\t\tdevno, mem->id, types[mem->buffer.dev.type]);\n\t\tlen += snprintf(page + len, count - len,\n\t\t\t\t"  addr = 0x%lx, size = %d bytes\\n",\n\t\t\t\t(unsigned long)mem->buffer.addr, (int)mem->buffer.bytes);\n\t}\n\tmutex_unlock(&list_mutex);\n\treturn len;\n}'
    DATASET_EXAMPLE_OUTPUT = """static int snd_mem_proc_read(char *page, char **start, off_t off, int count, int *eof, void *data)
    {
    	int len = 0;
    	long pages = snd_allocated_pages >> (PAGE_SHIFT-12);
    	struct snd_mem_list *mem;
    	int devno;
    	static char *types[] = { "UNKNOWN", "CONT", "DEV", "DEV-SG", "SBUS" };

    	mutex_lock(&list_mutex);
    	len += snprintf(page + len, count - len,
    			"pages  : %li bytes (%li pages per %likB)\n",
    			pages * PAGE_SIZE, pages, PAGE_SIZE / 1024);
    	devno = 0;
    	list_for_each_entry(mem, &mem_list_head, list) {
    		devno++;
    		len += snprintf(page + len, count - len,
    				"buffer %d : ID %08x : type %s\n",
    				devno, mem->id, types[mem->buffer.dev.type]);
    		len += snprintf(page + len, count - len,
    				"  addr = 0x%lx, size = %d bytes\n",
    				(unsigned long)mem->buffer.addr, (int)mem->buffer.bytes);
    	}
    	mutex_unlock(&list_mutex);
    	return len;
    }"""
    # ---

    # --- Integration Test ---
    test_data_decode_escaped = [
        pytest.param(SIMPLE_FUNC_INPUT, SIMPLE_FUNC_OUTPUT, id="simple_func_input"),
        pytest.param(REAL_LIKE_EXAMPLE_INPUT, REAL_LIKE_EXAMPLE_OUTPUT, id="real_like_example"),
        pytest.param(DATASET_EXAMPLE_INPUT, DATASET_EXAMPLE_OUTPUT, id="dataset_example"),
    ]
    @pytest.mark.parametrize("raw_string, expected_string", test_data_decode_escaped)
    def test_decode_escaped_string(self, raw_string: str, expected_string: str):
        decoded_string: str = decode_escaped_string(raw_string)
        assert _get_refactored_code(decoded_string, lang_name="c", fp="./tmp_candidate.c") == _get_refactored_code(expected_string, lang_name="c", fp="./tmp_expected.c")

    # ==============================================================================================================
    # `is_cpp` TESTS
    # =============================================================================================================
    @pytest.mark.parametrize(
        "code, expected",
        [
            ("void func() {}", False),
            ("int main(int argc, char *argv[])", False),
            ("template <typename T> void func() {}", True),
            ("namespace my_space {}", True),
            ("class MyClass { public: int x; };", True),
            ("void func(const std::string& s) {}", True),
            ("void func(int x = 0) {}", True),
            ("""namespace cimg {
                inline FILE *_stdin(const bool throw_exception) {
                #ifndef cimg_use_r
                  cimg::unused(throw_exception);
                  return stdin;
                #else
                  if(throw_exception) {
                    cimg::exception_mode(0);
                    throw CImgIOException("cimg::stdin(): Reference to 'stdin' stream not allowed in R mode "
                                          "('cimg_use_r' is defined).");
                  }
                #endif
                }
                }
            """, True),
            ("template <typename t>\n CImg<T> & label(const CImg<t> &connectivity_mask, const Tfloat tolerance = 0) {\n return get_label(connectivity_mask, tolerance).move_to(*this); }", True)
        ],
    )
    def test_is_cpp(self, code: str, expected: bool):
        """Tests the language detection heuristic."""
        assert is_cpp(code) == expected

    # ==============================================================================================================
    # `remove_comments` TESTS
    # =============================================================================================================
    @pytest.mark.parametrize(
        "code, expected", [
            ("    //! Wait for any event occuring either on the display \\c disp1 or \\c disp2.\n    static void wait(CImgDisplay& disp1, CImgDisplay& disp2) {\n      disp1._is_event = disp2._is_event = false;\n      while ((!disp1._is_closed || !disp2._is_closed) &&\n             !disp1._is_event && !disp2._is_event) wait_all();", "static void wait(CImgDisplay& disp1, CImgDisplay& disp2) {\n      disp1._is_event = disp2._is_event = false;\n      while ((!disp1._is_closed || !disp2._is_closed) &&\n             !disp1._is_event && !disp2._is_event) wait_all();"),
            ('void func() {\n  // This is a line comment.\n  printf("Fake comment // here");\n  /* This is a\n     block comment. */\n}', 'void func() {\n  \n  printf("Fake comment // here");\n  \n}'
            )
        ]
    )
    def test_remove_comments(self, code_sanitizer_instance: CodeSanitizer, code: str, expected: str):
        assert code_sanitizer_instance.remove_comments(code, tsp=TreeSitterParser("c")) == expected

    # ==============================================================================================================
    # `_validate_and_extract_body` TESTS
    # =============================================================================================================
    @pytest.mark.parametrize(
        "code, expected_type, expected_code",
        [
            ("} int main() { return 0; }", "function_definition", "int main() { return 0; }"),  # Leading garbage
            ("int x = 5; struct MyStruct { int y; };", None, None),  # Discard struct/global
            ("enum Color { RED };", None, None),  # Discard enum
            ("{ int x = 1; }", "compound_statement", "{ int x = 1; }"),  # Valid K&R style
            ("if (true) { int y = 2; }", None, None),  # Discard nested block
            ("/// @c true if the connection is transparent.", None, None), # discard comment
            ("", None, None), # discard empty entry

            # synthetic real-like entries
            ("#if __KERNEL__==4\nstatic inline struct ext4_sb_info *EXT4_SB(struct super_block *sb)\n{\n\treturn sb->s_fs_info;}",
            "function_definition", "static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb)\n{\n\treturn sb->s_fs_info;}"),
            ("#if __KERNEL__<=4\nstatic inline struct ext4_sb_info *EXT4_SB(struct super_block *sb)\n{\n\treturn sb->s_fs_info;}",
             "function_definition", "static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb)\n{\n\treturn sb->s_fs_info;}"),
        ],
    )
    def test_validate_and_extract_body(
        self,
        code_sanitizer_instance: CodeSanitizer,
        code: str,
        expected_type: str|None,
        expected_code: str|None,
    ):
        """Tests the logic for validating and extracting desirable code blocks."""

        tsp = TreeSitterParser("c")
        result = code_sanitizer_instance._validate_and_extract_body(code, tsp=tsp)
        if expected_type is None:
            assert result is None
        else:
            assert result is not None

            tree = tsp.parse(result)
            assert tree.root_node.children[0].type == expected_type
            assert result == expected_code

    # ==============================================================================================================
    # `_validate_and_extract_body` TESTS
    # =============================================================================================================
    @pytest.mark.parametrize(
        "code, expected_type, expected_code",
        [
            ("template<class T> void func(){}", "template_declaration", "template<class T> void func(){}"),  # Valid template
            ("int x = 5; struct MyStruct { int y; };", None, None),  # Discard struct/global
            ("enum Color { RED };", None, None),  # Discard enum
            ("{ int x = 1; }", "compound_statement", "{ int x = 1; }"),  # Valid K&R style
            ("if (true) { int y = 2; }", None, None),  # Discard nested block
            ("/// @c true if the connection is transparent.", None, None), # discard comment
            ("", None, None), # discard empty entry

            # real-like entries
            (
                '#if cimg_verbosity>=3\n\tCImg<T> *data(const unsigned int pos) {\n\t\tif (pos>=size())\n\t\t\tcimg::warn(_cimglist_instance\n\t\t\t"data(): Invalid pointer request, at position [%u].",\n\t\t\t\tcimglist_instance,\n\t\t\t\tpos);\n\t\t\treturn _data + pos;\n}',
                "function_definition",
                'CImg<T> *data(const unsigned int pos) {\n\t\tif (pos>=size())\n\t\t\tcimg::warn(_cimglist_instance\n\t\t\t"data(): Invalid pointer request, at position [%u].",\n\t\t\t\tcimglist_instance,\n\t\t\t\tpos);\n\t\t\treturn _data + pos;\n}',
            ),
            (
                '#if cimg_verbosity==3\n\tCImg<T> *data(const unsigned int pos) {\n\t\tif (pos>=size())\n\t\t\tcimg::warn(_cimglist_instance\n\t\t\t"data(): Invalid pointer request, at position [%u].",\n\t\t\t\tcimglist_instance,\n\t\t\t\tpos);\n\t\t\treturn _data + pos;\n}',
                "function_definition",
                'CImg<T> *data(const unsigned int pos) {\n\t\tif (pos>=size())\n\t\t\tcimg::warn(_cimglist_instance\n\t\t\t"data(): Invalid pointer request, at position [%u].",\n\t\t\t\tcimglist_instance,\n\t\t\t\tpos);\n\t\t\treturn _data + pos;\n}',
            ),
            (
                '#if cimg_verbosity<=3\n\tCImg<T> *data(const unsigned int pos) {\n\t\tif (pos>=size())\n\t\t\tcimg::warn(_cimglist_instance\n\t\t\t"data(): Invalid pointer request, at position [%u].",\n\t\t\t\tcimglist_instance,\n\t\t\t\tpos);\n\t\t\treturn _data + pos;\n}',
                "function_definition",
                'CImg<T> *data(const unsigned int pos) {\n\t\tif (pos>=size())\n\t\t\tcimg::warn(_cimglist_instance\n\t\t\t"data(): Invalid pointer request, at position [%u].",\n\t\t\t\tcimglist_instance,\n\t\t\t\tpos);\n\t\t\treturn _data + pos;\n}',
            ),
            (
                "#ifdef __KERNEL__\nstatic inline struct ext4_sb_info *EXT4_SB(struct super_block *sb)\n{\n\treturn sb->s_fs_info;\n}",
                "function_definition",
                "static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb)\n{\n\treturn sb->s_fs_info;\n}",
            )
        ],
    )
    def test_validate_and_extract_body_cpp(
        self,
        code_sanitizer_instance: CodeSanitizer,
        code: str,
        expected_type: str|None,
        expected_code: str|None,
    ):
        """Tests the logic for validating and extracting desirable code blocks."""

        tsp = TreeSitterParser("cpp")
        result = code_sanitizer_instance._validate_and_extract_body(code, tsp=tsp)
        if expected_type is None:
            assert result is None
        else:
            assert result is not None

            tree = tsp.parse(result)
            assert tree.root_node.children[0].type == expected_type
            assert result == expected_code

    # ==============================================================================================================
    # `test_preprocess_directives` TESTS
    # =============================================================================================================
    @pytest.mark.parametrize(
        "input_code, expected_code",
        [
            ("#if 1\nint x;\n#endif", "int x;"),  # Simple #if 1
            ("#if 0\nint x;\n#endif", ""),  # Simple #if 0
            ("#if 1\nint x;\n#else\nint y;\n#endif", "int x;"), #if 1 with else
            ("#if 0\nint x;\n#else\nint y;\n#endif", "int y;"), #if 0 with else
            ("#if 1\n#if 0\nint x;\n#endif\nint y;\n#endif", "int y;"),  # Nested
            ("#if 0\n#if 1\nint x;\n#if B\nint f;\n#endif\n#endif\nint y;\n#else\nint k;\n#endif", "int k;"),  # Nested
        ],
    )
    def test_preprocess_directives(
        self,
        code_sanitizer_instance: CodeSanitizer,
        input_code: str,
        expected_code: str,
    ):
        """Tests the semantic simplification of #if 0 and #if 1."""

        result = code_sanitizer_instance._preprocess_directives(input_code, tsp=TreeSitterParser("c"))
        assert "".join(result.split()) == "".join(expected_code.split())

    # ==============================================================================================================
    # `add_missing_braces` TESTS
    # =============================================================================================================
    @pytest.mark.parametrize(
        "input_code, expected_code",
        [
            ("void func() { #if A", "void func() { #if A\n}"),
            ("void func() {", "void func() {\n}"),
            (
                """ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
                    ssize_t ret;
                    do {
                        ret = recvfrom(s, buf, len, flags, from, fromlen);
                    #if defined(EWOULDBLOCK)
                    } while(ret == -1 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK));
                    #else
                    } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
                    #endif
                """,
                """ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
                    ssize_t ret;
                    do {
                        ret = recvfrom(s, buf, len, flags, from, fromlen);
                    #if defined(EWOULDBLOCK)
                    } while(ret == -1 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK));
                    #else
                    } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
                    #endif
                    }
                """,
            )
        ],
    )
    def test_add_missing_braces(
        self,
        code_sanitizer_instance: CodeSanitizer,
        interleaved_block_fixer_instance: InterleavedBlockFixer,
        input_code: str,
        expected_code: str,
    ):
        input_code = interleaved_block_fixer_instance.full_structural_refactor(input_code)
        expected_code = interleaved_block_fixer_instance.full_structural_refactor(expected_code)
        result = code_sanitizer_instance.add_missing_braces(input_code)

        assert "".join(result.split()) == "".join(expected_code.split())


    # ==============================================================================================================
    # `_balance_directives` TESTS
    # =============================================================================================================
    @pytest.mark.parametrize(
        "input_code, expected_code",
        [
            (
                """ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
                    ssize_t ret;
                    do {
                        ret = recvfrom(s, buf, len, flags, from, fromlen);
                    #if defined(EWOULDBLOCK)
                    } while(ret == -1 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK));
                    #else
                    } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
                """,
                """ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
                    ssize_t ret;
                    do {
                        ret = recvfrom(s, buf, len, flags, from, fromlen);
                    #if defined(EWOULDBLOCK)
                    } while(ret == -1 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK));
                    #else
                    } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
                    #endif
                """,
            ),

            (
                """void zend_set_timeout(zend_long seconds, int reset_signals) {
                   EG(timeout_seconds) = seconds;
                   #ifdef ZEND_WIN32
                     if(!seconds) { return; }
                     if(NULL != tq_timer) {
                       if(!DeleteTimerQueueTimer(NULL, tq_timer, NULL)) {
                         EG(timed_out) = 0;
                         q_timer = NULL;
                         zend_error_noreturn(E_ERROR, "Could not delete queued timer");
                         return;
                       }
                       q_timer = NULL;
                     }
                     if(!CreateTimerQueueTimer(&tq_timer, NULL, (WAITORTIMERCALLBACK)tq_timer_cb,
                                               (VOID *)&EG(timed_out), seconds * 1000, 0, WT_EXECUTEONLYONCE)) {
                       EG(timed_out) = 0;
                       tq_timer = NULL;
                       zend_error_noreturn(E_ERROR, "Could not queue new timer");
                       return;
                     }
                     EG(timed_out) = 0;
                   #else
                     #ifdef HAVE_SETITIMER
                     {
                       struct itimerval t_r;
                       int signo;
                       if(seconds) {
                         t_r.it_value.tv_sec = seconds;
                         t_r.it_value.tv_usec = t_r.it_interval.tv_sec = t_r.it_interval.tv_usec = 0;
                       #ifdef __CYGWIN__
                         setitimer(ITIMER_REAL, &t_r, NULL);
                       }
                       signo = SIGALRM;
                       #else
                         setitimer(ITIMER_PROF, &t_r, NULL);
                       }
                       signo = SIGPROF;
                       #endif
                       if(reset_signals) {
                       #ifdef ZEND_SIGNALS
                         zend_signal(signo, zend_timeout);
                       #else
                         sigset_t sigset;
                         signal(signo, zend_timeout);
                         sigemptyset(&sigset);
                         sigaddset(&sigset, signo);
                         sigprocmask(SIG_UNBLOCK, &sigset, NULL);
                       #endif
                       }
                   }""",
                """void zend_set_timeout(zend_long seconds, int reset_signals) {
                   EG(timeout_seconds) = seconds;
                   #ifdef ZEND_WIN32
                     if(!seconds) { return; }
                     if(NULL != tq_timer) {
                       if(!DeleteTimerQueueTimer(NULL, tq_timer, NULL)) {
                         EG(timed_out) = 0;
                         q_timer = NULL;
                         zend_error_noreturn(E_ERROR, "Could not delete queued timer");
                         return;
                       }
                       q_timer = NULL;
                     }
                     if(!CreateTimerQueueTimer(&tq_timer, NULL, (WAITORTIMERCALLBACK)tq_timer_cb,
                                               (VOID *)&EG(timed_out), seconds * 1000, 0, WT_EXECUTEONLYONCE)) {
                       EG(timed_out) = 0;
                       tq_timer = NULL;
                       zend_error_noreturn(E_ERROR, "Could not queue new timer");
                       return;
                     }
                     EG(timed_out) = 0;
                   #else
                     #ifdef HAVE_SETITIMER
                     {
                       struct itimerval t_r;
                       int signo;
                       if(seconds) {
                         t_r.it_value.tv_sec = seconds;
                         t_r.it_value.tv_usec = t_r.it_interval.tv_sec = t_r.it_interval.tv_usec = 0;
                       #ifdef __CYGWIN__
                         setitimer(ITIMER_REAL, &t_r, NULL);
                       }
                       signo = SIGALRM;
                       #else
                         setitimer(ITIMER_PROF, &t_r, NULL);
                       }
                       signo = SIGPROF;
                       #endif
                       if(reset_signals) {
                       #ifdef ZEND_SIGNALS
                         zend_signal(signo, zend_timeout);
                       #else
                         sigset_t sigset;
                         signal(signo, zend_timeout);
                         sigemptyset(&sigset);
                         sigaddset(&sigset, signo);
                         sigprocmask(SIG_UNBLOCK, &sigset, NULL);
                       #endif
                       }
                     }
                   #endif
                   #endif"""
            )
        ],
    )
    def test_balance_directives(self, code_sanitizer_instance: CodeSanitizer, input_code:str, expected_code:str):
        tsp = TreeSitterParser("c")
        result: str = code_sanitizer_instance._balance_directives(code=input_code, tsp=tsp)
        assert "".join(result.split()) == "".join(expected_code.split())

    # ==============================================================================================================
    # `_kr_style_to_ansi` TESTS
    # =============================================================================================================
    # ---
    SIMPLE_KR_INPUT = """
    funcname(a, b) int a; int b; {
        return a > b ? a : b;
    }
    """
    SIMPLE_KR_EXPECTED = """
    int funcname(int a, int b) {
        return a > b ? a : b;
    }
    """
    # ---

    # ---
    SIMPLE_KR_SINGLE_DECL_INPUT = """
    funcname(a, b) int a, b; {
        return a > b ? a : b;
    }
    """
    SIMPLE_KR_SINGLE_DECL_OUTPUT = SIMPLE_KR_EXPECTED
    # ---

    # ---
    SIMPLE_KR_SINGLE_DECL_TYPE_INPUT = """
    int funcname(a, b) int a, b; {
        return a > b ? a : b;
    }
    """
    SIMPLE_KR_SINGLE_DECL_TYPE_OUTPUT = SIMPLE_KR_EXPECTED
    # ---

    # ---
    COMPLEX_KR_INPUT = """
    inline volatile static void process_data(ptr1, ptr2, size, count)
        char *ptr1, *ptr2;
        unsigned int size, count;
    {
        /* function body */
    }
    """
    COMPLEX_KR_EXPECTED = """
    inline volatile static void process_data(char * ptr1, char * ptr2, unsigned int size, unsigned int count)
    {
        /* function body */
    }
    """
    # ---

    # ---
    MACRO_LIKE_INPUT = """
    glue(a, b)(int x, int y)
    {
        return x + y;
    }
    """
    MACRO_LIKE_EXPECTED = MACRO_LIKE_INPUT
    # ---

    # ---
    COMPLEX_MACRO_LIKE_INPUT = """
    glue(cirrus_bitblt_rop_fwd_, ROP_NAME)(CirrusVGAState *s, uint8_t *dst, const uint8_t *src,
                                           int dstpitch, int srcpitch, int bltwidth, int bltheight) {
      int x, y;
      dstpitch -= bltwidth;
      srcpitch -= bltwidth;
      for(y = 0; y < bltheight; y++) {
        for(x = 0; x < bltwidth; x++) {
          ROP_OP(*dst, *src);
          dst++;
          src++;
        }
        dst += dstpitch;
        src += srcpitch;
      }
    }
    """
    COMPLEX_MACRO_LIKE_OUTPUT = COMPLEX_MACRO_LIKE_INPUT
    # ---

    # ---
    MODERN_C_INPUT = """
    int multiply(int x, int y) {
        // This is a valid modern function
        return x * y;
    }
    """
    MODERN_C_EXPECTED = MODERN_C_INPUT
    # ---

    # --- Integration Test ---
    kr_test_cases = [
        pytest.param(SIMPLE_KR_INPUT, SIMPLE_KR_EXPECTED, id="simple_kr_implicit_int"),
        pytest.param(SIMPLE_KR_SINGLE_DECL_INPUT, SIMPLE_KR_SINGLE_DECL_OUTPUT, id="single_decl_kr_implicit_int"),
        pytest.param(SIMPLE_KR_SINGLE_DECL_TYPE_INPUT, SIMPLE_KR_SINGLE_DECL_TYPE_OUTPUT, id="single_decl_kr_explicit_int"),
        pytest.param(COMPLEX_KR_INPUT, COMPLEX_KR_EXPECTED, id="complex_kr_multi_declarations"),
        pytest.param(COMPLEX_MACRO_LIKE_INPUT, COMPLEX_MACRO_LIKE_OUTPUT, id="complex_macro_like"),
        pytest.param(MACRO_LIKE_INPUT, MACRO_LIKE_EXPECTED, id="macro_style_no_change"),
        pytest.param(MODERN_C_INPUT, MODERN_C_EXPECTED, id="modern_c_no_change"),
    ]

    @pytest.mark.parametrize( "input_code, expected_code", kr_test_cases)
    def test_fix_kr_style_function_integration(self, code_sanitizer_instance: CodeSanitizer, input_code:str, expected_code:str):
        # --- Act ---
        actual_code = code_sanitizer_instance._kr_style_to_ansi(input_code, tsp=TreeSitterParser())
        # --- Assert ---
        assert _get_refactored_code(actual_code, lang_name="c", fp="./tmp_in.c")  == _get_refactored_code(expected_code, lang_name="c", fp="./tmp_out.c")


    # ==============================================================================================================
    # `add_missing_return_types` TESTS
    # =============================================================================================================

    # ---
    SIMPLE_KR_INPUT = "my_func() { return 0; }"
    SIMPLE_KR_EXPECTED = "int my_func() { return 0; }"
    # ---
  
    # ---
    MACRO_KR_INPUT = """
    glue(cirrus_bitblt_rop_fwd_, ROP_NAME)(CirrusVGAState *s, uint8_t *dst, const uint8_t *src,
                                           int dstpitch, int srcpitch, int bltwidth, int bltheight) {
      int x, y;
      dstpitch -= bltwidth;
      srcpitch -= bltwidth;
      for(y = 0; y < bltheight; y++) {
        for(x = 0; x < bltwidth; x++) {
          ROP_OP(*dst, *src);
          dst++;
          src++;
        }
        dst += dstpitch;
        src += srcpitch;
      }
    }
    """
    MACRO_KR_EXPECTED = """
    int glue(cirrus_bitblt_rop_fwd_, ROP_NAME)(CirrusVGAState *s, uint8_t *dst, const uint8_t *src,
                                           int dstpitch, int srcpitch, int bltwidth, int bltheight) {
      int x, y;
      dstpitch -= bltwidth;
      srcpitch -= bltwidth;
      for(y = 0; y < bltheight; y++) {
        for(x = 0; x < bltwidth; x++) {
          ROP_OP(*dst, *src);
          dst++;
          src++;
        }
        dst += dstpitch;
        src += srcpitch;
      }
    }
    """
    # ---

    # ---
    MACRO_INPUT = """
    glue(cirrus_bitblt_rop_fwd_, ROP_NAME) {
      int x, y;
      dstpitch -= bltwidth;
      srcpitch -= bltwidth;
      for(y = 0; y < bltheight; y++) {
        for(x = 0; x < bltwidth; x++) {
          ROP_OP(*dst, *src);
          dst++;
          src++;
        }
        dst += dstpitch;
        src += srcpitch;
      }
    }
    """
    MACRO_OUTPUT = """
    int glue(cirrus_bitblt_rop_fwd_, ROP_NAME) {
      int x, y;
      dstpitch -= bltwidth;
      srcpitch -= bltwidth;
      for(y = 0; y < bltheight; y++) {
        for(x = 0; x < bltwidth; x++) {
          ROP_OP(*dst, *src);
          dst++;
          src++;
        }
        dst += dstpitch;
        src += srcpitch;
      }
    }
    """
    # ---
    # ---
    MODERN_FUNC_INPUT = "void modern_func(int x) { return; }"
    MODERN_FUNC_EXPECTED = MODERN_FUNC_INPUT
    # ---

    # ---
    KR_WITH_TYPE_INPUT = """
    static char *kr_with_type(p) char *p; {
        return p;
    }
    """
    KR_WITH_TYPE_EXPECTED = KR_WITH_TYPE_INPUT
    # ---

    # ---
    GLOBAL_VAR_INPUT = "int global_x = 10;"
    GLOBAL_VAR_EXPECTED = GLOBAL_VAR_INPUT

    # --- Integration Test ---
    tests = [
        pytest.param(SIMPLE_KR_INPUT, SIMPLE_KR_EXPECTED, id="simple_kr_function"),
        pytest.param(MACRO_KR_INPUT, MACRO_KR_EXPECTED, id="macro_based_function"),
        pytest.param(MACRO_INPUT, MACRO_OUTPUT, id="macro_function"),
        pytest.param(MODERN_FUNC_INPUT, MODERN_FUNC_EXPECTED, id="valid_modern_function"),
        pytest.param(KR_WITH_TYPE_INPUT, KR_WITH_TYPE_EXPECTED, id="valid_kr_with_type"),
        pytest.param(GLOBAL_VAR_INPUT, GLOBAL_VAR_EXPECTED, id="non_function_global_var"),
    ]

    @pytest.mark.parametrize("input_code, expected_code", tests)
    def test_add_missing_return_types(self, code_sanitizer_instance: CodeSanitizer, input_code: str, expected_code: str):
        # --- Act ---
        actual_code = code_sanitizer_instance.add_missing_return_types(code=input_code, tsp=TreeSitterParser("c"))
        # --- Assert ---
        assert _get_refactored_code(actual_code, lang_name="c", fp="./tmp_candidate.c")  == _get_refactored_code(expected_code, lang_name="c", fp="./tmp_expected.c")
    
