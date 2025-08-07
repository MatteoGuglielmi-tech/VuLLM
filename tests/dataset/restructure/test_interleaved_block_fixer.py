import pytest
import os

from dataset.restructure.proc_utils import write2file, spawn_clang_format
from dataset.restructure.interleaved_block_fixer import InterleavedBlockFixer


@pytest.fixture(scope="module")
def interleaved_block_fixer():
    yield InterleavedBlockFixer()


project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
clang_format_file_path: str = os.path.join(project_root, ".clang-format")


def _get_refactored_code(code: str, lang_name: str, fp: str):
    write2file(fp=fp, content=code)
    formatted_result = spawn_clang_format(fp, lang_name, clang_format_file_path)
    os.remove(fp)

    return formatted_result


class TestRestructurerUnit:
    # ==============================================================================================================
    # INTERLEAVED DO-WHILE STATEMENT TESTS
    # =============================================================================================================
    # ---
    CORRECT_DOWHILE_LOOP_INPUT = """
    void process_data(char* buffer) {
      int i = 0;
      do {
        printf("Processing index %d\n", i);
      #if defined(VERBOSE_MODE)
        log_event(i);
      #endif
        i++;
        } while (i < 10);
    }
    """
    CORRECT_DOWHILE_LOOP_OUTPUT = CORRECT_DOWHILE_LOOP_INPUT
    # ---

    # ---
    BROKEN_NESTED_DO_WHILE_INPUT = """
    void process_data(char *buffer) {
      char *ptr = buffer;
      do {
        *ptr = get_next_char();
    #if defined(PLATFORM_X)
      } while(*ptr != '\\0' && fast_validate(ptr));
    #elif defined(PLATFORM_Y)
      } while(*ptr != '\\0' && platform_y_validate(ptr));
    #else
      } while(*ptr != '\\0');
    #endif
      printf("Finished processing data.");
    }
    """
    BROKEN_NESTED_DO_WHILE_OUTPUT = """
    void process_data(char *buffer) {
      char *ptr = buffer;
      #if defined(PLATFORM_X)
        do { *ptr = get_next_char(); } while(*ptr != '\\0' && fast_validate(ptr));
      #elif defined(PLATFORM_Y)
        do { *ptr = get_next_char(); } while(*ptr != '\\0' && platform_y_validate(ptr));
      #else
        do { *ptr = get_next_char(); } while(*ptr != '\\0');
      #endif
        printf("Finished processing data.");
      }
    """
    # ---

    # ---
    BROKEN_DOUBLE_NESTED_DO_WHILE_INPUT = """
    void process_data(char *buffer) {
      char *ptr = buffer;
      do {
        *ptr = get_next_char();
    #if defined(PLATFORM_X)
      #if defined(FAST_VALIDATION)
      } while(*ptr != '\\0' && fast_validate(ptr));
      #else
      } while(*ptr != '\\0' && full_validate(ptr));
      #endif
    #elif defined(PLATFORM_Y)
      } while(*ptr != '\\0' && platform_y_validate(ptr));
    #else
      } while(*ptr != '\\0');
    #endif
      printf("Finished processing data.");
    }
    """
    BROKEN_DOUBLE_NESTED_DO_WHILE_OUTPUT = """
    void process_data(char *buffer) {
    char *ptr = buffer;
    #if defined(PLATFORM_X)
       #if defined(FAST_VALIDATION)
       do { *ptr = get_next_char(); } while(*ptr != '\\0' && fast_validate(ptr));
       #else
       do { *ptr = get_next_char(); } while(*ptr != '\\0' && full_validate(ptr));
       #endif
     #elif defined(PLATFORM_Y)
       do { *ptr = get_next_char(); } while(*ptr != '\\0' && platform_y_validate(ptr));
     #else
       do { *ptr = get_next_char(); } while(*ptr != '\\0');
     #endif
       printf("Finished processing data.");
     }
    """
    # ---

    # ---
    INTERLEAVED_IF_MULTIPLE_OPENERS_ONE_CLOSER_INPUT = """
    ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
      ssize_t ret = 1;
    #if defined(EWOULDBLOCK)
      if(ret > 1) {
    #elif defined(EWOULDBLOCK)
      if(ret < 1) {
    #else
      if(ret == 1) {
    #endif
        printf("this is another complex scenario where multiple openers and one closer");
    }
    """
    # ---

    # ---
    LONG_PROBLEMATIC_REAL_INPUT = """
    unpack_Z_stream(int fd_in, int fd_out) {
        IF_DESKTOP(long long total_written = 0;)
        IF_DESKTOP(long long) int retval = -1;
        unsigned char *stackp;
        long code;
        int finchar;
        long oldcode;
        long incode;
        int inbits;
        int posbits;
        int outpos;
        int insize;
        int bitmask;
        long free_ent;
        long maxcode;
        long maxmaxcode;
        int n_bits;
        int rsize = 0;
        unsigned char *inbuf;
        unsigned char *outbuf;
        unsigned char *htab;
        unsigned short *codetab;

        int maxbits;

        int block_mode;

        inbuf = xzalloc(IBUFSIZ + 64);
        outbuf = xzalloc(OBUFSIZ + 2048);
        htab = xzalloc(HSIZE);
        codetab = xzalloc(HSIZE * sizeof(codetab[0]));

        insize = 0;

        if(full_read(fd_in, inbuf, 1) != 1) {
          bb_error_msg("short read");
          goto err;
        }

        maxbits = inbuf[0] & BIT_MASK;
        block_mode = inbuf[0] & BLOCK_MODE;
        maxmaxcode = MAXCODE(maxbits);

        if(maxbits > BITS) {
          bb_error_msg("compressed with %d bits, can only handle " BITS_STR " bits", maxbits);
          goto err;
        }

        n_bits = INIT_BITS;
        maxcode = MAXCODE(INIT_BITS) - 1;
        bitmask = (1 << INIT_BITS) - 1;
        oldcode = -1;
        finchar = 0;
        outpos = 0;
        posbits = 0 << 3;

        free_ent = ((block_mode) ? FIRST : 256);

        for(code = 255; code >= 0; --code) { tab_suffixof(code) = (unsigned char)code; }

        do {
        resetbuf: {
          int i;
          int e;
          int o;

          o = posbits >> 3;
          e = insize - o;

          for(i = 0; i < e; ++i) inbuf[i] = inbuf[i + o];

          insize = e;
          posbits = 0;
        }

          if(insize < (int)(IBUFSIZ + 64) - IBUFSIZ) {
            rsize = safe_read(fd_in, inbuf + insize, IBUFSIZ);

            insize += rsize;
          }

          inbits = ((rsize > 0) ? (insize - insize % n_bits) << 3 : (insize << 3) - (n_bits - 1));

          while(inbits > posbits) {
            if(free_ent > maxcode) {
              posbits = ((posbits - 1) + ((n_bits << 3) - (posbits - 1 + (n_bits << 3)) % (n_bits << 3)));
              ++n_bits;
              if(n_bits == maxbits) {
                maxcode = maxmaxcode;
              } else {
                maxcode = MAXCODE(n_bits) - 1;
              }
              bitmask = (1 << n_bits) - 1;
              goto resetbuf;
            }
            {
              unsigned char *p = &inbuf[posbits >> 3];

              code = ((((long)(p[0])) | ((long)(p[1]) << 8) | ((long)(p[2]) << 16)) >> (posbits & 0x7)) &
                     bitmask;
            }
            posbits += n_bits;

            if(oldcode == -1) {
              oldcode = code;
              finchar = (int)oldcode;
              outbuf[outpos++] = (unsigned char)finchar;
              continue;
            }

            if(code == CLEAR && block_mode) {
              clear_tab_prefixof();
              free_ent = FIRST - 1;
              posbits = ((posbits - 1) + ((n_bits << 3) - (posbits - 1 + (n_bits << 3)) % (n_bits << 3)));
              n_bits = INIT_BITS;
              maxcode = MAXCODE(INIT_BITS) - 1;
              bitmask = (1 << INIT_BITS) - 1;
              goto resetbuf;
            }

            incode = code;
            stackp = de_stack;

            if(code >= free_ent) {
              if(code > free_ent) {
                unsigned char *p;

                posbits -= n_bits;
                p = &inbuf[posbits >> 3];

                bb_error_msg("insize:%d posbits:%d inbuf:%02X %02X %02X %02X %02X (%d)", insize, posbits,
                             p[-1], p[0], p[1], p[2], p[3], posbits & 07);
                bb_error_msg("corrupted data");
                goto err;
              }

              *--stackp = (unsigned char)finchar;
              code = oldcode;
            }

            while((long)code >= (long)256) {
              *--stackp = tab_suffixof(code);
              code = tab_prefixof(code);
            }

            finchar = tab_suffixof(code);
            *--stackp = (unsigned char)finchar;

            {
              int i;

              i = de_stack - stackp;
              if(outpos + i >= OBUFSIZ) {
                do {
                  if(i > OBUFSIZ - outpos) { i = OBUFSIZ - outpos; }

                  if(i > 0) {
                    memcpy(outbuf + outpos, stackp, i);
                    outpos += i;
                  }

                  if(outpos >= OBUFSIZ) {
                    full_write(fd_out, outbuf, outpos);

                    IF_DESKTOP(total_written += outpos;)
                    outpos = 0;
                  }
                  stackp += i;
                  i = de_stack - stackp;
                } while(i > 0);
              } else {
                memcpy(outbuf + outpos, stackp, i);
                outpos += i;
              }
            }

            code = free_ent;
            if(code < maxmaxcode) {
              tab_prefixof(code) = (unsigned short)oldcode;
              tab_suffixof(code) = (unsigned char)finchar;
              free_ent = code + 1;
            }

            oldcode = incode;
          }

        } while(rsize > 0);

        if(outpos > 0) {
          full_write(fd_out, outbuf, outpos);

          IF_DESKTOP(total_written += outpos;)
        }

        retval = IF_DESKTOP(total_written) + 0;

    err:
      free(inbuf);
      free(outbuf);
      free(htab);
      free(codetab);
      return retval;
    }
    """
    LONG_PROBLEMATIC_REAL_OUTPUT = LONG_PROBLEMATIC_REAL_INPUT
    # ---

    interleaved_dowhile_test_cases = [
        pytest.param(CORRECT_DOWHILE_LOOP_INPUT, CORRECT_DOWHILE_LOOP_OUTPUT, "c", id="interleaved_loop"),
        pytest.param(BROKEN_NESTED_DO_WHILE_INPUT, BROKEN_NESTED_DO_WHILE_OUTPUT, "c", id="interleaved_do_while"),
        pytest.param(BROKEN_DOUBLE_NESTED_DO_WHILE_INPUT, BROKEN_DOUBLE_NESTED_DO_WHILE_OUTPUT, "c", id="interleaved_double_do_while"),
        pytest.param(INTERLEAVED_IF_MULTIPLE_OPENERS_ONE_CLOSER_INPUT, INTERLEAVED_IF_MULTIPLE_OPENERS_ONE_CLOSER_INPUT, "c", id="non_dowhile_interleaved_statement")
    ]
    @pytest.mark.parametrize("input, output, lang", interleaved_dowhile_test_cases)
    def test_fix_interleaved_do_while(self, interleaved_block_fixer: InterleavedBlockFixer, input: str, output: str, lang: str):

        raw_result = interleaved_block_fixer._fix_interleaved_do_while(input)
        formatted_result = _get_refactored_code(code=raw_result, lang_name=lang, fp=f"./tmp_in_do_while.{lang}")
        formatted_output = _get_refactored_code(code=output, lang_name=lang, fp=f"./tmp_out_do_while.{lang}")

        assert formatted_result.strip() == formatted_output.strip()

    # ==============================================================================================================
    # INTERLEAVED CONTENT TESTS
    # ==============================================================================================================

    # ---
    BROKEN_INTERLEAVED_FOR_CONTENT_INPUT = """
    void print_values(int max) {
      for(int i = 0; i < max; i++) {
    #if defined(debug)
        printf("value: %d", i);
      }
    #else
        printf("%d", i);
      }
    #endif
    """
    BROKEN_INTERLEAVED_FOR_CONTENT_OUTPUT = """
    void print_values(int max) {
    #if defined(debug)
      for(int i = 0; i < max; i++) {
        printf("value: %d", i);
      }
    #else
      for(int i = 0; i < max; i++) {
        printf("%d", i);
      }
    #endif
    """

    interleaved_content_test_cases = [
        pytest.param(BROKEN_INTERLEAVED_FOR_CONTENT_INPUT, BROKEN_INTERLEAVED_FOR_CONTENT_OUTPUT, "c", id="interleaved_content_for_loop"),
    ]
    @pytest.mark.parametrize("input, output, lang", interleaved_content_test_cases)
    def test_fix_interleaved_content(
        self,
        interleaved_block_fixer: InterleavedBlockFixer,
        input: str,
        output: str,
        lang: str,
    ):

        raw_result = interleaved_block_fixer._fix_interleaved_statement_content(input)
        formatted_result = _get_refactored_code(code=raw_result, lang_name=lang, fp=f"./tmp_in_do_while.{lang}")
        formatted_output = _get_refactored_code(code=output, lang_name=lang, fp=f"./tmp_out_do_while.{lang}")

        assert formatted_result.strip() == formatted_output.strip()

    # ==============================================================================================================
    # FULL PIPELINE TESTS
    # ==============================================================================================================
    
    # --- 
    INTERLEAVED_IF_MULTIPLE_OPENERS_ONE_CLOSER_OUPUT = """
    ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
      ssize_t ret = 1;
    #if defined(EWOULDBLOCK)
      if(ret > 1) { printf("this is another complex scenario where multiple openers and one closer"); }
    #elif defined(EWOULDBLOCK)
      if(ret < 1) { printf("this is another complex scenario where multiple openers and one closer"); }
    #else
      if(ret == 1) { printf("this is another complex scenario where multiple openers and one closer"); }
    #endif
    """
    # --- 

    # ---
    NO_INTERLEAVED_STATEMENTS_INPUT = """
    ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
      ssize_t ret = 1;
    #if defined(EWOULDBLOCK)
      if(ret > 1) { printf("this is another complex scenario where multiple openers and one closer"); }
    #elif defined(EWOULDBLOCK)
      if(ret < 1) { printf("this is another complex scenario where multiple openers and one closer"); }
    #else
      if(ret == 1) { printf("this is another complex scenario where multiple openers and one closer"); }
    #endif
    """
    NO_INTERLEAVED_STATEMENTS_OUTPUT = NO_INTERLEAVED_STATEMENTS_INPUT
    # ---
  
    # ---
    BROKEN_INTERLEAVED_IF_ELIF_ELSE_ENDIF_INPUT = """
    #if defined(CONFIG_A)
    if(x > 100 && check_status(y)) {
    #elif defined(CONFIG_B)
      #if defined(SUB_CONFIG)
    if(z == 0) {
      #else
    if(z != 0) {
      #endif
    #else
    if(x < 0) {
    #endif
      int result = process_data(x, y, z);
      log_result(result);
    }
    """
    BROKEN_INTERLEAVED_IF_ELIF_ELSE_ENDIF_OUTPUT = """
    #if defined(CONFIG_A)
    if(x > 100 && check_status(y)) {
      int result = process_data(x, y, z);
      log_result(result);
    }
    #elif defined(CONFIG_B)
      #if defined(SUB_CONFIG)
    if(z == 0) {
      int result = process_data(x, y, z);
      log_result(result);
    }
      #else
    if(z != 0) {
      int result = process_data(x, y, z);
      log_result(result);
    }
      #endif
    #else
    if(x < 0) {
      int result = process_data(x, y, z);
      log_result(result);
    }
    #endif
    """
    # ---

    # ---
    BROKEN_NESTED_INTERLEAVED_SWITCH_INPUT = """
    #if defined(use_complex_id)
    switch(get_id(user, true)) {
    #else
      #if defined(simple_id)
    switch(user->id) {
      #else
    switch(0) {
      #endif
    #endif
      case 0 : handle_guest(); break;
      case 10 :
      case 20 : handle_admin(); break;
      default : handle_error(); break;
    }
    """
    BROKEN_NESTED_INTERLEAVED_SWITCH_OUTPUT = """
    #if defined(use_complex_id)
    switch(get_id(user, true)) {
      case 0 : handle_guest(); break;
      case 10 :
      case 20 : handle_admin(); break;
      default : handle_error(); break;
    }
    #else
      #if defined(simple_id)
    switch(user->id) {
      case 0 : handle_guest(); break;
      case 10 :
      case 20 : handle_admin(); break;
      default : handle_error(); break;
    }
      #else
    switch(0) {
      case 0 : handle_guest(); break;
      case 10 :
      case 20 : handle_admin(); break;
      default : handle_error(); break;
    }
      #endif
    #endif
    """
    # ---
  
    # ---
    BROKEN_SIMPLE_INTERLEAVED_SWITCH_INPUT = """
    #if defined(use_integer_codes)
    switch(get_int_code()) {
    #else
    switch(get_char_code()) {
    #endif
      case 1 :
        //...
        break;
      default :
        //...
        break;
    }
    """
    BROKEN_SIMPLE_INTERLEAVED_SWITCH_OUTPUT = """
    #if defined(use_integer_codes)
    switch(get_int_code()) {
      case 1 :
        //...
        break;
      default :
        //...
        break;
    }
    #else
    switch(get_char_code()) {
      case 1 :
        //...
        break;
      default :
        //...
        break;
    }
    #endif
    """
    # ---
    BROKEN_INTERLEAVED_SIMPLE_WHILE_LOOP_INPUT = """
    #ifdef fast_mode
    while(fast_check()) {
    #else
    while(slow_check()) {
    #endif
      do_work();
    }
    """
    BROKEN_INTERLEAVED_SIMPLE_WHILE_LOOP_OUTPUT = """
    #ifdef fast_mode
    while(fast_check()) { do_work(); }
    #else
    while(slow_check()) { do_work(); }
    #endif
    """
    # ---

    # ---
    BROKEN_INTERLEAVED_WHILE_LOOP_INPUT = """
    #if mode == 1
    while((c = getchar()) != eof) {
    #elif mode == 2
    while(read_buffer(buf) > 0) {
    #else
    while(true) {
    #endif
      if(is_special(c)) { process_special(c); }
      counter++;
    }
    """
    BROKEN_INTERLEAVED_WHILE_LOOP_OUTPUT = """
    #if mode == 1
    while((c = getchar()) != eof) {
      if(is_special(c)) { process_special(c); }
      counter++;
    }
    #elif mode == 2
    while(read_buffer(buf) > 0) {
      if(is_special(c)) { process_special(c); }
      counter++;
    }
    #else
    while(true) {
      if(is_special(c)) { process_special(c); }
      counter++;
    }
    #endif
    """
    # ---

    # ---
    BROKEN_INTERLEAVED_SIMPLE_FOR_LOOP_INPUT = """
    #ifdef fast_mode
    for(fast_check()) {
    #else
    for(slow_check()) {
    #endif
      do_work();
    }
    """
    BROKEN_INTERLEAVED_SIMPLE_FOR_LOOP_OUTPUT = """
    #ifdef fast_mode
    for(fast_check()) { do_work(); }
    #else
    for(slow_check()) { do_work(); }
    #endif
    """
    # ---

    # ---
    BROKEN_INTERLEAVED_FOR_LOOP_INPUT = """
    #if defined(iterate_forward)
    for(int i = 0; i < max_items; i++) {
    #elif defined(iterate_backward)
    for(int i = max_items - 1; i >= 0; i--) {
    #else
    for(item_t *p = list_head; p != null; p = p->next) {
    #endif
      process_item(p);
    }
    """
    BROKEN_INTERLEAVED_FOR_LOOP_OUTPUT = """
    #if defined(iterate_forward)
    for(int i = 0; i < max_items; i++) { process_item(p); }
    #elif defined(iterate_backward)
    for(int i = max_items - 1; i >= 0; i--) { process_item(p); }
    #else
    for(item_t *p = list_head; p != null; p = p->next) { process_item(p); }
    #endif
    """
    # ---

    full_pipeline_test_cases = [
        pytest.param(NO_INTERLEAVED_STATEMENTS_INPUT, NO_INTERLEAVED_STATEMENTS_OUTPUT, "c", id="no_interleaved_statements"),
        pytest.param(INTERLEAVED_IF_MULTIPLE_OPENERS_ONE_CLOSER_INPUT, INTERLEAVED_IF_MULTIPLE_OPENERS_ONE_CLOSER_OUPUT, "c", id="interleaved_if_multiple_openers"),
        pytest.param(BROKEN_INTERLEAVED_IF_ELIF_ELSE_ENDIF_INPUT, BROKEN_INTERLEAVED_IF_ELIF_ELSE_ENDIF_OUTPUT, "c", id="interleaved_if_elif_else_endif"),
        pytest.param(BROKEN_SIMPLE_INTERLEAVED_SWITCH_INPUT, BROKEN_SIMPLE_INTERLEAVED_SWITCH_OUTPUT, "c", id="interleaved_switch_simple"),
        pytest.param(BROKEN_NESTED_INTERLEAVED_SWITCH_INPUT, BROKEN_NESTED_INTERLEAVED_SWITCH_OUTPUT, "c", id="interleaved_switch"),
        pytest.param(BROKEN_INTERLEAVED_SIMPLE_WHILE_LOOP_INPUT, BROKEN_INTERLEAVED_SIMPLE_WHILE_LOOP_OUTPUT, "c", id="interleaved_simple_while"),
        pytest.param(BROKEN_INTERLEAVED_WHILE_LOOP_INPUT, BROKEN_INTERLEAVED_WHILE_LOOP_OUTPUT, "c", id="interleaved_while"),
        pytest.param(BROKEN_INTERLEAVED_SIMPLE_FOR_LOOP_INPUT, BROKEN_INTERLEAVED_SIMPLE_FOR_LOOP_OUTPUT, "c", id="interleaved_simple_for"),
        pytest.param(BROKEN_INTERLEAVED_FOR_LOOP_INPUT, BROKEN_INTERLEAVED_FOR_LOOP_OUTPUT, "c", id="interleaved_for"),
        pytest.param(BROKEN_INTERLEAVED_FOR_CONTENT_INPUT, BROKEN_INTERLEAVED_FOR_CONTENT_OUTPUT, "c", id="interleaved_content_for_loop"),
        pytest.param(LONG_PROBLEMATIC_REAL_INPUT, LONG_PROBLEMATIC_REAL_OUTPUT, "c", id="problematic_long_real_testcase"),
    ]
    @pytest.mark.parametrize("input, output, lang", full_pipeline_test_cases)
    def test_interleaved(self, interleaved_block_fixer: InterleavedBlockFixer, input: str, output: str, lang: str):

        code = interleaved_block_fixer.full_structural_refactor(input)
        formatted_result = _get_refactored_code(code=code, lang_name=lang, fp=f"./tmp_in.{lang}")
        formatted_output = _get_refactored_code(code=output, lang_name=lang, fp=f"./tmp_out.{lang}")

        assert formatted_result.strip() == formatted_output.strip()
    # ==============================================================================================================
