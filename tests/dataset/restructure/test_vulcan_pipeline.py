import pytest
import json
import os

from dataset.restructure.shared.proc_utils import write2file, spawn_clang_format


# ============================ UTILITIES ==============================
project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
clang_format_file_path: str = os.path.join(project_root, ".clang-format")

def _get_refactored_code(code: str, lang_name: str, fp: str):
    write2file(fp=fp, content=code)
    formatted_result = spawn_clang_format(fp, lang_name, clang_format_file_path)
    os.remove(fp)

    return formatted_result
# ======================================================================

# ---
SIMPLE_CASE_INPUT = """
int add(int a, int b) {
    return a + b;
}
"""
SIMPLE_CASE_EXPECTED = """
int add(int a, int b) {
    return a + b;
}
"""

# ---
REAL_LIKE_INPUT = """
int process_data(int *data, int size) {
    int total = 0;
    // Process all elements
    for (int i = 0; i < size; ++i) {
        if (data[i] > 0) {{ // Extra braces
            total += data[i];
        }}
    }
    return total;
}
"""
REAL_LIKE_EXPECTED = """
int process_data(int *data, int size) {
    int total = 0;
    for (int i = 0; i < size; ++i) {
        if (data[i] > 0) {
            total += data[i];
        }
    }
    return total;
}
"""

# ---
UNBALANCED_DIRECTIVES_INPUT = """
int get_status(void) {
    int status = 0;
#else
    status = -1;
#endif
    return status;
#endif
"""
UNBALANCED_DIRECTIVES_EXPECTED = """
int get_status(void) {
    int status = 0;
    status = -1;
    return status;
}
"""

# ==============================================================================================================
# INTERLEAVED DO-WHILE STATEMENT TESTS
# =============================================================================================================
# ---
CORRECT_DOWHILE_LOOP_INPUT = """
void process_data(char* buffer) {
  int i = 0;
  do {
    printf("Processing index %d\\n", i);
  #if defined(VERBOSE_MODE)
    log_event(i);
  #endif
    i++;
    } while (i < 10);
}
"""
CORRECT_DOWHILE_LOOP_OUTPUT = """
void process_data(char* buffer) {
  int i = 0;
  do {
    printf("Processing index %d\\n", i);
  #if defined(VERBOSE_MODE)
    log_event(i);
  #endif
    i++;
    } while (i < 10);
}
"""
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


interleaved_dowhile_test_cases = [
    pytest.param(CORRECT_DOWHILE_LOOP_INPUT, CORRECT_DOWHILE_LOOP_OUTPUT, "c", id="interleaved_loop"),
    pytest.param(BROKEN_NESTED_DO_WHILE_INPUT, BROKEN_NESTED_DO_WHILE_OUTPUT, "c", id="interleaved_do_while"),
    pytest.param(BROKEN_DOUBLE_NESTED_DO_WHILE_INPUT, BROKEN_DOUBLE_NESTED_DO_WHILE_OUTPUT, "c", id="interleaved_double_do_while"),
]


# ==============================================================================================================
# INTERLEAVED CONTENT TESTS
# ==============================================================================================================

# ---
BROKEN_INTERLEAVED_FOR_CONTENT_INPUT = """
void print_values(int max) {
  for(int i = 0; i < max; i++) {
#if defined(debug)
    printf("value: %d\\n", i);
  }
#else
    printf("%d\\n", i);
  }
#endif
"""
BROKEN_INTERLEAVED_FOR_CONTENT_OUTPUT = """
void print_values(int max) {
#if defined(debug)
  for(int i = 0; i < max; i++) {
    printf("value: %d\\n", i);
  }
#else
  for(int i = 0; i < max; i++) {
    printf("%d\\n", i);
  }
#endif
}
"""

interleaved_content_test_cases = [
    pytest.param(BROKEN_INTERLEAVED_FOR_CONTENT_INPUT, BROKEN_INTERLEAVED_FOR_CONTENT_OUTPUT, "c", id="interleaved_content_for_loop"),
]


# ==============================================================================================================
# FULL PIPELINE TESTS
# ==============================================================================================================

# ---
# input previously defined in do_while test
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
}
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
}
"""
NO_INTERLEAVED_STATEMENTS_OUTPUT = NO_INTERLEAVED_STATEMENTS_INPUT
# ---

# ---
BROKEN_INTERLEAVED_IF_ELIF_ELSE_ENDIF_INPUT = """
int interleaved_if(void){
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
}
"""
BROKEN_INTERLEAVED_IF_ELIF_ELSE_ENDIF_OUTPUT = """
int interleaved_if(void){
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
}
"""
# ---

# ---
BROKEN_NESTED_INTERLEAVED_SWITCH_INPUT = """
int interleaved_switch(void){
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
}
"""
BROKEN_NESTED_INTERLEAVED_SWITCH_OUTPUT = """
int interleaved_switch(void){
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
}
"""
# ---

# ---
BROKEN_SIMPLE_INTERLEAVED_SWITCH_INPUT = """
int interleaved_switch(void){
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
}
"""
BROKEN_SIMPLE_INTERLEAVED_SWITCH_OUTPUT = """
int interleaved_switch(void){
  #if defined(use_integer_codes)
  switch(get_int_code()) {
    case 1 :
      break;
    default :
      break;
  }
  #else
  switch(get_char_code()) {
    case 1 :
      break;
    default :
      break;
  }
  #endif
}
"""
# ---
BROKEN_INTERLEAVED_SIMPLE_WHILE_LOOP_INPUT = """
int interleaved_while(void){
  #ifdef fast_mode
  while(fast_check()) {
  #else
  while(slow_check()) {
  #endif
    do_work();
  }
}
"""
BROKEN_INTERLEAVED_SIMPLE_WHILE_LOOP_OUTPUT = """
int interleaved_while(void){
  #ifdef fast_mode
  while(fast_check()) { do_work(); }
  #else
  while(slow_check()) { do_work(); }
  #endif
}
"""
# ---

# ---
BROKEN_INTERLEAVED_WHILE_LOOP_INPUT = """
int interleaved_while(void){
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
}
"""
BROKEN_INTERLEAVED_WHILE_LOOP_OUTPUT = """
int interleaved_while(void){
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
}
"""
# ---

# ---
BROKEN_INTERLEAVED_SIMPLE_FOR_LOOP_INPUT = """
int interleaved_for(void){
  #ifdef fast_mode
  for(;fast_check();) {
  #else
  for(;slow_check();) {
  #endif
    do_work();
  }
}
"""
BROKEN_INTERLEAVED_SIMPLE_FOR_LOOP_OUTPUT = """
int interleaved_for(void){
  #ifdef fast_mode
  for(;fast_check();) { do_work(); }
  #else
  for(;slow_check();) { do_work(); }
  #endif
}
"""
# ---

# ---
BROKEN_INTERLEAVED_FOR_LOOP_INPUT = """
int interleaved_for(void){
  #if defined(iterate_forward)
  for(int i = 0; i < max_items; i++) {
  #elif defined(iterate_backward)
  for(int i = max_items - 1; i >= 0; i--) {
  #else
  for(item_t *p = list_head; p != null; p = p->next) {
  #endif
    process_item(p);
  }
}
"""
BROKEN_INTERLEAVED_FOR_LOOP_OUTPUT = """
int interleaved_for(void){
 #if defined(iterate_forward)
 for(int i = 0; i < max_items; i++) { process_item(p); }
 #elif defined(iterate_backward)
 for(int i = max_items - 1; i >= 0; i--) { process_item(p); }
 #else
 for(item_t *p = list_head; p != null; p = p->next) { process_item(p); }
 #endif
}
"""
# ---


full_pipeline_test_cases = [
    pytest.param(INTERLEAVED_IF_MULTIPLE_OPENERS_ONE_CLOSER_INPUT, INTERLEAVED_IF_MULTIPLE_OPENERS_ONE_CLOSER_OUPUT, "c", id="interleaved_if_multiple_openers"),
    pytest.param(NO_INTERLEAVED_STATEMENTS_INPUT, NO_INTERLEAVED_STATEMENTS_OUTPUT, "c", id="no_interleaved_statements"),
    pytest.param(BROKEN_INTERLEAVED_IF_ELIF_ELSE_ENDIF_INPUT, BROKEN_INTERLEAVED_IF_ELIF_ELSE_ENDIF_OUTPUT, "c", id="interleaved_if_elif_else_endif"),
    pytest.param(BROKEN_SIMPLE_INTERLEAVED_SWITCH_INPUT, BROKEN_SIMPLE_INTERLEAVED_SWITCH_OUTPUT, "c", id="interleaved_switch_simple"),
    pytest.param(BROKEN_NESTED_INTERLEAVED_SWITCH_INPUT, BROKEN_NESTED_INTERLEAVED_SWITCH_OUTPUT, "c", id="interleaved_switch"),
    pytest.param(BROKEN_INTERLEAVED_SIMPLE_WHILE_LOOP_INPUT, BROKEN_INTERLEAVED_SIMPLE_WHILE_LOOP_OUTPUT, "c", id="interleaved_simple_while"),
    pytest.param(BROKEN_INTERLEAVED_WHILE_LOOP_INPUT, BROKEN_INTERLEAVED_WHILE_LOOP_OUTPUT, "c", id="interleaved_while"),
    pytest.param(BROKEN_INTERLEAVED_SIMPLE_FOR_LOOP_INPUT, BROKEN_INTERLEAVED_SIMPLE_FOR_LOOP_OUTPUT, "c", id="interleaved_simple_for"),
    pytest.param(BROKEN_INTERLEAVED_FOR_LOOP_INPUT, BROKEN_INTERLEAVED_FOR_LOOP_OUTPUT, "c", id="interleaved_for"),
]

# --- Pytest Parametrization ---
test_cases = interleaved_content_test_cases + interleaved_dowhile_test_cases + full_pipeline_test_cases + [
    pytest.param(SIMPLE_CASE_INPUT, SIMPLE_CASE_EXPECTED, "c", id="simple_case"),
    pytest.param(REAL_LIKE_INPUT, REAL_LIKE_EXPECTED, "c", id="real_like_case"),
    pytest.param(UNBALANCED_DIRECTIVES_INPUT, UNBALANCED_DIRECTIVES_EXPECTED, "c", id="unbalanced_directives"),
]


@pytest.mark.parametrize("input_code, expected_code, lang", test_cases)
def test_vulcan_full_pipeline(vulcan_pipeline, input_code: str, expected_code: str, lang: str):
    """An integration test that runs a code snippet through the entire Vulcan pipeline."""
    # --- 1. Setup ---
    input_path = vulcan_pipeline.config["default_input_path"]
    output_path = vulcan_pipeline.config["default_output_path"]

    input_entry = {"func": input_code, "cwe": "CWE-125"} # Added CWE for full enrichment
    with open(input_path, "w") as f:
        f.write(json.dumps(input_entry) + "\n")

    # --- 2. Execute ---
    vulcan_pipeline.run()

    # --- 3. Assert ---
    assert os.path.exists(output_path), "Output file was not created"

    with open(output_path, "r") as f:
        result_line = f.readline()
    assert result_line, "Output file is empty"
    result_data = json.loads(result_line)

    processed_func = result_data.get("func", "").strip()
    expected_code = _get_refactored_code(expected_code, lang, f"./tmp_expected.{lang}")
    assert processed_func == expected_code

    # Assert that the enrichment from the MOCK object worked
    assert result_data.get("function_description") == "This is a mock function description."
    assert result_data.get("vulnerability_description") == "This is a mock CWE description."
    assert "cyclomatic_complexity" in result_data
    assert result_data.get("language") == lang


# --- Pytest Parametrization ---
NON_VALID_FUNCTION_1 = """
int lookup_count;
bool is_ram_cache_hit;

_CacheLookupInfo()
: action(CACHE_DO_UNDEFINED),
transform_action(CACHE_DO_UNDEFINED),
write_status(NO_CACHE_WRITE),
transform_write_status(NO_CACHE_WRITE),
lookup_url(NULL),
lookup_url_storage(),
original_url(),
object_read(NULL),
second_object_read(NULL),
object_store(),
transform_store(),
config(),
directives(),
open_read_retries(0),
open_write_retries(0),
write_lock_state(CACHE_WL_INIT),
"""

NON_VALID_FUNCTION_2 = """
bool logging_enabled;
bool retry_intercept_failures;

_HttpApiInfo()
: parent_proxy_name(NULL),
"""

NON_VALID_FUNCTION_3 = """
ServerState_t state;
int attempts;

_CurrentInfo()
"""

test_cases = [
        pytest.param(" ", "error: non valid function (empty or comments only)", id="empty_entry"),
        pytest.param("/// @c true if the connection is transparent.", "error: non valid function (empty or comments only)", id="only_comment"),
        pytest.param(NON_VALID_FUNCTION_1, "error: non valid function (no function definition found)", id="non_valid_function_1"),
        pytest.param(NON_VALID_FUNCTION_2, "error: non valid function (no function definition found)", id="non_valid_function_2"),
        pytest.param(NON_VALID_FUNCTION_3, "error: non valid function (no function definition found)", id="non_valid_function_3"),
]

@pytest.mark.parametrize("input_code, expected_code", test_cases)
def test_pipeline_rejects_empty_string_and_only_comments(vulcan_pipeline, input_code:str, expected_code:str):
    input_data = {"func": input_code}
    result = vulcan_pipeline._process_snippet(input_data)
    assert result["func"] == expected_code


def test_vulcan_full_pipeline_with_real_dataset(vulcan_pipeline):
    # --- 1. Setup ---
    real_dataset_path = "/home/matteo/dev/VuLLM/DiverseVul/raw/small_dataset.jsonl"
    # Get the input and output paths from the pipeline's configuration
    # pipeline_input_path = vulcan_pipeline.config["default_input_path"]
    pipeline_output_path = vulcan_pipeline.config["default_output_path"]
    assert os.path.exists(real_dataset_path), "The real dataset file was not found."

    # --- 2. Execute ---
    vulcan_pipeline.run()
    # --- 3. Assert ---
    assert os.path.exists(pipeline_output_path), "Output file was not created"

    with open(real_dataset_path, "r") as f_orig, open(pipeline_output_path, "r") as f_proc:
        original_lines = f_orig.readlines()
        processed_lines = f_proc.readlines()

    assert len(original_lines) == len(processed_lines), "Output file has a different number of lines than the input."

    for i, (original_line, processed_line) in enumerate(zip(original_lines, processed_lines)):
        # original_data = json.loads(original_line)
        processed_data = json.loads(processed_line)
        if not processed_data.get("func", "").startswith(("error:", "skipped:")):
            assert "function_description" in processed_data, f"Missing 'function_description' on line {i}"
            assert "vulnerability_description" in processed_data, f"Missing 'vulnerability_description' on line {i}"
            assert "language" in processed_data, f"Missing 'language' on line {i}"
            assert "cyclomatic_complexity" in processed_data, f"Missing 'cyclomatic_complexity' on line {i}"
            assert "language" in processed_data, f"Missing 'language' on line {i}"



