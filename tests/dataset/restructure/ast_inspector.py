from tree_sitter import Language, Parser, Node
from collections.abc import Generator
from tree_sitter_c import language as c_language

class TreeSitter:
    def __init__(self) -> None:
        C_LANGUAGE = Language(c_language())
        self.language : Language = C_LANGUAGE
        self.parser = Parser(self.language)

    def parse_input(self, code_snippet: str):
        self.tree = self.parser.parse(bytes(code_snippet, "utf8"))

    def traverse_tree(self) -> Generator[Node, None, None]:
        if not self.tree: return
        cursor = self.tree.walk()
        reached_root = False
        while not reached_root:
            assert cursor.node is not None
            yield cursor.node
            if cursor.goto_first_child(): continue
            if cursor.goto_next_sibling(): continue
            while True:
                if not cursor.goto_parent():
                    reached_root = True
                    break
                if cursor.goto_next_sibling():
                    break

# The complex C code from the failing test
# c_code = """ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from,
#                      socklen_t *fromlen) {
#   ssize_t ret;
#   do {
#     ret = recvfrom(s, buf, len, flags, from, fromlen);
# #if defined(EWOULDBLOCK)
#   } while(ret == -1 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK));
# #else
# } while (ret == -1 && (errno == EINTR || errno == EAGAIN));"""

# c_code = """ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
#     ssize_t ret;
#     do {
#         ret = recvfrom(s, buf, len, flags, from, fromlen);
#         #if defined(EWOULDBLOCK)
#             } while (ret == -1 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK));
#         #else
#             } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
#         #endif"""
#
# c_code = """#if A
#       #if B
#         #if C
#         #endif
#       #endif
#     #if D
#     """
# c_code = """#endif
#             #if A"""

broken_code = """ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
  ssize_t ret;
  do {
      ret = recvfrom(s, buf, len, flags, from, fromlen);
  #if defined(EWOULDBLOCK)
    printf("this is another complex scenario where multiple openers and one closer");
    printf("this is another complex scenario where multiple openers and one closer");
  } while(ret == -1 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK));
  #elif defined(EWOULDBLOCK)
    printf("this is another complex scenario where multiple openers and one closer");
    printf("this is another complex scenario where multiple openers and one closer");
  } while(ret == -1 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK));
  #else
    printf("this is another complex scenario where multiple openers and one closer");
    printf("this is another complex scenario where multiple openers and one closer");
  } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
  #endif
  }"""

if_code = """ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
  ssize_t ret = 1;
  #if defined(EWOULDBLOCK)
  if (ret > 1){
  #elif defined(EWOULDBLOCK)
  if (ret < 1){
  #else
  if (ret == 1){
  #endif
    printf("this is another complex scenario where multiple openers and one closer");
    }
  }"""

switch_code = """
#if defined(USE_INTEGER_CODES)
    switch (get_int_code()) {
#else
    switch (get_char_code()) {
#endif
        case 1:
            //...
            break;
        default:
            //...
            break;
    } 
"""

while_code = """
#ifdef FAST_MODE
    while (fast_check()) {
#else
    while (slow_check()) {
#endif
        do_work();
    }
"""

if_complex_code = """
#if defined(CONFIG_A)
  if (x > 100 && check_status(y)) {
#elif defined(CONFIG_B)
  #if defined(SUB_CONFIG)
  if (z == 0) {
  #else
  if (z != 0) {
  #endif
#else
  if (x < 0) {
#endif
  int result = process_data(x, y, z);
  log_result(result);
}
"""


switch_complex_code = """
    #if defined(USE_COMPLEX_ID)
      switch (get_id(user, true)) { // Function call in condition
    #else
      #if defined(SIMPLE_ID)
      switch(user->id) { // Pointer access
      #else
      switch(0) { // Default
      #endif
    #endif
      case 0: // Start of shared body
        handle_guest();
        break;
      case 10:
      case 20:
        handle_admin();
        break;
      default:
        handle_error();
        break;
    }
"""

# non_broken = """
# ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from,
#                      socklen_t *fromlen) {
#   ssize_t ret;
# #if defined(EWOULDBLOCK)
#   do {
#     ret = recvfrom(s, buf, len, flags, from, fromlen);
#   } while(ret == -1 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK));
# #else
#   do {
#     ret = recvfrom(s, buf, len, flags, from, fromlen);
#   } while(ret == -1 && (errno == EINTR || errno == EAGAIN));
# #endif
# }
# """

interleaved_content_statement = """
void print_values(int max) {
  for (int i = 0; i < max; i++) {
  #if defined(DEBUG)
    printf("Value: %d\n", i);
  }
  #else
    printf("%d\n", i);
  }
  #endif
"""

BROKEN_INTERLEAVED_SIMPLE_FOR_LOOP_OUTPUT = """
#ifdef fast_mode
for(;fast_check();) { do_work(); }
#else
for(;slow_check();) { do_work(); }
#endif
"""


# Initialize the parser and generate the node list
c_parser = TreeSitter()
c_parser.parse_input(BROKEN_INTERLEAVED_SIMPLE_FOR_LOOP_OUTPUT)
print(c_parser.tree.root_node)
actual_node_types = [( node.type.encode(), node.text ) for node in c_parser.traverse_tree()]

print("real-tree:")
print(c_parser.tree.root_node)

print("visited nodes = [")
for node_tuple in actual_node_types:
    print(f"\t{node_tuple} ,")
print("]")

