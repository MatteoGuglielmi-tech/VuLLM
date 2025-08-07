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


real_like = """
static char *make_filename_safe(const char *filename TSRMLS_DC) {
  if(*filename && strncmp(filename, ":memory:", sizeof(":memory:") - 1)) {
    char *fullpath = expand_filepath(filename, NULL TSRMLS_CC);
    if(!fullpath) { return NULL; }
    if(PG(safe_mode) && (!php_checkuid(fullpath, NULL, CHECKUID_CHECK_FILE_AND_DIR))) {
      efree(fullpath);
      return NULL;
    }
    if(php_check_open_basedir(fullpath TSRMLS_CC)) {
      efree(fullpath);
      return NULL;
    }
    return fullpath;
  }
  return estrdup(filename);
}
"""


glue = """
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

complex_kr_input = """
static void process_data(ptr1, ptr2, size, count)
    char *ptr1, *ptr2;
    unsigned int size, count;
{
    /* function body */
}
"""

tt = """
my_func() {
    return 0;
}
"""


kr = """
_gnutls_ciphertext2compressed(session, compress_data) 
    gnutls_session_t session; 
    opaque *compress_data;
{
  uint8 MAC[MAX_HASH_SIZE];
  uint16 c_length;
  uint8 pad;
  int length;
  mac_hd_t td;
  uint16 blocksize;
  int ret, i, pad_failed = 0;
  uint8 major, minor;
  gnutls_protocol_t ver;
  int hash_size = _gnutls_hash_get_algo_len(session->security_parameters.read_mac_algorithm);
  ver = gnutls_protocol_get_version(session);
  minor = _gnutls_version_get_minor(ver);
  major = _gnutls_version_get_major(ver);
}
"""

kr = """
funcname(a, b) int a; int b; {
  return a > b ? a : b; 
}
"""

pp = """
process_data(ptr1, ptr2, size, count) char *ptr1, *ptr2;
unsigned int size, count;
{ /* function body */ }
"""


real_shady = """
unpack_Z_stream(int fd_in, int fd_out)\n{\n\tIF_DESKTOP(long long total_written = 0;)\n\tIF_DESKTOP(long long) int retval = -1;\n\tunsigned char *stackp;\n\tlong code;\n\tint finchar;\n\tlong oldcode;\n\tlong incode;\n\tint inbits;\n\tint posbits;\n\tint outpos;\n\tint insize;\n\tint bitmask;\n\tlong free_ent;\n\tlong maxcode;\n\tlong maxmaxcode;\n\tint n_bits;\n\tint rsize = 0;\n\tunsigned char *inbuf; \n\tunsigned char *outbuf; \n\tunsigned char *htab;\n\tunsigned short *codetab;\n\tint maxbits; \n\tint block_mode; \n\tinbuf = xzalloc(IBUFSIZ + 64);\n\toutbuf = xzalloc(OBUFSIZ + 2048);\n\thtab = xzalloc(HSIZE);  \n\tcodetab = xzalloc(HSIZE * sizeof(codetab[0]));\n\tinsize = 0;\n\tif (full_read(fd_in, inbuf, 1) != 1) {\n\t\tbb_error_msg("short read");\n\t\tgoto err;\n\t}\n\tmaxbits = inbuf[0] & BIT_MASK;\n\tblock_mode = inbuf[0] & BLOCK_MODE;\n\tmaxmaxcode = MAXCODE(maxbits);\n\tif (maxbits > BITS) {\n\t\tbb_error_msg("compressed with %d bits, can only handle "\n\t\t\t\tBITS_STR" bits", maxbits);\n\t\tgoto err;\n\t}\n\tn_bits = INIT_BITS;\n\tmaxcode = MAXCODE(INIT_BITS) - 1;\n\tbitmask = (1 << INIT_BITS) - 1;\n\toldcode = -1;\n\tfinchar = 0;\n\toutpos = 0;\n\tposbits = 0 << 3;\n\tfree_ent = ((block_mode) ? FIRST : 256);\n\tfor (code = 255; code >= 0; --code) {\n\t\ttab_suffixof(code) = (unsigned char) code;\n\t}\n\tdo {\n resetbuf:\n\t\t{\n\t\t\tint i;\n\t\t\tint e;\n\t\t\tint o;\n\t\t\to = posbits >> 3;\n\t\t\te = insize - o;\n\t\t\tfor (i = 0; i < e; ++i)\n\t\t\t\tinbuf[i] = inbuf[i + o];\n\t\t\tinsize = e;\n\t\t\tposbits = 0;\n\t\t}\n\t\tif (insize < (int) (IBUFSIZ + 64) - IBUFSIZ) {\n\t\t\trsize = safe_read(fd_in, inbuf + insize, IBUFSIZ);\n\t\t\tinsize += rsize;\n\t\t}\n\t\tinbits = ((rsize > 0) ? (insize - insize % n_bits) << 3 :\n\t\t\t\t  (insize << 3) - (n_bits - 1));\n\t\twhile (inbits > posbits) {\n\t\t\tif (free_ent > maxcode) {\n\t\t\t\tposbits =\n\t\t\t\t\t((posbits - 1) +\n\t\t\t\t\t ((n_bits << 3) -\n\t\t\t\t\t  (posbits - 1 + (n_bits << 3)) % (n_bits << 3)));\n\t\t\t\t++n_bits;\n\t\t\t\tif (n_bits == maxbits) {\n\t\t\t\t\tmaxcode = maxmaxcode;\n\t\t\t\t} else {\n\t\t\t\t\tmaxcode = MAXCODE(n_bits) - 1;\n\t\t\t\t}\n\t\t\t\tbitmask = (1 << n_bits) - 1;\n\t\t\t\tgoto resetbuf;\n\t\t\t}\n\t\t\t\tunsigned char *p = &inbuf[posbits >> 3];\n\t\t\t\tcode = ((((long) (p[0])) | ((long) (p[1]) << 8) |\n\t\t\t\t         ((long) (p[2]) << 16)) >> (posbits & 0x7)) & bitmask;\n\t\t\tposbits += n_bits;\n\t\t\tif (oldcode == -1) {\n\t\t\t\toldcode = code;\n\t\t\t\tfinchar = (int) oldcode;\n\t\t\t\toutbuf[outpos++] = (unsigned char) finchar;\n\t\t\t\tcontinue;\n\t\t\t}\n\t\t\tif (code == CLEAR && block_mode) {\n\t\t\t\tclear_tab_prefixof();\n\t\t\t\tfree_ent = FIRST - 1;\n\t\t\t\tposbits =\n\t\t\t\t\t((posbits - 1) +\n\t\t\t\t\t ((n_bits << 3) -\n\t\t\t\t\t  (posbits - 1 + (n_bits << 3)) % (n_bits << 3)));\n\t\t\t\tn_bits = INIT_BITS;\n\t\t\t\tmaxcode = MAXCODE(INIT_BITS) - 1;\n\t\t\t\tbitmask = (1 << INIT_BITS) - 1;\n\t\t\t\tgoto resetbuf;\n\t\t\t}\n\t\t\tincode = code;\n\t\t\tstackp = de_stack;\n\t\t\tif (code >= free_ent) {\n\t\t\t\tif (code > free_ent) {\n\t\t\t\t\tunsigned char *p;\n\t\t\t\t\tposbits -= n_bits;\n\t\t\t\t\tp = &inbuf[posbits >> 3];\n\t\t\t\t\tbb_error_msg\n\t\t\t\t\t\t("insize:%d posbits:%d inbuf:%02X %02X %02X %02X %02X (%d)",\n\t\t\t\t\t\t insize, posbits, p[-1], p[0], p[1], p[2], p[3],\n\t\t\t\t\t\t (posbits & 07));\n\t\t\t\t\tbb_error_msg("corrupted data");\n\t\t\t\t\tgoto err;\n\t\t\t\t}\n\t\t\t\t*--stackp = (unsigned char) finchar;\n\t\t\t\tcode = oldcode;\n\t\t\t}\n\t\t\twhile ((long) code >= (long) 256) {\n\t\t\t\t*--stackp = tab_suffixof(code);\n\t\t\t\tcode = tab_prefixof(code);\n\t\t\t}\n\t\t\tfinchar = tab_suffixof(code);\n\t\t\t*--stackp = (unsigned char) finchar;\n\t\t\t\tint i;\n\t\t\t\ti = de_stack - stackp;\n\t\t\t\tif (outpos + i >= OBUFSIZ) {\n\t\t\t\t\tdo {\n\t\t\t\t\t\tif (i > OBUFSIZ - outpos) {\n\t\t\t\t\t\t\ti = OBUFSIZ - outpos;\n\t\t\t\t\t\t}\n\t\t\t\t\t\tif (i > 0) {\n\t\t\t\t\t\t\tmemcpy(outbuf + outpos, stackp, i);\n\t\t\t\t\t\t\toutpos += i;\n\t\t\t\t\t\t}\n\t\t\t\t\t\tif (outpos >= OBUFSIZ) {\n\t\t\t\t\t\t\tfull_write(fd_out, outbuf, outpos);\n\t\t\t\t\t\t\tIF_DESKTOP(total_written += outpos;)\n\t\t\t\t\t\t\toutpos = 0;\n\t\t\t\t\t\t}\n\t\t\t\t\t\tstackp += i;\n\t\t\t\t\t\ti = de_stack - stackp;\n\t\t\t\t\t} while (i > 0);\n\t\t\t\t} else {\n\t\t\t\t\tmemcpy(outbuf + outpos, stackp, i);\n\t\t\t\t\toutpos += i;\n\t\t\t\t}\n\t\t\tcode = free_ent;\n\t\t\tif (code < maxmaxcode) {\n\t\t\t\ttab_prefixof(code) = (unsigned short) oldcode;\n\t\t\t\ttab_suffixof(code) = (unsigned char) finchar;\n\t\t\t\tfree_ent = code + 1;\n\t\t\t}\n\t\t\toldcode = incode;\n\t\t}\n\t} while (rsize > 0);\n\tif (outpos > 0) {\n\t\tfull_write(fd_out, outbuf, outpos);\n\t\tIF_DESKTOP(total_written += outpos;)\n\t}\n\tretval = IF_DESKTOP(total_written) + 0;\n err:\n\tfree(inbuf);\n\tfree(outbuf);\n\tfree(htab);\n\tfree(codetab);\n\treturn retval;\n}
"""

# Initialize the parser and generate the node list
c_parser = TreeSitter()
c_parser.parse_input(real_shady)
actual_node_types = [( node.type.encode(), node.text ) for node in c_parser.traverse_tree()]

print("real-tree:")
print(c_parser.tree.root_node)

print("visited nodes = [")
for node_tuple in actual_node_types:
    print(f"\t{node_tuple} ,")
print("]")

