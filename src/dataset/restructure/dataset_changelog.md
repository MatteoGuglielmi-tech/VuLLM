# List of made changes as fix

- id $2107$ `ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen)`
  - add closing `#endif`
  - add closing function scope `}`
- id $4898$ `STATIC LONG WINAPI GC_write_fault_handler(struct _EXCEPTION_POINTERS *exc_info)`
  - remove `#endif` in function prototype line
- id $4949$ `addChar(char c, Lineprop mode)`
  - remove `#endif` in function prototype line
- id $5437$ `static int input(yyscan_% yyscanner)`
  - remove #else and #endif without #if directive in prototype
- ids $8322$ and $8324$ `static int do_seekable(__G__ lastchance)`
  - here a lot of functions have been condensed into one. It would be impossible, even for the largest models, to get context on such conglomerate functions.
    All dataset entries for this function have been _removed_ from the original dataset
- ids $11131$ and $11133$ `__attribute__((no_sanitize ("undefined")))`
  - remove #endif
- id $12674$ `int main(int argc, char **argv)`
  - add missing #endif
  - remove `}` in excess (probably due to a regex in my pre-proc)
- id $13480$ `int wolfSSH_SFTP_RecvOpen(WOLFSSH* ssh, int reqId, byte* data, word32 maxSz)`
  - missing `#endif`. Removed `#ifdef` in function signature since it wraps all the function to save some tokens
- id $13481$ `int wolfSSH_SFTP_RecvOpenDir(WOLFSSH *ssh, int reqId, byte *data, word32 maxSz)`
  - missing `#endif`. Removed `#ifdef` in function signature since it wraps all the function to save some tokens
- id $13484$ `int wolfSSH_SFTP_RecvRead(WOLFSSH* ssh, int reqId, byte* data, word32 maxSz)`
  - missing `#endif`. Removed `#ifdef` in function signature since it wraps all the function to save some tokens
- id $13485$ `int wolfSSH_SFTP_RecvWrite(WOLFSSH* ssh, int reqId, byte* data, word32 maxSz)`
  - missing `#endif`. Removed `#ifdef` in function signature since it wraps all the function to save some tokens
- id $16553$ `SPH_XCAT(sph_, HASH)(void *cc, const void *data, size_t len)`
  - remove `#endif` in function prototype
- id $16984$
  - remove `#ifdef` obscuring whole function
- id $17508$ -> `yyparse()`
  - remove two `#endif` in function prototype
- id $17715$, $17754$ -> `void OpenSSL_add_all_ciphers(void)`
  - remove `#endif` not matching any #if at line $128$
- id $17755$ -> `void OpenSSL_add_all_ciphers(void)`
  - remove `#endif` not matching any #if at line $22$
- id $18528$ -> `report_error(format, va_alist) const char *format; va_dcl`
  - remove `#endif` in function proto
- id $23559$ -> `inline static struct ext4_sb_info *EXT4_SB(struct super_block *sb)`
  - remove `#ifdef __KERNEL__`
- id $28732$ -> `static void xmlXPathCompStep(xmlXPathParserContextPtr ctxt)`
  - add closing `#endif` for `#ifdef DEBUG_STEP`
- id $28900$ ->

  - remove following piece of code since it's ignored anyhow (save tokens)
    ```c
    #if 0
    ctxt->value->floatval = ceil(ctxt->value->floatval);
    #else
    ```

- id $28933$ -> `static void xmlXPathDebugDumpStepAxis(xmlXPathStepOpPtr op, int nbNodes)`
  - remove `#ifdef DEBUG_STEP` obscuring function implementation
- id $28934$ -> `static int xmlXPathCompOpEvalFilterFirst(xmlXPathParserContextPtr ctxt, xmlXPathStepOpPtr op, xmlNodePtr *first)`
  - remove `#ifdef XP_OPTIMIZED_FILTER_FIRST`
  - add closing function scope `}`
- id $28959$ -> `static void xmlXPathCompStep(xmlXPathParserContextPtr ctxt)`
  - add closing `#endif` at line $108$
- id $33010$ -> `void untty(void)`
  - remove `#ifdef HAVE_SETSID` from func proto
- id $35316$ -> `int suhosin_header_handler(sapi_header_struct *sapi_header, sapi_headers_struct *sapi_headers TSRMLS_DC)`
  - remove unmatched `#endif` in func proto
- id $36989$ -> `GC_API void *GC_CALL GC_malloc(size_t lb)`
  - remove `#endif` in func proto
- id $36997$ -> `GC_API void *GC_CALL GC_malloc_atomic(size_t lb)`
  - remove `#endif` from func proto
- id $40908$ -> `static void crm_smtp_debug(const char *buf, int buflen, int writing, void *arg)`
  - remove `#ifdef ENABLE_ESMTP`
  - add closing function body `}`
- id $41461$, $47349$ -> `static inline int php_openssl_config_check_syntax(const char * section_label, const char * config_filename, const char * section, LHASH * config TSRMLS_DC)`
  - remove `#endif` from func proto
- id $47793$ -> `void CLASS apply_profile (const char *input, const char *output)`
  - change closing `#else` with `#endif`
- id $5035$ -> `(Bigint *a, int *e)`
  - remove `#endif` from func proto
- id $56223$ -> `ssize_t sys_writev(int fd, const struct iovec *iov, int iovcnt)`
  - remove following piece of dead code (save tokens)
  ```c
    #if 0
    if ((random() % 5) == 0) {
      return sys_write(fd, iov[0].iov_base, iov[0].iov_len);
    }
    if (iov[0].iov_len > 1) {
      return sys_write(fd, iov[0].iov_base,  (random() % (iov[0].iov_len-1)) + 1);
    }
    #endif
  ```
  - add closing function scope `}` at line $10$
  - add closing `#endif` at line $9$
- id $56231$ -> `ssize_t sys_send(int s, const void *msg, size_t len, int flags)`
  - add closing function scope `}` at line $9$
  - add closing `#endif` at line $8$
- id $56253$ -> `ssize_t sys_write(int fd, const void *buf, size_t count)`
  - add closing function scope `}` at line $10$
  - add closing `#endif` at line $9$
- id $56256$ -> `ssize_t sys_read(int fd, void *buf, size_t count)`
  - add closing function scope `}` at line $10$
  - add closing `#endif` at line $9$
- id $56443$ -> `static apr_status_t wsgi_python_parent_cleanup(void *data)`
  - remove `#endif` in func proto
- id $56466$ -> `static apr_status_t wsgi_python_child_cleanup(void *data)`
  - remove `#endif` in func proto
- id $58272$ -> `static PHP_FUNCTION(tidy_get_opt_doc)`
  - remove `#if HAVE_TIDYOPTGETDOC`
- id $58438$ -> `PHP_FE(tidy_get_head, arginfo_tidy_get_head) PHP_FE(tidy_get_html, arginfo_tidy_get_html)`
  - add missing opening and close `{` `}` for function scope definition
