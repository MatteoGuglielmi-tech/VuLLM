# List of made changes as fix

> [!NOTE] Line numbers are intended after refactoring

> [!IMPORTANT] Closing curvy braces have been introduced as semi-automatic. It hasn't been possible to fully automate this procedure.

- id $2107$ `ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen)`

  - add closing `#endif`
  - add closing function scope `}`

> [!NOTE] This is automatic now
>
> - id $4898$ `STATIC LONG WINAPI GC_write_fault_handler(struct _EXCEPTION_POINTERS *exc_info)`
>   - remove `#endif` in function prototype line
> - id $4949$ `addChar(char c, Lineprop mode)`
>   - remove `#endif` in function prototype line
> - id $5437$ `static int input(yyscan_% yyscanner)`
>   - remove `#else` and `#endif` without `#if` directive in prototype

> [!WARNING]
>
> - ids $8322$, $8324$ and $150985$ `static int do_seekable(__G__ lastchance)`
>   - here a lot of functions have been condensed into one. It would be impossible, even for the largest models, to get context on such conglomerate functions.
>     All dataset entries for this function have been _removed_ from the original dataset

> [!NOTE] This is automatic now
>
> - ids $11131$ and $11133$ `__attribute__((no_sanitize ("undefined")))`
>   - remove `#endif`
> - id $12674$ `int main(int argc, char **argv)`
>   - add missing `#endif`
>   - remove `}` in excess (probably due to a regex in my pre-proc)
> - id $13480$ `int wolfSSH_SFTP_RecvOpen(WOLFSSH* ssh, int reqId, byte* data, word32 maxSz)`
>   - missing `#endif`. Removed `#ifdef` in function signature since it wraps all the function to save some tokens
> - id $13481$ `int wolfSSH_SFTP_RecvOpenDir(WOLFSSH *ssh, int reqId, byte *data, word32 maxSz)`
>   - missing `#endif`. Removed `#ifdef` in function signature since it wraps all the function to save some tokens
> - id $13484$ `int wolfSSH_SFTP_RecvRead(WOLFSSH* ssh, int reqId, byte* data, word32 maxSz)`
>   - missing `#endif`. Removed `#ifdef` in function signature since it wraps all the function to save some tokens
> - id $13485$ `int wolfSSH_SFTP_RecvWrite(WOLFSSH* ssh, int reqId, byte* data, word32 maxSz)`
>   - missing `#endif`. Removed `#ifdef` in function signature since it wraps all the function to save some tokens
> - id $16553$ `SPH_XCAT(sph_, HASH)(void *cc, const void *data, size_t len)`
>   - remove `#endif` in function prototype
> - id $16984$
>   - remove `#ifdef` obscuring whole function
> - id $17508$ -> `yyparse()`
>   - remove two `#endif` in function prototype

- id $17715$, $17754$ -> `void OpenSSL_add_all_ciphers(void)`

  - remove `#endif` not matching any `#if` at line $128$
  - remove following pieces of code since it's ignored anyhow (save tokens)

    - line $99$

      ```C
      #if 0
        EVP_add_cipher(EVP_aes_128_ctr());
      #endif
      ```

    - line $110$

      ```C
      #if 0
        EVP_add_cipher(EVP_aes_192_ctr());
      #endif
      ```

    - line $121$

      ```C
      #if 0
        EVP_add_cipher(EVP_aes_256_ctr());
      #endif
      ```

- id $17755$ -> `int SSL_library_init(void)`

  - remove `#endif` not matching any `#if` at line $26$
  - remove following piece of code since it's ignored anyhow (save tokens) (line $69$)

  ```C
  #if 0
    EVP_add_digest(EVP_sha());  EVP_add_digest(EVP_dss());
  #endif
  ```

> [!NOTE] This is automatic now
>
> - id $18528$ -> `report_error(format, va_alist) const char *format; va_dcl`
>   - remove `#endif` in function proto
> - id $23559$ -> `inline static struct ext4_sb_info *EXT4_SB(struct super_block *sb)`
>   - remove `#ifdef __KERNEL__`

- id $28732$ -> `static void xmlXPathCompStep(xmlXPathParserContextPtr ctxt)`
  - add closing `#endif` for `#ifdef DEBUG_STEP` at line $105$
- id $28900$ -> `void xmlXPathCeilingFunction(xmlXPathParserContextPtr ctxt, int nargs)`

  - add closing function scope `}` at line $17$
  - remove following piece of code since it's ignored anyhow (save tokens)

    ```C
    #if 0
    ctxt->value->floatval = ceil(ctxt->value->floatval);
    #else
    ```

> [!NOTE] This is automatic now
>
> - id $28933$ -> `static void xmlXPathDebugDumpStepAxis(xmlXPathStepOpPtr op, int nbNodes)`
>   - remove `#ifdef DEBUG_STEP` obscuring function implementation
> - id $28934$ -> `static int xmlXPathCompOpEvalFilterFirst(xmlXPathParserContextPtr ctxt, xmlXPathStepOpPtr op, xmlNodePtr *first)`
>   - remove `#ifdef XP_OPTIMIZED_FILTER_FIRST`
>   - add closing function scope `}`

- id $28959$ -> `static void xmlXPathCompStep(xmlXPathParserContextPtr ctxt)`
  - add closing `#endif` at line $108$

> [!NOTE] This is automatic now
>
> - id $33010$ -> `void untty(void)`
>   - remove `#ifdef HAVE_SETSID` from func proto
> - id $35316$ -> `int suhosin_header_handler(sapi_header_struct *sapi_header, sapi_headers_struct *sapi_headers TSRMLS_DC)`
>   - remove unmatched `#endif` in func proto
> - id $36989$ -> `GC_API void *GC_CALL GC_malloc(size_t lb)`
>   - remove `#endif` in func proto
> - id $36997$ -> `GC_API void *GC_CALL GC_malloc_atomic(size_t lb)`
>   - remove `#endif` from func proto
> - id $40908$ -> `static void crm_smtp_debug(const char *buf, int buflen, int writing, void *arg)`
>   - remove `#ifdef ENABLE_ESMTP`
>   - add closing function body `}`
> - id $41461$, $47349$ -> `static inline int php_openssl_config_check_syntax(const char * section_label, const char * config_filename, const char * section, LHASH * config TSRMLS_DC)`
>   - remove `#endif` from func proto

- id $47793$ -> `void CLASS apply_profile (const char *input, const char *output)`
  - change closing `#else` with `#endif`

> [!NOTE] This is automatic now
>
> - id $5035$ -> `(Bigint *a, int *e)`
>   - remove `#endif` from func proto

- id $56223$ -> `ssize_t sys_writev(int fd, const struct iovec *iov, int iovcnt)`

  - remove following piece of dead code (save tokens)

    ```C
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
  - add closing function scope `}` at line $10$
  - add closing `#endif` at line $9$
- id $56253$ -> `ssize_t sys_write(int fd, const void *buf, size_t count)`
  - add closing function scope `}` at line $10$
  - add closing `#endif` at line $9$
- id $56256$ -> `ssize_t sys_read(int fd, void *buf, size_t count)`
  - add closing function scope `}` at line $10$
  - add closing `#endif` at line $9$

> [!NOTE] This is automatic now
>
> - id $56443$ -> `static apr_status_t wsgi_python_parent_cleanup(void *data)`
>   - remove `#endif` in func proto
> - id $56466$ -> `static apr_status_t wsgi_python_child_cleanup(void *data)`
>   - remove `#endif` in func proto
> - id $58272$ -> `static PHP_FUNCTION(tidy_get_opt_doc)`
>   - remove `#if HAVE_TIDYOPTGETDOC`
> - id $58438$ -> `PHP_FE(tidy_get_head, arginfo_tidy_get_head) PHP_FE(tidy_get_html, arginfo_tidy_get_html)`
>   - add missing opening and close `{` `}` for function scope definition
> - id $61603$ -> ``
>   - empty function, removed from dataset
> - id $61604$ -> ``
>
>   - not a function, removed from dataset
>
>     ```C
>     bool logging_enabled;
>     bool retry_intercept_failures;
>
>     _HttpApiInfo()
>     : parent_proxy_name(NULL),
>     ```
>
> - id $61612$ -> ``
>
>   - not a function, removed from dataset
>
>     ```C
>     ServerState_t state;
>     int attempts;
>
>     _CurrentInfo()
>     ```
>
> - id $61620$ -> ``
>
>   - not a function, removed from dataset
>
>     ```C
>
>     int lookup_count;
>     bool is_ram_cache_hit;
>
>     _CacheLookupInfo()
>     : action(CACHE_DO_UNDEFINED),
>     transform_action(CACHE_DO_UNDEFINED),
>     write_status(NO_CACHE_WRITE),
>     transform_write_status(NO_CACHE_WRITE),
>     lookup_url(NULL),
>     lookup_url_storage(),
>     original_url(),
>     object_read(NULL),
>     second_object_read(NULL),
>     object_store(),
>     transform_store(),
>     config(),
>     directives(),
>     open_read_retries(0),
>     open_write_retries(0),
>     write_lock_state(CACHE_WL_INIT),
>     ```
>
> - id $61623$ -> ``
>
>   - not a function, removed from dataset
>
>     ```C
>     bool does_server_permit_lookup;
>     bool does_server_permit_storing;
>
>     _CacheDirectives()
>     : does_client_permit_lookup(true),
>     does_client_permit_storing(true),
>     does_client_permit_dns_storing(true),
>     ```
>
> - id $61643$ -> ``
>
>   - not a function, removed from dataset
>
>     ```C
>     bool is_transparent;
>     ```
>
> - id $61665$ -> `void destroy()`
>   - remove extraneous closing brace (`}`) before function implementation (line $0$)
> - id $61696$ -> `/// @c true if the connection is transparent.`
>   - not a function, removed from dataset

- id $65859$ -> `main(int argc, char **argv)`
  - remove "indent : Standard input : 179 : Error : Unmatched `#endif`", it seems like a residual of an lsp error message
- id $71457$ -> `OPJ_BOOL j2k_read_ppm_v3 ( opj_j2k_t *p_j2k, OPJ_BYTE * p_header_data, OPJ_UINT32 p_header_size, struct opj_event_mgr * p_manager )`
  - add `#endif` and `return OPJ_TRUE` at lines $219$ respectively
- id $72903$ -> `PNG_FUNCTION(png_structp, PNGAPI png_create_write_struct, (png_const_charp user_png_ver, png_voidp error_ptr, png_error_ptr error_fn, png_error_ptr warn_fn), PNG_ALLOCATED)`
  - add `#endif` at line $10$
- id $72927$ -> `PNG_FUNCTION(png_structp, PNGAPI png_create_write_struct_2, (png_const_charp user_png_ver, png_voidp error_ptr, png_error_ptr error_fn, png_error_ptr warn_fn, png_voidp mem_ptr, png_malloc_ptr malloc_fn, png_free_ptr free_fn), PNG_ALLOCATED)`
  - remove `#endif` at line $8$
- id $74993$ -> `zend_set_timeout(zend_long seconds, int reset_signals)`
  - add `#endif` at lines $51$ and $52$
- id $121004$ -> `namespace cimg { inline FILE *_stdin(const bool throw_exception) [...] }`
  - add `#endif` at lines $13$
  - add `}` at line $15$
- id $152754$ -> `namespace cimg { inline FILE *_stdin(const bool throw_exception) [...] }`
  - add `#endif` at lines $13$
  - add `}` at line $15$ and $16$
- id $198966$ -> `scsi_co_writev(BlockDriverState *bs, int64_t sector_num, int nb_sectors, QEMUIOVector *iov, int flags)`
  - remove `#else` at line $33$
  - add `}` at line $33$ and $35$
  - add `#endif` at lines $34$
- id $198969$ -> `scsi_co_writev(BlockDriverState *bs, int64_t sector_num, int nb_sectors, QEMUIOVector *iov, int flags)`
  - remove `#else` at line $54$
  - add `}` at line $54$ and $56$
  - add `#endif` at lines $55$
- id $226482$ -> `buflist_findname_stat(char_u *ffname, stat_T *stp)`
  - remove `#endif` at lines $2$
- id $226520$ -> `buflist_findname(char_u *ffname)`
  - remove `#endif` at lines $5$
- id $232716$ -> `inline FILE* _stdin(const bool throw_exception)`
  - remove `#endif` at lines $12$
  - remove `}` at lines $13$
- id $234328$ -> `int TTF_Init(void)`
  - remove `#endif` at line $11$
- id $253792$ -> `static inline void f2fs_set_encrypted_inode(struct inode *inode)`
  - remove `#endif` at line $11$
- id $253792$ -> `inline static void f2fs_set_encrypted_inode(struct inode *inode)`
  - add `#endif` at line $5$
- id $253894$ -> `inline static void f2fs_set_encrypted_inode(struct inode *inode)`
  - add `#endif` at line $8$
- id $254337$ -> `static void dump_isom_obu(GF_ISOFile *file, GF_ISOTrackID trackID, FILE *dump, Bool dump_crc)`
  - add `#endif` at line $8$
  - remove `#ifndef GPAC_DISABLE_AV_PARSERS` at line $2$ (missing `#endif`)
- id $254346$ -> `static void dump_isom_nal_ex(GF_ISOFile *file, GF_ISOTrackID trackID, FILE *dump, u32 dump_flags)`
  - add `#endif` at line $244$
- id $254360$ -> `static void dump_qt_prores(GF_ISOFile *file, u32 trackID, FILE *dump, Bool dump_crc)`
  - remove `#ifndef GPAC_DISABLE_AV_PARSERS` at line $2$ (missing `#endif`)
- id $256041$ -> `static SSL_SESSION *getCloseCb(SSL *ssl, const unsigned char *, int, int *)`
  - remove `#else` at line $2$
  - remove `#endif` at line $4$
- id $267134$ -> `GF_EXPORT void gf_isom_reset_fragment_info(GF_ISOFile *movie, Bool keep_sample_count)`
  - add `#endif` at line $13$
  - add `}` at line $14$
- id $279926$ -> `void gf_isom_reset_fragment_info(GF_ISOFile *movie, Bool keep_sample_count)`
  - add `#endif` at line $13$
  - add `}` at line $14$
- id $280844$ -> `generate_hash(char *data, unsigned int datasize, PE_COFF_LOADER_IMAGE_CONTEXT *context, UINT8 *sha256hash, UINT8 *sha1hash)`
  - add `#endif` at line $13$
  - add `}` at line $14$
  - substitute the following code snippet:

    ```C
    #if 1
      }
    #else 
    ```

    with:

    ```C
    }
    ```

- id $285986$ -> `REQUEST *received_proxy_response(RADIUS_PACKET *packet)`
  - remove following block of code:

    ```C
    #if 0
      if ((request->num_proxied_responses == 1)
          int rtt;
          home_server *home = request->home_server;
          rtt = now.tv_sec - request->proxy_when.tv_sec;
          rtt *= USEC;
          rtt += now.tv_usec;
          rtt -= request->proxy_when.tv_usec;
          if (!home->has_rtt) {
              home->has_rtt = TRUE;
              home->srtt = rtt;
              home->rttvar = rtt / 2;
          }
          else {
            home->rttvar -= home->rttvar >> 2;
            home->rttvar += (home->srtt - rtt);
            home->srtt -= home->srtt >> 3;
            home->srtt += rtt >> 3;
          }
          home->rto = home->srtt; 
          if (home->rttvar > (USEC / 4)
      ) {
          home->rto += home->rttvar * 4;
      } else {
          home->rto += USEC;
      }
    ```

- id $302501$ -> `static void dump_isom_obu(GF_ISOFile *file, GF_ISOTrackID trackID, FILE *dump, Bool dump_crc)`
  - remove `#ifndef GPAC_DISABLE_AV_PARSERS` at line $2$
