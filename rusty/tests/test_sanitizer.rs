mod test_helpers;

#[cfg(test)]
mod tests {
    use super::test_helpers;
    use data_processing::processor_lib::{
        code_sanitizer::CodeSanitizer, tree_sitter_parser::TreeSitterParser,
    };
    use rstest::{fixture, rstest};
    use tree_sitter::{Language, Parser};

    #[fixture]
    fn sanitizer() -> CodeSanitizer {
        CodeSanitizer
    }

    #[fixture]
    fn ts() -> TreeSitterParser {
        TreeSitterParser::new(tree_sitter_c::LANGUAGE.into())
    }

    #[fixture]
    fn language() -> Language {
        tree_sitter_c::LANGUAGE.into()
    }

    #[fixture]
    fn parser() -> Parser {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_c::LANGUAGE.into())
            .unwrap();

        parser
    }

    #[rstest]
    #[case(
        "define",
        r#"#define SQUARE(x) ((x) * (x))
        int main() {
          return SQUARE(5);
        }"#,
        r#"int main() {
          return ((5) * (5));
        }
        "#
    )]
    #[case(
        "define",
        r#"#define KEEEP
        int main() {
          do {
            //some statement
          #ifdef KEEP
          } while(0);
          #else
          }while(1);
          #endif
        }"#,
        r#"int main() {
          do {
          }while(1);
        }
        "#
    )]
    #[case(
        "simple #if 1",
        r#"#if 1
            int x;
        #endif"#,
        "int x;"
    )]
    #[case(
        "simple #if 0",
        r#"
        #if 0
            int x;
        #endif"#,
        ""
    )]
    #[case(
        "#if 1 with else",
        r#"
        #if 1
            int x;
        #else
            int y;
        #endif"#,
        "int x;"
    )]
    #[case(
        "#if 0 with else",
        r#"
        #if 0
            int x;
        #else
            int y;
        #endif"#,
        "int y;"
    )]
    #[case(
        "nested",
        r#"
        #if 1
            #if 0
                int x;
            #endif
                int y;
            #endif"#,
        "int y;"
    )]
    #[case(
        "nested 2",
        r#"
        #if 0
            #if 1
                int x;
                #if B
                    int f;
                #endif
            #endif
            int y;
        #else
            int k;
        #endif"#,
        "int k;"
    )]
    fn test_gcc_preprocessor(
        sanitizer: CodeSanitizer,
        #[case] name: &str,
        #[case] code: &str,
        #[case] expected: &str,
    ) {
        let result = sanitizer
            .call_gcc_preprocessor(code, None)
            .expect("Impossible to call gcc preprocessor");

        test_helpers::assert_code_eq(
            tree_sitter_ext_c::language(),
            result.trim(),
            expected.trim(),
            "Test case failed:",
            name,
        );
    }

    #[rstest]
    #[case(
        "inline comment",
        r#"//! Wait for any event occuring either on the display \\c disp1 or \\c disp2.
        static void wait(CImgDisplay& disp1, CImgDisplay& disp2) {
            disp1._is_event = disp2._is_event = false;
            while ((!disp1._is_closed || !disp2._is_closed) && !disp1._is_event && !disp2._is_event) wait_all();
        "#,
        r#"static void wait(CImgDisplay& disp1, CImgDisplay& disp2) {
            disp1._is_event = disp2._is_event = false;
            while ((!disp1._is_closed || !disp2._is_closed) && !disp1._is_event && !disp2._is_event) wait_all();
        "#
    )]
    #[case(
        "inline comment",
        r#"static void wait(CImgDisplay& disp1, CImgDisplay& disp2) {
            //! Wait for any event occuring either on the display \\c disp1 or \\c disp2.
            disp1._is_event = disp2._is_event = false;
            /* this is a multi-line comment
             * this is another line
             * this is the end
             */
            while ((!disp1._is_closed || !disp2._is_closed) && !disp1._is_event && !disp2._is_event) wait_all();
        "#,
        r#"static void wait(CImgDisplay& disp1, CImgDisplay& disp2) {
            disp1._is_event = disp2._is_event = false;
            while ((!disp1._is_closed || !disp2._is_closed) && !disp1._is_event && !disp2._is_event) wait_all();
        "#
    )]
    fn test_remove_comments(
        sanitizer: CodeSanitizer,
        ts: TreeSitterParser,
        #[case] name: &str,
        #[case] test_code: &str,
        #[case] expected_code: &str,
    ) {
        let result = sanitizer.remove_comments(test_code, &ts);
        test_helpers::assert_code_eq(
            tree_sitter_c::LANGUAGE.into(),
            result.trim(),
            expected_code.trim(),
            "Test case failed :",
            name,
        );
    }

    #[rstest]
    #[case(
        "leading braces",
        "} int main() { return 0; }",
        "int main() { return 0; }"
    )]
    #[case(
        "leading directive",
        r#"
        #if __KERNEL__==4
        static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb) {
            return sb->s_fs_info;
        }"#,
        r#"static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb) {
            return sb->s_fs_info;
        }"#
    )]
    #[case(
        "leading kernel",
        r#"
        #if __KERNEL__<=4
        static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb) {
            return sb->s_fs_info;
        }"#,
        r#"static inline struct ext4_sb_info *EXT4_SB(struct super_block *sb) {
            return sb->s_fs_info;
        }"#
    )]
    #[case(
        "ok",
        r#"
int _gnutls_ciphertext2compressed(gnutls_session_t session, opaque *compress_data,
                                  int compress_size, gnutls_datum_t ciphertext, uint8 type) {
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
  blocksize =
    _gnutls_cipher_get_block_size(session->security_parameters.read_bulk_cipher_algorithm);
  switch(_gnutls_cipher_is_block(session->security_parameters.read_bulk_cipher_algorithm)) {
    case CIPHER_STREAM :
      if((ret = _gnutls_cipher_decrypt(session->connection_state.read_cipher_state, ciphertext.data,
                                       ciphertext.size)) < 0) {
        gnutls_assert();
        return ret;
      }
      length = ciphertext.size - hash_size;
      break;
    case CIPHER_BLOCK :
      if((ciphertext.size < blocksize) || (ciphertext.size % blocksize != 0)) {
        gnutls_assert();
        return GNUTLS_E_DECRYPTION_FAILED;
      }
      if((ret = _gnutls_cipher_decrypt(session->connection_state.read_cipher_state, ciphertext.data,
                                       ciphertext.size)) < 0) {
        gnutls_assert();
        return ret;
      }
      if(session->security_parameters.version >= GNUTLS_TLS1_1) {
        ciphertext.size -= blocksize;
        ciphertext.data += blocksize;
        if(ciphertext.size == 0) {
          gnutls_assert();
          return GNUTLS_E_DECRYPTION_FAILED;
        }
      }
      pad = ciphertext.data[ciphertext.size - 1] + 1;
      length = ciphertext.size - hash_size - pad;
      if(pad > ciphertext.size - hash_size) {
        gnutls_assert();
        pad_failed = GNUTLS_E_DECRYPTION_FAILED;
      }
      if(ver >= GNUTLS_TLS1)
        for(i = 2; i < pad; i++) {
          if(ciphertext.data[ciphertext.size - i] != ciphertext.data[ciphertext.size - 1])
            pad_failed = GNUTLS_E_DECRYPTION_FAILED;
        }
      break;
    default : gnutls_assert(); return GNUTLS_E_INTERNAL_ERROR;
  }
  if(length < 0) length = 0;
  c_length = _gnutls_conv_uint16((uint16)length);
  if(td != GNUTLS_MAC_FAILED) {
    _gnutls_hmac(td, UINT64DATA(session->connection_state.read_sequence_number), 8);
    _gnutls_hmac(td, &type, 1);
    if(ver >= GNUTLS_TLS1) {
      _gnutls_hmac(td, &major, 1);
      _gnutls_hmac(td, &minor, 1);
    }
    _gnutls_hmac(td, &c_length, 2);
    if(length > 0) _gnutls_hmac(td, ciphertext.data, length);
    mac_deinit(td, MAC, ver);
  }
  if(pad_failed != 0) return pad_failed;
  if(memcmp(MAC, &ciphertext.data[length], hash_size) != 0) {
    gnutls_assert();
    return GNUTLS_E_DECRYPTION_FAILED;
  }
  if(compress_size < length) {
    gnutls_assert();
    return GNUTLS_E_INTERNAL_ERROR;
  }
  memcpy(compress_data, ciphertext.data, length);
  return length;
}"#,
        r#"
int _gnutls_ciphertext2compressed(gnutls_session_t session, opaque *compress_data,
                                  int compress_size, gnutls_datum_t ciphertext, uint8 type) {
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
  blocksize =
    _gnutls_cipher_get_block_size(session->security_parameters.read_bulk_cipher_algorithm);
  switch(_gnutls_cipher_is_block(session->security_parameters.read_bulk_cipher_algorithm)) {
    case CIPHER_STREAM :
      if((ret = _gnutls_cipher_decrypt(session->connection_state.read_cipher_state, ciphertext.data,
                                       ciphertext.size)) < 0) {
        gnutls_assert();
        return ret;
      }
      length = ciphertext.size - hash_size;
      break;
    case CIPHER_BLOCK :
      if((ciphertext.size < blocksize) || (ciphertext.size % blocksize != 0)) {
        gnutls_assert();
        return GNUTLS_E_DECRYPTION_FAILED;
      }
      if((ret = _gnutls_cipher_decrypt(session->connection_state.read_cipher_state, ciphertext.data,
                                       ciphertext.size)) < 0) {
        gnutls_assert();
        return ret;
      }
      if(session->security_parameters.version >= GNUTLS_TLS1_1) {
        ciphertext.size -= blocksize;
        ciphertext.data += blocksize;
        if(ciphertext.size == 0) {
          gnutls_assert();
          return GNUTLS_E_DECRYPTION_FAILED;
        }
      }
      pad = ciphertext.data[ciphertext.size - 1] + 1;
      length = ciphertext.size - hash_size - pad;
      if(pad > ciphertext.size - hash_size) {
        gnutls_assert();
        pad_failed = GNUTLS_E_DECRYPTION_FAILED;
      }
      if(ver >= GNUTLS_TLS1)
        for(i = 2; i < pad; i++) {
          if(ciphertext.data[ciphertext.size - i] != ciphertext.data[ciphertext.size - 1])
            pad_failed = GNUTLS_E_DECRYPTION_FAILED;
        }
      break;
    default : gnutls_assert(); return GNUTLS_E_INTERNAL_ERROR;
  }
  if(length < 0) length = 0;
  c_length = _gnutls_conv_uint16((uint16)length);
  if(td != GNUTLS_MAC_FAILED) {
    _gnutls_hmac(td, UINT64DATA(session->connection_state.read_sequence_number), 8);
    _gnutls_hmac(td, &type, 1);
    if(ver >= GNUTLS_TLS1) {
      _gnutls_hmac(td, &major, 1);
      _gnutls_hmac(td, &minor, 1);
    }
    _gnutls_hmac(td, &c_length, 2);
    if(length > 0) _gnutls_hmac(td, ciphertext.data, length);
    mac_deinit(td, MAC, ver);
  }
  if(pad_failed != 0) return pad_failed;
  if(memcmp(MAC, &ciphertext.data[length], hash_size) != 0) {
    gnutls_assert();
    return GNUTLS_E_DECRYPTION_FAILED;
  }
  if(compress_size < length) {
    gnutls_assert();
    return GNUTLS_E_INTERNAL_ERROR;
  }
  memcpy(compress_data, ciphertext.data, length);
  return length;
}"#,
    )]
    fn test_validate_and_extract_body(
        sanitizer: CodeSanitizer,
        #[case] name: &str,
        #[case] input: &str,
        #[case] expected: &str,
    ) {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_c::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(input, None).unwrap();

        let result = sanitizer
            .validate_and_extract_body(input, &tree)
            .expect("Error");

        test_helpers::assert_code_eq(
            tree_sitter_c::LANGUAGE.into(),
            result.trim(),
            expected.trim(),
            "Test case failed :",
            name,
        );
    }

    #[rstest]
    #[case(
        "very simple case",
        "void func() { #if A",
        r#"void func() {
            #if A
        }"#
    )]
    #[case("simple empty func", "void func() {", "void func() { }")]
    #[case(
        "real world case",
        r#"ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
            ssize_t ret;
            do {
                ret = recvfrom(s, buf, len, flags, from, fromlen);
            } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
        "#,
        r#"ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
            ssize_t ret;
            do {
                ret = recvfrom(s, buf, len, flags, from, fromlen);
            } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
            }
        "#,
    )]
    fn test_add_missing_braces(
        sanitizer: CodeSanitizer,
        mut parser: Parser,
        language: Language,
        #[case] name: &str,
        #[case] input: &str,
        #[case] expected: &str,
    ) {
        let tree = parser.parse(input, None).unwrap();
        let result = sanitizer.add_missing_braces(input, &tree);
        test_helpers::assert_code_eq(
            language,
            result.trim(),
            expected.trim(),
            "Test case failed :",
            name,
        );
    }

    #[rstest]
    #[case(
        "real world case",
        r#"ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
            ssize_t ret;
            do {
                ret = recvfrom(s, buf, len, flags, from, fromlen);
            #ifdef EWOULDBLOCK
            } while(ret == -1 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK));
            #else
            } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
        "#,
        r#"ssize_t sys_recvfrom(int s, void *buf, size_t len, int flags, struct sockaddr *from, socklen_t *fromlen) {
            ssize_t ret;
            do {
                ret = recvfrom(s, buf, len, flags, from, fromlen);
            #ifdef EWOULDBLOCK
            } while(ret == -1 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK));
            #else
            } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
            #endif
        "#,
    )]
    fn test_balance_directives(
        sanitizer: CodeSanitizer,
        language: Language,
        ts: TreeSitterParser,
        #[case] name: &str,
        #[case] code: &str,
        #[case] expected: &str,
    ) {
        let result = sanitizer.balance_directives(code, &ts);
        test_helpers::assert_code_eq(
            language,
            result.trim(),
            expected.trim(),
            "Test case failed :",
            name,
        );
    }

    #[rstest]
    #[case(
        "simple func",
        r#"my_func() {
            return 0;
        }"#,
        r#"int my_func() {
            return 0;
        }"#
    )]
    #[case(
        "complex function",
        r#"glue(cirrus_bitblt_rop_fwd_, ROP_NAME)(CirrusVGAState *s, uint8_t *dst, const uint8_t *src,
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
        }"#,
        r#"int glue(cirrus_bitblt_rop_fwd_, ROP_NAME)(CirrusVGAState *s, uint8_t *dst, const uint8_t *src,
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
        }"#
    )]
    #[case(
        "glue",
        r#"glue(cirrus_bitblt_rop_fwd_, ROP_NAME) {
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
        }"#,
        r#"int glue(cirrus_bitblt_rop_fwd_, ROP_NAME) {
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
        }"#
    )]
    #[case(
        "correct",
        "void modern_func(int x) { return; }",
        "void modern_func(int x) { return; }"
    )]
    #[case(
        "kr",
        "static char *kr_with_type(p) char *p; { return p; }",
        "static char *kr_with_type(p) char *p; { return p; }"
    )]
    #[case(
        "global var",
        "int global_x = 10;",
        "int global_x = 10;"
    )]
    #[case(
        "isdn_ioctl",
        r#"isdn_ioctl(struct inode *inode, struct file *file, uint cmd, ulong arg) {
            uint minor = iminor(inode);
            isdn_ctrl c;
            int drvidx;
            int chidx;
            int ret;
            int i;
            char *p;
            char *s;
            union iocpar {
                char name[10];
                char bname[22];
                isdn_ioctl_struct iocts;
                isdn_net_ioctl_phone phone;
                isdn_net_ioctl_cfg cfg;
            } iocpar;
            void *argp = (void *)arg;
        }"#,
        r#"int isdn_ioctl(struct inode *inode, struct file *file, uint cmd, ulong arg) {
            uint minor = iminor(inode);
            isdn_ctrl c;
            int drvidx;
            int chidx;
            int ret;
            int i;
            char *p;
            char *s;
            union iocpar {
                char name[10];
                char bname[22];
                isdn_ioctl_struct iocts;
                isdn_net_ioctl_phone phone;
                isdn_net_ioctl_cfg cfg;
            } iocpar;
            void *argp = (void *)arg;
        }"#
    )]
    fn test_add_missing_return_types(
        sanitizer: CodeSanitizer,
        mut parser: Parser,
        language: Language,
        #[case] name: &str,
        #[case] input: &str,
        #[case] expected: &str,
    ) {
        let tree = parser.parse(input, None).unwrap();
        let result = sanitizer.add_missing_return_types(input, &tree).unwrap();
        test_helpers::assert_code_eq(
            language,
            result.trim(),
            expected.trim(),
            "Test case failed :",
            name,
        );
    }
}
