import os

import google.generativeai as genai
from absl.testing import absltest

# get API key from .zshrc
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


class UnitTests(absltest.TestCase):
    def test_json_controlled_generation(self):
        # get API key from .zshrc
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        # decide which model to load
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")

        result = model.generate_content(
            """Generate an exaustive description, in English prose, of what the following C/C++ function could be used for. Don't describe parameters neither returned data. Don't use special characters.
            int _gnutls_ciphertext2compressed(gnutls_session_t session, opaque *compress_data, int compress_size, gnutls_datum_t ciphertext, uint8 type) {\n    uint8 MAC[MAX_HASH_SIZE];\n    uint16 c_length;\n    uint8 pad;\n    int length;\n    mac_hd_t td;\n    uint16 blocksize;\n    int ret, i, pad_failed = 0;\n    uint8 major, minor;\n    gnutls_protocol_t ver;\n    int hash_size = _gnutls_hash_get_algo_len(session->security_parameters.read_mac_algorithm);\n    ver = gnutls_protocol_get_version(session);\n    minor = _gnutls_version_get_minor(ver);\n    major = _gnutls_version_get_major(ver);\n    blocksize = _gnutls_cipher_get_block_size(session->security_parameters.read_bulk_cipher_algorithm);\n    td = mac_init(session->security_parameters.read_mac_algorithm, session->connection_state.read_mac_secret.data, session->connection_state.read_mac_secret.size, ver);\n    if (td == GNUTLS_MAC_FAILED && session->security_parameters.read_mac_algorithm != GNUTLS_MAC_NULL) {\n        gnutls_assert();\n        return GNUTLS_E_INTERNAL_ERROR;\n    }\n    switch (_gnutls_cipher_is_block(session->security_parameters.read_bulk_cipher_algorithm)) {\n    case CIPHER_STREAM:\n        if ((ret = _gnutls_cipher_decrypt(session->connection_state.read_cipher_state, ciphertext.data, ciphertext.size)) < 0) {\n            gnutls_assert();\n            return ret;\n        }\n        length = ciphertext.size - hash_size;\n        break;\n    case CIPHER_BLOCK:\n        if ((ciphertext.size < blocksize) || (ciphertext.size % blocksize != 0)) {\n            gnutls_assert();\n            return GNUTLS_E_DECRYPTION_FAILED;\n        }\n        if ((ret = _gnutls_cipher_decrypt(session->connection_state.read_cipher_state, ciphertext.data, ciphertext.size)) < 0) {\n            gnutls_assert();\n            return ret;\n        }\n        if (session->security_parameters.version >= GNUTLS_TLS1_1) {\n            ciphertext.size -= blocksize;\n            ciphertext.data += blocksize;\n            if (ciphertext.size == 0) {\n                gnutls_assert();\n                return GNUTLS_E_DECRYPTION_FAILED;\n            }\n        }\n        pad = ciphertext.data[ciphertext.size - 1] + 1;\n        length = ciphertext.size - hash_size - pad;\n        if (pad > ciphertext.size - hash_size) {\n            gnutls_assert();\n            pad_failed = GNUTLS_E_DECRYPTION_FAILED;\n        }\n        if (ver >= GNUTLS_TLS1)\n            for (i = 2; i < pad; i++) {\n                if (ciphertext.data[ciphertext.size - i] != ciphertext.data[ciphertext.size - 1])\n                    pad_failed = GNUTLS_E_DECRYPTION_FAILED;\n            }\n        break;\n    default:\n        gnutls_assert();\n        return GNUTLS_E_INTERNAL_ERROR;\n    }\n    if (length < 0)\n        length = 0;\n    c_length = _gnutls_conv_uint16((uint16) length);\n    if (td != GNUTLS_MAC_FAILED) {\n        _gnutls_hmac(td, UINT64DATA(session->connection_state.read_sequence_number), 8);\n        _gnutls_hmac(td, &type, 1);\n        if (ver >= GNUTLS_TLS1) {\n            _gnutls_hmac(td, &major, 1);\n            _gnutls_hmac(td, &minor, 1);\n        }\n        _gnutls_hmac(td, &c_length, 2);\n        if (length > 0)\n            _gnutls_hmac(td, ciphertext.data, length);\n        mac_deinit(td, MAC, ver);\n    }\n    if (pad_failed != 0)\n        return pad_failed;\n    if (memcmp(MAC, &ciphertext.data[length], hash_size) != 0) {\n        gnutls_assert();\n        return GNUTLS_E_DECRYPTION_FAILED;\n    }\n    if (compress_size < length) {\n        gnutls_assert();\n        return GNUTLS_E_INTERNAL_ERROR;\n    }\n    memcpy(compress_data, ciphertext.data, length);\n    return length;\n}
            """,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=str,
                candidate_count=1,
            ),
        )
        print(result.text)


class Gemini:
    def __init__(self, model_name: str = "gemini-1.5-pro") -> None:
        # decide which model to load
        self.model = genai.GenerativeModel(model_name=model_name)

    def generate_description(self, func_str: str) -> str:
        prompt: str = (
            "Generate one exaustive description, in English prose, of what the C/C++ function starting at the next line can be used for. Don't describe function parameters and neither the returned data. Don't use special characters."
        )

        prompt = "\n".join([prompt, func_str])

        synth_desc: str = self.model.generate_content(
            contents=prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=str,
                candidate_count=1,
            ),
        ).text

        return synth_desc

    def test(self) -> None:
        absltest.main()


if __name__ == "__main__":
    # absltest.main()
    Gemini().test()
