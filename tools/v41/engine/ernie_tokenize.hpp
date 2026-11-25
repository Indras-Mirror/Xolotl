// tools/v41/engine/ernie_tokenize.hpp - ERNIE-5.5-Flash's text encoder: byte-level GPT-2 BPE through the `ernie5`
// pre-tokenizer (tools/strata_tokenizer.py's Tokenizer, `_bpe`, `_pre_split`, PRE_PATTERNS["ernie5"]).
//
// The invariant this header serves: `ernie_generate --encode-only -p TEXT` prints exactly the id list
// `llama-tokenize -m <same gguf> --ids -p TEXT` prints.  Both the vocabulary and the merge ranks come from the
// GGUF, so only the algorithm is code.
//
// The pre-tokenizer follows llama.cpp's `unicode_regex_split` for a pattern list with `\p{..}` classes: every
// codepoint is COLLAPSED to one byte (ASCII as itself, a non-ASCII \p{N}/\p{L}/\p{P}/\p{M}/\p{S} to 0xD1..0xD5,
// other non-ASCII to 0xD0, non-ASCII whitespace to 0x0B) and the classes in the pattern are replaced by the
// collapsed byte plus the class's ASCII ranges.  Each pattern in the list then splits the pieces the previous
// one left.  That is the reference's own byte-for-byte behaviour, so the id lists agree by construction.
#pragma once

#include <array>
#include <cstdint>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::v41 {

struct ErnieTokenizer {
    // ---- from the GGUF ----
    std::vector<std::string> tokens;      // tokenizer.ggml.tokens
    std::vector<std::string> merges;      // tokenizer.ggml.merges ("left right")
    std::vector<int32_t> token_types;     // tokenizer.ggml.token_type (empty when the file carries none)
    std::string pre;                      // tokenizer.ggml.pre ("ernie5")
    int32_t bos_id = 2;                   // tokenizer.ggml.bos_token_id

    // ---- built by ernie_tokenizer_load ----
    std::unordered_map<std::string, int32_t> ids;          // token text -> id
    std::unordered_map<std::string, int32_t> merge_ranks;  // left + '\x1f' + right -> rank
    std::vector<std::pair<std::string, int32_t>> always_literals;   // type 4, longest first
    std::vector<std::pair<std::string, int32_t>> special_literals;  // types 3+4, longest first
    std::regex re_digits;    // pattern 1 on the collapsed text
    std::regex re_words;     // pattern 3 on the collapsed text
    std::string byte_char[256];          // GPT-2 byte -> the one-codepoint token text it maps to
    std::array<int32_t, 0x200> cp_to_byte{};   // mapped codepoint -> byte (-1 when not in the byte alphabet)
};

/// Reads the tokenizer from `gguf` (no weights are touched).  False + `err` on a non-gpt2 / non-ernie5 file.
bool ernie_tokenizer_load(const std::string& gguf, ErnieTokenizer& tk, std::string& err);

/// Token ids of `text`.  `parse_special` controls only the type-3 CONTROL literals (`<|ernie:im_start|>`); the
/// type-4 USER_DEFINED ones (`<ernie:think>`) are matched either way, as llama.cpp's tokenizer does.
/// Throws std::runtime_error when BPE produces a token outside the vocabulary (the Python reference raises too).
std::vector<int> ernie_tokenizer_encode(const ErnieTokenizer& tk, const std::string& text, bool parse_special = false);

/// The text of `ids`: bytes through the GPT-2 alphabet; a type-3/4 token contributes its literal string as UTF-8.
/// For ids produced by ernie_tokenizer_encode this is the original text exactly (the byte layer is lossless).
std::string ernie_tokenizer_decode(const ErnieTokenizer& tk, const std::vector<int>& ids);

}  // namespace strata::v41
