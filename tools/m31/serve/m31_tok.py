"""MiniMax's byte BPE adapter; the vocabulary and merges come from the GGUF.

The split is llama.cpp's MINIMAX_M2 pattern (PR #31192). NFC is applied to
ordinary spans, after special-token partitioning, as in llama_tokenize.
"""
import unicodedata

import strata_tokenizer as ST


MINIMAX_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
    r"|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
ST.PRE_PATTERNS.setdefault("minimax-m2", [MINIMAX_PATTERN])


class M31Tokenizer(ST.Tokenizer):
    def __init__(self, tokens, merges, token_types=None, pre="minimax-m2", special_ids=None, nfc=True):
        if pre != "minimax-m2":
            raise ValueError("M31 needs GGUF pre-tokenizer minimax-m2")
        super().__init__(tokens, merges, token_types, pre=pre, special_ids=special_ids)
        self.nfc = nfc
        self.vocab_size = len(tokens)
        self._literal_bytes = {i: text.encode("utf-8") for text, i in self.special_tokens.items()}

    def _encode_plain(self, text):
        return super()._encode_plain(unicodedata.normalize("NFC", text) if self.nfc else text)

    def token_bytes(self, token):
        value = self._literal_bytes.get(token)
        return value if value is not None else super().token_bytes(token)
