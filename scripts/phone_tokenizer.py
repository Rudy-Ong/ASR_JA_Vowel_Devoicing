"""
scripts/phone_tokenizer.py
---------------------------
Japanese ASR phone-level tokenizer for PER evaluation.

Vocabulary is built by scanning transcript_phone.txt for all unique
space-separated IPA tokens.  Devoiced vowels ɯ̥ and i̥ are the two
devoicing token types (analogs of the 12 romaji <X devo> tokens).

Special tokens
--------------
  <pad>  – padding
  <sos>  – start-of-sequence
  <eos>  – end-of-sequence
  <unk>  – unknown symbol
  <sp>   – word boundary space
"""
from __future__ import annotations

import json
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>", "<sp>"]

# Devoiced vowels: all 12 romaji devoicing cases reduce to one of these two.
DEVO_PHONES = ["ɯ̥", "i̥"]


def _build_vocab(transcript_path: Path) -> tuple[list[str], dict[str, int]]:
    tokens: set[str] = set()
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        _, phone_str = line.split(":", 1)
        tokens.update(phone_str.strip().split())
    vocab = SPECIAL_TOKENS + sorted(tokens)
    token2id = {t: i for i, t in enumerate(vocab)}
    return vocab, token2id


class JapanesePhoneTokenizer:
    """
    Tokenizer for IPA phone sequences with devoicing detection.

    encode(text) → list[int]
    decode(ids)  → str
    """

    PAD_ID = 0
    SOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    SP_ID  = 4

    def __init__(self, transcript_path: str | Path) -> None:
        self.vocab, self.token2id = _build_vocab(Path(transcript_path))
        self.id2token = {i: t for t, i in self.token2id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(
        self,
        text: str,
        add_sos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = []
        if add_sos:
            ids.append(self.SOS_ID)
        for tok in text.strip().split():
            ids.append(self.token2id.get(tok, self.UNK_ID))
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special = {self.PAD_ID, self.SOS_ID, self.EOS_ID}
        parts = []
        for i in ids:
            if skip_special and i in special:
                continue
            parts.append(self.id2token.get(i, "<unk>"))
        return " ".join(parts)

    def devo_token_ids(self) -> list[int]:
        return [self.token2id[p] for p in DEVO_PHONES if p in self.token2id]

    def is_devo_token(self, token_id: int) -> bool:
        return self.id2token.get(token_id, "") in DEVO_PHONES

    def save(self, path: str | Path) -> None:
        """Write the vocab JSON; skip when unchanged so a training run doesn't
        touch (and potentially dirty) the tracked vocab file."""
        path = Path(path)
        text = json.dumps({"vocab": self.vocab}, ensure_ascii=False, indent=2)
        if path.exists() and path.read_text(encoding="utf-8") == text:
            return
        path.write_text(text, encoding="utf-8")

    @classmethod
    def from_vocab(cls, path: str | Path) -> "JapanesePhoneTokenizer":
        obj = cls.__new__(cls)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj.vocab = data["vocab"]
        obj.token2id = {t: i for i, t in enumerate(obj.vocab)}
        obj.id2token = {i: t for t, i in obj.token2id.items()}
        return obj


# ── Phoneme-level tokenizer (capital I / U for devoiced vowels) ───────────────
DEVO_PHONEMES = ["I", "U"]


class JapanesePhonemeTokenizer(JapanesePhoneTokenizer):
    """Tokenizer for phoneme-level IPA transcripts (transcript_phoneme.txt).

    Identical to JapanesePhoneTokenizer except that devoiced vowels are
    represented as capital 'I' and 'U' rather than the combining-ring
    diacritics ɯ̥ / i̥ used in phone-level transcripts.
    """

    def devo_token_ids(self) -> list[int]:
        return [self.token2id[p] for p in DEVO_PHONEMES if p in self.token2id]

    def is_devo_token(self, token_id: int) -> bool:
        return self.id2token.get(token_id, "") in DEVO_PHONEMES


# ── Romaji-rev tokenizer (whole-syllable romaji, pau, capital I/U devoicing) ──
PAU_TOKEN = "pau"


class JapaneseRomajiRevTokenizer(JapanesePhoneTokenizer):
    """Tokenizer for whole-syllable romaji transcripts (transcript_romaji_rev.txt).

    Vocabulary is built by scanning the transcript for space-separated tokens
    (e.g. ``mi``, ``zu``, ``:``, ``pau``, ``shI``, ``kU``).  A devoiced vowel is
    encoded *inside* its syllable by capitalising the vowel letter
    (し→``shI``, く→``kU``, す→``sU``); any token carrying an upper-case letter is
    therefore a devoicing token.  ``pau`` (from an ideographic comma 、) is a
    normal vocab token but is excluded from PER scoring by callers.
    """

    def devo_token_ids(self) -> list[int]:
        return [i for t, i in self.token2id.items()
                if t not in SPECIAL_TOKENS and any(c.isupper() for c in t)]

    def is_devo_token(self, token_id: int) -> bool:
        tok = self.id2token.get(token_id, "")
        return tok not in SPECIAL_TOKENS and any(c.isupper() for c in tok)

    @property
    def pau_id(self) -> int:
        """Token id of the ``pau`` symbol, or -1 if absent from the vocab."""
        return self.token2id.get(PAU_TOKEN, -1)

    def render_pau(self, text: str) -> str:
        """Display helper: replace standalone ``pau`` tokens with a comma ``,``.

        Operates on a space-separated romaji-token string (decode output or a raw
        transcript line). Display-only — do not feed the result back into encode().
        """
        return " ".join("," if t == PAU_TOKEN else t for t in text.split())

    def vowel_groups(self) -> dict[str, dict[str, set[int]]]:
        """Map vowel family ('i'/'u') → {'voiced': {ids}, 'devoiced': {ids}}.

        The devoiced syllable (e.g. ``shI``) is paired with its voiced
        counterpart (``shi``) obtained by lower-casing; both must be present in
        the vocab for the pair to be added.
        """
        groups: dict[str, dict[str, set[int]]] = {
            "i": {"voiced": set(), "devoiced": set()},
            "u": {"voiced": set(), "devoiced": set()},
        }
        for tok, tid in self.token2id.items():
            if tok in SPECIAL_TOKENS or not any(c.isupper() for c in tok):
                continue
            label = "i" if "I" in tok else "u" if "U" in tok else None
            if label is None:
                continue
            groups[label]["devoiced"].add(tid)
            voiced = tok.lower()
            if voiced in self.token2id:
                groups[label]["voiced"].add(self.token2id[voiced])
        return {lab: g for lab, g in groups.items() if g["devoiced"]}


# ── phone_level3-based romaji-rev tokenizer (cl / N / n / pau, I·U devoicing) ─
class JapaneseRomajiRevTokenizer3(JapanesePhonemeTokenizer):
    """Tokenizer for phone_level3-derived transcripts (transcript_phone3_rev.txt).

    Vocabulary is the set of JSUT ``phone_level3`` units (one token per phone:
    ``k``, ``ky``, ``sh``, ``ts``, ``ry`` …) extended with rule-based devoicing:

      • ``cl``  – geminate / double consonant (the sokuon っ closure)
      • ``N``   – moraic nasal ん
      • ``n``   – na-row onset consonant
      • ``pau`` – pause emitted for an ideographic comma 、
      • ``I``/``U`` – devoiced high vowels (voiced counterparts stay ``i``/``u``)

    Devoicing is encoded by upper-casing the vowel token, so ``I``/``U`` are the
    two devoicing token types (inherited :meth:`devo_token_ids`). ``pau`` is a
    normal vocab token but is excluded from PER and KER scoring by callers
    (rendered as 、 for display only).
    """

    @property
    def pau_id(self) -> int:
        """Token id of the ``pau`` symbol, or -1 if absent from the vocab."""
        return self.token2id.get(PAU_TOKEN, -1)

    def render_pau(self, text: str) -> str:
        """Display helper: replace standalone ``pau`` tokens with a comma ``,``.

        Operates on a space-separated token string (decode output or a raw
        transcript line). Display-only — do not feed the result back to encode().
        """
        return " ".join("," if t == PAU_TOKEN else t for t in text.split())

    def vowel_groups(self) -> dict[str, dict[str, set[int]]]:
        """Map vowel family ('i'/'u') → {'voiced': {ids}, 'devoiced': {ids}}.

        For phone_level3 tokens the devoiced vowels are the single tokens ``I``
        and ``U``; their voiced counterparts are ``i`` / ``u`` (obtained by
        lower-casing). ``N`` (moraic nasal) carries no I/U and is skipped. Used
        by the devoiced-vowel confusion-matrix evaluation (devoicing_eval).
        """
        groups: dict[str, dict[str, set[int]]] = {
            "i": {"voiced": set(), "devoiced": set()},
            "u": {"voiced": set(), "devoiced": set()},
        }
        for tok, tid in self.token2id.items():
            if tok in SPECIAL_TOKENS or not any(c.isupper() for c in tok):
                continue
            label = "i" if "I" in tok else "u" if "U" in tok else None
            if label is None:
                continue
            groups[label]["devoiced"].add(tid)
            voiced = tok.lower()
            if voiced in self.token2id:
                groups[label]["voiced"].add(self.token2id[voiced])
        return {lab: g for lab, g in groups.items() if g["devoiced"]}
