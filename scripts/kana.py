"""
scripts/kana.py
---------------
Kana rendering of phone3 token sequences + KER (kana error rate).

The repo's headline "CER" was historically computed as an edit distance over
phone3 *tokens* (``k``, ``sh``, ``cl``, ``I``/``U`` …), i.e. a phone error rate
(PER).  This module provides the complementary *character*-level metric — the
**kana error rate (KER)**: both hypothesis and reference phone3 sequences are
rendered as kana strings and the Levenshtein distance is taken over kana
characters (a character error rate, CER, computed on kana).

Conventions follow nyosegawa/hiragana-asr
(https://github.com/nyosegawa/hiragana-asr), which reports JSUT KER:

* **Punctuation is excluded from scoring** — their converter drops
  ``、。？！…`` before evaluation, so ``pau`` is stripped from both hypothesis
  and reference here as well (same treatment as PER).  ``pau`` still renders
  as 、 for display via :func:`phone3_tokens_to_kana`.
* Kana characters only; word boundaries/spaces carry no score.

Deliberate differences from hiragana-asr (kept self-consistent with this
repo's phone-output model, no extra dependencies):

* References are derived from the phone3 transcripts rather than from
  pyopenjtalk G2P of the surface text, so both sides share identical
  conventions (pronunciation-faithful, mora-transparent).
* Long vowels stay as doubled vowel kana (こお, not こー) — no ー collapsing —
  so the mapping from phone3 tokens is unambiguous; the convention cancels
  between reference and hypothesis.
* KER is aggregated corpus-level (total edits / total reference characters),
  matching this repo's PER/F1 reporting; hiragana-asr averages per-utterance.

Other design notes
------------------
* One kana per mora: onset tokens combine with the following vowel token
  (``k a`` → か, ``sh i`` → し, ``ky a`` → きゃ); ``cl`` → っ, ``N`` → ん.
* Devoiced ``I``/``U`` fold to their voiced kana (い/う): orthography does not
  mark devoicing, so KER is deliberately devoicing-blind — devoicing quality
  is measured by the P/R/F1, CCDA and DEO metrics.
* Robustness: model hypotheses can contain illegal phone sequences (e.g. an
  onset with no following vowel).  A lone onset falls back to its -u row kana
  (``k`` → く) so decoding never crashes; reference transcripts never hit the
  fallback (checked over the full corpus).
"""
from __future__ import annotations

_VOWEL_KANA = {
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
    "I": "い", "U": "う",                      # devoiced fold to voiced kana
}

# onset token -> {vowel-family: kana}; vowel key folds I->i, U->u
_ONSET_KANA: dict[str, dict[str, str]] = {
    "k":  {"a": "か", "i": "き", "u": "く", "e": "け", "o": "こ"},
    "ky": {"a": "きゃ", "i": "きぃ", "u": "きゅ", "e": "きぇ", "o": "きょ"},
    "s":  {"a": "さ", "i": "し", "u": "す", "e": "せ", "o": "そ"},
    "sh": {"a": "しゃ", "i": "し", "u": "しゅ", "e": "しぇ", "o": "しょ"},
    "t":  {"a": "た", "i": "ち", "u": "つ", "e": "て", "o": "と"},
    "ts": {"a": "つぁ", "i": "つぃ", "u": "つ", "e": "つぇ", "o": "つぉ"},
    "ch": {"a": "ちゃ", "i": "ち", "u": "ちゅ", "e": "ちぇ", "o": "ちょ"},
    "n":  {"a": "な", "i": "に", "u": "ぬ", "e": "ね", "o": "の"},
    "ny": {"a": "にゃ", "i": "にぃ", "u": "にゅ", "e": "にぇ", "o": "にょ"},
    "h":  {"a": "は", "i": "ひ", "u": "ふ", "e": "へ", "o": "ほ"},
    "hy": {"a": "ひゃ", "i": "ひぃ", "u": "ひゅ", "e": "ひぇ", "o": "ひょ"},
    "f":  {"a": "ふぁ", "i": "ふぃ", "u": "ふ", "e": "ふぇ", "o": "ふぉ"},
    "m":  {"a": "ま", "i": "み", "u": "む", "e": "め", "o": "も"},
    "my": {"a": "みゃ", "i": "みぃ", "u": "みゅ", "e": "みぇ", "o": "みょ"},
    "y":  {"a": "や", "i": "い", "u": "ゆ", "e": "え", "o": "よ"},
    "r":  {"a": "ら", "i": "り", "u": "る", "e": "れ", "o": "ろ"},
    "ry": {"a": "りゃ", "i": "りぃ", "u": "りゅ", "e": "りぇ", "o": "りょ"},
    "w":  {"a": "わ", "i": "うぃ", "u": "う", "e": "うぇ", "o": "を"},
    "g":  {"a": "が", "i": "ぎ", "u": "ぐ", "e": "げ", "o": "ご"},
    "gy": {"a": "ぎゃ", "i": "ぎぃ", "u": "ぎゅ", "e": "ぎぇ", "o": "ぎょ"},
    "z":  {"a": "ざ", "i": "じ", "u": "ず", "e": "ぜ", "o": "ぞ"},
    "j":  {"a": "じゃ", "i": "じ", "u": "じゅ", "e": "じぇ", "o": "じょ"},
    "d":  {"a": "だ", "i": "ぢ", "u": "づ", "e": "で", "o": "ど"},
    "dy": {"a": "ぢゃ", "i": "ぢぃ", "u": "ぢゅ", "e": "ぢぇ", "o": "ぢょ"},
    "b":  {"a": "ば", "i": "び", "u": "ぶ", "e": "べ", "o": "ぼ"},
    "by": {"a": "びゃ", "i": "びぃ", "u": "びゅ", "e": "びぇ", "o": "びょ"},
    "p":  {"a": "ぱ", "i": "ぴ", "u": "ぷ", "e": "ぺ", "o": "ぽ"},
    "py": {"a": "ぴゃ", "i": "ぴぃ", "u": "ぴゅ", "e": "ぴぇ", "o": "ぴょ"},
    "v":  {"a": "ゔぁ", "i": "ゔぃ", "u": "ゔ", "e": "ゔぇ", "o": "ゔぉ"},
}

_SPECIAL_KANA = {"cl": "っ", "N": "ん", "pau": "、"}

_SKIP_TOKENS = {"<pad>", "<sos>", "<eos>", "<unk>", "<sp>"}



def phone3_tokens_to_kana(tokens: list[str]) -> str:
    """Render a phone3 token sequence as a kana string.

    ``cl``→っ, ``N``→ん, ``pau``→、; devoiced I/U fold to い/う.  A lone onset
    (no following vowel — only possible in model hypotheses) falls back to its
    -u row kana so decoding never crashes.
    """
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _SKIP_TOKENS:
            i += 1
            continue
        if tok in _SPECIAL_KANA:
            out.append(_SPECIAL_KANA[tok])
            i += 1
            continue
        if tok in _VOWEL_KANA:
            out.append(_VOWEL_KANA[tok])
            i += 1
            continue
        # onset: needs the following vowel to pick the kana
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        vowel = {"I": "i", "U": "u"}.get(nxt, nxt)
        if vowel in ("a", "i", "u", "e", "o") and tok in _ONSET_KANA:
            out.append(_ONSET_KANA[tok][vowel])
            i += 2
        elif tok in _ONSET_KANA:
            out.append(_ONSET_KANA[tok]["u"])   # lone-onset fallback (hypotheses only)
            i += 1
        else:
            # unknown token — surface it visibly rather than dropping silently
            out.append(f"�{tok}�")
            i += 1
    return "".join(out)


def ids_to_kana(ids: list[int], id2token: dict[int, str]) -> str:
    """Kana rendering of a token-id sequence (special ids skipped)."""
    return phone3_tokens_to_kana([id2token.get(t, "<unk>") for t in ids])


def _edit_distance(a: list, b: list) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        new = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            new[j] = dp[j - 1] if a[i - 1] == b[j - 1] else 1 + min(dp[j], new[j - 1], dp[j - 1])
        dp = new
    return dp[len(b)]


def compute_kana_error_rate(pred_ids, ref_ids, tokenizer) -> float:
    """Kana error rate, KER (%): character error rate over the kana rendering
    of phone3 sequences.

    Unlike the PER metric (edit distance over phone tokens), the Levenshtein
    distance here runs over kana *characters*, so this is a true character
    error rate (CER) on kana — the "kana error rate" of
    github.com/nyosegawa/hiragana-asr.  Devoicing is invisible to it (I/U fold
    to い/う).  ``<pad>``/``<sos>``/``<eos>`` **and ``pau``** are stripped
    before scoring (punctuation excluded, as in hiragana-asr's converter), so
    KER and PER see the same spoken content at different granularities.

    Input:
    - pred_ids / ref_ids: token-id sequences (with or without SOS/EOS/PAD)
    - tokenizer: JapaneseRomajiRevTokenizer3 (uses id2token + special ids)
    """
    strip = {tokenizer.PAD_ID, tokenizer.SOS_ID, tokenizer.EOS_ID, tokenizer.pau_id}
    total_dist = total_ref = 0
    for pred, ref in zip(pred_ids, ref_ids):
        pred_kana = ids_to_kana([t for t in pred if t not in strip], tokenizer.id2token)
        ref_kana = ids_to_kana([t for t in ref if t not in strip], tokenizer.id2token)
        total_dist += _edit_distance(list(pred_kana), list(ref_kana))
        total_ref += len(ref_kana)
    return total_dist / max(1, total_ref) * 100
