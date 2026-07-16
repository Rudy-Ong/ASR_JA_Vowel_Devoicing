"""
make_transcript_phone3_rev.py
─────────────────────────────
Build transcript_phone3_rev.txt from the JSUT ``phone_level3`` field in
basic5000.yaml, then save the matching vocab (tokenizer_romaji_rev_vocab.json).

Pipeline:
  basic5000.yaml  phone_level3      (hyphen-separated phone units, with pau/cl/N)
      │  (this script — space-separate + rule-based I/U devoicing)
      ▼
  transcript_phone3_rev.txt
      │  (JapaneseRomajiRevTokenizer3 — scan space-separated tokens)
      ▼
  tokenizer_romaji_rev_vocab.json

Token scheme (one token per phone, as in phone_level3):
  • cl   – geminate / double consonant (sokuon っ)
  • N    – moraic nasal ん
  • n    – na-row onset consonant
  • pau  – pause emitted for an ideographic comma 、
  • I/U  – devoiced high vowels (voiced counterparts stay i / u)

Devoicing rule (high vowel i/u between voiceless consonants):
  i/u is devoiced when its onset consonant is voiceless AND
    - the next mora's onset is voiceless, or
    - it is utterance-final after a voiceless onset (です / ます), or
    - it precedes a geminate cl whose doubled consonant is voiceless.
  A pause breaks the context: the mora right before, or right after, a pau is
  never devoiced.

Usage:
    python scripts/make_transcript_phone3_rev.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phone_tokenizer import JapaneseRomajiRevTokenizer3

PAU = "pau"
CL = "cl"

# Voiceless onsets among phone_level3 consonant tokens.
VOICELESS = frozenset({"k", "ky", "s", "sh", "t", "ts", "ch", "h", "hy", "f", "p", "py"})
# High vowels eligible for devoicing.
DEVO_VOWELS = frozenset({"i", "u"})


def parse_phone_level3(yaml_path: Path) -> dict[str, list[str]]:
    """Extract {utt_id: [phone tokens]} from the phone_level3 lines of the yaml.

    Lightweight line scan (no yaml dep): an utterance id is a non-indented
    ``KEY:`` line; phone_level3 is its indented ``phone_level3: a-b-c`` child.
    """
    out: dict[str, list[str]] = {}
    cur: str | None = None
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace() and line.rstrip().endswith(":"):
            cur = line.strip()[:-1]
            continue
        s = line.strip()
        if s.startswith("phone_level3:") and cur is not None:
            seq = s.split(":", 1)[1].strip()
            out[cur] = seq.split("-") if seq else []
    return out


def devoice(tokens: list[str]) -> list[str]:
    """Upper-case devoiced high vowels (i→I, u→U) per the voiceless-onset rule."""
    out = list(tokens)
    n = len(tokens)
    for idx, tok in enumerate(tokens):
        if tok not in DEVO_VOWELS:
            continue
        prev = tokens[idx - 1] if idx > 0 else None
        if prev not in VOICELESS:               # onset must be voiceless
            continue
        if idx >= 2 and tokens[idx - 2] == PAU:  # first mora right after a pau
            continue
        nxt = tokens[idx + 1] if idx + 1 < n else None
        if nxt is None:
            dv = True                            # utterance-final (です/ます)
        elif nxt == PAU:
            dv = False                           # mora right before a pau
        elif nxt == CL:
            after = tokens[idx + 2] if idx + 2 < n else None
            dv = (after in VOICELESS) if after is not None else True
        elif nxt in VOICELESS:
            dv = True
        else:
            dv = False
        if dv:
            out[idx] = tok.upper()
    return out


def process(yaml_path: Path, dst_paths: list[Path]) -> Path:
    p3 = parse_phone_level3(yaml_path)
    lines = [f"{uid}:{' '.join(devoice(toks))}" for uid, toks in p3.items()]
    text = "\n".join(lines) + "\n"
    for dst in dst_paths:
        dst.write_text(text, encoding="utf-8")
        print(f"Wrote {len(lines)} utterances → {dst}")
    print("\nSample output:")
    for row in lines[:5]:
        print(" ", row)
    return dst_paths[0]


if __name__ == "__main__":
    _DATA = ROOT / "dataset" / "jsut_ver1.1" / "basic5000"
    _VOCAB = ROOT / "doc" / "tokenizer_romaji_rev_vocab.json"

    parser = argparse.ArgumentParser(
        description="Generate transcript_phone3_rev.txt + tokenizer_romaji_rev_vocab.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--yaml", default=str(_DATA / "basic5000.yaml"))
    parser.add_argument("--dst", nargs="+",
                        default=[str(_DATA / "transcript_phone3_rev.txt")],
                        help="Output transcript path(s); written to each.")
    parser.add_argument("--vocab", default=str(_VOCAB))
    args = parser.parse_args()

    first = process(Path(args.yaml), [Path(p) for p in args.dst])

    tok = JapaneseRomajiRevTokenizer3(first)
    tok.save(args.vocab)
    print(f"\nVocab size: {tok.vocab_size} → {args.vocab}")
    print("Devoicing token ids (I/U):", tok.devo_token_ids())
    print("Vocab:", json.dumps(tok.vocab, ensure_ascii=False))
