"""
scripts/analyze_devoicing_ojt.py
─────────────────────────────────
Devoicing distribution/frequency analysis for the (manually-corrected)
phone3 transcript at dataset/jsut_ver1.1/basic5000/transcript_phone3_rev.txt
(the file train_r3.py trains on — formerly transcript_phone3_ojt.txt, renamed
in place by the user).

Devoicing rule: a token is a devoiced-vowel token iff 'I' in tok or 'U' in tok,
excluding the moraic nasal 'N'. This is the same rule the project's own
tokenizer/eval use (scripts/phone_tokenizer.py:vowel_groups(),
scripts/devoicing_eval.py:_DEVOICED_SYMBOL) — simpler than the sibling
ja_devoicing_vowel repo's 14-candidate-C1V restriction (which excludes a few
loanword-only onsets like 'ti' for a specific linguistic paper analysis); here
we count every observed devoicing token so the corpus totals are complete.

Writes to doc/:
  devoicing_distribution_per_sentence.csv  — per-utterance devoicing count + I/U split
  devoicing_token_frequency.csv            — count per observed devoicing token
                                              (onset consonant + devoiced I/U)
  devoicing_summary.json                   — corpus-wide totals

Usage:
    python scripts/analyze_devoicing_ojt.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / 'dataset' / 'jsut_ver1.1' / 'basic5000'
TRANSCRIPT   = DATA_DIR / 'transcript_phone3_rev.txt'
DOC_DIR      = PROJECT_ROOT / 'doc'

OUT_PER_SENTENCE = DOC_DIR / 'devoicing_distribution_per_sentence.csv'
OUT_FREQ         = DOC_DIR / 'devoicing_token_frequency.csv'
OUT_SUMMARY      = DOC_DIR / 'devoicing_summary.json'

NON_ONSET = {'pau', 'cl', 'N'}  # not a consonant onset that can precede a devoiced vowel


def is_devoiced(tok: str) -> bool:
    return tok != 'N' and ('I' in tok or 'U' in tok)


def parse_transcript(path: Path) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.rstrip()
        if not line or ':' not in line:
            continue
        utt_id, text = line.split(':', 1)
        entries[utt_id] = text.split()
    return entries


def main() -> None:
    entries = parse_transcript(TRANSCRIPT)

    per_sentence_rows: list[tuple[str, int, int, int]] = []  # utt_id, n_i, n_u, n_total_tokens
    token_counts: Counter[str] = Counter()
    total_tokens = 0
    total_devoiced = 0

    for utt_id, toks in entries.items():
        total_tokens += len(toks)
        n_i = n_u = 0
        for i, t in enumerate(toks):
            if not is_devoiced(t):
                continue
            prev = toks[i - 1] if i > 0 else None
            onset = prev if (prev is not None and prev not in NON_ONSET and not prev.isupper()) else 'bare'
            vowel = 'I' if 'I' in t else 'U'
            token_counts[f'{onset}{vowel}'] += 1
            if vowel == 'I':
                n_i += 1
            else:
                n_u += 1
        total_devoiced += n_i + n_u
        per_sentence_rows.append((utt_id, n_i, n_u, len(toks)))

    # ── Per-sentence CSV ──
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PER_SENTENCE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['utterance_id', 'n_i_devoiced', 'n_u_devoiced', 'n_devoiced_total', 'n_phone_tokens'])
        for utt_id, n_i, n_u, n_tok in per_sentence_rows:
            writer.writerow([utt_id, n_i, n_u, n_i + n_u, n_tok])

    # ── Token frequency CSV ──
    grand_total_tokens = sum(token_counts.values())
    with open(OUT_FREQ, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['token', 'count', 'pct_of_all_devoicing_events'])
        for token, count in token_counts.most_common():
            pct = 100 * count / grand_total_tokens if grand_total_tokens else 0
            writer.writerow([token, count, f'{pct:.2f}'])
        writer.writerow(['TOTAL', grand_total_tokens, '100.00'])

    # ── Corpus summary ──
    totals = [n_i + n_u for _, n_i, n_u, _ in per_sentence_rows]
    n_utt = len(per_sentence_rows)
    n_zero = sum(1 for t in totals if t == 0)
    summary = {
        'n_utterances': n_utt,
        'total_phone_tokens': total_tokens,
        'total_devoiced_tokens': total_devoiced,
        'pct_devoiced_of_all_tokens': round(100 * total_devoiced / total_tokens, 3) if total_tokens else 0.0,
        'mean_devoicing_per_sentence': round(total_devoiced / n_utt, 3) if n_utt else 0.0,
        'max_devoicing_in_a_sentence': max(totals) if totals else 0,
        'sentences_with_zero_devoicing': n_zero,
        'pct_sentences_with_at_least_one_devoicing': round(100 * (n_utt - n_zero) / n_utt, 2) if n_utt else 0.0,
        'devoicing_token_counts': token_counts.most_common(),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'utterances                : {n_utt}')
    print(f'total phone tokens        : {total_tokens}')
    print(f'total devoiced tokens     : {total_devoiced}  ({summary["pct_devoiced_of_all_tokens"]}% of all tokens)')
    print(f'mean devoicing / sentence : {summary["mean_devoicing_per_sentence"]}')
    print(f'max devoicing in a sentence: {summary["max_devoicing_in_a_sentence"]}')
    print(f'sentences w/ 0 devoicing  : {n_zero} ({100 - summary["pct_sentences_with_at_least_one_devoicing"]:.2f}%)')
    print('top 10 devoicing tokens   :')
    for token, count in token_counts.most_common(10):
        print(f'  {token:6s} {count}')
    print(f'wrote {OUT_PER_SENTENCE}')
    print(f'wrote {OUT_FREQ}')
    print(f'wrote {OUT_SUMMARY}')


if __name__ == '__main__':
    main()
