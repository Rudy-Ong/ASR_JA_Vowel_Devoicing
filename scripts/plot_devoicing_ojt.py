"""
scripts/plot_devoicing_ojt.py
──────────────────────────────
Devoicing distribution charts for
dataset/jsut_ver1.1/basic5000/transcript_phone3_rev.txt, reading the CSVs/JSON
written by scripts/analyze_devoicing_ojt.py. Labels are in Japanese for use in
the paper. Each chart is written as its own standalone image (no multi-panel
figure):

  devoicing_ojt_distribution_per_sentence.png  — per-sentence devoicing-count
                                                  histogram, stacked by I/U
  devoicing_ojt_distribution_token_freq.png    — token-frequency bar chart
                                                  (top devoicing tokens: onset + I/U)
  devoicing_ojt_distribution_corpus_totals.png — corpus-wide phoneme vs.
                                                  devoiced-phoneme totals

Usage:
    python scripts/analyze_devoicing_ojt.py   # first, to generate the CSVs/JSON
    python scripts/plot_devoicing_ojt.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CJK_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(_CJK_FONT_PATH).exists():
    font_manager.fontManager.addfont(_CJK_FONT_PATH)
    _CJK_FONT = font_manager.FontProperties(fname=_CJK_FONT_PATH).get_name()
    plt.rcParams['font.family'] = [_CJK_FONT, 'DejaVu Sans']

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_DIR      = PROJECT_ROOT / 'doc'
IMG_DIR      = PROJECT_ROOT / 'img'

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.kana import phone3_tokens_to_kana  # noqa: E402


def token_to_kana(token: str) -> str:
    """Hiragana rendering of a devoicing-frequency token (onset + I/U), e.g.
    'kI' -> 'き', 'shU' -> 'しゅ'. 'bare' onset (vowel-initial) falls back to
    the plain vowel kana."""
    onset, vowel_letter = token[:-1], token[-1]
    toks = [vowel_letter] if onset in ('', 'bare') else [onset, vowel_letter]
    return phone3_tokens_to_kana(toks)

IN_PER_SENTENCE = DOC_DIR / 'devoicing_distribution_per_sentence.csv'
IN_FREQ         = DOC_DIR / 'devoicing_token_frequency.csv'
IN_SUMMARY      = DOC_DIR / 'devoicing_summary.json'

OUT_PER_SENTENCE_PNG = IMG_DIR / 'devoicing_ojt_distribution_per_sentence.png'
OUT_TOKEN_FREQ_PNG    = IMG_DIR / 'devoicing_ojt_distribution_token_freq.png'
OUT_CORPUS_TOTALS_PNG = IMG_DIR / 'devoicing_ojt_distribution_corpus_totals.png'

# Categorical palette (validated blue/orange pair, see dataviz skill), neutral
# gray for the non-devoiced baseline category.
COLOR_I    = '#2a78d6'
COLOR_U    = '#eb6834'
COLOR_NONE = '#999999'
TOP_N_TOKENS = 12
BAR_EDGE = dict(edgecolor='white', linewidth=0.8)


def load_per_sentence():
    rows = []
    with open(IN_PER_SENTENCE, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append((int(row['n_i_devoiced']), int(row['n_u_devoiced'])))
    return rows


def load_freq():
    toks, counts = [], []
    with open(IN_FREQ, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['token'] == 'TOTAL':
                continue
            toks.append(row['token'])
            counts.append(int(row['count']))
    return toks[:TOP_N_TOKENS], counts[:TOP_N_TOKENS]


def plot_per_sentence(per_sentence: list[tuple[int, int]], n_utterances: int) -> None:
    totals = [n_i + n_u for n_i, n_u in per_sentence]
    max_bin = max(totals)

    bins = {k: {'n': 0, 'i': 0, 'u': 0} for k in range(max_bin + 1)}
    for n_i, n_u in per_sentence:
        k = n_i + n_u
        bins[k]['n'] += 1
        bins[k]['i'] += n_i
        bins[k]['u'] += n_u

    ks = list(range(max_bin + 1))
    none_h, i_h, u_h = [], [], []
    for k in ks:
        b = bins[k]
        if k == 0:
            none_h.append(b['n']); i_h.append(0); u_h.append(0)
        else:
            denom = b['i'] + b['u']
            frac_i = b['i'] / denom if denom else 0.5
            none_h.append(0)
            i_h.append(b['n'] * frac_i)
            u_h.append(b['n'] * (1 - frac_i))

    n_sentences = [bins[k]['n'] for k in ks]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(ks, none_h, color=COLOR_NONE, label='無声化なし', **BAR_EDGE)
    ax.bar(ks, i_h, color=COLOR_I, label='無声化/I/', **BAR_EDGE)
    ax.bar(ks, u_h, bottom=i_h, color=COLOR_U, label='無声化/U/', **BAR_EDGE)
    for k, n in zip(ks, n_sentences):
        ax.text(k, n + max(n_sentences) * 0.012, f'{n:,}', ha='center', va='bottom', fontsize=15)
    ax.set_xlabel('文あたりの無声化回数')
    ax.set_ylabel('文数')
    ax.set_xticks(ks)
    ax.set_title(f'文あたりの母音無声化回数の分布')
    ax.legend(fontsize=15)
    ax.set_ylim(0, max(n_sentences) * 1.15)

    fig.tight_layout()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PER_SENTENCE_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT_PER_SENTENCE_PNG}')


def plot_token_freq(toks: list[str], counts: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [COLOR_I if t.endswith('I') else COLOR_U for t in toks]
    labels = [f'{t} ({token_to_kana(t)})' for t in toks]
    bars = ax.bar(labels, counts, color=colors, **BAR_EDGE)
    ax.bar_label(bars, padding=2, fontsize=9)
    ax.set_xlabel('無声化トークン')
    ax.set_ylabel('出現回数')
    ax.set_title(f'各無声化トークンの頻出')
    ax.tick_params(axis='x', rotation=45)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_I), plt.Rectangle((0, 0), 1, 1, color=COLOR_U)]
    ax.legend(handles, ['無声化/I/', '無声化/U/'], fontsize=15)
    ax.set_ylim(0, max(counts) * 1.15)
    
    fig.tight_layout()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_TOKEN_FREQ_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT_TOKEN_FREQ_PNG}')


def plot_corpus_totals(summary: dict) -> None:
    total_tok = summary['total_phone_tokens']
    total_devo = summary['total_devoiced_tokens']

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(['全音素トークン', '無声化トークン'], [total_tok, total_devo],
                   color=[COLOR_NONE, COLOR_I], **BAR_EDGE)
    ax.bar_label(bars, fmt='{:,.0f}', padding=3, fontsize=10)
    ax.set_title(
        f'コーパス全体の無声化集計\n'
        f'無声化トークンは全音素トークンの{summary["pct_devoiced_of_all_tokens"]:.2f}%\n'
        f'文あたり平均{summary["mean_devoicing_per_sentence"]:.2f}回、'
        f'最大{summary["max_devoicing_in_a_sentence"]}回、'
        f'{summary["pct_sentences_with_at_least_one_devoicing"]:.1f}%の文が1回以上の無声化を含む'
    )
    ax.set_ylabel('トークン数')
    ax.set_ylim(0, total_tok * 1.15)

    fig.tight_layout()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_CORPUS_TOTALS_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT_CORPUS_TOTALS_PNG}')


def main() -> None:
    per_sentence = load_per_sentence()
    toks, counts = load_freq()
    summary = json.loads(IN_SUMMARY.read_text(encoding='utf-8'))

    plot_per_sentence(per_sentence, summary['n_utterances'])
    plot_token_freq(toks, counts)
    plot_corpus_totals(summary)


if __name__ == '__main__':
    main()
