"""
scripts/stratified_sampling.py
─────────────────────────────
Stratified sampling of JSUT basic5000 utterances by transcript length.
Also computes per-utterance devoiced-token counts for distribution analysis.

Standalone usage (generate manifest + optional plot):
    python scripts/stratified_sampling.py
    python scripts/stratified_sampling.py --plot

Import bridge for train_p.py / train.py:
    from scripts.stratified_sampling import load_splits
    train_ids, val_ids, test_ids = load_splits()
"""

from __future__ import annotations

import wave as _wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# ── Constants ─────────────────────────────────────────────────────────────────
SEED             = 42
_DEVO_CHAR       = '̥'   # combining ring below — marks devoiced vowels in IPA
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR         = PROJECT_ROOT / 'dataset' / 'jsut_ver1.1' / 'basic5000'
WAV_DIR          = DATA_DIR / 'wav'
DEFAULT_MANIFEST = DATA_DIR / 'stratified_manifest.csv'
DEFAULT_IMG_DIR  = PROJECT_ROOT / 'img'


# ── Internal helpers ──────────────────────────────────────────────────────────
def _parse_transcript(path: Path) -> dict[str, str]:
    """Parse 'UTT_ID:text' transcript file into {utt_id: text} dict."""
    data: dict[str, str] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if ':' not in line:
            continue
        utt_id, text = line.split(':', 1)
        data[utt_id.strip()] = text.strip()
    return data


def _assign_strata(
    phone_lengths: np.ndarray,
    q1: float,
    q2: float,
    q3: float,
) -> np.ndarray:
    """Assign each utterance to strata group 1–4 based on phone token count.

    Group 1 (short):      phone_len <= Q1
    Group 2 (med-short):  Q1 < phone_len <= Q2
    Group 3 (med-long):   Q2 < phone_len <= Q3
    Group 4 (long):       phone_len > Q3
    """
    s = np.ones(len(phone_lengths), dtype=int)
    s[(phone_lengths > q1) & (phone_lengths <= q2)] = 2
    s[(phone_lengths > q2) & (phone_lengths <= q3)] = 3
    s[phone_lengths > q3]                           = 4
    return s


def _audio_dur(wav_path: Path) -> float:
    """Return audio duration in seconds via stdlib wave module."""
    with _wave.open(str(wav_path)) as w:
        return w.getnframes() / w.getframerate()


# ── Core: build manifest ──────────────────────────────────────────────────────
def build_manifest(
    data_dir: Path = DATA_DIR,
    wav_dir:  Path = WAV_DIR,
    seed:     int  = SEED,
) -> pd.DataFrame:
    """Compute stratified split and return a full metadata DataFrame.

    Columns: utt_id, wav_path, phone_len, devo_count,
             audio_dur_s, strata_group, split
    """
    # 1. Locate phone transcript (primary stratification key)
    phone_path = data_dir / 'transcript_phone3_rev.txt'
    if not phone_path.exists():
        raise FileNotFoundError(
            f"Phone transcript not found in {data_dir}\n"
            "Expected: transcript_phone3_rev.txt"
        )
    phone_raw  = _parse_transcript(phone_path)

    # 2. Filter to utterances that have a matching WAV file
    all_ids = sorted(k for k in phone_raw if (wav_dir / f'{k}.wav').exists())
    if not all_ids:
        raise RuntimeError(f"No utterances with WAV files found in {wav_dir}")
    print(f'Total utterances with WAV: {len(all_ids)}')

    # 3. Compute transcript token lengths (space-separated tokens)
    phone_lens  = np.array([len(phone_raw[k].split())          for k in all_ids])

    # 4. Count devoiced tokens per utterance (IPA combining ring below U+0325)
    devo_counts = np.array([
        sum(1 for t in phone_raw[k].split() if _DEVO_CHAR in t)
        for k in all_ids
    ])

    # 5. Quartile boundaries from phone lengths
    q1, q2, q3 = np.percentile(phone_lens, [25, 50, 75])
    print(f'Phone token quartiles: Q1={q1:.1f}  Q2={q2:.1f}  Q3={q3:.1f}')
    print(f'Devoiced tokens/utt  : mean={devo_counts.mean():.2f}  '
          f'max={devo_counts.max()}  '
          f'zero-devo={int((devo_counts == 0).sum())} utterances')

    # 6. Assign strata groups
    strata = _assign_strata(phone_lens, q1, q2, q3)

    # 7. Stratified 80 / 10 / 10 split
    ids_arr = np.array(all_ids)

    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, tmp_idx = next(sss_outer.split(ids_arr, strata))

    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(sss_inner.split(ids_arr[tmp_idx], strata[tmp_idx]))
    val_idx  = tmp_idx[val_rel]
    test_idx = tmp_idx[test_rel]

    split_labels = np.empty(len(ids_arr), dtype=object)
    split_labels[train_idx] = 'train'
    split_labels[val_idx]   = 'val'
    split_labels[test_idx]  = 'test'

    # 8. Audio durations via stdlib wave (no extra dependencies)
    print('Reading audio durations from WAV headers...')
    audio_durs = np.array([_audio_dur(wav_dir / f'{k}.wav') for k in all_ids])

    # 9. Relative wav_path from project ROOT
    wav_paths = [f'dataset/jsut_ver1.1/basic5000/wav/{k}.wav' for k in all_ids]

    return pd.DataFrame({
        'utt_id':       all_ids,
        'wav_path':     wav_paths,
        'phone_len':    phone_lens,
        'devo_count':   devo_counts,
        'audio_dur_s':  audio_durs,
        'strata_group': strata,
        'split':        split_labels,
    })


# ── Print report ──────────────────────────────────────────────────────────────
def print_report(df: pd.DataFrame) -> None:
    """Print statistics report to stdout."""
    pq1 = df['phone_len'].quantile(0.25)
    pq2 = df['phone_len'].quantile(0.50)
    pq3 = df['phone_len'].quantile(0.75)

    LABELS = {
        1: f'short   (<=Q1={pq1:.0f})',
        2: f'med-sh  (<=Q2={pq2:.0f})',
        3: f'med-lg  (<=Q3={pq3:.0f})',
        4: f'long    (>Q3={pq3:.0f})',
    }

    print('\n' + '=' * 62)
    print('  JSUT basic5000 — Stratified Split Report')
    print('=' * 62)
    print(f'  Total utterances : {len(df):,}')
    print(f'  Quartile bounds  : Q1={pq1:.0f}  Q2={pq2:.0f}  Q3={pq3:.0f}  (phone tokens)')
    print()

    # ── Strata × split table ──
    hdr = f"{'Grp':<5}{'Label':<22}{'Total':>6}{'Train':>7}{'Val':>6}{'Test':>6}  {'Tr%':>5}{'Va%':>5}{'Te%':>5}"
    print(hdr)
    print('-' * len(hdr))

    tot_n = tot_tr = tot_va = tot_te = 0
    for g in [1, 2, 3, 4]:
        sub  = df[df['strata_group'] == g]
        n    = len(sub)
        n_tr = (sub['split'] == 'train').sum()
        n_va = (sub['split'] == 'val').sum()
        n_te = (sub['split'] == 'test').sum()
        tot_n += n; tot_tr += n_tr; tot_va += n_va; tot_te += n_te
        print(
            f"{g:<5}{LABELS[g]:<22}{n:>6}{n_tr:>7}{n_va:>6}{n_te:>6}"
            f"  {n_tr/n*100:>5.1f}{n_va/n*100:>5.1f}{n_te/n*100:>5.1f}"
        )
    print('-' * len(hdr))
    print(
        f"{'TOT':<5}{'':22}{tot_n:>6}{tot_tr:>7}{tot_va:>6}{tot_te:>6}"
        f"  {tot_tr/tot_n*100:>5.1f}{tot_va/tot_n*100:>5.1f}{tot_te/tot_n*100:>5.1f}"
    )

    # ── Continuous variable stats ──
    ph  = df['phone_len']
    dur = df['audio_dur_s']
    dv  = df['devo_count']
    r_ph_dur = float(np.corrcoef(ph.values, dur.values)[0, 1])
    r_ph_dv  = float(np.corrcoef(ph.values, dv.values)[0, 1])
    print()
    print(f'  Phone  len  — mean={ph.mean():.1f}, std={ph.std():.1f}, min={ph.min()}, max={ph.max()}')
    print(f'  Audio  dur  — mean={dur.mean():.2f}s, std={dur.std():.2f}s, '
          f'min={dur.min():.2f}s, max={dur.max():.2f}s')
    print(f'  Devo count  — mean={dv.mean():.2f}, std={dv.std():.2f}, '
          f'min={dv.min()}, max={dv.max()}, '
          f'zero-devo={int((dv == 0).sum())} ({(dv == 0).mean()*100:.1f}%)')
    print(f'  Pearson r(phone_len, audio_dur_s) = {r_ph_dur:.4f}')
    print(f'  Pearson r(phone_len, devo_count)  = {r_ph_dv:.4f}')

    # ── Devoicing distribution per split ──
    print()
    print('  Devoiced token count per split:')
    print(f"    {'split':<7} {'mean':>6} {'std':>6} {'min':>4} {'max':>4} {'total':>7}  {'zero%':>6}")
    print('    ' + '-' * 47)
    for sp in ['train', 'val', 'test']:
        sub = df.loc[df['split'] == sp, 'devo_count']
        print(
            f"    {sp:<7} {sub.mean():>6.2f} {sub.std():>6.2f} "
            f"{sub.min():>4} {sub.max():>4} {sub.sum():>7}  "
            f"{(sub == 0).mean()*100:>6.1f}%"
        )

    # ── Devoicing counts per strata group ──
    print()
    print('  Devoiced token count per strata group:')
    print(f"    {'grp':<5} {'mean':>6} {'std':>6} {'min':>4} {'max':>4} {'total':>7}  {'zero%':>6}")
    print('    ' + '-' * 47)
    for g in [1, 2, 3, 4]:
        sub = df.loc[df['strata_group'] == g, 'devo_count']
        print(
            f"    {g:<5} {sub.mean():>6.2f} {sub.std():>6.2f} "
            f"{sub.min():>4} {sub.max():>4} {sub.sum():>7}  "
            f"{(sub == 0).mean()*100:>6.1f}%"
        )
    print('=' * 62 + '\n')


# ── Optional plot ─────────────────────────────────────────────────────────────
# Sized for a single column of a two-column A4 paper (column width ≈ 85 mm /
# 3.35in, e.g. IEICE/IPSJ/ASJ-style proceedings) — each chart is meant to be
# placed at native size (or \columnwidth) without further shrinking, so text
# is set to the point size it should read at in print, not scaled up for an
# on-screen preview.
_FIG_W_IN, _FIG_H_IN = 3.35, 2.7
_PLOT_DPI    = 300
_FONT_TITLE  = 9
_FONT_LABEL  = 8.5
_FONT_TICK   = 7.5
_FONT_LEGEND = 7

# Categorical palette: fixed-order, colour-vision-deficiency-safe hues
# (blue / orange / aqua / yellow) for the 4 strata groups.
_COLOURS = {1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a', 4: '#eda100'}
# Q1/Q2/Q3 quartile-line colours, each its own legend entry (navy/orange/green).
_Q_LINE_COLOURS = {'Q1': 'navy', 'Q2': 'darkorange', 'Q3': 'darkgreen'}

_CJK_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
_cjk_font_ready = False


def _ensure_cjk_font() -> None:
    """Register the Noto CJK font so Japanese labels don't render as tofu
    boxes — matplotlib's default (DejaVu Sans) has no CJK glyphs. Also pins
    the Agg backend for headless rendering. Must run before pyplot is first
    imported, so call this before ``import matplotlib.pyplot``."""
    global _cjk_font_ready
    if _cjk_font_ready:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if Path(_CJK_FONT_PATH).exists():
        font_manager.fontManager.addfont(_CJK_FONT_PATH)
        cjk_font = font_manager.FontProperties(fname=_CJK_FONT_PATH).get_name()
        plt.rcParams['font.family'] = [cjk_font, 'DejaVu Sans']
    _cjk_font_ready = True


def _strata_labels(df: pd.DataFrame) -> dict[int, str]:
    """Japanese strata-group legend labels, bucketed by phone-token quartile."""
    pq1 = df['phone_len'].quantile(0.25)
    pq2 = df['phone_len'].quantile(0.50)
    pq3 = df['phone_len'].quantile(0.75)
    return {
        1: f'G1: 短い（音素数≤{pq1:.0f}）',
        2: f'G2: 中短（音素数≤{pq2:.0f}）',
        3: f'G3: 中長（音素数≤{pq3:.0f}）',
        4: f'G4: 長い（音素数>{pq3:.0f}）',
    }


def _apply_paper_style(ax: 'Axes', title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=_FONT_TITLE)
    ax.set_xlabel(xlabel, fontsize=_FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=_FONT_LABEL)
    ax.tick_params(axis='both', labelsize=_FONT_TICK)


def _legend_groups_then_quartiles(ax: 'Axes') -> None:
    """One combined legend, strata-group patches (G1-G4) before quartile
    lines (Q1-Q3) — matplotlib's default handle order puts ax.bar()
    containers after ax.axvline() lines regardless of draw order, which
    would otherwise put Q1-Q3 first."""
    handles, labels = ax.get_legend_handles_labels()
    pairs = sorted(zip(labels, handles), key=lambda hl: hl[0].startswith('Q'))
    labels, handles = zip(*pairs)
    ax.legend(handles, labels, fontsize=_FONT_LEGEND, loc='upper right')


def _mark_quartiles(ax: 'Axes', q1: float, q2: float, q3: float, fmt: str) -> None:
    """Draw dashed quartile lines, each its own colour + legend entry
    (navy/orange/green), combined into the same legend as the strata-group
    patches — matching img/stratified_distribution_2_plot.png's style."""
    for name, q in (('Q1', q1), ('Q2', q2), ('Q3', q3)):
        ax.axvline(q, color=_Q_LINE_COLOURS[name], ls='--', lw=1.2,
                   label=f'{name}={q:{fmt}}')


def plot_phone_length(df: pd.DataFrame, save_path: Path) -> None:
    """発話あたりの音素トークン数の分布 (strata-coloured histogram)."""
    _ensure_cjk_font()
    import matplotlib.pyplot as plt

    labels = _strata_labels(df)
    pq1, pq2, pq3 = (df['phone_len'].quantile(q) for q in (0.25, 0.50, 0.75))

    fig, ax = plt.subplots(figsize=(_FIG_W_IN, _FIG_H_IN))
    bins = np.arange(10, 215, 5)
    for g in [1, 2, 3, 4]:
        ax.hist(
            df[df['strata_group'] == g]['phone_len'].values,
            bins=bins, color=_COLOURS[g], alpha=0.85,
            label=labels[g], edgecolor='white', linewidth=0.3,
        )
    _mark_quartiles(ax, pq1, pq2, pq3, '.0f')
    _apply_paper_style(ax, '発話あたりの音素トークン数の分布', '音素トークン数', '発話数')
    _legend_groups_then_quartiles(ax)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot saved → {save_path}')


def plot_audio_duration(df: pd.DataFrame, save_path: Path) -> None:
    """発話あたりの音声長の分布 (strata-coloured, stacked histogram)."""
    _ensure_cjk_font()
    import matplotlib.pyplot as plt

    labels = _strata_labels(df)
    dq1, dq2, dq3 = (df['audio_dur_s'].quantile(q) for q in (0.25, 0.50, 0.75))

    fig, ax = plt.subplots(figsize=(_FIG_W_IN, _FIG_H_IN))
    bins    = np.arange(1.0, 18.0, 0.5)
    ctrs    = (bins[:-1] + bins[1:]) / 2
    bottoms = np.zeros(len(ctrs))
    for g in [1, 2, 3, 4]:
        counts, _ = np.histogram(df[df['strata_group'] == g]['audio_dur_s'].values, bins=bins)
        ax.bar(ctrs, counts, width=0.5, bottom=bottoms, color=_COLOURS[g], alpha=0.85,
               label=labels[g], edgecolor='white', linewidth=0.3)
        bottoms += counts
    _mark_quartiles(ax, dq1, dq2, dq3, '.2f')
    _apply_paper_style(ax, '発話あたりの音声長の分布', '音声長（s）', '発話数')
    _legend_groups_then_quartiles(ax)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot saved → {save_path}')


def plot_devoicing_count(df: pd.DataFrame, save_path: Path) -> None:
    """発話あたりの無声化トークン数の分布 (strata-coloured, stacked histogram)."""
    _ensure_cjk_font()
    import matplotlib.pyplot as plt

    labels = _strata_labels(df)
    dv = df['devo_count']
    dvq1, dvq2, dvq3 = (dv.quantile(q) for q in (0.25, 0.50, 0.75))
    zero_pct = (dv == 0).mean() * 100

    fig, ax = plt.subplots(figsize=(_FIG_W_IN, _FIG_H_IN))
    dmax    = int(dv.max())
    bins    = np.arange(-0.5, dmax + 1.5, 1)
    ctrs    = np.arange(0, dmax + 1)
    bottoms = np.zeros(len(ctrs))
    for g in [1, 2, 3, 4]:
        counts, _ = np.histogram(df[df['strata_group'] == g]['devo_count'].values, bins=bins)
        ax.bar(ctrs, counts, width=0.8, bottom=bottoms, color=_COLOURS[g], alpha=0.85,
               label=labels[g], edgecolor='white', linewidth=0.3)
        bottoms += counts
    _mark_quartiles(ax, dvq1, dvq2, dvq3, '.1f')
    _apply_paper_style(
        ax,
        f'発話あたりの無声化トークン数の分布\n（無声化なし：発話の{zero_pct:.1f}%）',
        '無声化トークン数', '発話数',
    )
    _legend_groups_then_quartiles(ax)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot saved → {save_path}')


def plot_distributions(df: pd.DataFrame, save_dir: Path | None = None) -> None:
    """Three standalone charts (phone length / audio duration / devoiced token
    count per utterance, each strata-coloured), written as separate PNGs to
    ``save_dir`` (default: img/) rather than one multi-panel figure."""
    save_dir = Path(save_dir) if save_dir is not None else DEFAULT_IMG_DIR
    plot_phone_length(df, save_dir / 'stratified_distribution_phone_length.png')
    plot_audio_duration(df, save_dir / 'stratified_distribution_audio_duration.png')
    plot_devoicing_count(df, save_dir / 'stratified_distribution_devoicing_count.png')


# ── Public bridge: imported by train_p.py and train.py ───────────────────────
def load_splits(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    wav_dir:       Path | str = WAV_DIR,
) -> tuple[list[str], list[str], list[str]]:
    """Load pre-generated stratified manifest and return split ID lists.

    Returns:
        (train_ids, val_ids, test_ids) as lists of utterance ID strings.

    Raises:
        FileNotFoundError: manifest CSV not found.
            Fix: run  python scripts/stratified_sampling.py
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Stratified manifest not found: {manifest_path}\n"
            f"Generate it first:\n"
            f"    python scripts/stratified_sampling.py"
        )
    df      = pd.read_csv(manifest_path)
    wav_dir = Path(wav_dir)

    exists  = df['utt_id'].apply(lambda k: (wav_dir / f'{k}.wav').exists())
    dropped = int((~exists).sum())
    if dropped:
        print(f'[stratified_sampling] Warning: {dropped} rows skipped '
              f'(WAV not found); using {int(exists.sum())} utterances.')
    df = df[exists]

    return (
        df.loc[df['split'] == 'train', 'utt_id'].tolist(),
        df.loc[df['split'] == 'val',   'utt_id'].tolist(),
        df.loc[df['split'] == 'test',  'utt_id'].tolist(),
    )


# ── Standalone entry-point ────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Build stratified train/val/test manifest for JSUT basic5000.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir', default=str(DATA_DIR),
                        help='Dir with transcript_phone3_rev.txt')
    parser.add_argument('--wav-dir',  default=str(WAV_DIR),
                        help='Dir containing BASIC5000_XXXX.wav files')
    parser.add_argument('--out',      default=str(DEFAULT_MANIFEST),
                        help='Output path for stratified_manifest.csv')
    parser.add_argument('--seed',     type=int, default=SEED)
    parser.add_argument('--plot',     action='store_true',
                        help='Save phone-length / audio-duration / devoicing-count '
                             'distribution charts (3 separate PNGs) to img/')
    parser.add_argument('--plot-dir', default=None,
                        help='Directory for the 3 PNGs (default: img/)')
    args = parser.parse_args()

    df = build_manifest(
        data_dir=Path(args.data_dir),
        wav_dir=Path(args.wav_dir),
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Manifest saved → {out_path}')

    print_report(df)

    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else DEFAULT_IMG_DIR
        plot_distributions(df, save_dir=plot_dir)
