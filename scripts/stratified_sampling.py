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

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

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

    Columns: utt_id, wav_path, phone_len, romaji_len, devo_count,
             audio_dur_s, strata_group, split
    """
    # 1. Locate phone transcript (primary stratification key)
    phone_path = data_dir / 'transcript_phone.txt'
    if not phone_path.exists():
        phone_path = data_dir / 'transcript_phoneme.txt'
    if not phone_path.exists():
        raise FileNotFoundError(
            f"Phone transcript not found in {data_dir}\n"
            "Expected: transcript_phone.txt or transcript_phoneme.txt"
        )
    phone_raw  = _parse_transcript(phone_path)
    romaji_raw = _parse_transcript(data_dir / 'transcript_romaji.txt')

    # 2. Filter to utterances that have a matching WAV file
    all_ids = sorted(k for k in phone_raw if (wav_dir / f'{k}.wav').exists())
    if not all_ids:
        raise RuntimeError(f"No utterances with WAV files found in {wav_dir}")
    print(f'Total utterances with WAV: {len(all_ids)}')

    # 3. Compute transcript token lengths (space-separated tokens)
    phone_lens  = np.array([len(phone_raw[k].split())          for k in all_ids])
    romaji_lens = np.array([len(romaji_raw.get(k, '').split()) for k in all_ids])

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
        'romaji_len':   romaji_lens,
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
def plot_distributions(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """Three-panel histogram: phone length, audio duration, devoiced token count.

    All bars colour-coded by strata group.  Saved to img/ by default.
    """
    import matplotlib.pyplot as plt

    COLOURS = {1: '#4C72B0', 2: '#DD8452', 3: '#55A868', 4: '#C44E52'}
    pq1 = df['phone_len'].quantile(0.25)
    pq2 = df['phone_len'].quantile(0.50)
    pq3 = df['phone_len'].quantile(0.75)
    LABELS = {
        1: f'G1 short  (<=Q1={pq1:.0f})',
        2: f'G2 med-sh (<=Q2={pq2:.0f})',
        3: f'G3 med-lg (<=Q3={pq3:.0f})',
        4: f'G4 long   (>Q3={pq3:.0f})',
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))

    # ── Panel 1: phone token length ──
    bins1 = np.arange(10, 215, 5)
    for g in [1, 2, 3, 4]:
        ax1.hist(
            df[df['strata_group'] == g]['phone_len'].values,
            bins=bins1, color=COLOURS[g], alpha=0.85,
            label=LABELS[g], edgecolor='white', linewidth=0.3,
        )
    ax1.axvline(pq1, color='navy',       ls='--', lw=1.5, label=f'Q1={pq1:.0f}')
    ax1.axvline(pq2, color='darkorange', ls='--', lw=1.5, label=f'Q2={pq2:.0f}')
    ax1.axvline(pq3, color='darkgreen',  ls='--', lw=1.5, label=f'Q3={pq3:.0f}')
    ax1.set_title('Phone Token Length per Utterance')
    ax1.set_xlabel('Number of phone tokens')
    ax1.set_ylabel('Count (utterances)')
    ax1.legend(fontsize=8)

    # ── Panel 2: audio duration (stacked by strata) ──
    bins2    = np.arange(1.0, 18.0, 0.5)
    bin_ctrs = (bins2[:-1] + bins2[1:]) / 2
    bottoms  = np.zeros(len(bin_ctrs))
    for g in [1, 2, 3, 4]:
        counts, _ = np.histogram(
            df[df['strata_group'] == g]['audio_dur_s'].values, bins=bins2
        )
        ax2.bar(bin_ctrs, counts, width=0.5, bottom=bottoms,
                color=COLOURS[g], alpha=0.85, label=LABELS[g],
                edgecolor='white', linewidth=0.3)
        bottoms += counts
    dq1 = df['audio_dur_s'].quantile(0.25)
    dq2 = df['audio_dur_s'].quantile(0.50)
    dq3 = df['audio_dur_s'].quantile(0.75)
    ax2.axvline(dq1, color='navy',       ls='--', lw=1.5, label=f'Q1={dq1:.2f}s')
    ax2.axvline(dq2, color='darkorange', ls='--', lw=1.5, label=f'Q2={dq2:.2f}s')
    ax2.axvline(dq3, color='darkgreen',  ls='--', lw=1.5, label=f'Q3={dq3:.2f}s')
    ax2.set_title('Audio Duration per Utterance\n(bars coloured by phone-length strata)')
    ax2.set_xlabel('Duration (seconds)')
    ax2.set_ylabel('Count (utterances)')
    ax2.legend(fontsize=8)

    # ── Panel 3: devoiced token count (stacked by strata) ──
    dmax  = int(df['devo_count'].max())
    bins3 = np.arange(-0.5, dmax + 1.5, 1)
    bin_ctrs3 = np.arange(0, dmax + 1)
    bottoms3  = np.zeros(len(bin_ctrs3))
    for g in [1, 2, 3, 4]:
        counts, _ = np.histogram(
            df[df['strata_group'] == g]['devo_count'].values, bins=bins3
        )
        ax3.bar(bin_ctrs3, counts, width=0.8, bottom=bottoms3,
                color=COLOURS[g], alpha=0.85, label=LABELS[g],
                edgecolor='white', linewidth=0.3)
        bottoms3 += counts
    dv   = df['devo_count']
    dvq1 = dv.quantile(0.25)
    dvq2 = dv.quantile(0.50)
    dvq3 = dv.quantile(0.75)
    ax3.axvline(dvq1, color='navy',       ls='--', lw=1.5, label=f'Q1={dvq1:.1f}')
    ax3.axvline(dvq2, color='darkorange', ls='--', lw=1.5, label=f'Q2={dvq2:.1f}')
    ax3.axvline(dvq3, color='darkgreen',  ls='--', lw=1.5, label=f'Q3={dvq3:.1f}')
    zero_pct = (dv == 0).mean() * 100
    ax3.set_title(
        f'Devoiced Token Count per Utterance\n'
        f'(zero-devo: {zero_pct:.1f}% of utterances)'
    )
    ax3.set_xlabel('Number of devoiced tokens')
    ax3.set_ylabel('Count (utterances)')
    ax3.legend(fontsize=8)

    r_ph_dur = float(np.corrcoef(df['phone_len'].values, df['audio_dur_s'].values)[0, 1])
    r_ph_dv  = float(np.corrcoef(df['phone_len'].values, df['devo_count'].values)[0, 1])
    fig.suptitle(
        f'JSUT basic5000  —  N={len(df):,}\n'
        f'Phone: mean={df["phone_len"].mean():.1f} tokens  '
        f'|  Audio: mean={df["audio_dur_s"].mean():.2f}s  '
        f'|  Devo: mean={df["devo_count"].mean():.2f} tokens/utt  '
        f'|  r(len,dur)={r_ph_dur:.3f}  r(len,devo)={r_ph_dv:.3f}',
        fontsize=10,
    )
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Plot saved → {save_path}')
    plt.show()


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
                        help='Dir with transcript_phone.txt & transcript_romaji.txt')
    parser.add_argument('--wav-dir',  default=str(WAV_DIR),
                        help='Dir containing BASIC5000_XXXX.wav files')
    parser.add_argument('--out',      default=str(DEFAULT_MANIFEST),
                        help='Output path for stratified_manifest.csv')
    parser.add_argument('--seed',     type=int, default=SEED)
    parser.add_argument('--plot',     action='store_true',
                        help='Save three-panel distribution plot to img/')
    parser.add_argument('--plot-out', default=None,
                        help='PNG save path (default: img/stratified_distribution.png)')
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
        plot_out = (
            Path(args.plot_out)
            if args.plot_out
            else DEFAULT_IMG_DIR / 'stratified_distribution.png'
        )
        plot_distributions(df, save_path=plot_out)
