"""
scripts/devoicing_eval.py
------------------------------
Reference-vs-prediction confusion-matrix evaluation for devoiced vowels.

Treats devoicing as a per-vowel binary detection problem: every reference
vowel of a given family (i-family or u-family) is a ground-truth sample whose
label is *devoiced* (positive) or *voiced* (negative). The aligned prediction
is scored as devoiced or not, giving the four confusion-matrix cells:

    TP  reference devoiced  &  predicted devoiced
    FN  reference devoiced  &  predicted NOT devoiced  (missed devoicing)
    FP  reference voiced     &  predicted devoiced      (spurious devoicing)
    TN  reference voiced     &  predicted NOT devoiced

Reference and prediction token sequences are aligned with Needleman–Wunsch
(unit substitution / indel cost) so the comparison is position-aware and
robust to insertions/deletions. Counts are grounded on *reference* vowels;
hallucinated devoiced vowels with no reference vowel in their aligned column
are ASR insertion errors (captured by PER) and are not counted here.

This module is tokenizer-agnostic: callers pass ``vowel_groups`` mapping each
vowel label to its set of voiced and devoiced token-ids, so the same logic
serves the phone tokenizer (i̥ / ɯ̥) and the syllable tokenizer (<X devo>).
"""
from __future__ import annotations

from pathlib import Path

# ── Sequence alignment ────────────────────────────────────────────────────────
GAP = None  # sentinel for an alignment gap (insertion/deletion)


def nw_align(pred: list[int], ref: list[int]) -> list[tuple[int | None, int | None]]:
    """Needleman–Wunsch global alignment of two token-id sequences.

    Match cost 0, substitution/indel cost 1. Returns a list of aligned
    columns ``(pred_tok, ref_tok)`` where either side may be ``GAP``.
    """
    m, n = len(pred), len(ref)
    # dp[i][j] = min edit distance between pred[:i] and ref[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        pi = pred[i - 1]
        for j in range(1, n + 1):
            sub = dp[i - 1][j - 1] + (0 if pi == ref[j - 1] else 1)
            dp[i][j] = min(sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    # Backtrace
    cols: list[tuple[int | None, int | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if pred[i - 1] == ref[j - 1] else 1):
            cols.append((pred[i - 1], ref[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            cols.append((pred[i - 1], GAP))  # insertion in prediction
            i -= 1
        else:
            cols.append((GAP, ref[j - 1]))   # deletion (ref token missed)
            j -= 1
    cols.reverse()
    return cols


# ── Confusion computation ─────────────────────────────────────────────────────
def _strip_specials(ids: list[int], special: set[int]) -> list[int]:
    return [t for t in ids if t not in special]


def compute_vowel_confusion(
    pred_ids_list: list[list[int]],
    ref_ids_list: list[list[int]],
    special_ids: set[int],
    vowel_groups: dict[str, dict[str, set[int]]],
) -> dict[str, dict[str, int]]:
    """Accumulate TP/FP/TN/FN per vowel family over a corpus.

    Args:
        pred_ids_list / ref_ids_list: parallel lists of token-id sequences
            (may contain PAD/SOS/EOS — they are stripped via ``special_ids``).
        special_ids: PAD/SOS/EOS ids to ignore during alignment.
        vowel_groups: ``{label: {"voiced": {ids}, "devoiced": {ids}}}``.

    Returns:
        ``{label: {"TP":.., "FP":.., "TN":.., "FN":..}}``.
    """
    stats = {lab: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for lab in vowel_groups}

    for pred_ids, ref_ids in zip(pred_ids_list, ref_ids_list):
        pred = _strip_specials(pred_ids, special_ids)
        ref = _strip_specials(ref_ids, special_ids)
        for p_tok, r_tok in nw_align(pred, ref):
            if r_tok is GAP:
                continue  # no reference vowel to ground a sample on
            for lab, grp in vowel_groups.items():
                voiced, devoiced = grp["voiced"], grp["devoiced"]
                ref_devoiced = r_tok in devoiced
                ref_voiced = r_tok in voiced
                if not (ref_devoiced or ref_voiced):
                    continue
                pred_devoiced = p_tok in devoiced
                if ref_devoiced:
                    stats[lab]["TP" if pred_devoiced else "FN"] += 1
                else:  # ref voiced
                    stats[lab]["FP" if pred_devoiced else "TN"] += 1
    return stats


def _ccda_correct_by_family(
    pred: list[int], ref: list[int], devoiced: set[int]
) -> int:
    """CCDA match count for one utterance / one vowel family.

    Counts reference devoiced vowels whose (C1, V, C2) triplet is also present
    in the prediction (EOS retained as a possible C2 by the caller).
    """
    pred_trip: dict[tuple, int] = {}
    for i, t in enumerate(pred):
        if t in devoiced:
            key = (pred[i - 1] if i > 0 else None, t,
                   pred[i + 1] if i + 1 < len(pred) else None)
            pred_trip[key] = pred_trip.get(key, 0) + 1
    correct = 0
    for i, t in enumerate(ref):
        if t in devoiced:
            key = (ref[i - 1] if i > 0 else None, t,
                   ref[i + 1] if i + 1 < len(ref) else None)
            if pred_trip.get(key, 0) > 0:
                correct += 1
                pred_trip[key] -= 1
    return correct


def compute_vowel_confusion_ccda(
    pred_ids_list: list[list[int]],
    ref_ids_list: list[list[int]],
    special_ids: set[int],
    vowel_groups: dict[str, dict[str, set[int]]],
    eos_id: int | None = None,
) -> dict[str, dict[str, int]]:
    """Confusion cells where the positive class is scored by CCDA context match.

    A reference devoiced vowel counts as a true positive only when its
    (C1·V·C2) context is reproduced in the prediction (CCDA criterion);
    otherwise it is a false negative. TP and FN therefore both use the CCDA
    definition, so ``TP + FN`` still equals the number of reference devoiced
    vowels — i.e. this stays a valid confusion matrix. FP/TN concern reference
    *voiced* vowels (which CCDA does not address) and are taken unchanged from
    the alignment-based :func:`compute_vowel_confusion`.

    Note: requiring agreement of DEO *and* CCDA would be identical to CCDA
    alone, since every CCDA match implies a DEO match (CCDA ⊆ DEO); the DEO
    term is therefore dropped as redundant.

    Args:
        pred_ids_list / ref_ids_list: parallel token-id sequences (may carry
            PAD/SOS/EOS).
        special_ids: PAD/SOS/EOS ids stripped for the FP/TN base.
        vowel_groups: ``{label: {"voiced": {ids}, "devoiced": {ids}}}``.
        eos_id: EOS id, kept as a valid right context (C2) for CCDA; if None,
            EOS is stripped like the other specials.

    Returns:
        ``{label: {"TP":.., "FP":.., "TN":.., "FN":..}}`` with CCDA-scored TP/FN.
    """
    base = compute_vowel_confusion(pred_ids_list, ref_ids_list, special_ids, vowel_groups)
    # CCDA keeps EOS as a possible C2; everything else is stripped.
    ccda_special = special_ids - {eos_id} if eos_id is not None else special_ids

    tp = {lab: 0 for lab in vowel_groups}
    for pred_ids, ref_ids in zip(pred_ids_list, ref_ids_list):
        pred_cc = _strip_specials(pred_ids, ccda_special)
        ref_cc  = _strip_specials(ref_ids, ccda_special)
        for lab, grp in vowel_groups.items():
            tp[lab] += _ccda_correct_by_family(pred_cc, ref_cc, grp["devoiced"])

    for lab in base:
        # Positives = reference devoiced vowels = base TP + base FN; CCDA splits
        # them into context-matched (TP) and the rest (FN), keeping TP+FN fixed.
        positives = base[lab]["TP"] + base[lab]["FN"]
        base[lab]["TP"] = tp[lab]
        base[lab]["FN"] = positives - tp[lab]
    return base


def derive_metrics(cell: dict[str, int]) -> dict[str, float]:
    """Precision/recall/F1/accuracy/specificity from one TP/FP/TN/FN cell."""
    tp, fp, tn, fn = cell["TP"], cell["FP"], cell["TN"], cell["FN"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {
        "precision": prec * 100,
        "recall": rec * 100,
        "f1": f1 * 100,
        "specificity": spec * 100,
        "accuracy": acc * 100,
        "support": tp + fn,            # reference devoiced count
        "support_voiced": tn + fp,     # reference voiced count
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
# Vowel-family label → devoiced vowel symbol (romaji style) shown in each title.
_DEVOICED_SYMBOL = {"i": "I", "u": "U"}
# Left-to-right order: i first, u second (others appended afterwards).
_LABEL_ORDER = {"i": 0, "u": 1}


def plot_confusion(
    stats: dict[str, dict[str, int]],
    out_path: str | Path,
    title: str = "Devoiced-vowel confusion matrices",
) -> Path:
    """Render one 2×2 confusion matrix per vowel family to a single PNG.

    Rows = reference (Devoiced / Voiced), Cols = prediction (Devoiced / Voiced).
    Matrices are ordered i (left) then u (right); each title names the
    devoiced vowel in romaji style (I / U) rather than the voiced vowel family.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(stats, key=lambda l: _LABEL_ORDER.get(l, len(_LABEL_ORDER)))
    fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 4.6))
    if len(labels) == 1:
        axes = [axes]

    for ax, lab in zip(axes, labels):
        c = stats[lab]
        # matrix[ref][pred]: ref 0=Devoiced 1=Voiced ; pred 0=Devoiced 1=Voiced
        matrix = [[c["TP"], c["FN"]],
                  [c["FP"], c["TN"]]]
        m = derive_metrics(c)
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], ["Pred Devoiced", "Pred Voiced"])
        ax.set_yticks([0, 1], ["Ref Devoiced", "Ref Voiced"])
        cell_labels = [["TP", "FN"], ["FP", "TN"]]
        vmax = max(max(row) for row in matrix) or 1
        for r in range(2):
            for col in range(2):
                val = matrix[r][col]
                ax.text(col, r, f"{cell_labels[r][col]}\n{val}",
                        ha="center", va="center",
                        color="white" if val > vmax * 0.5 else "black",
                        fontsize=12, fontweight="bold")
        sym = _DEVOICED_SYMBOL.get(lab, lab)
        ax.set_title(
            f"{sym}  P={m['precision']:.1f}  R={m['recall']:.1f}  "
            f"F1={m['f1']:.1f}\n(devoiced n={m['support']}, voiced n={m['support_voiced']})",
            fontsize=10,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_confusion_csv(
    stats: dict[str, dict[str, int]],
    out_path: str | Path,
    model_name: str = "",
    split: str = "",
) -> Path:
    """Write per-vowel TP/FP/TN/FN + derived metrics to a CSV."""
    import csv

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model_name", "split", "vowel", "TP", "FP", "TN", "FN",
              "precision", "recall", "f1", "specificity", "accuracy",
              "support_devoiced", "support_voiced"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for lab, c in stats.items():
            m = derive_metrics(c)
            w.writerow([
                model_name, split, lab, c["TP"], c["FP"], c["TN"], c["FN"],
                f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                f"{m['specificity']:.4f}", f"{m['accuracy']:.4f}",
                m["support"], m["support_voiced"],
            ])
    return out_path


def print_confusion(stats: dict[str, dict[str, int]]) -> None:
    """Pretty-print the confusion cells + metrics to stdout."""
    for lab, c in stats.items():
        m = derive_metrics(c)
        print(f"\n── Devoiced vowel /{lab}/ ──")
        print(f"  TP={c['TP']}  FP={c['FP']}  TN={c['TN']}  FN={c['FN']}")
        print(f"  precision={m['precision']:.2f}%  recall={m['recall']:.2f}%  "
              f"f1={m['f1']:.2f}%  specificity={m['specificity']:.2f}%  "
              f"accuracy={m['accuracy']:.2f}%")
        print(f"  reference devoiced n={m['support']}  voiced n={m['support_voiced']}")
