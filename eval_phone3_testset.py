"""
eval_phone3_testset.py
─────────────────────────────
Evaluate phone3 (train_phone3_*) checkpoints on the stratified TEST split with
the Option-B devoicing suite and emit a Markdown results table.

Reuses inference_r3's decode machinery (load_model / greedy_decode_batch /
load_mel / test-id + transcript loaders) and the train_r3 metric definitions
(PER, KER, slot-level Precision/Recall/F1, CCDA, DEO).

Usage:
    python eval_phone3_testset.py                 # all train_phone3_*.pt
    python eval_phone3_testset.py --checkpoint <path.pt>
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))

from inference_r3 import (
    MAX_TOKENS, load_mel, greedy_decode_batch, load_model,
    _load_test_ids, _load_transcripts, _is_lfs_pointer,
)
from scripts.phone_tokenizer import JapaneseRomajiRevTokenizer3
from scripts.kana import compute_kana_error_rate
from scripts import devoicing_eval as deval

DATA_DIR = ROOT / "dataset" / "jsut_ver1.1" / "basic5000"
WAV_DIR  = DATA_DIR / "wav"
VOCAB    = ROOT / "doc" / "tokenizer_romaji_rev_vocab.json"
MODELS   = ROOT / "models"
IMG_DIR  = ROOT / "img"
CSV_DIR  = ROOT / "doc"


# ── Metrics (mirror train_r3.py, parameterised by the tokenizer) ──────────────
def _edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        new = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            new[j] = dp[j-1] if a[i-1] == b[j-1] else 1 + min(dp[j], new[j-1], dp[j-1])
        dp = new
    return dp[len(b)]


def compute_per(tok, pred_ids, ref_ids):
    """Phone Error Rate (%): edit distance over phone3 tokens (pau/special
    stripped).  Historically reported as "CER" in this repo; the true
    character-level metric is ``compute_kana_error_rate`` (KER)."""
    special = {tok.PAD_ID, tok.SOS_ID, tok.EOS_ID, tok.pau_id}
    dist = ref = 0
    for p, r in zip(pred_ids, ref_ids):
        p = [t for t in p if t not in special]
        r = [t for t in r if t not in special]
        dist += _edit_distance(p, r); ref += len(r)
    return dist / max(1, ref) * 100


def _align(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            c = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+c)
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if ref[i-1] == hyp[j-1] else 1):
            pairs.append((i-1, j-1)); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            pairs.append((i-1, None)); i -= 1
        else:
            pairs.append((None, j-1)); j -= 1
    pairs.reverse()
    return pairs


def compute_detection(tok, pred_ids, ref_ids):
    special  = {tok.PAD_ID, tok.SOS_ID, tok.EOS_ID, tok.pau_id}
    id2t     = tok.id2token
    devo     = {"I", "U"}
    fold = lambda t: "i" if id2t.get(t) == "I" else "u" if id2t.get(t) == "U" else id2t.get(t, "")
    tp = ref_dev = pred_dev = 0
    for p, r in zip(pred_ids, ref_ids):
        rr = [t for t in r if t not in special]
        hh = [t for t in p if t not in special]
        ref_dev  += sum(1 for t in rr if id2t.get(t) in devo)
        pred_dev += sum(1 for t in hh if id2t.get(t) in devo)
        rf, hf = [fold(t) for t in rr], [fold(t) for t in hh]
        for ri, hi in _align(rf, hf):
            if ri is None or hi is None:
                continue
            if rf[ri] in ("i", "u") and rf[ri] == hf[hi] and id2t.get(rr[ri]) in devo and id2t.get(hh[hi]) in devo:
                tp += 1
    fp, fn = pred_dev - tp, ref_dev - tp
    precision = tp / pred_dev * 100 if pred_dev else float("nan")
    recall    = tp / ref_dev  * 100 if ref_dev  else float("nan")
    f1 = (2*precision*recall/(precision+recall)
          if not (math.isnan(precision) or math.isnan(recall)) and (precision+recall) else
          (0.0 if not (math.isnan(precision) or math.isnan(recall)) else float("nan")))
    return f1, precision, recall, tp, fp, fn


def compute_ccda(tok, pred_ids, ref_ids):
    skip = {tok.PAD_ID, tok.SOS_ID}
    devo = set(tok.devo_token_ids())
    correct = total = 0
    for p, r in zip(pred_ids, ref_ids):
        pc = [t for t in p if t not in skip]
        rc = [t for t in r if t not in skip]
        trip = {}
        for i, t in enumerate(pc):
            if t in devo:
                key = (pc[i-1] if i > 0 else None, t, pc[i+1] if i+1 < len(pc) else None)
                trip[key] = trip.get(key, 0) + 1
        for i, t in enumerate(rc):
            if t in devo:
                total += 1
                key = (rc[i-1] if i > 0 else None, t, rc[i+1] if i+1 < len(rc) else None)
                if trip.get(key, 0) > 0:
                    correct += 1; trip[key] -= 1
    return (correct / total * 100 if total else float("nan")), correct, total


def compute_deo(tok, pred_ids, ref_ids):
    special = {tok.PAD_ID, tok.SOS_ID, tok.EOS_ID}
    devo = set(tok.devo_token_ids())
    correct = total = 0
    for p, r in zip(pred_ids, ref_ids):
        pc = [t for t in p if t not in special]
        rc = [t for t in r if t not in special]
        total += sum(1 for t in rc if t in devo)
        for d in devo:
            correct += min(rc.count(d), pc.count(d))
    return (correct / total * 100 if total else float("nan")), correct, total


# ── Decode one split ──────────────────────────────────────────────────────────
@torch.no_grad()
def decode_split(model, tok, utt_ids, batch_size=16):
    cmvn = getattr(model, "cmvn", False)
    all_pred, all_ref = [], []
    transcripts = _load_transcripts(DATA_DIR)
    for s in range(0, len(utt_ids), batch_size):
        chunk = utt_ids[s:s + batch_size]
        mel = torch.stack([load_mel(WAV_DIR / f"{u}.wav", cmvn=cmvn) for u in chunk])
        all_pred.extend(greedy_decode_batch(model, mel, tok, max_len=MAX_TOKENS))
        all_ref.extend(tok.encode(transcripts[u], add_sos=True, add_eos=True) for u in chunk)
        print(f"    {min(s+batch_size, len(utt_ids))}/{len(utt_ids)}", end="\r", flush=True)
    print()
    return all_pred, all_ref


def _cfg_label(name: str) -> str:
    bs = re.search(r"_bs(\d+)", name)
    lr = re.search(r"_lr([0-9.e-]+?)_", name)
    lr_v = float(lr.group(1)) if lr else 0.0
    lr_s = f"{lr_v:.0e}".replace("e-0", "e-")              # 0.001 → 1e-3
    return f"Batch_Size: {bs.group(1) if bs else '?'}; Learning_Rate: {lr_s};"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Single checkpoint; default = all train_phone3_*.pt")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "doc" / "phone3_detection_table.md")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-confusion", action="store_true",
                    help="Skip the per-vowel devoiced/voiced confusion matrices")
    args = ap.parse_args()

    tok = JapaneseRomajiRevTokenizer3.from_vocab(VOCAB)
    test_ids = _load_test_ids(DATA_DIR, WAV_DIR)
    transcripts = _load_transcripts(DATA_DIR)
    test_ids = [u for u in test_ids if u in transcripts and (WAV_DIR / f"{u}.wav").exists()]
    print(f"Vocab {tok.vocab_size} | test utts {len(test_ids)}")

    ckpts = ([args.checkpoint] if args.checkpoint else
             sorted(MODELS.glob("train_phone3_*.pt"),
                    key=lambda p: (int(re.search(r'_bs(\d+)', p.name).group(1)),
                                   -float(re.search(r'_lr([0-9.e-]+?)_', p.name).group(1)))))
    lfs_stubs = [c for c in ckpts if _is_lfs_pointer(c)]
    if lfs_stubs:
        print(f"Skipping {len(lfs_stubs)} git-lfs pointer stub(s): "
              + ", ".join(c.name for c in lfs_stubs))
    ckpts = [c for c in ckpts if not _is_lfs_pointer(c)]
    if not ckpts:
        raise SystemExit(
            "No usable checkpoints found (only git-lfs pointer stubs). "
            "Run `git lfs pull`, or train with train_r3.py.")
    groups  = tok.vowel_groups()
    special = {tok.PAD_ID, tok.SOS_ID, tok.EOS_ID}

    rows, cm_rows = [], []
    for ck in ckpts:
        print(f"\n=== {ck.name} ===")
        model = load_model(ck, tok)
        pred, ref = decode_split(model, tok, test_ids, args.batch_size)
        per = compute_per(tok, pred, ref)
        ker = compute_kana_error_rate(pred, ref, tok)
        f1, P, R, tp, fp, fn = compute_detection(tok, pred, ref)
        ccda, _, devo_total = compute_ccda(tok, pred, ref)
        deo, _, _ = compute_deo(tok, pred, ref)
        print(f"  PER={per:.2f}  KER={ker:.2f}  P={P:.2f} R={R:.2f} F1={f1:.2f}  "
              f"CCDA={ccda:.2f} DEO={deo:.2f}  (TP={tp} FP={fp} FN={fn}/{devo_total})")
        rows.append((_cfg_label(ck.name), per, ker, P, R, f1, ccda, deo))

        # ── two-class (devoiced vs voiced) detection confusion matrix, per vowel i/u ──
        if not args.no_confusion:
            stats = deval.compute_vowel_confusion(pred, ref, special, groups)
            deval.print_confusion(stats)
            bs = re.search(r"_bs(\d+)", ck.name).group(1)
            lr_s = _cfg_label(ck.name).split("Learning_Rate: ")[1].rstrip(";")
            png = deval.plot_confusion(
                stats, IMG_DIR / f"cm_vw_phone3_bs{bs}_lr{lr_s}.png",
                title=f"Devoiced-vowel detection (i / u) — {ck.stem}")
            print(f"  confusion → {png}")
            for lab, c in stats.items():
                mm = deval.derive_metrics(c)
                cm_rows.append([_cfg_label(ck.name), lab, c["TP"], c["FP"], c["TN"], c["FN"],
                                f"{mm['precision']:.2f}", f"{mm['recall']:.2f}", f"{mm['f1']:.2f}",
                                f"{mm['specificity']:.2f}", f"{mm['accuracy']:.2f}",
                                mm["support"], mm["support_voiced"]])

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if cm_rows:
        cm_csv = CSV_DIR / "cm_vw_phone3_detection.csv"
        with cm_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model_config", "vowel", "TP", "FP", "TN", "FN",
                        "precision", "recall", "f1", "specificity", "accuracy",
                        "support_devoiced", "support_voiced"])
            w.writerows(cm_rows)
        print(f"\nConfusion CSV → {cm_csv}")

    # ── Markdown table (header separator "| === |" per request) ──
    head = ("| model config | PER ↓ (%) | KER ↓ (%) | Precision ↑ (%) | Recall ↑ (%) | "
            "F1 ↑ (%) | CCDA ↑ (%) | DEO ↑ (%) |")
    lines = ["# Phone-level3 Romaji-rev — Devoicing Detection (stratified, test split)",
             "", head, "| === |"]
    for cfg, per, ker, P, R, f1, ccda, deo in rows:
        lines.append(f"| {cfg} | {per:.2f} | {ker:.2f} | {P:.2f} | {R:.2f} | {f1:.2f} | {ccda:.2f} | {deo:.2f} |")
    table = "\n".join(lines) + "\n"
    args.out.write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"Table written → {args.out}")


if __name__ == "__main__":
    main()
