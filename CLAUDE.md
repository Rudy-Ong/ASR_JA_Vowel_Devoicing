# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Japanese ASR that outputs **explicit vowel-devoicing tokens**. A Conformer-encoder /
Transformer-decoder seq2seq model trained on JSUT basic5000, using `phone_level3`-style
transcripts where devoiced high vowels are upper-cased (e.g. `s U k i`). Recognition and
devoicing detection happen in a single pass. Best config (bs8 lr1e-3, dw5): PER 2.61%,
devoicing F1 93.26% — see `results.md`.

## Setup

Requires Python ≥ 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync && source .venv/bin/activate
```

or plain pip: `pip install -r requirements.txt`.

JSUT basic5000 audio (5000 WAVs) is not redistributed — download from
https://sites.google.com/site/shinnosuketakamichi/publication/jsut and place in
`dataset/jsut_ver1.1/basic5000/wav/`. Most other pipeline stages don't need the wavs
(preprocessing and manifest already committed); training and inference/eval do.

## Key commands

```bash
# Training (BS/LR from SWEEP_BATCH_SIZE / SWEEP_LR env vars, else defaults in train_r3.py)
python train_r3.py

# Inference — single/multiple files (defaults to newest models/train_phone3_*.pt)
python inference_r3.py path/to/audio.wav
python inference_r3.py path/to/audio.wav --checkpoint models/train_phone3_<stamp>_bs8_lr0.001_ks19_do0.1_stratified.pt

# Evaluate on JSUT test split + devoicing confusion matrix (needs wavs)
python inference_r3.py --eval-scope test

# Batch evaluation over all checkpoints (PER + KER + devoicing P/R/F1) -> img/ + doc/
python eval_phone3_testset.py

# Gradio demo (record mic or pick JSUT wav; launches with share=True, public link)
python gradio_demo.py

# Rebuild phone3 transcript + tokenizer vocab (overwrites 5 manual corrections, see below)
python scripts/make_transcript_phone3_rev.py

# Rebuild stratified split manifest (needs wavs + legacy r1 transcript files, see below)
python scripts/stratified_sampling.py
```

There is no test suite and no configured linter/formatter in this repo — don't assume
`pytest`/`ruff`/`black`/`mypy` conventions exist here.

Training logs to Weights & Biases: `wandb login` first, or set `WANDB_MODE=offline`. A
`.env` file at the repo root (gitignored) is auto-loaded for `WANDB_API_KEY`.

### Preprocessing caveats

- `transcript_phone3_rev.txt` carries 5 manual corrections (e.g. `BASIC5000_0047`) on top
  of the rule-based output of `make_transcript_phone3_rev.py`. Re-running it overwrites
  them — use `git checkout dataset/.../transcript_phone3_rev.txt` to restore, or pass a
  different `--dst`.
- The stratified split manifest **cannot be rebuilt exactly**: `stratified_sampling.py`
  needs `transcript_phone.txt` / `transcript_romaji.txt` from the older r1 pipeline, which
  aren't committed. Treat the committed `stratified_manifest.csv` as authoritative for
  reproducing published numbers.

## Architecture

End-to-end seq2seq: `ConformerASR` is defined **identically in both `train_r3.py` and
`scripts/inference_r.py`** — if you modify the model architecture in one, you must keep
the other in sync.

- **Encoder**: `ConvSubsampling` (Conv2d ×4 temporal downsampling) → 4 × `ConformerBlock`
  (d_model=256, 4 heads, FF=1024, depthwise conv kernel=19, SiLU activations)
- **Decoder**: 2 × `TransformerDecoderLayer` (Pre-LN, causal mask, batch_first)
- **Audio**: 16 kHz → 80-mel log-spectrogram, n_fft=1024, hop=256, `MAX_FRAMES=1024`,
  `MAX_TOKENS=256` (fixed-shape padding per batch so cudnn autotuning stays effective)

Checkpoints: `models/train_phone3_<YYYYMMDD_HHMM>_bs{BS}_lr{LR}_ks19_do0.1_stratified.pt`
(sweep runs via `scripts/sweep_runner.py` insert `_dw{DEVO_WEIGHT}` before `_stratified`),
tracked with **git-lfs**. Each training run also appends a row to `doc/results_phone3.csv`.
The checkpoint(s) published for public use are manually renamed to a stable,
date-free name (`model_bs{BS}_lr{LR}_dw{DW}.pt`) after training — checkpoint
globs across the repo (`gradio_demo.py`, `inference_r3.py`, `eval_phone3_testset.py`)
match `*.pt` so both naming schemes work.

### phone_level3 / devoicing convention

The model outputs **phone_level3** romaji tokens, not raw text. Devoiced high vowels are
**upper-case** `I`/`U`; their voiced counterparts are lower-case `i`/`u`. These are the
only tokens with any upper-case character (excluding special tokens).

- Tokenizer: `JapaneseRomajiRevTokenizer3` in `scripts/phone_tokenizer.py` (hierarchy:
  `JapanesePhoneTokenizer` → `JapaneseRomajiRevTokenizer3`, used by train/inference).
- Vocab file: `doc/tokenizer_romaji_rev_vocab.json` (auto-generated on training start).
- Transcript format, one line per utterance: `UNAME:space-separated tokens`, e.g.
  `BASIC5000_0001:k a r e w a m i z U m i n o h I t o d e s U`.
- Special tokens: `<pad>=0 <sos>=1 <eos>=2 <unk>=3 <sp>=4`.
- `pau` token represents an ideographic comma `、` — excluded from PER and KER scoring,
  rendered as `、` for display only.

### Training details

- `DEVO_WEIGHT = 5.0`: cross-entropy loss upweights devoiced `I`/`U` tokens 5× to
  counteract their low frequency.
- LR schedule: linear warmup (10% of steps) → cosine annealing.
- Stratified split: 80/10/10 by phone token length quartile, manifest at
  `dataset/jsut_ver1.1/basic5000/stratified_manifest.csv` (pre-committed). Set
  `USE_STRATIFIED_SPLIT = False` in `train_r3.py` for a random split instead.
- Sweep hyperparameters via `SWEEP_BATCH_SIZE` / `SWEEP_LR` env vars.

### Devoicing evaluation

`scripts/devoicing_eval.py` scores devoicing as a per-vowel-family binary detection
problem using Needleman–Wunsch alignment (`nw_align`) to handle insertions/deletions
before computing TP/FP/TN/FN. Two modes, both tokenizer-agnostic (callers pass
`vowel_groups` from the tokenizer):

- `compute_vowel_confusion` — alignment-based DEO (Devoicing Error Only)
- `compute_vowel_confusion_ccda` — requires matching phonetic context (C1·V·C2)

KER (kana error rate) is a character error rate over the kana rendering of the phone3
output, punctuation excluded — see `scripts/kana.py` (conventions follow
[nyosegawa/hiragana-asr](https://github.com/nyosegawa/hiragana-asr)). Note: PER was
reported as "CER" in results before 2026-07-29.

### `scripts/` module layout

`scripts/` is on `sys.path` at runtime via `sys.path.insert(0, ROOT)` at the top of
entry-point scripts. Import as `from scripts.phone_tokenizer import ...`.

| Module | Role |
|--------|------|
| `phone_tokenizer.py` | Tokenizer hierarchy, phone3 vocab |
| `stratified_sampling.py` | Manifest builder + `load_splits()` for train |
| `devoicing_eval.py` | NW alignment + confusion matrix + plotting |
| `inference_r.py` | Shared decode/model-loading (`ConformerASR` copy) for romaji-rev checkpoints |
| `gradio_demo_r.py` | Shared Gradio UI helpers (plots, playhead) |
| `audio.py` | Pitch / RMS extraction for demo plots |
| `make_transcript_phone3_rev.py` | Preprocessing: `basic5000.yaml` → phone3 transcript + vocab |
| `apply_manual_corrections.py` | Propagates reviewed rows from the manual-check xlsx into transcript `.txt` files |
| `analyze_devoicing_ojt.py` | Devoicing distribution/frequency stats over `transcript_phone3_rev.txt` |
| `plot_devoicing_ojt.py` | Renders the CSVs/JSON from `analyze_devoicing_ojt.py` as standalone charts |
| `sweep_runner.py` | Per-GPU sequential runner + hang watchdog for the BS×LR×DW training sweep |

## Repository layout

```
train_r3.py               training entrypoint (phone3, stratified split)
inference_r3.py           CLI inference + devoicing evaluation
eval_phone3_testset.py    batch test-set evaluation over checkpoints
gradio_demo.py            browser demo (record / pick JSUT wav)
results.md                results tables
models/                   trained weights (git-lfs)
doc/                      tokenizer vocabularies + run notes
img/                      confusion matrices, distributions, architecture diagrams
dataset/jsut_ver1.1/basic5000/
  wav/                    EMPTY — download JSUT and place the 5000 wavs here
  transcript_phone3_rev.txt      training/eval transcript (devoiced I/U)
  stratified_manifest.csv        train/val/test split manifest (authoritative, see above)
```
